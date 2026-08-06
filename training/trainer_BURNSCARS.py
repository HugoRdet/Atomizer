"""
BurnScars Trainer (SKIP variant) for Atomizer
================================================

Ported from Model_SenFlood_Skip. The encoder is shared across tasks
(Atomiser_Senflood_Skip — name is historical, architecture is task-agnostic
per the Atomizer-IO methodology), so this is a near-identical port: renamed
class/log strings, same loss/metrics/optimizer/sliding-window guard logic.

IMPORTANT — sliding window: unchanged from Sen1Floods11. BurnScars is also
512x512 (HLS common grid, see BurnScarsSkipDataset.IMG_SIZE), so the same
guard against use_decoder_skip + slide=True applies, and the same
recommendation (trainer.slide=False, decode whole image) holds.

FLAG — class imbalance: BurnScars is a heavily imbalanced binary task (burn
scars are a small minority of pixels). This port keeps the loss/metrics
exactly as in Sen1Floods11Skip (plain CrossEntropyLoss, macro-averaged
IoU/accuracy with ignore_index=255) since no prior BurnScars trainer was
available to diff against. If your old BurnScars runs used a weighted
CrossEntropyLoss, Dice loss, or focal loss to handle the imbalance, that
logic is NOT present here — flagging explicitly so it isn't silently lost.
Macro IoU on a 2-class problem will heavily penalize a model that ignores
the minority (burn) class, so it's a reasonable default metric even without
loss reweighting, but you may still want to add class weights to self.loss
if training collapses to majority-class prediction.

All changes vs Model_SenFlood_Skip are tagged  # >>> RENAME (cosmetic) or
# >>> FLAG (behavioral, needs your confirmation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

# >>> RENAME: same encoder class, shared across tasks
from training.atomiser.Atomiser_senflood_skip import Atomiser_Senflood_Skip
from training.atomiser.error_supervision import (
    compute_latent_errors,
    compute_error_predictor_loss,
)
from training.utils.datasets.sliding_window import stitch_predictions


class Model_BurnScars_Skip(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table,
                 class_names=None):
        super().__init__()
        self.strict_loading = False
        self.config         = config
        self.transform      = transform
        self.wand           = wand
        self.name           = name
        self.lookup_table   = lookup_table

        self.num_classes  = config["trainer"]["num_classes"]
        self.ignore_index = 255
        self.use_sliding  = config["trainer"].get("slide", False)

        # Guard unchanged from Sen1Floods11: BurnScars is also 512x512
        # (see BurnScarsSkipDataset.IMG_SIZE), so _forward_crop dropping
        # query_token_idx/valid is the same silent-disable risk.
        use_decoder_skip = config["Atomiser"].get("use_decoder_skip", False)
        if use_decoder_skip and self.use_sliding:
            raise ValueError(
                "use_decoder_skip=True is incompatible with slide=True: the "
                "sliding-window path drops query_token_idx/valid and would "
                "silently disable the skip cascade at eval time, invalidating "
                "the comparison. Set trainer.slide=False for the skip run "
                "(BurnScars is 512x512 and decodes whole)."
            )

        self.class_names = (
            class_names
            or config["trainer"].get("class_names", None)
            or [f"class_{i}" for i in range(self.num_classes)]
        )

        # =====================================================================
        # METRICS
        # =====================================================================
        self.metric_IoU_train = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_IoU_val = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_IoU_test = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index,
        )
        self.metric_acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_acc_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index,
        )

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood_Skip(
            config=self.config, lookup_table=self.lookup_table)

        # =====================================================================
        # LOSS
        # =====================================================================
        # >>> FLAG: plain CE, no class weighting. See module docstring.
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # =====================================================================
        # ERROR PREDICTOR SUPERVISION
        # =====================================================================
        self.use_error_predictor = config["Atomiser"].get(
            "use_error_predictor", False)
        if self.use_error_predictor:
            self.lambda_error = float(
                config["Atomiser"].get("lambda_error", 0.1))
            self.error_warmup = int(
                config["Atomiser"].get("error_supervision_warmup_epochs", 0))
            print(f"[Trainer] Error predictor supervision ENABLED "
                  f"(lambda={self.lambda_error}, warmup={self.error_warmup} epochs)")
        else:
            print(f"[Trainer] Error predictor supervision DISABLED")

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _should_supervise_error(self) -> bool:
        return (self.use_error_predictor
                and self.current_epoch >= self.error_warmup)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False, return_for_error=False):
        return self.encoder(
            batch, training=training, return_for_error=return_for_error)

    # =========================================================================
    # SHARED STEP LOGIC
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        supervise_error = training and self._should_supervise_error()

        result = self.forward(
            batch, training=training, return_for_error=supervise_error)

        y_hat  = result["predictions"] if isinstance(result, dict) else result
        labels = batch["queries"][:, :, 4].long()

        y_hat_flat  = rearrange(y_hat,  "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        seg_loss    = self.loss(y_hat_flat, labels_flat)

        total_loss = seg_loss

        if supervise_error and isinstance(result, dict):
            predicted_errors = result.get("predicted_errors")
            topk_indices     = result.get("topk_indices")
            topk_dists_sq    = result.get("topk_dists_sq")
            num_latents      = result.get("num_latents")

            if (predicted_errors is not None
                    and topk_indices is not None
                    and topk_dists_sq is not None):
                zone_error, valid_mask = compute_latent_errors(
                    logits        = y_hat.detach(),
                    labels        = labels,
                    topk_indices  = topk_indices,
                    topk_dists_sq = topk_dists_sq,
                    num_latents   = num_latents,
                    ignore_index  = self.ignore_index,
                )
                L_pred = predicted_errors.shape[1]
                error_loss = compute_error_predictor_loss(
                    predicted_errors = predicted_errors,
                    zone_error       = zone_error[:, :L_pred],
                    valid_mask       = valid_mask[:, :L_pred],
                )
                total_loss = seg_loss + self.lambda_error * error_loss
                self.log("train_error_loss", error_loss,
                         on_step=False, on_epoch=True, logger=True)

        preds = torch.argmax(y_hat, dim=-1)
        return total_loss, seg_loss, preds, labels

    # =========================================================================
    # SLIDING WINDOW INFERENCE
    # =========================================================================
    # Kept intact (guarded off when use_decoder_skip=True, see __init__), so
    # this trainer remains a drop-in for the non-skip BurnScars baseline too.

    def _forward_crop(self, batch, crop_idx):
        mini_batch = {
            "groups":       {},
            "queries":      batch["queries"][crop_idx:crop_idx + 1],
            "queries_mask": batch["queries_mask"][crop_idx:crop_idx + 1],
        }
        for res, grp in batch["groups"].items():
            mini_batch["groups"][res] = {
                "tokens": grp["tokens"][crop_idx:crop_idx + 1],
                "mask":   grp["mask"][crop_idx:crop_idx + 1],
                "shape":  grp["shape"],
            }
        result = self.forward(mini_batch, training=False)
        return result["predictions"] if isinstance(result, dict) else result

    def _sliding_window_step(self, batch):
        num_crops      = batch["queries"].shape[0]
        positions      = batch["crop_positions"]
        crop_h, crop_w = batch["crop_size"]
        full_h, full_w = batch["full_size"]

        crop_logits_list = []
        for i in range(num_crops):
            with torch.no_grad():
                logits = self._forward_crop(batch, i)
            crop_logits_list.append(logits.squeeze(0))

        preds_full, logits_avg = stitch_predictions(
            crop_logits_list=crop_logits_list,
            crop_positions=positions,
            crop_h=crop_h, crop_w=crop_w,
            full_h=full_h, full_w=full_w,
            num_classes=self.num_classes,
        )
        return preds_full, batch["label"].to(self.device), logits_avg

    # =========================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        total_loss, seg_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=True)

        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)

        self.log("train_loss",       total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_class_loss", seg_loss,   on_step=False, on_epoch=True,
                 logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            loss  = self.loss(logits_avg.unsqueeze(0), label_full.unsqueeze(0))
            valid = label_full != self.ignore_index
            if valid.sum() > 0:
                self.metric_IoU_val.update(preds_full[valid], label_full[valid])
                self.metric_acc_val.update(preds_full[valid], label_full[valid])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            return loss

        _, seg_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False)
        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)
        self.log("val_loss", seg_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return seg_loss

    def test_step(self, batch, batch_idx):
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            loss  = self.loss(logits_avg.unsqueeze(0), label_full.unsqueeze(0))
            valid = label_full != self.ignore_index
            if valid.sum() > 0:
                self.metric_IoU_test.update(preds_full[valid], label_full[valid])
                self.metric_acc_test.update(preds_full[valid], label_full[valid])
            self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
            return loss

        _, seg_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False)
        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)
        self.log("test_loss", seg_loss, on_step=False, on_epoch=True, logger=True)
        return seg_loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU",     self.metric_IoU_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.metric_acc_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_mIoU",     self.metric_IoU_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self.metric_acc_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou = self.metric_IoU_test.compute()
        test_acc = self.metric_acc_test.compute()

        self.log("test_mIoU",     test_iou.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc.mean(), on_epoch=True, logger=True)

        for i, name in enumerate(self.class_names):
            if i < len(test_iou):
                self.log(f"test_IoU_{name}", test_iou[i], on_epoch=True, logger=True)
            if i < len(test_acc):
                self.log(f"test_acc_{name}", test_acc[i], on_epoch=True, logger=True)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}")
        print(f"  mIoU:     {test_iou.mean():.4f}")
        print(f"  Accuracy: {test_acc.mean():.4f}")
        for i, name in enumerate(self.class_names):
            if i < len(test_iou):
                print(f"  IoU {name}: {test_iou[i]:.4f}")
        print(f"{'='*60}\n")

        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()

    # =========================================================================
    # MODEL SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[BurnScars] Model saved to {file_path}")  # >>> RENAME

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[BurnScars] Model loaded from {file_path}")  # >>> RENAME

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[Trainer] WARN: cannot estimate total_steps. "
                  f"Falling back to {fallback}.")
            return fallback

        print(f"[Trainer] total_steps estimate: {est}")
        return est

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps  = self._compute_total_steps()
        warmup_steps = self.config.get("optimizer", {}).get(
            "warmup_steps", max(1, int(0.05 * total_steps))
        )

        print(f"[Trainer] LR schedule final: "
              f"total_steps={total_steps}, warmup={warmup_steps}, "
              f"peak_lr={self.lr}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
            },
        }
