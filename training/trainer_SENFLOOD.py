"""
Sen1Floods11 Trainer for Atomizer
===================================

Semantic segmentation trainer with optional sliding window inference
and error predictor supervision via soft cross-entropy per latent zone.

Config:
    config["trainer"]["slide"]:   bool  (default False)
    config["trainer"]["total_steps"]: int (optional, override)

    config["Atomiser"]["use_error_predictor"]:             bool  (default False)
    config["Atomiser"]["lambda_error"]:                    float (default 0.1)
    config["Atomiser"]["error_supervision_warmup_epochs"]: int   (default 0)

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.atomiser.error_supervision import (
    compute_latent_errors,
    compute_error_predictor_loss,
)
from training.utils.datasets.sliding_window import stitch_predictions


class Model_SenFlood(pl.LightningModule):
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

        # Class names for logging — can be overridden per dataset
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
        self.encoder = Atomiser_Senflood(
            config=self.config, lookup_table=self.lookup_table)

        # =====================================================================
        # LOSS
        # =====================================================================
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
        print(f"[SenFlood] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[SenFlood] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def _compute_total_steps(self) -> int:
        """
        Compute total optimizer steps for the LR scheduler.

        Why manual: Lightning's `estimated_stepping_batches` and
        `num_training_batches` can both be wrong depending on:
          - DDP + `use_distributed_sampler=False` + manual DistributedSampler
          - Lightning version
          - Whether num_training_batches is set at configure_optimizers time

        We compute from dataset length directly using the DDP formula:
            total_samples_per_worker = len(dataset) // num_devices  (DistributedSampler drops remainder)
            batches_per_worker       = total_samples_per_worker // batch_size
            steps_per_epoch          = batches_per_worker // accumulate_grad_batches
            total_steps              = steps_per_epoch × max_epochs

        Priority:
          1. Config override `trainer.total_steps`
          2. Dataset-based manual computation
          3. Fallback: num_training_batches × max_epochs / accum
          4. Last resort: estimated_stepping_batches
        """
        # 1. Config override
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[Trainer] total_steps override from config: {override}")
            return int(override)

        max_epochs  = self.trainer.max_epochs
        accum       = max(1, int(self.trainer.accumulate_grad_batches))
        num_devices = max(1, self.trainer.num_devices * self.trainer.num_nodes)

        # Fallback: if trainer.accumulate_grad_batches is still 1 but config
        # specifies grad_accum > 1, use config value. This handles the case
        # where GradientAccumulationScheduler callback runs after
        # configure_optimizers (trainer attr not yet updated).
        if accum == 1:
            config_accum = int(
                self.config.get("trainer", {}).get("grad_accum", 1)
            )
            if config_accum > 1:
                print(f"[Trainer] Using grad_accum={config_accum} from config "
                      f"(trainer.accumulate_grad_batches still 1 at optimizer setup)")
                accum = config_accum

        # Read batch size from config
        batch_size = int(
            self.config.get("trainer", {}).get(
                "train_batch_size",
                self.config.get("trainer", {}).get("batchsize", 1)
            )
        )

        # 2. Dataset-based manual computation (most reliable)
        dataset_len = None
        try:
            dm = self.trainer.datamodule
            train_ds = getattr(dm, "train_dataset", None)
            if train_ds is not None and hasattr(train_ds, "__len__"):
                dataset_len = len(train_ds)
        except Exception:
            pass

        # Also try: self.trainer.num_training_batches and the dataloader
        num_training_batches_raw = None
        try:
            ntb = self.trainer.num_training_batches
            if ntb is not None and ntb != float("inf") and ntb > 0:
                num_training_batches_raw = int(ntb)
        except Exception:
            pass

        # Pick the most trustworthy source
        if dataset_len is not None:
            # Ground truth from dataset
            samples_per_worker = dataset_len // num_devices
            batches_per_worker = samples_per_worker // batch_size
            steps_per_epoch    = max(1, batches_per_worker // accum)
            source = "dataset"
        elif num_training_batches_raw is not None:
            # Fallback: use Lightning's count, with a heuristic check.
            # If num_training_batches >> expected per-worker count, it's the
            # full-dataset count, so divide by num_devices.
            # Heuristic: if `num_training_batches * num_devices` is close to
            # the Lightning estimate per epoch, it's per-worker. Otherwise
            # it's full-dataset.
            # Simpler: assume it's full-dataset if num_devices > 1 and there's
            # no clear per-worker behavior. This matches Lightning versions
            # where use_distributed_sampler=False means num_training_batches
            # reflects the raw dataloader length (which for manual
            # DistributedSampler is already per-worker).
            #
            # Safest approach: compute both and pick the one that divides evenly.
            assumed_per_worker = num_training_batches_raw
            assumed_full       = num_training_batches_raw // num_devices

            steps_per_epoch = max(1, assumed_per_worker // accum)
            source = "num_training_batches (as-is)"

            # Note: if the user reports training ends before LR reaches 0,
            # this is where the overcount comes from. User can set
            # trainer.total_steps explicitly in config to override.
        else:
            # Last resort
            est = int(self.trainer.estimated_stepping_batches)
            total_steps = est
            print(f"[Trainer] [WARNING] No dataset length, no num_training_batches. "
                  f"Using estimated_stepping_batches={est}")
            return total_steps

        total_steps = steps_per_epoch * max_epochs

        # Diagnostic
        try:
            lightning_est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            lightning_est = -1

        print(f"[Trainer] LR schedule computation (source: {source}):")
        if dataset_len is not None:
            print(f"[Trainer]   dataset length:                 {dataset_len}")
            print(f"[Trainer]   batch size:                     {batch_size}")
            print(f"[Trainer]   samples per worker:             {samples_per_worker}")
            print(f"[Trainer]   batches per worker:             {batches_per_worker}")
        if num_training_batches_raw is not None:
            print(f"[Trainer]   num_training_batches (raw):     {num_training_batches_raw}")
        print(f"[Trainer]   num_devices:                    {num_devices}")
        print(f"[Trainer]   accumulate_grad_batches:        {accum}")
        print(f"[Trainer]   max_epochs:                     {max_epochs}")
        print(f"[Trainer]   steps_per_epoch:                {steps_per_epoch}")
        print(f"[Trainer]   total_steps (manual):           {total_steps}")
        print(f"[Trainer]   total_steps (Lightning est):    {lightning_est}")

        return total_steps

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