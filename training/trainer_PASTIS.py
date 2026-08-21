"""
PASTIS Crop Segmentation Trainer
==================================

PyTorch Lightning module for training Atomizer (± LTAE ± decoder skip ±
temporal transformer) on PASTIS-HD for multi-temporal crop type segmentation.

Encoder selection (priority order):
    1. config["Atomiser"]["use_temporal_transformer"] == True
         -> AtomiserTemporal   (per-timestep Atomiser_Senflood_Skip encoding
                                 + TemporalTransformer aggregation; internally
                                 wraps Atomiser_Senflood_Skip, so set
                                 use_decoder_skip: True too if you want the
                                 pixel-skip cascade active)
    2. config["Atomiser"]["use_decoder_skip"] == True
         -> Atomiser_Senflood_Skip   (decoder pixel-skip cascade)
    3. config["Atomiser"]["use_ltae"] == True
         -> AtomiserLTAE             (per-timestamp encoding + LTAE)
    4. otherwise
         -> Atomiser_Senflood        (no temporal reasoning; the 0.38 baseline)

Metric convention:
    mIoU is averaged over crop classes 1..18 (excluding background class 0).

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.atomiser.atomiser_ltae_wrapper import AtomiserLTAE
# >>> SKIP: decoder pixel-skip variant (adjust path if the class lives elsewhere)
from training.atomiser.Atomiser_senflood_skip import Atomiser_Senflood_Skip
# >>> TEMPORAL: per-timestep encoding + TemporalTransformer aggregation
from training.atomiser.Atomiser_temporal import AtomiserTemporal
from training.atomiser.error_supervision import (
    compute_latent_errors,
    compute_error_predictor_loss,
)


TASK_NAME = "pastis_segmentation"


class PASTISTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for PASTIS crop segmentation."""

    CROP_NAMES = [
        "background",
        "meadow",
        "soft_winter_wheat",
        "corn",
        "winter_barley",
        "winter_rapeseed",
        "spring_barley",
        "sunflower",
        "grapevine",
        "beet",
        "soy",
        "sorghum",
        "flax",
        "protein_crops",
        "other_cereals",
        "fruits_veg",
        "other_crops",
        "grassland",
        "shrub_forest",
    ]

    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config         = config
        self.transform      = transform
        self.wand           = wand
        self.name           = name
        self.lookup_table   = lookup_table

        self.num_classes  = config["trainer"]["num_classes"]
        self.ignore_index = 255

        # =====================================================================
        # METRICS — per-class, mIoU over classes 1-18 (exclude background)
        # =====================================================================
        for split in ("train", "val", "test"):
            setattr(self, f"metric_IoU_{split}", torchmetrics.JaccardIndex(
                task="multiclass", num_classes=self.num_classes,
                average=None, ignore_index=self.ignore_index,
            ))
            setattr(self, f"metric_acc_{split}", torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes,
                average=None, ignore_index=self.ignore_index,
            ))

        # =====================================================================
        # MODEL — encoder selection
        #   use_temporal_transformer > use_decoder_skip > use_ltae > base
        # =====================================================================
        self.use_temporal_transformer = config["Atomiser"].get(
            "use_temporal_transformer", False)
        self.use_decoder_skip = config["Atomiser"].get("use_decoder_skip", False)
        self.use_ltae         = config["Atomiser"].get("use_ltae", True)

        if self.use_temporal_transformer:
            print("[PASTIS] Using AtomiserTemporal "
                  "(per-timestep Atomiser_Senflood_Skip encoding + "
                  "TemporalTransformer aggregation)")
            if not self.use_decoder_skip:
                print("[PASTIS] [WARNING] use_temporal_transformer=True but "
                      "use_decoder_skip=False — the SKIP cascade "
                      "(query_token_idx/query_token_valid) will be ignored "
                      "by AtomiserTemporal.forward. Set "
                      "use_decoder_skip=True if you want it active.")
            self.encoder = AtomiserTemporal(
                config=self.config, lookup_table=self.lookup_table)
        elif self.use_decoder_skip:
            print("[PASTIS] Using Atomiser_Senflood_Skip "
                  "(decoder pixel-skip cascade)")
            self.encoder = Atomiser_Senflood_Skip(
                config=self.config, lookup_table=self.lookup_table)
        elif self.use_ltae:
            print("[PASTIS] Using AtomiserLTAE "
                  "(per-timestamp encoding + LTAE)")
            self.encoder = AtomiserLTAE(
                config=self.config, lookup_table=self.lookup_table)
        else:
            print("[PASTIS] Using Atomiser_Senflood (no temporal reasoning)")
            self.encoder = Atomiser_Senflood(
                config=self.config, lookup_table=self.lookup_table)

        # =====================================================================
        # LOSS
        # =====================================================================
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # =====================================================================
        # ERROR PREDICTOR SUPERVISION (zone-CE, same setup as Sen1Floods11)
        # Note: this is a no-op with AtomiserTemporal — its forward() does not
        # return predicted_errors/topk_indices/topk_dists_sq, so the block in
        # _compute_loss_and_preds below silently skips adding the error loss.
        # Leave use_error_predictor: false in temporal-transformer configs.
        # =====================================================================
        self.use_error_predictor = config["Atomiser"].get(
            "use_error_predictor", False)
        if self.use_error_predictor:
            self.lambda_error = float(
                config["Atomiser"].get("lambda_error", 0.1))
            self.error_warmup = int(
                config["Atomiser"].get("error_supervision_warmup_epochs", 0))
            print(f"[PASTIS] Error predictor supervision ENABLED "
                  f"(lambda={self.lambda_error}, warmup={self.error_warmup} epochs)")
            if self.use_temporal_transformer:
                print("[PASTIS] [WARNING] use_error_predictor=True has no "
                      "effect with AtomiserTemporal (no-op).")
        else:
            print(f"[PASTIS] Error predictor supervision DISABLED")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _should_supervise_error(self) -> bool:
        return (self.use_error_predictor
                and self.current_epoch >= self.error_warmup)

    def _ensure_flat_batch(self, batch):
        """Lift queries/queries_mask from batch['tasks'] to the top level.
        Preserves all other top-level keys (incl. query_token_idx /
        query_token_valid for the skip, and time_indices for the temporal
        transformer) via the shallow dict() copy."""
        if "queries" in batch and batch["queries"] is not None:
            return batch
        if "tasks" in batch and isinstance(batch["tasks"], dict) and len(batch["tasks"]) > 0:
            task_data = next(iter(batch["tasks"].values()))
            batch = dict(batch)
            batch["queries"]      = task_data["queries"]
            batch["queries_mask"] = task_data["queries_mask"]
        return batch

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False, return_for_error=False):
        batch = self._ensure_flat_batch(batch)
        return self.encoder(
            batch, training=training, return_for_error=return_for_error)

    # =========================================================================
    # SHARED STEP LOGIC
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        supervise_error = training and self._should_supervise_error()

        result = self.forward(
            batch, training=training, return_for_error=supervise_error)

        if isinstance(result, dict):
            y_hat = result.get("predictions")
        else:
            y_hat = result
            result = {}

        batch = self._ensure_flat_batch(batch)
        queries = batch["queries"]
        labels  = queries[:, :, 4].long()

        y_hat_flat  = rearrange(y_hat,  "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        class_loss  = self.loss(y_hat_flat, labels_flat)

        total_loss = class_loss

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
                total_loss = class_loss + self.lambda_error * error_loss
                self.log("train_error_loss", error_loss,
                         on_step=False, on_epoch=True, logger=True)

        preds = torch.argmax(y_hat, dim=-1)
        return total_loss, class_loss, preds, labels

    # =========================================================================
    # TRAIN / VAL / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=True)

        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)

        self.log("train_loss",       total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_class_loss", class_loss, on_step=False, on_epoch=True,
                 logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        _, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False)

        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)

        self.log("val_loss", class_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return class_loss

    def test_step(self, batch, batch_idx):
        _, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False)

        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)

        self.log("test_loss", class_loss, on_step=False, on_epoch=True,
                 logger=True)
        return class_loss

    # =========================================================================
    # METRIC HELPERS
    # =========================================================================

    def _compute_crop_miou(self, per_class_iou: torch.Tensor) -> torch.Tensor:
        """mIoU over crop classes 1..18 (exclude background 0)."""
        return per_class_iou[1:19].mean()

    def _compute_crop_acc(self, per_class_acc: torch.Tensor) -> torch.Tensor:
        """Mean accuracy over crop classes 1..end."""
        return per_class_acc[1:].mean()

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        iou = self.metric_IoU_train.compute()
        acc = self.metric_acc_train.compute()

        self.log("train_mIoU",     self._compute_crop_miou(iou),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self._compute_crop_acc(acc),
                 on_epoch=True, prog_bar=True, logger=True)

        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        iou = self.metric_IoU_val.compute()
        acc = self.metric_acc_val.compute()

        self.log("val_mIoU",     self._compute_crop_miou(iou),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self._compute_crop_acc(acc),
                 on_epoch=True, prog_bar=True, logger=True)

        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou = self.metric_IoU_test.compute()
        test_acc = self.metric_acc_test.compute()

        self.log("test_mIoU",     self._compute_crop_miou(test_iou),
                 on_epoch=True, logger=True)
        self.log("test_accuracy", self._compute_crop_acc(test_acc),
                 on_epoch=True, logger=True)

        for i, name in enumerate(self.CROP_NAMES):
            if i < len(test_iou):
                self.log(f"test_IoU_{name}", test_iou[i],
                         on_epoch=True, logger=True)
            if i < len(test_acc):
                self.log(f"test_acc_{name}", test_acc[i],
                         on_epoch=True, logger=True)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (crop mIoU = mean over classes 1-18)")
        print(f"{'='*60}")
        print(f"  mIoU:     {self._compute_crop_miou(test_iou):.4f}")
        print(f"  Accuracy: {self._compute_crop_acc(test_acc):.4f}")
        for i, name in enumerate(self.CROP_NAMES):
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
        print(f"[PASTIS] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[PASTIS] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[PASTIS] total_steps override from config: {override}")
            return int(override)

        max_epochs  = self.trainer.max_epochs
        accum       = max(1, int(self.trainer.accumulate_grad_batches))
        num_devices = max(1, self.trainer.num_devices * self.trainer.num_nodes)

        if accum == 1:
            cfg_trainer = self.config.get("trainer", {})
            config_accum = int(
                cfg_trainer.get("accumulate_grad_batches",
                cfg_trainer.get("grad_accum", 1))
            )
            if config_accum > 1:
                print(f"[PASTIS] Using accumulate_grad_batches={config_accum} "
                      f"from config (trainer attr still 1)")
                accum = config_accum

        batch_size = int(
            self.config.get("trainer", {}).get(
                "train_batch_size",
                self.config.get("trainer", {}).get("batchsize", 1)
            )
        )

        dataset_len = None
        try:
            dm = self.trainer.datamodule
            train_ds = getattr(dm, "train_dataset", None)
            if train_ds is not None and hasattr(train_ds, "__len__"):
                dataset_len = len(train_ds)
        except Exception:
            pass

        if dataset_len is not None:
            samples_per_worker = dataset_len // num_devices
            batches_per_worker = samples_per_worker // batch_size
            steps_per_epoch    = max(1, batches_per_worker // accum)
            total_steps        = steps_per_epoch * max_epochs

            try:
                lightning_est = int(self.trainer.estimated_stepping_batches)
            except Exception:
                lightning_est = -1

            print(f"[PASTIS] LR schedule (source: dataset):")
            print(f"[PASTIS]   dataset length:          {dataset_len}")
            print(f"[PASTIS]   batch size:              {batch_size}")
            print(f"[PASTIS]   samples per worker:      {samples_per_worker}")
            print(f"[PASTIS]   batches per worker:      {batches_per_worker}")
            print(f"[PASTIS]   num_devices:             {num_devices}")
            print(f"[PASTIS]   grad_accum:              {accum}")
            print(f"[PASTIS]   max_epochs:              {max_epochs}")
            print(f"[PASTIS]   steps_per_epoch:         {steps_per_epoch}")
            print(f"[PASTIS]   total_steps (manual):    {total_steps}")
            print(f"[PASTIS]   total_steps (Lightning): {lightning_est}")
            return total_steps

        try:
            ntb = int(self.trainer.num_training_batches)
            if ntb > 0:
                steps_per_epoch = max(1, ntb // accum)
                total_steps = steps_per_epoch * max_epochs
                print(f"[PASTIS] LR schedule (source: num_training_batches): "
                      f"total_steps={total_steps}")
                return total_steps
        except Exception:
            pass

        fallback = int(self.trainer.estimated_stepping_batches)
        print(f"[PASTIS] [WARNING] Using estimated_stepping_batches={fallback}")
        return fallback

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

        print(f"[PASTIS] LR schedule final: "
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
