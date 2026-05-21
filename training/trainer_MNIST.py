"""
MNIST Trainer for Atomizer
==========================

Per-image classification trainer. Mirrors the structure of Model_SenFlood but
adapted for the differences between segmentation and classification:

  - labels come from batch["label"] (per-image scalar), not queries col 4
  - the encoder is expected to output [B, num_classes] after attention pooling;
    if it returns [B, M, num_classes] we mean-pool over the query dimension
    (respecting queries_mask) as a fallback
  - no ignore_index, no sliding window, no per-query loss reshape
  - metrics: Accuracy + F1Score (multiclass); per-class accuracy in test

Config:
    config["trainer"]["num_classes"]:   int  (default 10)
    config["trainer"]["total_steps"]:   int  (optional override)

    config["Atomiser"]["use_error_predictor"]: ignored. Classification has no
        spatial loss to distribute, so the mechanism is moot. If True in
        config, a warning is printed and it stays disabled.

Token format (unchanged, 8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup
from training.atomiser import Atomiser_Senflood


class Model_MNIST(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table,
                 class_names=None):
        super().__init__()
        self.strict_loading = False
        self.config         = config
        self.transform      = transform
        self.wand           = wand
        self.name           = name
        self.lookup_table   = lookup_table

        self.num_classes = config["trainer"].get("num_classes", 10)

        # Class names for logging — default to digit labels 0..9
        self.class_names = (
            class_names
            or config["trainer"].get("class_names", None)
            or [str(i) for i in range(self.num_classes)]
        )

        # =====================================================================
        # METRICS
        # =====================================================================
        self.metric_acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, average="micro",
        )
        self.metric_acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, average="micro",
        )
        self.metric_acc_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, average="micro",
        )
        self.metric_acc_test_per_class = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, average=None,
        )
        self.metric_f1_train = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro",
        )
        self.metric_f1_val = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro",
        )
        self.metric_f1_test = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro",
        )

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood(
            config=self.config, lookup_table=self.lookup_table)

        # =====================================================================
        # LOSS
        # =====================================================================
        # No ignore_index for MNIST — every sample has a valid digit class.
        self.loss = nn.CrossEntropyLoss()

        # =====================================================================
        # ERROR PREDICTOR SUPERVISION — N/A for classification
        # =====================================================================
        # Sen1Floods11 distributes per-pixel CE loss to spatially-nearby
        # latents via topk_indices / topk_dists_sq. For MNIST there's a single
        # scalar label per image, so there's no spatial loss to distribute and
        # the mechanism is moot. We disable it explicitly and warn if
        # requested in config.
        if config.get("Atomiser", {}).get("use_error_predictor", False):
            print(f"[Trainer-MNIST] WARNING: use_error_predictor=True in config "
                  f"but classification has no spatial loss to distribute. Disabled.")
        self.use_error_predictor = False

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False):
        # task="classification" routes through self.classify() →
        # LatentAttentionPooling → [B, num_classes], skipping the per-query
        # decoder entirely. The default ("reconstruction") would build a
        # [B*M, ...] cross-attention call that blows past the SDPA kernel's
        # grid limit for MNIST batch sizes (B=256, M=784 → BM ≈ 200k).
        return self.encoder(batch, training=training, task="classification")

    # =========================================================================
    # SHARED STEP LOGIC
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        result = self.forward(batch, training=training)
        y_hat  = result["predictions"] if isinstance(result, dict) else result

        # Expected shape: [B, num_classes]. If the encoder returns per-query
        # logits [B, M, num_classes] (e.g. it still uses the segmentation
        # head), mean-pool over the query dimension while respecting
        # queries_mask (0 = valid, like Sen1Floods11).
        if y_hat.dim() == 3:
            qmask = batch.get("queries_mask", None)
            if qmask is not None:
                valid = (qmask == 0).float().unsqueeze(-1)               # [B, M, 1]
                y_hat = (y_hat * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            else:
                y_hat = y_hat.mean(dim=1)

        labels = batch["label"].long()
        loss   = self.loss(y_hat, labels)
        preds  = torch.argmax(y_hat, dim=-1)
        return loss, preds, labels

    # =========================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._compute_loss_and_preds(batch, training=True)
        self.metric_acc_train.update(preds, labels)
        self.metric_f1_train.update(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        self.metric_acc_val.update(preds, labels)
        self.metric_f1_val.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        self.metric_acc_test.update(preds, labels)
        self.metric_acc_test_per_class.update(preds, labels)
        self.metric_f1_test.update(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.metric_acc_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.metric_f1_train.compute(),
                 on_epoch=True, logger=True)
        self.metric_acc_train.reset()
        self.metric_f1_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.metric_acc_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.metric_f1_val.compute(),
                 on_epoch=True, logger=True)
        self.metric_acc_val.reset()
        self.metric_f1_val.reset()

    def on_test_epoch_end(self):
        test_acc           = self.metric_acc_test.compute()
        test_acc_per_class = self.metric_acc_test_per_class.compute()
        test_f1            = self.metric_f1_test.compute()

        self.log("test_accuracy", test_acc, on_epoch=True, logger=True)
        self.log("test_f1",       test_f1,  on_epoch=True, logger=True)

        for i, name in enumerate(self.class_names):
            if i < len(test_acc_per_class):
                self.log(f"test_acc_{name}", test_acc_per_class[i],
                         on_epoch=True, logger=True)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (MNIST)")
        print(f"{'='*60}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1:       {test_f1:.4f}")
        for i, name in enumerate(self.class_names):
            if i < len(test_acc_per_class):
                print(f"  Acc[{name}]: {test_acc_per_class[i]:.4f}")
        print(f"{'='*60}\n")

        self.metric_acc_test.reset()
        self.metric_acc_test_per_class.reset()
        self.metric_f1_test.reset()

    # =========================================================================
    # MODEL SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[MNIST] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[MNIST] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def _compute_total_steps(self) -> int:
        """
        Compute total optimizer steps for the LR scheduler.

        Same priority chain as Model_SenFlood — see that file for the full
        rationale on why we can't trust Lightning's
        `estimated_stepping_batches` / `num_training_batches`:

          1. Config override `trainer.total_steps`
          2. Dataset-based manual computation
          3. Fallback to `num_training_batches` as-is
          4. Last resort: `estimated_stepping_batches`
        """
        # 1. Config override
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[Trainer-MNIST] total_steps override from config: {override}")
            return int(override)

        max_epochs  = self.trainer.max_epochs
        accum       = max(1, int(self.trainer.accumulate_grad_batches))
        num_devices = max(1, self.trainer.num_devices * self.trainer.num_nodes)

        # GradientAccumulationScheduler may run after configure_optimizers
        if accum == 1:
            config_accum = int(
                self.config.get("trainer", {}).get("grad_accum", 1)
            )
            if config_accum > 1:
                print(f"[Trainer-MNIST] Using grad_accum={config_accum} from config "
                      f"(trainer.accumulate_grad_batches still 1 at optimizer setup)")
                accum = config_accum

        batch_size = int(
            self.config.get("trainer", {}).get(
                "train_batch_size",
                self.config.get("trainer", {}).get("batchsize", 1)
            )
        )

        # 2. Dataset-based manual computation
        dataset_len = None
        try:
            dm = self.trainer.datamodule
            train_ds = getattr(dm, "train_dataset", None)
            if train_ds is not None and hasattr(train_ds, "__len__"):
                dataset_len = len(train_ds)
        except Exception:
            pass

        num_training_batches_raw = None
        try:
            ntb = self.trainer.num_training_batches
            if ntb is not None and ntb != float("inf") and ntb > 0:
                num_training_batches_raw = int(ntb)
        except Exception:
            pass

        if dataset_len is not None:
            samples_per_worker = dataset_len // num_devices
            batches_per_worker = samples_per_worker // batch_size
            steps_per_epoch    = max(1, batches_per_worker // accum)
            source = "dataset"
        elif num_training_batches_raw is not None:
            steps_per_epoch = max(1, num_training_batches_raw // accum)
            source = "num_training_batches (as-is)"
        else:
            est = int(self.trainer.estimated_stepping_batches)
            print(f"[Trainer-MNIST] [WARNING] No dataset length, no "
                  f"num_training_batches. Using estimated_stepping_batches={est}")
            return est

        total_steps = steps_per_epoch * max_epochs

        try:
            lightning_est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            lightning_est = -1

        print(f"[Trainer-MNIST] LR schedule computation (source: {source}):")
        if dataset_len is not None:
            print(f"[Trainer-MNIST]   dataset length:                 {dataset_len}")
            print(f"[Trainer-MNIST]   batch size:                     {batch_size}")
            print(f"[Trainer-MNIST]   samples per worker:             {samples_per_worker}")
            print(f"[Trainer-MNIST]   batches per worker:             {batches_per_worker}")
        if num_training_batches_raw is not None:
            print(f"[Trainer-MNIST]   num_training_batches (raw):     {num_training_batches_raw}")
        print(f"[Trainer-MNIST]   num_devices:                    {num_devices}")
        print(f"[Trainer-MNIST]   accumulate_grad_batches:        {accum}")
        print(f"[Trainer-MNIST]   max_epochs:                     {max_epochs}")
        print(f"[Trainer-MNIST]   steps_per_epoch:                {steps_per_epoch}")
        print(f"[Trainer-MNIST]   total_steps (manual):           {total_steps}")
        print(f"[Trainer-MNIST]   total_steps (Lightning est):    {lightning_est}")

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

        print(f"[Trainer-MNIST] LR schedule final: "
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