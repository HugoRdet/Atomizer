"""
Sen1Floods11 Trainer for Atomizer
===================================

Semantic segmentation trainer with optional sliding window inference.

Config:
    config["trainer"]["slide"]: bool (default False)
        If True, val/test datasets return sliding window batches.
        The trainer detects these via batch["sliding"] == True and
        processes each crop independently, stitching predictions
        before computing metrics on the full tile.

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]

Batch format (normal):
    groups[res]["tokens"]: [B, N, 8]
    queries:               [B, M, 8]
    label:                 [B, H, W]

Batch format (sliding window, after collate):
    groups[res]["tokens"]: [num_crops, N, 8]   ← crops stacked on dim 0
    queries:               [num_crops, M, 8]
    label:                 [H, W]              ← full tile, no batch dim
    crop_positions:        [(y0, x0), ...]
    crop_size:             (crop_h, crop_w)
    full_size:             (full_h, full_w)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.utils.datasets.sliding_window import stitch_predictions


class Model_SenFlood(pl.LightningModule):
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

        # Sliding window inference for val/test
        self.use_sliding = config["trainer"].get("slide", False)

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
            average=None,    # per-class for test reporting
            ignore_index=self.ignore_index,
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
            average=None,    # per-class for test reporting
            ignore_index=self.ignore_index,
        )

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)
        self.loss     = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False):
        """Forward pass — delegates entirely to encoder."""
        return self.encoder(batch, training=training)

    # =========================================================================
    # SHARED STEP LOGIC
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        """
        Run forward, compute loss, return (total_loss, preds, labels).

        Labels come from column 4 of query tokens [B, M, 8] → [B, M].
        Logits are flattened to [B*M, C] before cross-entropy.
        """
        result = self.forward(batch, training=training)

        y_hat  = result["predictions"] if isinstance(result, dict) else result
        labels = batch["queries"][:, :, 4].long()  # col 4 = class label

        # Flatten for loss: [B, M, C] → [B*M, C],  [B, M] → [B*M]
        y_hat_flat  = rearrange(y_hat,   "b t c -> (b t) c")
        labels_flat = rearrange(labels,  "b n   -> (b n)")

        loss  = self.loss(y_hat_flat, labels_flat)
        preds = torch.argmax(y_hat, dim=-1)  # [B, M]

        return loss, preds, labels

    # =========================================================================
    # SLIDING WINDOW INFERENCE
    # =========================================================================

    def _forward_crop(self, batch, crop_idx):
        """
        Run forward on a single crop slice from a sliding window batch.
        Returns logits [1, M, C].
        """
        mini_batch = {
            "groups":      {},
            "queries":     batch["queries"][crop_idx:crop_idx + 1],
            "queries_mask": batch["queries_mask"][crop_idx:crop_idx + 1],
        }
        for res, grp in batch["groups"].items():
            mini_batch["groups"][res] = {
                "tokens": grp["tokens"][crop_idx:crop_idx + 1],
                "mask":   grp["mask"][crop_idx:crop_idx + 1],
                "shape":  grp["shape"],
            }

        result = self.forward(mini_batch, training=False)
        return result["predictions"] if isinstance(result, dict) else result  # [1, M, C]

    def _sliding_window_step(self, batch):
        """
        Process all crops, stitch predictions, compute metrics on full tile.

        Returns:
            preds_full: [H, W]   full-tile argmax predictions
            label_full: [H, W]   full-tile labels
            logits_avg: [H*W, C] averaged logits (for loss computation)
        """
        num_crops       = batch["queries"].shape[0]
        positions       = batch["crop_positions"]
        crop_h, crop_w  = batch["crop_size"]
        full_h, full_w  = batch["full_size"]

        crop_logits_list = []
        for i in range(num_crops):
            with torch.no_grad():
                logits = self._forward_crop(batch, i)   # [1, crop_h*crop_w, C]
            crop_logits_list.append(logits.squeeze(0))  # [crop_h*crop_w, C]

        preds_full, logits_avg = stitch_predictions(
            crop_logits_list=crop_logits_list,
            crop_positions=positions,
            crop_h=crop_h, crop_w=crop_w,
            full_h=full_h, full_w=full_w,
            num_classes=self.num_classes,
        )

        label_full = batch["label"].to(self.device)
        return preds_full, label_full, logits_avg

    # =========================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._compute_loss_and_preds(batch, training=True)

        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ── Sliding window path ───────────────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            loss = self.loss(logits_avg.unsqueeze(0), label_full.unsqueeze(0))

            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_val.update(preds_full[valid], label_full[valid])
                self.metric_acc_val.update(preds_full[valid], label_full[valid])

            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            return loss

        # ── Normal path ───────────────────────────────────────────────
        loss, preds, labels = self._compute_loss_and_preds(batch, training=False)

        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # ── Sliding window path ───────────────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            loss = self.loss(logits_avg.unsqueeze(0), label_full.unsqueeze(0))

            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_test.update(preds_full[valid], label_full[valid])
                self.metric_acc_test.update(preds_full[valid], label_full[valid])

            self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
            return loss

        # ── Normal path ───────────────────────────────────────────────
        loss, preds, labels = self._compute_loss_and_preds(batch, training=False)

        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU",    self.metric_IoU_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.metric_acc_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_mIoU",    self.metric_IoU_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self.metric_acc_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou = self.metric_IoU_test.compute()  # [num_classes]
        test_acc = self.metric_acc_test.compute()  # [num_classes]

        self.log("test_mIoU",    test_iou.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc.mean(), on_epoch=True, logger=True)

        for i, name in enumerate(["no_flood", "flood"]):
            if i < len(test_iou):
                self.log(f"test_IoU_{name}", test_iou[i], on_epoch=True, logger=True)
            if i < len(test_acc):
                self.log(f"test_acc_{name}", test_acc[i], on_epoch=True, logger=True)

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps  = int(self.trainer.estimated_stepping_batches)
        warmup_steps = self.config["optimizer"].get(
            "warmup_steps", max(1, int(0.05 * total_steps))
        )

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