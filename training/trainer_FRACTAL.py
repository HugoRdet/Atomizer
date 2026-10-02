"""
FRACTAL Atomizer Trainer (single-task)
========================================

LIDAR + VHR semantic segmentation on FRACTAL. Mirrors Model_FlairHub:
per-pixel cross-entropy, multi-resolution input groups, mIoU + accuracy
metrics. Differences:
  - 7 classes (FRACTAL), configured via config["trainer"]["num_classes"]
  - ignore_index=255 by default (matches FractalDataset's padding label
    for variable-length LIDAR point counts)
  - Per-class IoU at test time matches FRACTAL paper's reporting format
  - Queries are sparse (one per LIDAR point), not dense per-pixel
  - Unweighted cross-entropy (no class weighting) — confirmed intentional.

Forward contract (unchanged from FLAIR-HUB):
    model(batch, training=...) -> [B, M, K]
    where M is the number of queries per sample (padded to a fixed count).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

# Atomizer architecture — FRACTAL-specific subclass. Inherits from
# Atomiser_Senflood_Skip; overrides _apply_pruning/encode/forward to thread
# a precomputed per-token Voronoi assignment (token_latent_assignment) and
# patch_ids fallback through to its own GeographicPruning (reused from
# geographic_pruning_dales.py — dataset-agnostic despite the filename), and
# overrides the decoder to combine the pixel-skip cascade (own-pixel VHR
# band-token attention, gated by config["Atomiser"]["use_decoder_skip"])
# additively with a z-aware query projection, so the model can distinguish
# LIDAR points sharing (x, y) but differing in z (e.g. bridge over road,
# tree canopy over ground).
from training.atomiser.Atomiser_Fractal import Atomiser_Fractal


# ────────────────────────────────────────────────────────────────────
# FRACTAL class metadata
# ────────────────────────────────────────────────────────────────────
# Class order matches FractalDataset.FRACTAL_CLASSES and the LAS→FRACTAL
# remap in utils_dataset_fractal.py.

FRACTAL_CLASS_NAMES = [
    "other",                 # 0
    "ground",                # 1
    "vegetation",            # 2
    "building",              # 3
    "water",                 # 4
    "bridge",                # 5
    "permanent_structure",   # 6
]
NUM_CLASSES_FRACTAL = 7


# ────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────

class Model_Fractal(pl.LightningModule):
    """
    FRACTAL LIDAR + VHR segmentation Lightning module.

    Args:
        config:           Atomizer config dict. config["trainer"]["num_classes"]
                          MUST already be set to the class count (7 for
                          FRACTAL) — this trainer reads it, not the other
                          way around.
        wand:             Whether W&B logging is active (caller-managed).
        name:             Experiment name.
        transform:        Unused; API parity with other single-task trainers.
        lookup_table:     Lookup_encoding instance.
        ignore_index:     Class index to ignore in loss/metrics.
                          Default 255 (matches FractalDataset's padding label
                          for variable-length LIDAR point counts).
                          Set to None to score all positions including padding.
    """

    def __init__(
        self,
        config: dict,
        wand: bool,
        name: str,
        transform=None,
        lookup_table=None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.strict_loading = False
        self.config       = config
        self.transform    = transform
        self.wand         = wand
        self.name         = name
        self.lookup_table = lookup_table
        self.ignore_index = ignore_index
        self.num_classes  = NUM_CLASSES_FRACTAL
        self.class_names  = FRACTAL_CLASS_NAMES

        # ── Build Atomizer model ─────────────────────────────────
        # config["trainer"]["num_classes"] is read directly by
        # Atomiser_Senflood_Skip.__init__ (self.num_classes = config
        # ["trainer"]["num_classes"]) — set it in the YAML, not here.
        self.model = Atomiser_Fractal(
            config=config,
            lookup_table=lookup_table,
        )

        # ── Loss (unweighted CE) ────────────────────────────────
        ce_kwargs = {}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = int(self.ignore_index)
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        # ── Metrics ──────────────────────────────────────────────
        # Macro mIoU is the primary metric (matches FRACTAL paper).
        metric_kwargs = dict(num_classes=self.num_classes, average="macro")
        if self.ignore_index is not None:
            metric_kwargs["ignore_index"] = int(self.ignore_index)

        self.train_miou      = MulticlassJaccardIndex(**metric_kwargs)
        self.val_miou        = MulticlassJaccardIndex(**metric_kwargs)
        self.test_miou       = MulticlassJaccardIndex(**metric_kwargs)

        self.train_macro_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_macro_acc   = MulticlassAccuracy(**metric_kwargs)
        self.test_macro_acc  = MulticlassAccuracy(**metric_kwargs)

        # Per-class IoU at test time (one number per class).
        per_class_kwargs = dict(num_classes=self.num_classes, average=None)
        if self.ignore_index is not None:
            per_class_kwargs["ignore_index"] = int(self.ignore_index)
        self.test_per_class_iou = MulticlassJaccardIndex(**per_class_kwargs)

        # ── Optimizer config ─────────────────────────────────────
        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        print(f"[FRACTAL-Trainer] {self.num_classes} classes, "
              f"ignore_index={self.ignore_index}, unweighted CE.")

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch, training: bool = True):
        return self.model(batch, training=training)

    # ─────────────────────────────────────────────────────────────────
    # Shared step
    # ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        """
        Forward + loss + metrics. Labels come from queries[:, :, 4] which
        the FractalDataset populates with per-point LIDAR labels (with
        IGNORE_INDEX padding to fix the query count per batch).

        The CE ignore_index + per-class metrics ignore_index handle the
        padding correctly without needing the queries_mask explicitly.
        """
        is_train = (stage == "train")
        logits = self.forward(batch, training=is_train)        # [B, M, K]

        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[FRACTAL-Trainer] Model returned {logits.shape[-1]} "
                f"classes, expected {self.num_classes}."
            )

        labels = batch["queries"][:, :, 4].long()              # [B, M]

        # Flatten for CE: [B*M, K] vs [B*M]
        logits_flat = rearrange(logits, "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        loss = self.loss_fn(logits_flat, labels_flat)

        with torch.no_grad():
            preds_flat = logits_flat.argmax(dim=-1)

            if stage == "train":
                self.train_miou.update(preds_flat, labels_flat)
                self.train_macro_acc.update(preds_flat, labels_flat)
            elif stage == "val":
                self.val_miou.update(preds_flat, labels_flat)
                self.val_macro_acc.update(preds_flat, labels_flat)
            elif stage == "test":
                self.test_miou.update(preds_flat, labels_flat)
                self.test_macro_acc.update(preds_flat, labels_flat)
                self.test_per_class_iou.update(preds_flat, labels_flat)

        bs = batch["queries"].shape[0]
        on_step = is_train
        self.log(
            f"{stage}_loss", loss,
            on_step=on_step, on_epoch=True,
            prog_bar=is_train, sync_dist=True,
            batch_size=bs,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ─────────────────────────────────────────────────────────────────
    # End-of-epoch logging
    # ─────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self):
        self.log("train_mIoU",      self.train_miou,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_acc", self.train_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_mIoU",      self.val_miou,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_acc", self.val_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        # Per-class IoU — log each one with its class name.
        per_class = self.test_per_class_iou.compute()          # [7]
        self.test_per_class_iou.reset()
        for i, class_name in enumerate(self.class_names):
            if self.ignore_index is not None and i == self.ignore_index:
                continue
            value = per_class[i].item()
            self.log(f"test_IoU/{class_name}", value,
                     on_epoch=True, sync_dist=True)

        self.log("test_mIoU",      self.test_miou,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_macro_acc", self.test_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    # ─────────────────────────────────────────────────────────────────
    # Optimizer (AdamW + cosine warmup) — mirrors FLAIR-HUB
    # ─────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[FRACTAL-Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[FRACTAL-Trainer] WARN: cannot estimate total_steps. "
                  f"Falling back to {fallback}.")
            return fallback

        print(f"[FRACTAL-Trainer] total_steps estimate: {est}")
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

        print(f"[FRACTAL-Trainer] LR sched: total_steps={total_steps}, "
              f"warmup={warmup_steps}, peak_lr={self.lr}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
