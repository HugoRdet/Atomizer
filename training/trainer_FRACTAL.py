"""
FRACTAL Atomizer Trainer (single-task)
========================================

LIDAR + VHR semantic segmentation on FRACTAL. Mirrors Model_FlairHub:
per-pixel cross-entropy, multi-resolution input groups, mIoU + accuracy
metrics. Differences:
  - 7 classes (FRACTAL), severe imbalance → class weighting ON by default
  - ignore_index=255 by default (matches FractalDataset's padding label
    for variable-length LIDAR point counts)
  - Per-class IoU at test time matches FRACTAL paper's reporting format
  - Queries are sparse (one per LIDAR point), not dense per-pixel

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

# Atomizer architecture — FRACTAL-specific subclass with z-aware decoder.
# Atomiser_Fractal inherits from Atomiser_Senflood and overrides only the
# decoder's query construction to derive per-pixel Q from query z values.
# This lets the model distinguish LIDAR points sharing (x, y) but with
# different z (e.g. bridge over road, tree canopy over ground).
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

# Global class frequencies (computed across all 80k train + 10k val patches;
# matches FRACTAL paper Table). Used for inverse-frequency class weighting.
# Source: aggregate diagnostic over the full dataset.
FRACTAL_CLASS_FREQS = [
    0.0056,   # other
    0.3897,   # ground
    0.5698,   # vegetation
    0.0280,   # building
    0.0052,   # water
    0.0013,   # bridge
    0.0005,   # permanent_structure
]


def default_fractal_class_weights(
    freqs=FRACTAL_CLASS_FREQS,
    weight_clip: float = 50.0,
) -> torch.Tensor:
    """
    Inverse-frequency weights, clipped to avoid destabilizing extreme weights
    on the rarest classes (permanent_structure raw weight is ~1900).

    Median-frequency balancing is an alternative; we use clipped inverse-freq
    because it's simpler and gives the rare classes meaningful gradient
    signal without overwhelming the dominant ones.
    """
    raw = torch.tensor([1.0 / max(f, 1e-9) for f in freqs], dtype=torch.float32)
    return raw.clamp(max=weight_clip)


# ────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────

class Model_Fractal(pl.LightningModule):
    """
    FRACTAL LIDAR + VHR segmentation Lightning module.

    Args:
        config:           Atomizer config dict.
        wand:             Whether W&B logging is active (caller-managed).
        name:             Experiment name.
        transform:        Unused; API parity with other single-task trainers.
        lookup_table:     Lookup_encoding instance.
        ignore_index:     Class index to ignore in loss/metrics.
                          Default 255 (matches FractalDataset's padding label
                          for variable-length LIDAR point counts).
                          Set to None to score all positions including padding.
        class_weights:    Optional [7]-tensor of CE class weights.
                          - "auto" (default): use inverse-frequency weights
                                              clipped at 50 (see default_fractal_class_weights).
                          - None:             unweighted CE.
                          - tensor/list:      caller-provided weights.
    """

    def __init__(
        self,
        config: dict,
        wand: bool,
        name: str,
        transform=None,
        lookup_table=None,
        ignore_index: int = 255,
        class_weights="auto",
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

        # Force the model's output head to 7 classes regardless of YAML.
        config = dict(config)
        config_model = dict(config.get("model", {}))
        config_model["num_classes"] = self.num_classes
        config["model"] = config_model
        self.config = config

        # ── Build Atomizer model ─────────────────────────────────
        self.model = Atomiser_Fractal(
            config=config,
            lookup_table=lookup_table,
        )

        # ── Class weights ────────────────────────────────────────
        if class_weights == "auto":
            class_weights = default_fractal_class_weights()
            print(f"[FRACTAL-Trainer] Using auto inverse-frequency weights "
                  f"(clipped at 50): {class_weights.tolist()}")
        elif class_weights is None:
            print(f"[FRACTAL-Trainer] No class weighting (unweighted CE).")
        else:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"[FRACTAL-Trainer] Custom class weights: "
                  f"{class_weights.tolist()}")

        # ── Loss ─────────────────────────────────────────────────
        ce_kwargs = {}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = int(self.ignore_index)
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights)
            ce_kwargs["weight"] = self._class_weights
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
              f"ignore_index={self.ignore_index}, "
              f"class_weighted={'yes' if class_weights is not None else 'no'}")

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
