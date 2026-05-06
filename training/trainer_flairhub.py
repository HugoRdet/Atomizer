"""
FLAIR-HUB Atomizer Trainer (single-task)
==========================================

Land cover semantic segmentation on FLAIR-HUB with multi-modal multi-resolution
input. Mirrors Model_SenFlood: per-pixel cross-entropy, multi-resolution input
groups, mIoU + accuracy metrics. Differences:
  - 19 classes (COSIA), severe class imbalance → macro mIoU as primary metric
  - Per-class IoU logged at test time for fine-grained reporting
  - Optional class weighting for loss (off by default; FLAIR-1 baselines didn't use)

Forward contract (matches Atomiser_Senflood):
    model(batch, training=...) -> [B, M, K]
    where M is the number of queries per sample (per-pixel at VHR resolution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

# Atomizer architecture used as-is. Same model as Sen1Floods11/PASTIS but
# with 19-class output head built via config["model"]["num_classes"].
from training.atomiser.Atomiser_SENFLOOD import Atomiser_Senflood


# ────────────────────────────────────────────────────────────────────
# COSIA class metadata
# ────────────────────────────────────────────────────────────────────

COSIA_CLASS_NAMES = [
    "building",         # 0
    "greenhouse",       # 1
    "swimming_pool",    # 2
    "impervious",       # 3  (Impervious surface)
    "pervious",         # 4  (Pervious surface)
    "bare_soil",        # 5
    "water",            # 6
    "snow",             # 7
    "herbaceous",       # 8  (Herbaceous vegetation)
    "agricultural",     # 9  (Agricultural land)
    "plowed",           # 10 (Plowed land)
    "vineyard",         # 11
    "deciduous",        # 12
    "coniferous",       # 13
    "brushwood",        # 14
    "clear_cut",        # 15
    "ligneous",         # 16
    "mixed",            # 17
    "undefined",        # 18
]
NUM_CLASSES_FLAIR = 19


# ────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────

class Model_FlairHub(pl.LightningModule):
    """
    Single-task FLAIR-HUB land cover segmentation Lightning module.

    Args:
        config:        Atomizer config dict.
        wand:          Whether W&B logging is active (caller-managed).
        name:          Experiment name.
        transform:     Unused; API parity with single-task trainers.
        lookup_table:  Lookup_encoding instance.
        ignore_index:  Class index to ignore in loss/metrics. None means
                       no ignore (all 19 classes scored). Set to 18 if
                       "undefined" should be excluded.
        class_weights: Optional [19]-tensor of CE class weights. Default None.
    """

    def __init__(
        self,
        config: dict,
        wand: bool,
        name: str,
        transform=None,
        lookup_table=None,
        ignore_index: int = None,
        class_weights=None,
    ):
        super().__init__()
        self.strict_loading = False
        self.config       = config
        self.transform    = transform
        self.wand         = wand
        self.name         = name
        self.lookup_table = lookup_table
        self.ignore_index = ignore_index
        self.num_classes  = NUM_CLASSES_FLAIR
        self.class_names  = COSIA_CLASS_NAMES

        # Force the model's output head to 19 classes regardless of what
        # the YAML config says (FLAIR-HUB-specific).
        config = dict(config)
        config_model = dict(config.get("model", {}))
        config_model["num_classes"] = self.num_classes
        config["model"] = config_model
        self.config = config

        # ── Build Atomizer model ─────────────────────────────────
        self.model = Atomiser_Senflood(
            config=config,
            lookup_table=lookup_table,
        )

        # ── Loss ─────────────────────────────────────────────────
        # Cross-entropy with optional ignore_index and class weights.
        ce_kwargs = {}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = int(self.ignore_index)
        if class_weights is not None:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("_class_weights", class_weights)
            ce_kwargs["weight"] = self._class_weights
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        # ── Metrics: per-split, per-task ─────────────────────────
        # mIoU (primary), macro_acc, and per-class IoU at test time.
        # Setting ignore_index in metrics aligns with loss when set.
        metric_kwargs = dict(num_classes=self.num_classes, average="macro")
        if self.ignore_index is not None:
            metric_kwargs["ignore_index"] = int(self.ignore_index)

        # Macro metrics (primary)
        self.train_miou      = MulticlassJaccardIndex(**metric_kwargs)
        self.val_miou        = MulticlassJaccardIndex(**metric_kwargs)
        self.test_miou       = MulticlassJaccardIndex(**metric_kwargs)

        self.train_macro_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_macro_acc   = MulticlassAccuracy(**metric_kwargs)
        self.test_macro_acc  = MulticlassAccuracy(**metric_kwargs)

        # Per-class IoU at test time (one number per class).
        # Use average=None to get per-class values.
        per_class_kwargs = dict(num_classes=self.num_classes, average=None)
        if self.ignore_index is not None:
            per_class_kwargs["ignore_index"] = int(self.ignore_index)
        self.test_per_class_iou = MulticlassJaccardIndex(**per_class_kwargs)

        # ── Optimizer config ─────────────────────────────────────
        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        print(f"[FLAIR-HUB-Trainer] {self.num_classes} classes, "
              f"ignore_index={self.ignore_index}, "
              f"class_weighted={'yes' if class_weights is not None else 'no'}")

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch, training: bool = True):
        return self.model(batch, training=training)

    # ─────────────────────────────────────────────────────────────────
    # Step (shared)
    # ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        """
        Forward + loss + metrics. Stage-specific torchmetrics object
        is selected via stage name. Returns loss for backprop (train).

        Forward returns logits [B, M, K] where:
            B = batch size
            M = num queries per sample (per-pixel at VHR resolution)
            K = num_classes (19 for FLAIR-HUB)

        Labels come from queries[:, :, 4] (per-pixel class labels stored
        in the query tokens — set by the dataset's TokenBuilder).
        """
        is_train = (stage == "train")
        logits = self.forward(batch, training=is_train)        # [B, M, K]

        # Defensive shape check
        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[FLAIR-HUB-Trainer] Model returned {logits.shape[-1]} "
                f"classes, expected {self.num_classes}."
            )

        labels = batch["queries"][:, :, 4].long()              # [B, M]

        # Flatten for CE: [B*M, K] vs [B*M]
        logits_flat = rearrange(logits, "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        loss = self.loss_fn(logits_flat, labels_flat)

        # Metrics — argmax once, update all stage-relevant metrics.
        with torch.no_grad():
            preds_flat = logits_flat.argmax(dim=-1)            # [B*M]

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
        # Pass Metric objects to self.log → Lightning calls compute()/reset().
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
        # Compute per-class IoU and log named values.
        per_class = self.test_per_class_iou.compute()          # [19]
        self.test_per_class_iou.reset()
        for i, class_name in enumerate(self.class_names):
            # Skip ignore class in logs to keep dashboard clean.
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
    # Optimizer (AdamW + cosine warmup) — mirrors Model_SenFlood
    # ─────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        """
        Estimate total optimizer steps across the run. Priority:
          1. Config override `trainer.total_steps`
          2. Lightning's estimated_stepping_batches
          3. Fallback: epochs × 1000
        """
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[FLAIR-HUB-Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[FLAIR-HUB-Trainer] WARN: cannot estimate total_steps. "
                  f"Falling back to {fallback}.")
            return fallback

        print(f"[FLAIR-HUB-Trainer] total_steps estimate: {est}")
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

        print(f"[FLAIR-HUB-Trainer] LR sched: total_steps={total_steps}, "
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