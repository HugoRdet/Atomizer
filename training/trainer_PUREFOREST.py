"""
PureForest Atomizer Trainer (single-task classification)
==========================================================

13-class pure forest tree species classification.
Input: RGB+NIR ortho (4 bands, 0.2 m/px) + LiDAR point cloud (optional).

Mirrors Model_ForestNet for the classification contract:
    model(batch, training=..., task="classification") → [B, num_classes]
    CE loss on [B] scalar targets from batch["label"]

Differences from Model_ForestNet:
  - 13 classes (PureForest) with severe imbalance →
      inverse-frequency class weighting ON by default ("auto")
  - Per-class top-1 accuracy logged at test time
  - Cosine schedule with linear warmup (from FRACTAL trainer) rather than
    plain CosineAnnealingLR — more robust with large datasets / multi-GPU

Class frequencies are computed from the full PureForest-patches.csv
(all splits combined, 135 569 scenes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


# ─────────────────────────────────────────────────────────────────────────────
# PureForest class metadata
# ─────────────────────────────────────────────────────────────────────────────

PUREFOREST_CLASS_NAMES = [
    "Deciduous oak",          # 0  — 35.4 %
    "Evergreen oak",          # 1  — 16.5 %
    "Beech",                  # 2  —  9.3 %
    "Chestnut",               # 3  —  2.7 %
    "Black locust",           # 4  —  1.7 %
    "Maritime pine",          # 5  —  5.6 %
    "Scotch pine",            # 6  — 13.5 %
    "Black pine",             # 7  —  5.3 %
    "Aleppo pine",            # 8  —  3.5 %
    "Fir",                    # 9  —  0.6 %
    "Spruce",                 # 10 —  3.0 %
    "Larch",                  # 11 —  2.4 %
    "Douglas",                # 12 —  0.4 %
]
NUM_CLASSES_PUREFOREST = 13

# Global class frequencies from PureForest-patches.csv (all 135 569 scenes).
# Each entry is count / total.  Used for inverse-frequency class weighting.
PUREFOREST_CLASS_FREQS = [
    0.3545,   # 0  Deciduous oak       48 055
    0.1649,   # 1  Evergreen oak       22 361
    0.0934,   # 2  Beech               12 670
    0.0272,   # 3  Chestnut             3 684
    0.0170,   # 4  Black locust         2 303
    0.0558,   # 5  Maritime pine        7 568
    0.1347,   # 6  Scotch pine         18 265
    0.0533,   # 7  Black pine           7 226
    0.0346,   # 8  Aleppo pine          4 699
    0.0062,   # 9  Fir                    840
    0.0300,   # 10 Spruce               4 074
    0.0243,   # 11 Larch                3 294
    0.0039,   # 12 Douglas                530
]


def default_pureforest_class_weights(
    freqs=PUREFOREST_CLASS_FREQS,
    weight_clip: float = 20.0,
) -> torch.Tensor:
    """
    Inverse-frequency weights, clipped to avoid destabilising extreme values
    for the rarest classes (Douglas raw weight ≈ 256, Fir ≈ 161).

    A clip of 20 keeps the signal on rare species meaningful while preventing
    them from swamping the gradient signal from common species.
    Tune weight_clip if you want more / less emphasis on rare classes.
    """
    raw = torch.tensor(
        [1.0 / max(f, 1e-9) for f in freqs], dtype=torch.float32
    )
    return raw.clamp(max=weight_clip)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Model_PureForest(pl.LightningModule):
    """
    PureForest tree species classification Lightning module.

    Args:
        config:           Atomizer config dict (YAML loaded).
        wand:             Whether W&B logging is active.
        name:             Experiment name string.
        transform:        Unused; kept for API parity with other trainers.
        lookup_table:     Lookup_encoding instance.
        class_weights:    CE class weights.
                            "auto"  (default) → inverse-frequency, clipped at 20
                            None              → unweighted CE
                            Tensor / list     → caller-provided weights [13]
        label_smoothing:  CE label smoothing. Default 0.0 — kept off for
                          fair comparison with the RandLA-Net baseline.
    """

    def __init__(
        self,
        config: dict,
        wand: bool = False,
        name: str = "pureforest",
        transform=None,
        lookup_table=None,
        class_weights="auto",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters(ignore=["lookup_table", "transform", "class_weights"])

        self.config          = config
        self.name            = name
        self.wand            = wand
        self.num_classes     = NUM_CLASSES_PUREFOREST
        self.class_names     = PUREFOREST_CLASS_NAMES
        self.label_smoothing = label_smoothing

        # ── Force num_classes in config ───────────────────────────
        config = dict(config)
        config_trainer = dict(config.get("trainer", {}))
        config_trainer["num_classes"] = self.num_classes
        config["trainer"] = config_trainer
        self.config = config

        # ── Build Atomizer model ──────────────────────────────────
        from training.atomiser.Atomiser_SENFLOOD import Atomiser_Senflood
        self.model = Atomiser_Senflood(
            config=config,
            lookup_table=lookup_table,
        )

        # ── Class weights ─────────────────────────────────────────
        if class_weights == "auto":
            cw = default_pureforest_class_weights()
            print(
                f"[PureForest-Trainer] Auto inverse-frequency weights "
                f"(clip=20): {[round(w, 2) for w in cw.tolist()]}"
            )
        elif class_weights is None:
            cw = None
            print("[PureForest-Trainer] No class weighting (unweighted CE).")
        else:
            cw = torch.tensor(class_weights, dtype=torch.float32) \
                if not torch.is_tensor(class_weights) else class_weights
            print(f"[PureForest-Trainer] Custom class weights: "
                  f"{[round(w, 2) for w in cw.tolist()]}")

        if cw is not None:
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

        # ── Optimizer config ──────────────────────────────────────
        trainer_cfg       = config["trainer"]
        self.lr           = float(trainer_cfg.get("lr", 1e-4))
        self.weight_decay = float(trainer_cfg.get("weight_decay", 1e-2))

        # ── Metrics ───────────────────────────────────────────────
        def _make_metrics(prefix: str):
            return nn.ModuleDict({
                f"{prefix}_top1": MulticlassAccuracy(
                    num_classes=self.num_classes, top_k=1, average="micro"),
                f"{prefix}_top5": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    top_k=min(5, self.num_classes), average="micro"),
                f"{prefix}_macro_acc": MulticlassAccuracy(
                    num_classes=self.num_classes, average="macro"),
                f"{prefix}_macro_f1": MulticlassF1Score(
                    num_classes=self.num_classes, average="macro"),
            })

        self.train_metrics = _make_metrics("train")
        self.val_metrics   = _make_metrics("val")
        self.test_metrics  = _make_metrics("test")

        # Per-class top-1 accuracy at test time only
        self.test_per_class_acc = MulticlassAccuracy(
            num_classes=self.num_classes, average=None
        )

        print(
            f"[PureForest-Trainer] {self.num_classes} classes, "
            f"label_smoothing={self.label_smoothing}, "
            f"lr={self.lr}, weight_decay={self.weight_decay}"
        )

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training: bool = True):
        return self.model(batch, training=training, task="classification")

    # =========================================================================
    # SHARED STEP
    # =========================================================================

    def _shared_step(self, batch, stage: str):
        is_train = stage == "train"

        target = batch["label"].long()     # [B]
        bs     = target.shape[0]

        logits = self.forward(batch, training=is_train)   # [B, 13]

        # Defensive checks
        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[PureForest-Trainer] Logits last dim {logits.shape[-1]} != "
                f"num_classes {self.num_classes}. "
                f"Check config['trainer']['num_classes']."
            )
        if target.numel() > 0:
            tmin, tmax = int(target.min()), int(target.max())
            if tmin < 0 or tmax >= self.num_classes:
                raise RuntimeError(
                    f"[PureForest-Trainer] Targets out of range "
                    f"[{tmin}, {tmax}], num_classes={self.num_classes}."
                )

        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing if is_train else 0.0,
        )

        # Update metrics
        metrics = getattr(self, f"{stage}_metrics")
        with torch.no_grad():
            for metric in metrics.values():
                metric.update(logits, target)
            if stage == "test":
                self.test_per_class_acc.update(logits, target)

        self.log(
            f"{stage}_loss", loss,
            on_step=is_train, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=bs,
        )
        return loss

    # =========================================================================
    # LIGHTNING HOOKS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _epoch_end(self, stage: str):
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            self.log(name, metric,
                     prog_bar=("top1" in name), sync_dist=True)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

        # Per-class accuracy
        per_class = self.test_per_class_acc.compute()   # [13]
        self.test_per_class_acc.reset()
        for i, class_name in enumerate(self.class_names):
            self.log(
                f"test_acc/{class_name}", per_class[i].item(),
                on_epoch=True, sync_dist=True,
            )

    # =========================================================================
    # OPTIMIZER  — AdamW + cosine warmup (mirrors FRACTAL trainer)
    # =========================================================================

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[PureForest-Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(
                f"[PureForest-Trainer] WARN: cannot estimate total_steps, "
                f"falling back to {fallback}."
            )
            return fallback

        print(f"[PureForest-Trainer] total_steps estimate: {est}")
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

        print(
            f"[PureForest-Trainer] LR schedule: "
            f"total_steps={total_steps}, warmup={warmup_steps}, "
            f"peak_lr={self.lr}"
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
