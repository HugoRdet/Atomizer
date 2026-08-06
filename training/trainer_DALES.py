"""
DALES Atomizer Trainer (single-task)
========================================

LIDAR-only semantic segmentation on DALES. Adapted from Model_Fractal:
per-point cross-entropy, single-resolution input group (LIDAR only, no
VHR), mIoU + accuracy metrics.

Differences from Model_Fractal:
  - 8 classes (DALES: ground, vegetation, cars, trucks, power_lines,
    fences, poles, buildings) instead of FRACTAL's 7
  - LIDAR-only input (no VHR group) — groups[PIXEL_RESOLUTION] holds a
    single-channel-per-point token stream (elevation, with intensity
    riding in col 6 — see DalesTokenProcessor)
  - Uses DalesTokenProcessor (echo routing on col 7 AND intensity routing
    on col 6) instead of FractalTokenProcessor (echo routing only)
  - Severe class imbalance is even more extreme than FRACTAL's (power
    lines/trucks/poles each <0.2% of points per the DALES inspection
    numbers) — class weighting default clip raised accordingly, see
    DALES_CLASS_FREQS below (VERIFY against your actual full-dataset
    aggregate before trusting these — they're taken from the single-file
    + 11-file aggregate inspection run earlier in this conversation, not
    a full-dataset pass)

REQUIRES: an `Atomiser_Dales` model class (analogous to Atomiser_Fractal),
subclassing Atomiser_Senflood_Skip and swapping in DalesTokenProcessor
instead of the base TokenProcessor. Not yet defined here — see the import
below and the note in script_train_dales.py.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

# STILL NEEDED: an Atomiser_Dales subclass analogous to Atomiser_Fractal,
# swapping in DalesTokenProcessor (echo + intensity routing) instead of
# FractalTokenProcessor (echo routing only). If Atomiser_Fractal's z-aware
# decoder override applies equally to DALES (LIDAR points sharing (x,y)
# but differing in z — e.g. a building point above a ground point), that
# override should carry over too; not yet confirmed without seeing
# Atomiser_Fractal's source.
from training.atomiser.Atomiser_dales import Atomiser_Dales


# ────────────────────────────────────────────────────────────────────
# DALES class metadata
# ────────────────────────────────────────────────────────────────────
# Class order matches DalesDataset.DALES_CLASSES and DALES_TO_ATOMIZER
# in utils_dataset_dales.py.

DALES_CLASS_NAMES = [
    "ground",        # 0
    "vegetation",    # 1
    "cars",          # 2
    "trucks",        # 3
    "power_lines",   # 4
    "fences",        # 5
    "poles",         # 6
    "buildings",     # 7
]
NUM_CLASSES_DALES = 8

# Class frequencies — taken from the single-file + 11-file test-split
# aggregate inspected earlier in this conversation. VERIFY against a full
# train-split aggregate before trusting these for weighting; they are a
# reasonable starting point but not a substitute for computing frequencies
# over your actual full training set.
DALES_CLASS_FREQS = [
    0.5040,   # ground
    0.3034,   # vegetation
    0.0078,   # cars
    0.0011,   # trucks
    0.0017,   # power_lines
    0.0046,   # fences
    0.0007,   # poles
    0.1716,   # buildings
]


def default_dales_class_weights(
    freqs=DALES_CLASS_FREQS,
    weight_clip: float = 50.0,
) -> torch.Tensor:
    """
    SQRT-inverse-frequency weights (was raw inverse-frequency).

    Now that DalesDataset also does class-balanced QUERY SAMPLING
    (sqrt(1/freq) weighted, see QUERY_SAMPLING_WEIGHT_LUT in
    utils_dataset_dales.py), stacking that with RAW inverse-frequency loss
    weighting would compound multiplicatively: a rare-class point's
    combined training signal would scale roughly as
    (oversampling factor) x (loss weight) ~= freq^-0.5 x freq^-1 = freq^-1.5
    -- a much stronger correction than either mechanism was designed for
    in isolation, risking destabilizing training / overcorrecting onto the
    rare classes at the expense of the majority ones.

    Using sqrt(1/freq) here too means the COMBINED effect of sampling x
    loss weight is sqrt(1/freq) x sqrt(1/freq) = 1/freq -- a single,
    standard inverse-frequency correction, just applied through two
    complementary gentle mechanisms instead of one aggressive one.

    Raw (unclipped) sqrt weights for DALES' actual frequencies top out
    around ~38 (poles), vs raw inverse-frequency's ~1400 -- so the clip
    is lowered accordingly (100 -> 50) since it's providing a much smaller
    safety margin than before, not doing most of the work.
    """
    raw = torch.tensor(
        [1.0 / max(f, 1e-9) ** 0.5 for f in freqs], dtype=torch.float32
    )
    return raw.clamp(max=weight_clip)


# ────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────

class Model_Dales(pl.LightningModule):
    """
    DALES LIDAR-only segmentation Lightning module.

    Args:
        config:           Atomizer config dict.
        wand:             Whether W&B logging is active (caller-managed).
        name:             Experiment name.
        transform:        Unused; API parity with other single-task trainers.
        lookup_table:     Lookup_encoding instance.
        ignore_index:     Class index to ignore in loss/metrics.
                          Default 255 (matches DalesDataset's padding label
                          for variable-length LIDAR point counts).
        class_weights:    Optional [8]-tensor of CE class weights.
                          - "auto" (default): inverse-frequency, clipped at 100.
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
        self.num_classes  = NUM_CLASSES_DALES
        self.class_names  = DALES_CLASS_NAMES

        # Force the model's output head to 8 classes regardless of YAML.
        # NOTE: Atomiser_Senflood_Skip.__init__ reads
        # `config["trainer"]["num_classes"]` directly — NOT
        # config["model"]["num_classes"]. The FRACTAL trainer this was
        # copied from wrote to config["model"], which the architecture
        # never actually reads (a latent no-op there too). Write to the
        # key that's actually consulted, as a safety net — but the YAML's
        # own trainer.num_classes should also be set correctly, since
        # this dict mutation happens after config parsing but the same
        # dict object is what gets passed to Atomiser_Dales below.
        config = dict(config)
        config_trainer = dict(config.get("trainer", {}))
        config_trainer["num_classes"] = self.num_classes
        config["trainer"] = config_trainer
        self.config = config

        # ── Build Atomizer model ─────────────────────────────────
        self.model = Atomiser_Dales(
            config=config,
            lookup_table=lookup_table,
        )

        # ── Class weights ────────────────────────────────────────
        if class_weights == "auto":
            class_weights = default_dales_class_weights()
            print(f"[DALES-Trainer] Using auto inverse-frequency weights "
                  f"(clipped at 100): {class_weights.tolist()}")
        elif class_weights is None:
            print(f"[DALES-Trainer] No class weighting (unweighted CE).")
        else:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"[DALES-Trainer] Custom class weights: "
                  f"{class_weights.tolist()}")

        # ── Loss ─────────────────────────────────────────────────
        ce_kwargs = {}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = int(self.ignore_index)
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights)
            # Applied to the loss (unlike the FRACTAL trainer this was
            # copied from, where this line stayed commented out). For
            # DALES, rare classes (poles ~0.07%, trucks ~0.1%, fences
            # ~0.46% of points) get near-zero gradient signal under
            # unweighted CE, which is exactly what's dragging down macro
            # mIoU relative to specialized point-cloud baselines that
            # weren't fighting this same imbalance issue.
            ce_kwargs["weight"] = self._class_weights
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        # ── Metrics ──────────────────────────────────────────────
        metric_kwargs = dict(num_classes=self.num_classes, average="macro")
        if self.ignore_index is not None:
            metric_kwargs["ignore_index"] = int(self.ignore_index)

        self.train_miou      = MulticlassJaccardIndex(**metric_kwargs)
        self.val_miou        = MulticlassJaccardIndex(**metric_kwargs)
        self.test_miou       = MulticlassJaccardIndex(**metric_kwargs)

        self.train_macro_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_macro_acc   = MulticlassAccuracy(**metric_kwargs)
        self.test_macro_acc  = MulticlassAccuracy(**metric_kwargs)

        per_class_kwargs = dict(num_classes=self.num_classes, average=None)
        if self.ignore_index is not None:
            per_class_kwargs["ignore_index"] = int(self.ignore_index)
        self.test_per_class_iou = MulticlassJaccardIndex(**per_class_kwargs)

        # ── Optimizer config ─────────────────────────────────────
        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        print(f"[DALES-Trainer] {self.num_classes} classes, "
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
        DalesDataset populates with per-point DALES labels (with
        IGNORE_INDEX padding to fix the query count per batch).
        """
        is_train = (stage == "train")
        logits = self.forward(batch, training=is_train)        # [B, M, K]

        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[DALES-Trainer] Model returned {logits.shape[-1]} "
                f"classes, expected {self.num_classes}."
            )

        labels = batch["queries"][:, :, 4].long()              # [B, M]

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
        per_class = self.test_per_class_iou.compute()          # [8]
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
    # Optimizer (AdamW + cosine warmup) — identical to FRACTAL trainer
    # ─────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[DALES-Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[DALES-Trainer] WARN: cannot estimate total_steps. "
                  f"Falling back to {fallback}.")
            return fallback

        print(f"[DALES-Trainer] total_steps estimate: {est}")
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

        print(f"[DALES-Trainer] LR sched: total_steps={total_steps}, "
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
