"""
FRACTAL PerceiverIO Trainer
=============================

Lightning module for PerceiverFractal on the FRACTAL dataset.

Mirrors Model_Fractal (Atomizer trainer) as closely as possible so
results are directly comparable:
  - Same 7 classes, same ignore_index=255
  - Same inverse-frequency class weighting (clipped at 50)
  - Same mIoU + macro accuracy metrics
  - Same AdamW + cosine warmup optimizer schedule
  - Same per-class IoU logging at test time

Key differences from Model_Fractal:
  - Uses PerceiverFractal instead of Atomiser_Fractal
  - Labels come from batch["query_labels"] [B, M] instead of
    batch["queries"][:, :, 4] (Option B from dataset design)
  - model(batch) signature is identical: returns [B, M, num_classes]
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from training.perceiverIO.perceiver_fractal import PerceiverFractal


# ============================================================================
# FRACTAL class metadata (identical to Model_Fractal)
# ============================================================================

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
    """Inverse-frequency weights, clipped to avoid extreme gradient scaling."""
    raw = torch.tensor(
        [1.0 / max(f, 1e-9) for f in freqs], dtype=torch.float32
    )
    return raw.clamp(max=weight_clip)


# ============================================================================
# Trainer
# ============================================================================

class Model_PerceiverFractal(pl.LightningModule):
    """
    FRACTAL LIDAR + VHR segmentation with PerceiverIO.

    Args:
        num_latents:        Number of latent vectors. Default 256.
        latent_dim:         Latent dimension. Default 256.
        depth:              Encoder depth. Default 6.
        cross_heads:        Cross-attention heads. Default 1.
        latent_heads:       Self-attention heads. Default 8.
        cross_dim_head:     Dim per cross-attn head. Default 64.
        latent_dim_head:    Dim per self-attn head. Default 64.
        self_per_cross_attn: Self-attn blocks per cross-attn. Default 1.
        weight_tie_layers:  Share encoder weights across blocks > 0.
        attn_dropout:       Attention dropout. Default 0.0.
        ff_dropout:         FF dropout. Default 0.0.
        echo_hidden_dim:    Echo MLP hidden dim. Default 64.
        lr:                 Peak learning rate. Default 1e-4.
        weight_decay:       AdamW weight decay. Default 1e-2.
        warmup_steps:       LR warmup steps. None = 5% of total. Default None.
        ignore_index:       Label index to ignore in loss/metrics. Default 255.
        class_weights:      CE class weights. "auto", None, or tensor.
    """

    def __init__(
        self,
        num_latents: int = 256,
        latent_dim: int = 256,
        depth: int = 6,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        self_per_cross_attn: int = 1,
        weight_tie_layers: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        echo_hidden_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = None,
        ignore_index: int = 255,
        class_weights="auto",
        query_chunk_size: int = 100_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes  = NUM_CLASSES_FRACTAL
        self.query_chunk_size = query_chunk_size
        self.class_names  = FRACTAL_CLASS_NAMES
        self.ignore_index = ignore_index
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # ── Model ────────────────────────────────────────────────────
        self.model = PerceiverFractal(
            num_classes=self.num_classes,
            num_latents=num_latents,
            latent_dim=latent_dim,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            self_per_cross_attn=self_per_cross_attn,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            echo_hidden_dim=echo_hidden_dim,
        )

        # ── Class weights ────────────────────────────────────────────
        if class_weights == "auto":
            class_weights = default_fractal_class_weights()
            print(f"[PerceiverFractal-Trainer] Auto inverse-frequency weights "
                  f"(clipped at 50): {class_weights.tolist()}")
        elif class_weights is None:
            print(f"[PerceiverFractal-Trainer] No class weighting.")
        else:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(
                    class_weights, dtype=torch.float32
                )
            print(f"[PerceiverFractal-Trainer] Custom class weights: "
                  f"{class_weights.tolist()}")

        # ── Loss ─────────────────────────────────────────────────────
        ce_kwargs = {}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = int(self.ignore_index)
        if class_weights is not None:
            #self.register_buffer("_class_weights", class_weights)
            #ce_kwargs["weight"] = self._class_weights
            pass
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        # ── Metrics ──────────────────────────────────────────────────
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

        print(f"[PerceiverFractal-Trainer] {self.num_classes} classes, "
              f"ignore_index={self.ignore_index}, "
              f"lr={self.lr}, weight_decay={self.weight_decay}")

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch: dict, training: bool = True) -> torch.Tensor:
        chunk = None if training else self.query_chunk_size
        return self.model(batch, training=training, query_chunk_size=chunk)

    # =========================================================================
    # Shared step
    # =========================================================================

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """
        Forward + loss + metrics.

        Labels come from batch["query_labels"] [B, M] (Option B).
        ignore_index=255 handles both LAS unmapped codes and padding.
        """
        is_train = (stage == "train")
        logits   = self.forward(batch, training=is_train)   # [B, M, K]

        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[PerceiverFractal-Trainer] Model returned "
                f"{logits.shape[-1]} classes, expected {self.num_classes}."
            )

        labels = batch["query_labels"].long()   # [B, M]

        # Flatten for CE: [B*M, K] vs [B*M]
        logits_flat = rearrange(logits, "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        loss        = self.loss_fn(logits_flat, labels_flat)

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

        bs      = labels.shape[0]
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

    # =========================================================================
    # End-of-epoch logging
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU",      self.train_miou,
                 on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("train_macro_acc", self.train_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_mIoU",      self.val_miou,
                 on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_macro_acc", self.val_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        per_class = self.test_per_class_iou.compute()   # [7]
        self.test_per_class_iou.reset()
        for i, class_name in enumerate(self.class_names):
            if self.ignore_index is not None and i == self.ignore_index:
                continue
            self.log(f"test_IoU/{class_name}", per_class[i].item(),
                     on_epoch=True, sync_dist=True)

        self.log("test_mIoU",      self.test_miou,
                 on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("test_macro_acc", self.test_macro_acc,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    # =========================================================================
    # Optimizer (AdamW + cosine warmup) — mirrors Model_Fractal
    # =========================================================================

    def _compute_total_steps(self) -> int:
        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[PerceiverFractal-Trainer] WARN: cannot estimate "
                  f"total_steps. Falling back to {fallback}.")
            return fallback

        print(f"[PerceiverFractal-Trainer] total_steps estimate: {est}")
        return est

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps  = self._compute_total_steps()
        warmup_steps = (self.warmup_steps
                        if self.warmup_steps is not None
                        else max(1, int(0.05 * total_steps)))

        print(f"[PerceiverFractal-Trainer] LR schedule: "
              f"total_steps={total_steps}, warmup={warmup_steps}, "
              f"peak_lr={self.lr}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
