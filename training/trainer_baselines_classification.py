"""
ClassificationBaselineTrainer
===============================

PyTorch Lightning module for single-label image classification baselines.

Mirror of `BaselineTrainer` (segmentation) but with classification metrics
(top-1, top-5 accuracy, macro-F1) and CE loss on logits of shape [B, num_classes].

Expected batch format (from collate):
    {
        "image":  {modality_key: [B, C, H, W]},
        "target": [B] long (class indices in [0, num_classes)),
        "metadata": [...],
    }

Model contract:
    Standard models:  forward(image: [B, C, H, W]) -> logits: [B, num_classes]
    RAMEN-style:       forward(image: dict[modality] -> [B, C, H, W]) -> logits: [B, num_classes]
                       (model sets `expects_full_image_dict = True` — see _get_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


class ClassificationBaselineTrainer(pl.LightningModule):
    """
    Args:
        model:        nn.Module with forward(image) -> logits [B, num_classes].
                      Standard models take a single [B,C,H,W] tensor (selected
                      via `modality`). Multi-modal models (e.g. RAMEN) that
                      need the entire modality dict should set a class
                      attribute `expects_full_image_dict = True` — see
                      _get_image below. `modality` is then unused for input
                      extraction (only shown in logs), but is still required
                      as a constructor arg.
        modality:     dataset image dict key (e.g. "fused"). Ignored for
                      input extraction when model.expects_full_image_dict
                      is True.
        num_classes:  number of classes
        lr:           initial learning rate
        weight_decay: AdamW weight decay
        class_weights: optional [num_classes] tensor for weighted CE loss
        label_smoothing: float for CE label smoothing (0 = none)
    """

    def __init__(
        self,
        model: nn.Module,
        modality: str,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.model = model
        self.modality = modality
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Metrics — separate instances per split (Lightning handles DDP sync)
        def _make_metrics(prefix: str):
            return nn.ModuleDict({
                f"{prefix}_top1":  MulticlassAccuracy(
                    num_classes=num_classes, top_k=1, average="micro"),
                f"{prefix}_top5":  MulticlassAccuracy(
                    num_classes=num_classes,
                    top_k=min(5, num_classes), average="micro"),
                f"{prefix}_macro_acc": MulticlassAccuracy(
                    num_classes=num_classes, average="macro"),
                f"{prefix}_macro_f1":  MulticlassF1Score(
                    num_classes=num_classes, average="macro"),
            })

        self.train_metrics = _make_metrics("train")
        self.val_metrics   = _make_metrics("val")
        self.test_metrics  = _make_metrics("test")

    # ─────────────────────────────────────────────────────────────────────
    # Input extraction
    # ─────────────────────────────────────────────────────────────────────

    def _get_image(self, batch):
        """
        Extract the model input from a batch.

        Models that need the entire modality dict (e.g. a RAMEN-style
        classifier, which looks up a separate spectral projector per
        modality internally) set a class attribute
        `expects_full_image_dict = True`. Checked with getattr rather
        than isinstance so that wrappers around such a model (e.g. a
        modality-drop wrapper) are also recognized correctly as long as
        they carry the same attribute. Every other model expects a
        single modality's tensor, selected via self.modality.
        """
        if getattr(self.model, "expects_full_image_dict", False):
            return batch["image"]  # dict[modality] -> Tensor, passed through as-is
        return batch["image"][self.modality]  # [B, C, H, W]

    # ─────────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, image):
        return self.model(image)

    # ─────────────────────────────────────────────────────────────────────
    # SHARED STEP
    # ─────────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        image  = self._get_image(batch)
        target = batch["target"]                   # [B] long
        bs = target.shape[0]

        logits = self.model(image)                 # [B, num_classes]

        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Update metrics (torchmetrics handles its own DDP state)
        metrics = getattr(self, f"{stage}_metrics")
        with torch.no_grad():
            for name, metric in metrics.items():
                metric.update(logits, target)

        # Log loss explicitly with batch_size so Lightning's auto-inference
        # doesn't get confused by the dict structure of the batch.
        on_step = (stage == "train")
        self.log(
            f"{stage}_loss", loss,
            on_step=on_step, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=bs,
        )

        return loss

    def _epoch_end(self, stage: str):
        """
        Pass each Metric object directly to self.log (NOT metric.compute()).
        Lightning detects torchmetrics.Metric instances and handles
        compute / DDP-sync / reset automatically. Bypassing this with manual
        .compute() on the local rank produces unsynced scalars and can hang
        DDP at the next collective.
        """
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            self.log(name, metric, prog_bar=("top1" in name))

    # ─────────────────────────────────────────────────────────────────────
    # LIGHTNING HOOKS
    # ─────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    # ─────────────────────────────────────────────────────────────────────
    # OPTIMIZER
    # ─────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.trainer.max_epochs, eta_min=self.lr * 1e-2,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
