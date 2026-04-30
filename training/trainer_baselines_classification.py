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
    forward(image: [B, C, H, W]) -> logits: [B, num_classes]
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
        model:        nn.Module with forward(image) -> logits [B, num_classes]
        modality:     dataset image dict key (e.g. "landsat")
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
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, image):
        return self.model(image)

    # ─────────────────────────────────────────────────────────────────────
    # SHARED STEP
    # ─────────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        image  = batch["image"][self.modality]    # [B, C, H, W]
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