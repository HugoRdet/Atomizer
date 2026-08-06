"""
Atomiser Multi-Task Trainer (ForestNet — classification only for now)
======================================================================

Reads batch["task"] (set by the dataset, propagated by the collate) and
dispatches to the appropriate forward + loss + metric path.

Currently handles:
    "classification" — uses Atomiser's task="classification" path
                       returns [B, num_classes] logits
                       CE loss on [B] scalar targets
                       metrics: top-1, top-5, macro-F1

Designed to extend with "segmentation" and "reconstruction" branches
when multi-task training is added.

Round-robin assumption:
    All samples in a batch have the same task. We read the task from the
    batch dict (set by the collate) and branch a single time per step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


class Model_ForestNet(pl.LightningModule):

    def __init__(
        self,
        config: dict,
        wand: bool = False,
        name: str = "forestnet",
        transform=None,
        lookup_table=None,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters(ignore=["lookup_table", "transform", "class_weights"])

        self.config = config
        self.name   = name
        self.wand   = wand

        # Build Atomiser
        from training.atomiser.Atomiser_SENFLOOD import Atomiser_Senflood
        self.model = Atomiser_Senflood(
            config=config,
            lookup_table=lookup_table,
        )

        self.num_classes     = config["trainer"]["num_classes"]
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Optimizer
        trainer_cfg          = config["trainer"]
        self.lr              = float(trainer_cfg.get("lr", 1e-4))
        self.weight_decay    = float(trainer_cfg.get("weight_decay", 1e-2))

        # Classification metrics — pass Metric objects so Lightning auto-syncs
        def _make_cls_metrics(prefix: str):
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

        self.train_cls_metrics = _make_cls_metrics("train")
        self.val_cls_metrics   = _make_cls_metrics("val")
        self.test_cls_metrics  = _make_cls_metrics("test")

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training: bool = True):
        task = batch.get("task", "classification")
        return self.model(batch, training=training, task=task)

    # =========================================================================
    # SHARED STEP (dispatches on task)
    # =========================================================================

    def _shared_step(self, batch, stage: str):
        task = batch.get("task", "classification")

        if task == "classification":
            return self._classification_step(batch, stage)
        elif task == "segmentation":
            # Hook for future multi-task — not used yet.
            raise NotImplementedError(
                "Segmentation branch not implemented in this trainer. "
                "Add when wiring multi-task."
            )
        else:
            raise ValueError(f"Unknown task: {task!r}")

    # =========================================================================
    # CLASSIFICATION BRANCH
    # =========================================================================

    def _classification_step(self, batch, stage: str):
        is_train = (stage == "train")

        target = batch["label"]                    # [B] long
        if target.dtype != torch.long:
            target = target.long()
        bs = target.shape[0]

        # Atomiser's task="classification" path:
        # encode → spatial latents → self.classify(latents_per_res)
        # → LatentAttentionPooling → LayerNorm → Linear → [B, num_classes]
        logits = self.model(batch, training=is_train, task="classification")

        # Defensive checks — fail loud rather than CUDA-assert.
        if logits.shape[-1] != self.num_classes:
            raise RuntimeError(
                f"[Model_ForestNet] Logits last dim {logits.shape[-1]} != "
                f"num_classes {self.num_classes}. "
                f"Check config['trainer']['num_classes'] in your YAML."
            )
        if target.numel() > 0:
            tmin, tmax = target.min().item(), target.max().item()
            if tmin < 0 or tmax >= self.num_classes:
                raise RuntimeError(
                    f"[Model_ForestNet] Targets out of range: "
                    f"min={tmin}, max={tmax}, num_classes={self.num_classes}."
                )

        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Update classification metrics
        cls_metrics = getattr(self, f"{stage}_cls_metrics")
        with torch.no_grad():
            for name, metric in cls_metrics.items():
                metric.update(logits, target)

        on_step = is_train
        self.log(
            f"{stage}_loss", loss,
            on_step=on_step, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=bs,
        )

        return loss

    def _epoch_end(self, stage: str):
        # Log all classification metrics by passing Metric objects to self.log
        # (Lightning handles compute/sync/reset automatically).
        cls_metrics = getattr(self, f"{stage}_cls_metrics")
        for name, metric in cls_metrics.items():
            self.log(name, metric, prog_bar=("top1" in name))

    # =========================================================================
    # LIGHTNING HOOKS
    # =========================================================================

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

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 1e-2,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
