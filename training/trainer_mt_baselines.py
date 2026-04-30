"""
Multi-task Lightning trainer.

Drives training and validation for a multi-task baseline:
    - Dispatches each batch to the right per-task head via `batch["task"]`.
    - Computes CE loss (segmentation: ignore_index=255; classification:
      no ignore index). Loss is divided by the number of tasks so that
      with `accumulate_grad_batches=num_tasks`, the optimizer sees the
      gradient of the *mean* per-task loss — equal task weights with
      LR semantics that match single-task training.
    - Tracks per-task primary metric (mIoU for seg, top-1 acc for cls)
      on validation. Logs each, plus their mean as `val/mean_primary`,
      which is the metric used for `ModelCheckpoint(monitor=...)`.

Expected `model` interface:
    model(image, task) -> logits
        image: [B, 15, H, W] (single-frame) or [B, T, 15, H, W] (PASTIS).
        task:  string key into the model's per-task heads.
        seg returns [B, num_classes, H, W]; cls returns [B, num_classes].

Expected batch shape (from collate):
    {
        "task": str,
        "image": {"input": tensor},
        "target": tensor,
        "valid_mask": tensor (uint8),
        "original_size": tensor,
        "metadata": list,
    }
"""

import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)

IGNORE_INDEX=255


# ────────────────────────────────────────────────────────────────────
# Per-task metric containers
# ────────────────────────────────────────────────────────────────────

def _build_seg_metrics(num_classes: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "miou": MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            average="macro",
        ),
        "acc": MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            average="micro",
        ),
    })


def _build_cls_metrics(num_classes: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "top1": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "macro_acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
    })


def _build_metrics_for_task(spec: dict) -> nn.ModuleDict:
    if spec["type"] == "seg":
        return _build_seg_metrics(spec["num_classes"])
    if spec["type"] == "cls":
        return _build_cls_metrics(spec["num_classes"])
    raise ValueError(f"Unknown task type: {spec['type']}")


# ────────────────────────────────────────────────────────────────────
# MultiTaskTrainer
# ────────────────────────────────────────────────────────────────────

class MultiTaskTrainer(pl.LightningModule):
    """
    LightningModule for multi-task baseline training.

    Args:
        model:           nn.Module with `forward(image, task) -> logits`.
        task_configs:    {task_name: {"type": "seg"|"cls", "num_classes": int}}.
                         Order is preserved and used for the round-robin.
        lr:              base AdamW learning rate.
        weight_decay:    AdamW weight decay.
        max_steps:       total number of optimizer steps (used for cosine schedule).
        warmup_steps:    linear warmup steps.
        primary_metric_per_type:
                         which metric to use for the cross-task mean. Defaults to
                         {"seg": "miou", "cls": "top1"}.
    """

    def __init__(
        self,
        model: nn.Module,
        task_configs: dict,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_steps: int = 30000,
        warmup_steps: int = 1500,
        primary_metric_per_type: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        self.num_tasks = len(self.task_names)
        self.primary = primary_metric_per_type or {"seg": "miou", "cls": "top1"}

        # Per-task metric ModuleDicts, one set per split.
        self.val_metrics  = nn.ModuleDict({
            t: _build_metrics_for_task(c) for t, c in task_configs.items()
        })
        self.test_metrics = nn.ModuleDict({
            t: _build_metrics_for_task(c) for t, c in task_configs.items()
        })

    # ─────────────────────────────────────────────────────────────
    # Forward / loss
    # ─────────────────────────────────────────────────────────────

    def forward(self, image, task):
        return self.model(image, task)

    def _compute_loss(self, logits, target, task_type):
        if task_type == "seg":
            # logits: [B, K, H, W], target: [B, H, W] (255 = ignore + pad)
            return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        # cls: logits [B, K], target [B]
        return F.cross_entropy(logits, target)

    # ─────────────────────────────────────────────────────────────
    # Train / val / test steps
    # ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        task = batch["task"]
        cfg = self.task_configs[task]
        image = batch["image"]["input"]
        target = batch["target"]

        logits = self.model(image, task)
        loss = self._compute_loss(logits, target, cfg["type"])

        # Equal task weights: divide by num_tasks so accumulating N
        # micro-steps gives the gradient of the mean task loss.
        scaled = loss / self.num_tasks

        bs = image.shape[0]
        self.log(f"train/{task}/loss", loss, on_step=True, on_epoch=False,
                 batch_size=bs, prog_bar=False)
        self.log("train/loss_scaled", scaled, on_step=True, on_epoch=False,
                 batch_size=bs, prog_bar=True)
        return scaled

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        task = batch["task"]
        cfg = self.task_configs[task]
        image = batch["image"]["input"]
        target = batch["target"]

        logits = self.model(image, task)
        loss = self._compute_loss(logits, target, cfg["type"])
        preds = logits.argmax(dim=1)

        bs = image.shape[0]
        self.log(f"val/{task}/loss", loss, on_step=False, on_epoch=True,
                 batch_size=bs, add_dataloader_idx=False)

        # Update per-task metrics. torchmetrics handles ignore_index for seg.
        for m in self.val_metrics[task].values():
            m.update(preds, target)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        task = batch["task"]
        cfg = self.task_configs[task]
        image = batch["image"]["input"]
        target = batch["target"]

        logits = self.model(image, task)
        loss = self._compute_loss(logits, target, cfg["type"])
        preds = logits.argmax(dim=1)

        bs = image.shape[0]
        self.log(f"test/{task}/loss", loss, on_step=False, on_epoch=True,
                 batch_size=bs, add_dataloader_idx=False)
        for m in self.test_metrics[task].values():
            m.update(preds, target)

    # ─────────────────────────────────────────────────────────────
    # Epoch-end aggregation
    # ─────────────────────────────────────────────────────────────

    def _aggregate_split(self, metrics_dict: nn.ModuleDict, split: str):
        """Compute, log, and reset per-task metrics; return primary metric values."""
        primary_values = []
        for task in self.task_names:
            cfg = self.task_configs[task]
            m_set = metrics_dict[task]
            for name, metric in m_set.items():
                v = metric.compute()
                self.log(f"{split}/{task}/{name}", v, sync_dist=True,
                         add_dataloader_idx=False)
                metric.reset()
                if name == self.primary[cfg["type"]]:
                    primary_values.append(v)
        if primary_values:
            mean_primary = torch.stack(primary_values).mean()
            self.log(f"{split}/mean_primary", mean_primary,
                     sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    def on_validation_epoch_end(self):
        self._aggregate_split(self.val_metrics, "val")

    def on_test_epoch_end(self):
        self._aggregate_split(self.test_metrics, "test")

    # ─────────────────────────────────────────────────────────────
    # Optimizer + scheduler
    # ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        max_steps = max(1, self.hparams.max_steps)
        warmup    = max(0, self.hparams.warmup_steps)

        def lr_lambda(step: int):
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }