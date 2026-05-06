"""
Atomizer Multi-Task Trainer
============================

Lightning module for multi-task training of Atomizer across heterogeneous
EO benchmarks (BurnScars, Sen1Floods11, PASTIS, EuroSAT, ForestNet).

Mirrors the structure of MultiTaskTrainer (used for ResNet/ViT MT
baselines) but with Atomizer's forward signature:

    Baselines:  model(image, task) -> logits
    Atomizer:   model(batch, training=...) -> logits  (task read from batch["task"])

Key responsibilities:
  - Dispatch loss + metrics by task_type (seg vs cls).
  - Track per-task primary metrics with seen-task gating (skip metrics
    for tasks the model hasn't been evaluated on yet — avoids NaN/0.0
    pollution from torchmetrics computing on no updates).
  - Aggregate `mean_primary` across all tasks for checkpointing.
  - Round-robin assumption: one task per batch; batch["task"] is the
    task name (not type) and is set by the per-task collate.

Round-robin assumption
----------------------
All samples in a batch share the same task. The collate sets
batch["task"] = task_name and batch["task_type"] = "seg"|"cls".
Mixed-task batches are NOT supported here — would require per-sample
dispatch which we explicitly skip.

Token format reminder (for label extraction in seg):
    queries: [B, M, 8]
        cols [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
    Per-pixel class label is at queries[:, :, 4].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

# Path to the MultiTaskAtomiser model — adjust if your import path differs.
# (User created the file as Atomiser_mt.py; importable from
#  training.atomiser.atomiser_mt as MultiTaskAtomiser.)
from training.atomiser.Atomiser_mt import MultiTaskAtomiser


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_seg_metrics(num_classes: int, ignore_index: int) -> nn.ModuleDict:
    """
    Standard seg metrics for a single task. Macro-averaged so primary
    metric (mIoU) is comparable across tasks with different class counts.
    """
    return nn.ModuleDict({
        "mIoU": torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes,
            average="macro", ignore_index=ignore_index,
        ),
        "macro_acc": MulticlassAccuracy(
            num_classes=num_classes,
            average="macro", ignore_index=ignore_index,
        ),
    })


def _make_cls_metrics(num_classes: int) -> nn.ModuleDict:
    """
    Standard cls metrics for a single task. Primary metric is macro_f1
    (more robust to class imbalance than top1 — important since EuroSAT
    and ForestNet have skewed class distributions).
    """
    return nn.ModuleDict({
        "top1": MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro",
        ),
        "top5": MulticlassAccuracy(
            num_classes=num_classes,
            top_k=min(5, num_classes), average="micro",
        ),
        "macro_acc": MulticlassAccuracy(
            num_classes=num_classes, average="macro",
        ),
        "macro_f1": MulticlassF1Score(
            num_classes=num_classes, average="macro",
        ),
    })


# ────────────────────────────────────────────────────────────────────
# AtomizerMultiTaskTrainer
# ────────────────────────────────────────────────────────────────────

class AtomizerMultiTaskTrainer(pl.LightningModule):
    """
    Atomizer multi-task Lightning module.

    Args:
        config:        Atomizer config dict.
        lookup_table:  Spectral/resolution lookup.
        task_specs:    {task_name: {"type": "seg"|"cls",
                                    "num_classes": int,
                                    "primary": "mIoU"|"macro_f1",  (optional)
                                    "ignore_index": int           (optional, seg only)}}
                       Order of the dict determines metric/log ordering.
        wand:          Whether to log to W&B (forwarded by user; not used internally
                       beyond storing the flag).
        name:          Experiment name (used for ckpt-naming hooks).
        transform:     Unused; kept for API symmetry with single-task trainers.
    """

    DEFAULT_IGNORE_INDEX = 255

    def __init__(
        self,
        config: dict,
        wand: bool,
        name: str,
        transform=None,
        lookup_table=None,
        task_specs: dict = None,
    ):
        super().__init__()
        if not task_specs:
            raise ValueError("task_specs must contain at least one task.")

        self.strict_loading = False
        self.config       = config
        self.transform    = transform
        self.wand         = wand
        self.name         = name
        self.lookup_table = lookup_table
        self.task_specs   = dict(task_specs)
        self._task_names  = list(self.task_specs.keys())

        # ── Build the multi-task Atomizer model ───────────────────────
        # MultiTaskAtomiser disables error_predictor / refinement /
        # targeted_depth2 internally for clean MT comparison.
        self.model = MultiTaskAtomiser(
            config=config,
            lookup_table=lookup_table,
            task_specs=task_specs,
        )

        # ── Per-task losses ──────────────────────────────────────────
        # Seg: CE with ignore_index. Cls: vanilla CE.
        # We build one CE per task because seg ignore_index can differ
        # (e.g., PASTIS uses 255, BurnScars uses 255 — uniform here, but
        # config could vary).
        self.losses = nn.ModuleDict()
        for task, spec in self.task_specs.items():
            if spec["type"] == "seg":
                ignore_index = spec.get("ignore_index", self.DEFAULT_IGNORE_INDEX)
                self.losses[task] = nn.CrossEntropyLoss(ignore_index=ignore_index)
            elif spec["type"] == "cls":
                self.losses[task] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown type {spec['type']!r} for task {task!r}")

        # ── Per-task metrics, per-split ──────────────────────────────
        # Lightning's auto-sync handles distributed training when we pass
        # Metric objects to self.log directly.
        for split in ("train", "val", "test"):
            mdict = nn.ModuleDict()
            for task, spec in self.task_specs.items():
                if spec["type"] == "seg":
                    ignore_index = spec.get("ignore_index", self.DEFAULT_IGNORE_INDEX)
                    mdict[task] = _make_seg_metrics(spec["num_classes"], ignore_index)
                elif spec["type"] == "cls":
                    mdict[task] = _make_cls_metrics(spec["num_classes"])
            setattr(self, f"_metrics_{split}", mdict)

        # ── Seen-task tracking ───────────────────────────────────────
        # We only compute & log metrics for tasks that were actually
        # evaluated this epoch. Without this, tasks with no batches
        # (e.g., when the test loader for one task hasn't been called yet)
        # would log torchmetrics' default value (often 0.0 or NaN) and
        # poison the mean_primary aggregate.
        self._train_seen_tasks: set = set()
        self._val_seen_tasks:   set = set()
        self._test_seen_tasks:  set = set()

        # ── Optimizer config ─────────────────────────────────────────
        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        n_seg = sum(1 for s in self.task_specs.values() if s["type"] == "seg")
        n_cls = sum(1 for s in self.task_specs.values() if s["type"] == "cls")
        print(f"[MT-Atomizer] Trainer built: {n_seg} seg + {n_cls} cls task(s).")
        for task, spec in self.task_specs.items():
            primary = self._primary_metric_name(task)
            print(f"[MT-Atomizer]   {task:<14} type={spec['type']:<3} "
                  f"K={spec['num_classes']:<3} primary={primary}")

    # ─────────────────────────────────────────────────────────────────
    # Primary metric resolution
    # ─────────────────────────────────────────────────────────────────

    def _primary_metric_name(self, task: str) -> str:
        """
        Default primary metric:
          seg → mIoU
          cls → macro_f1
        Overridable per task via task_specs[task]["primary"].
        """
        spec = self.task_specs[task]
        if "primary" in spec:
            return spec["primary"]
        return "mIoU" if spec["type"] == "seg" else "macro_f1"

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch, training: bool = True):
        return self.model(batch, training=training)

    # ─────────────────────────────────────────────────────────────────
    # Per-task step
    # ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        """
        Dispatch by batch["task"] (set by per-task collate).
        Returns the task's loss; metrics are updated in-place on the
        per-stage ModuleDict.
        """
        task = batch.get("task")
        if task is None:
            raise KeyError(
                "AtomizerMultiTaskTrainer expects batch['task'] to be set "
                "by the per-task collate."
            )
        if task not in self.task_specs:
            raise KeyError(
                f"Unknown task {task!r}; expected one of {self._task_names}"
            )
        spec = self.task_specs[task]

        # Track which tasks we've seen — gates metric aggregation later.
        getattr(self, f"_{stage}_seen_tasks").add(task)

        is_train = (stage == "train")

        if spec["type"] == "seg":
            return self._seg_step(batch, task, spec, stage, is_train)
        if spec["type"] == "cls":
            return self._cls_step(batch, task, spec, stage, is_train)
        raise ValueError(f"Unknown type {spec['type']!r} for task {task!r}")

    # ─────────────────────────────────────────────────────────────────
    # Seg step
    # ─────────────────────────────────────────────────────────────────

    def _seg_step(self, batch, task: str, spec: dict, stage: str, is_train: bool):
        """
        Seg forward + loss + metrics.

        Forward returns [B, M, K_task] — per-pixel logits for the M queries.
        Labels come from queries[:, :, 4] (col 4 is per-pixel class label).
        """
        logits = self.forward(batch, training=is_train)        # [B, M, K]

        # Defensive shape check — fail loud rather than CUDA assert.
        if logits.shape[-1] != spec["num_classes"]:
            raise RuntimeError(
                f"[MT-Atomizer] Task {task!r}: model returned "
                f"{logits.shape[-1]} classes, expected {spec['num_classes']}. "
                f"Check task_specs vs MultiTaskAtomiser.seg_heads."
            )

        labels = batch["queries"][:, :, 4].long()                # [B, M]

        # Flatten for CE: [B*M, K] vs [B*M].
        logits_flat = rearrange(logits, "b m c -> (b m) c")
        labels_flat = rearrange(labels, "b m   -> (b m)")
        loss = self.losses[task](logits_flat, labels_flat)

        # Metrics — apply argmax once and update every metric for this task.
        with torch.no_grad():
            preds = logits.argmax(dim=-1)                        # [B, M]
            metrics = getattr(self, f"_metrics_{stage}")[task]
            for m in metrics.values():
                m.update(preds, labels)

        bs = batch["queries"].shape[0]
        on_step = is_train
        self.log(
            f"{stage}_loss/{task}", loss,
            on_step=on_step, on_epoch=True,
            prog_bar=is_train, sync_dist=True,
            batch_size=bs,
        )
        return loss

    # ─────────────────────────────────────────────────────────────────
    # Cls step
    # ─────────────────────────────────────────────────────────────────

    def _cls_step(self, batch, task: str, spec: dict, stage: str, is_train: bool):
        """
        Cls forward + loss + metrics.

        Forward returns [B, K_task]. Labels come from batch["label"]
        (scalar [B] — set by the cls collate).
        """
        logits = self.forward(batch, training=is_train)          # [B, K]

        if logits.shape[-1] != spec["num_classes"]:
            raise RuntimeError(
                f"[MT-Atomizer] Task {task!r}: model returned "
                f"{logits.shape[-1]} classes, expected {spec['num_classes']}. "
                f"Check task_specs vs MultiTaskAtomiser.cls_heads."
            )

        target = batch["label"]                                  # [B]
        if target.dtype != torch.long:
            target = target.long()

        # Defensive label-range check — catches mis-keyed datasets early.
        if target.numel() > 0:
            tmin, tmax = target.min().item(), target.max().item()
            if tmin < 0 or tmax >= spec["num_classes"]:
                raise RuntimeError(
                    f"[MT-Atomizer] Task {task!r}: targets out of range "
                    f"[{tmin}, {tmax}] vs num_classes={spec['num_classes']}."
                )

        loss = self.losses[task](logits, target)

        with torch.no_grad():
            metrics = getattr(self, f"_metrics_{stage}")[task]
            for m in metrics.values():
                m.update(logits, target)

        bs = target.shape[0]
        on_step = is_train
        self.log(
            f"{stage}_loss/{task}", loss,
            on_step=on_step, on_epoch=True,
            prog_bar=is_train, sync_dist=True,
            batch_size=bs,
        )
        return loss

    # ─────────────────────────────────────────────────────────────────
    # Lightning hooks
    # ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self._aggregate_split("train", self._train_seen_tasks)
        self._train_seen_tasks.clear()

    def on_validation_epoch_end(self):
        self._aggregate_split("val", self._val_seen_tasks)
        self._val_seen_tasks.clear()

    def on_test_epoch_end(self):
        self._aggregate_split("test", self._test_seen_tasks)
        self._test_seen_tasks.clear()

    # ─────────────────────────────────────────────────────────────────
    # Aggregate per-task metrics + mean_primary
    # ─────────────────────────────────────────────────────────────────

    def _aggregate_split(self, stage: str, seen_tasks: set):
        """
        Compute & log per-task metrics for tasks seen in this epoch,
        then a `<stage>_mean_primary` aggregate across them.

        Logging shape:
            <stage>_<metric>/<task>          per-task metric value
            <stage>_mean_primary             mean over tasks of each task's
                                              primary metric

        Metric reset is delegated to torchmetrics (we pass the Metric
        object to self.log so Lightning calls .compute() and .reset()
        for us). We additionally call .reset() defensively below for any
        unused tasks (they remain at default values otherwise; harmless,
        but cleaner state).
        """
        primary_values = []

        all_metrics = getattr(self, f"_metrics_{stage}")
        for task, mdict in all_metrics.items():
            if task not in seen_tasks:
                # Reset unused metrics so they don't accumulate stale
                # state across epochs (e.g., if a task is skipped one
                # epoch but evaluated the next).
                for m in mdict.values():
                    m.reset()
                continue

            primary = self._primary_metric_name(task)
            for metric_name, metric in mdict.items():
                # Pass the Metric object — Lightning syncs+resets.
                self.log(
                    f"{stage}_{metric_name}/{task}", metric,
                    on_epoch=True, prog_bar=(metric_name == primary),
                    sync_dist=True,
                )

                # Also track the primary metric value for the mean.
                # We compute() here to capture it for averaging; the
                # log() call above will compute() again, but torchmetrics
                # caches the result within the same step so this is cheap.
                if metric_name == primary:
                    val = metric.compute()
                    if torch.is_tensor(val):
                        val = val.detach()
                    primary_values.append(val)

        if primary_values:
            mean_primary = torch.stack([
                v if torch.is_tensor(v) else torch.tensor(float(v))
                for v in primary_values
            ]).mean()
            self.log(
                f"{stage}_mean_primary", mean_primary,
                on_epoch=True, prog_bar=True, sync_dist=True,
            )

    # ─────────────────────────────────────────────────────────────────
    # Optimizer (AdamW + cosine warmup, mirroring single-task SenFlood)
    # ─────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        """
        Estimate total optimizer steps across the full MT run.

        For MT we sum the per-task DataLoader lengths (the round-robin
        loader yields one batch per task per round, so its length is
        approximately num_tasks × max(per_task_batches_per_epoch) — but
        we don't have direct access to that here, so we fall back to
        Lightning's estimated_stepping_batches with a sanity print).

        Priority:
          1. Config override `trainer.total_steps`
          2. Lightning's estimated_stepping_batches
        """
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[MT-Atomizer] total_steps override from config: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            # Last resort: epochs × 1000 — clearly a fudge but safer than 0.
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[MT-Atomizer] [WARN] Cannot estimate total_steps. "
                  f"Falling back to {fallback}. Set trainer.total_steps "
                  f"in config for accurate cosine schedule.")
            return fallback

        print(f"[MT-Atomizer] total_steps from Lightning estimate: {est}")
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

        print(f"[MT-Atomizer] LR schedule: total_steps={total_steps}, "
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