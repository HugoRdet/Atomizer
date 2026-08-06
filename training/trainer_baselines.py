"""
Baseline Segmentation Trainer
==============================

PyTorch Lightning trainer for fixed-format baseline models (UNet, ViT,
TemporalUNet / U-TAE, RAMEN) on segmentation tasks.

Batch format:
    {
        "image":    {modality: [B, C, H, W] or [B, T, C, H, W]},
        "dates":    {modality: [B, T]},          (optional, for temporal models)
        "target":   [B, H, W],
        "metadata": [list of dicts],
    }

Architecture:
    Non-temporal: model(image) → logits [B, num_classes, H, W]
    Temporal:     model(image, doy=dates) → logits [B, num_classes, H, W]
    RAMEN:        model(image_dict) → logits [B, num_classes, H, W]
                  (image_dict = batch["image"] passed through as-is,
                   since RAMENUPerNet looks up a separate spectral
                   projector per modality internally)

Supports:
    - Single-modality training (one sensor key)
    - Temporal models (U-TAE) with day-of-year positional encoding
    - Multi-modal models (RAMEN) that consume the full modality dict
    - Cross-sensor evaluation
    - Per-class IoU logging
    - Cosine LR schedule with warmup
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup
from training.sliding_window import sliding_window_inference  # adjust import path


IGNORE_INDEX = 255


# ─────────────────────────────────────────────────────────────────────
# Task-specific class registries
# ─────────────────────────────────────────────────────────────────────

TASK_CLASS_NAMES = {
    "mdas": {
        0: "Pavement", 1: "Soil", 2: "Roof",
        3: "Low vegetation", 4: "Tree", 5: "Water",
    },
    "pastis": {
        0: "Background",      1: "Meadow",          2: "Soft Winter Wheat",
        3: "Corn",            4: "Winter Barley",    5: "Winter Rapeseed",
        6: "Spring Barley",   7: "Sunflower",        8: "Grapevine",
        9: "Beet",           10: "Soy",             11: "Sorghum",
        12: "Flax",          13: "Protein Crops",   14: "Other Cereals",
        15: "Fruits/Veg",    16: "Other Crops",     17: "Grassland",
        18: "Shrub/Forest",
    },
    "c2seg": {
        0: "Background",     1: "Urban Fabric",     2: "Industrial",
        3: "Street Network", 4: "Mine/Dump",        5: "Artif. Vegetated",
        6: "Arable Land",    7: "Low Vegetation",   8: "Forests",
        9: "Water",
    },
    "multiearth": {
        0: "Forest",         1: "Deforested",
    },
    "burnscars": {0: "No burn", 1: "Burn scar"},
}


class BaselineTrainer(pl.LightningModule):
    """
    Single-task segmentation trainer for baseline models.

    Handles standard models (UNet, ViT), temporal models (TemporalUNet /
    U-TAE), and multi-modal models (RAMEN) transparently. Temporal models
    receive day-of-year positional encoding from batch["dates"]. RAMEN
    receives the entire batch["image"] dict rather than a single
    modality's tensor.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
        Non-temporal: [B, C, H, W] → [B, num_classes, H, W]
        Temporal:     [B, T, C, H, W], doy=[B, T] → [B, num_classes, H, W]
        RAMEN:        dict[modality] -> Tensor → [B, num_classes, H, W]
    modality : str
        Key in batch["image"] to use as input (e.g., "s2", "hyspex").
        Ignored when model is a RAMENUPerNet — pass any placeholder
        (e.g. "optical+sar") since it's only used in the startup print.
    temporal : bool
        If True, pass dates to model and expect [B, T, C, H, W] input.
    task : str
        Task name for class registry ("pastis", "mdas", "c2seg").
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    num_classes : int
        Number of segmentation classes.
    ignore_index : int
        Label value to ignore in loss and metrics.
    window_size : int, optional
        For models that can't afford a full-resolution forward pass
        (currently: RAMEN, whose pixel-level tokenization makes full
        self-attention over a large image intractable), the spatial size
        the model was built/trained for. At val/test time, if the input
        is larger than window_size, sliding_window_inference tiles the
        input into overlapping windows of this size and stitches the
        logits back together. Ignored for all other models, and ignored
        during training (training already sees pre-cropped windows of
        this size directly from the dataset/collate, so no tiling is
        needed there).
    window_stride : int, optional
        Step between windows for sliding-window inference. Must be
        <= window_size; stride < window_size gives overlap (averaged in
        the output), stride == window_size gives non-overlapping tiling.
        Required if window_size is set.
    """

    def __init__(
        self,
        model: nn.Module,
        modality: str = "s2",
        temporal: bool = False,
        task: str = "pastis",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_classes: int = 20,
        ignore_index: int = IGNORE_INDEX,
        window_size: int = None,
        window_stride: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.modality = modality
        self.temporal = temporal
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.window_size = window_size
        self.window_stride = window_stride if window_stride is not None else window_size

        self.class_names = TASK_CLASS_NAMES.get(task, {})

        # ── Loss ────────────────────────────────────────────────────
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # ── Metrics ─────────────────────────────────────────────────
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_mIoU", torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
                ignore_index=self.ignore_index,
            ))
            setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
                ignore_index=self.ignore_index,
            ))
            setattr(self, f"{split}_iou_per_class", torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=num_classes,
                average="none",
                ignore_index=self.ignore_index,
            ))

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode_str = "temporal" if temporal else "standard"
        print(f"[BaselineTrainer] task='{task}', modality='{modality}', "
              f"mode={mode_str}, classes={num_classes}, params={param_count:,}")

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, x, doy=None):
        if self.temporal and doy is not None:
            return self.model(x, doy=doy)
        return self.model(x)

    # ─────────────────────────────────────────────────────────────────
    # Input extraction
    # ─────────────────────────────────────────────────────────────────

    def _get_image(self, batch):
        """
        Extract the model input from a batch.

        Models that need the entire modality dict (e.g. RAMENUPerNet,
        which looks up a separate spectral projector per modality
        internally) set a class attribute `expects_full_image_dict = True`.
        This is checked with getattr rather than isinstance so that
        wrappers around such a model (e.g. a modality-drop wrapper used
        for ablation studies) are also recognized correctly, as long as
        they carry the same attribute — no need for this trainer to
        import or know about every wrapper class.
        Every other model expects a single modality's tensor, selected
        via self.modality.
        """
        if getattr(self.model, "expects_full_image_dict", False):
            return batch["image"]  # dict[modality] -> Tensor, passed through as-is
        return batch["image"][self.modality]  # [B, C, H, W] or [B, T, C, H, W]

    # ─────────────────────────────────────────────────────────────────
    # Shared step
    # ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, split: str):
        image = self._get_image(batch)
        target = batch["target"]                # [B, H, W]

        # Get dates for temporal models
        doy = None
        if self.temporal and "dates" in batch:
            dates = batch["dates"]
            if isinstance(dates, dict) and self.modality in dates:
                doy = dates[self.modality]  # [B, T]
            elif isinstance(dates, torch.Tensor):
                doy = dates  # [B, T]

        # Forward
        use_sliding_window = (
            self.window_size is not None
            and split != "train"
            and not self.temporal
        )
        if use_sliding_window:
            # Training already sees pre-cropped windows of this exact
            # size (see dataset/collate crop_size), so tiling is only
            # needed for full-resolution val/test images.
            logits = sliding_window_inference(
                self.model,
                image,
                window_size=self.window_size,
                stride=self.window_stride,
                num_classes=self.num_classes,
            )
        elif self.temporal and doy is not None:
            logits = self.model(image, doy=doy)
        else:
            logits = self.model(image)

        # Handle spatial size mismatch
        if logits.shape[2:] != target.shape[1:]:
            logits = nn.functional.interpolate(
                logits, size=target.shape[1:],
                mode="bilinear", align_corners=False,
            )

        loss = self.loss_fn(logits, target.long())

        preds = logits.argmax(dim=1)  # [B, H, W]

        # Update metrics
        miou_metric = getattr(self, f"{split}_mIoU")
        acc_metric = getattr(self, f"{split}_acc")
        iou_per_class = getattr(self, f"{split}_iou_per_class")

        miou_metric.update(preds, target)
        acc_metric.update(preds, target)
        iou_per_class.update(preds, target)

        self.log(f"{split}_loss", loss,
                 on_step=(split == "train"),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=(split != "train"))

        return loss

    # ─────────────────────────────────────────────────────────────────
    # Train / Val / Test steps
    # ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test")

    # ─────────────────────────────────────────────────────────────────
    # Epoch end — log metrics
    # ─────────────────────────────────────────────────────────────────

    def _on_epoch_end(self, split: str):
        miou_metric = getattr(self, f"{split}_mIoU")
        acc_metric = getattr(self, f"{split}_acc")
        iou_per_class = getattr(self, f"{split}_iou_per_class")

        miou = miou_metric.compute()
        acc = acc_metric.compute()
        per_class = iou_per_class.compute()

        self.log(f"{split}_mIoU", miou,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{split}_acc", acc,
                 on_epoch=True, logger=True, sync_dist=True)

        # Per-class IoU
        for cls_idx in range(self.num_classes):
            cls_name = self.class_names.get(cls_idx, f"class_{cls_idx}")
            self.log(f"{split}_IoU_{cls_name}", per_class[cls_idx],
                     on_epoch=True, logger=True, sync_dist=True)

        miou_metric.reset()
        acc_metric.reset()
        iou_per_class.reset()

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    # ─────────────────────────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
