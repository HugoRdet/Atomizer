"""
Multi-Task Pre-training Visualization Callbacks (Encode-Once)
==============================================================

Two callbacks for the encode-once multi-task pre-training setup:

1. PretrainSegVizCallback  — Segmentation visualization (any seg task)
2. PretrainReconVizCallback — Reconstruction visualization

Both work with the encode-once batch format:
    batch["tasks"][task_name]["queries"]   (not batch["queries"])
    pl_module.forward_multitask(batch)     returns {task_name: preds}

Both find the multi-task dataset via trainer.datamodule.train_dataset
(MMEarthMultiTask or FlairHubMultiTask) and call .get_viz_sample().

get_viz_sample() returns:
    {
        "groups": {res: {"tokens": ..., "mask": ..., "shape": ...}},
        "tasks":  {task_name: {"queries": [M,8], "queries_mask": [M]}},
        "target_resolution": float,
        "image":       [C, H, W] raw unnormalized image,
        "labels":      {task_name: [H, W] label tensor},
        "image_shape": (C, H, W),
        "n_real":      int (non-padded token count),
    }

Usage in train_pretrain_v2.py:
    from training.viz_callbacks_pretrain import (
        PretrainSegVizCallback, PretrainReconVizCallback,
        ESA_CLASS_NAMES, DW_CLASS_NAMES,
        COSIA_CLASS_NAMES, LPIS_CLASS_NAMES,
    )

    callbacks = [
        PretrainSegVizCallback(task_name="esa_worldcover",  sample_indices=(0, 1)),
        PretrainSegVizCallback(task_name="flairhub_cosia",  sample_indices=(0, 1)),
        PretrainReconVizCallback(sample_indices=(0,)),
    ]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange

from training.utils.datasets.token_grouping import collate_multitask


# =============================================================================
# CLASS NAMES
# =============================================================================

ESA_CLASS_NAMES = [
    "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
    "Bare/sparse", "Snow/ice", "Water", "Wetland", "Mangroves", "Moss/lichen",
]

DW_CLASS_NAMES = [
    "Water", "Trees", "Grass", "Flooded veg", "Crops",
    "Shrub/scrub", "Built", "Bare", "Snow/ice",
]

COSIA_CLASS_NAMES = [
    "Building", "Pervious surface", "Impervious surface", "Bare soil",
    "Water", "Coniferous", "Deciduous", "Brushwood", "Vineyard",
    "Herbaceous veg", "Agricultural land", "Plowed land",
    "Swimming pool", "Snow", "Clear cut", "Mixed",
    "Ligneous", "Greenhouse",
]

LPIS_CLASS_NAMES = [f"LPIS_{i}" for i in range(23)]

# Task → defaults
_TASK_DEFAULTS = {
    "esa_worldcover": {"class_names": ESA_CLASS_NAMES,  "rgb_idx": [3, 2, 1]},
    "dynamic_world":  {"class_names": DW_CLASS_NAMES,   "rgb_idx": [3, 2, 1]},
    "flairhub_cosia": {"class_names": COSIA_CLASS_NAMES, "rgb_idx": [0, 1, 2]},
    "flairhub_lpis":  {"class_names": LPIS_CLASS_NAMES,  "rgb_idx": [0, 1, 2]},
}


# =============================================================================
# HELPERS
# =============================================================================

def _unwrap_model(pl_module):
    """
    Get the raw model from a PL module, bypassing DDP/FSDP wrappers.

    DDP wraps the model as pl_module.module. Calling forward() on the
    DDP-wrapped model triggers NCCL collectives — which deadlocks if
    only rank 0 is running the callback. Using the unwrapped model
    runs a plain forward pass with no distributed sync.
    """
    model = pl_module
    # DDP wraps as .module
    if hasattr(model, "module"):
        model = model.module
    return model


def _batch_to_device(batch, device):
    """Recursively move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


def _get_multitask_dataset(trainer):
    """
    Find the multi-task dataset from the DataModule.

    Works with both MMEarthMultiTaskDataModule and FlairHubMultiTaskDataModule,
    which store the dataset as trainer.datamodule.train_dataset.
    """
    dm = trainer.datamodule
    if dm is None:
        return None
    if hasattr(dm, "train_dataset") and hasattr(dm.train_dataset, "get_viz_sample"):
        return dm.train_dataset
    return None


def _normalize_rgb(rgb):
    """Percentile-based contrast stretch for display."""
    rgb = rgb.numpy() if isinstance(rgb, torch.Tensor) else rgb.copy()
    for c in range(min(3, rgb.shape[0])):
        lo = np.percentile(rgb[c], 2)
        hi = np.percentile(rgb[c], 98)
        if hi - lo > 1e-6:
            rgb[c] = (rgb[c] - lo) / (hi - lo)
        else:
            rgb[c] = 0.0
    return np.clip(rgb, 0, 1)


# =============================================================================
# SEGMENTATION CALLBACK
# =============================================================================

class PretrainSegVizCallback(pl.Callback):
    """
    Segmentation visualization for one task.

    Renders RGB | GT mask | Predicted mask for a few samples.
    Uses the encode-once path: forward_multitask(batch) → {task: preds}.

    Works with any seg task: esa_worldcover, dynamic_world,
    flairhub_cosia, flairhub_lpis.
    """

    def __init__(
        self,
        task_name: str,
        class_names: list = None,
        rgb_indices: list = None,
        sample_indices=(0, 1, 2),
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ):
        super().__init__()
        self.task_name = task_name
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb
        self.ignore_index = ignore_index

        defaults = _TASK_DEFAULTS.get(task_name, {})
        self.class_names = class_names or defaults.get("class_names", [])
        self.rgb_indices = rgb_indices or defaults.get("rgb_idx", [0, 1, 2])

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = _get_multitask_dataset(trainer)
        if dataset is None:
            return

        # Skip if this task isn't enabled in the dataset
        if hasattr(dataset, "enabled_tasks") and self.task_name not in dataset.enabled_tasks:
            return

        device = pl_module.device

        # ── CRITICAL: unwrap DDP to avoid NCCL deadlock ──
        model = _unwrap_model(pl_module)
        model.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                sample = dataset.get_viz_sample(idx)

                # Check task produced queries for this sample
                if self.task_name not in sample.get("tasks", {}):
                    continue

                # Spatial dims
                image_shape = sample.get("image_shape")
                if image_shape and len(image_shape) == 3:
                    C, H, W = image_shape
                else:
                    shape = list(sample["groups"].values())[0]["shape"]
                    C, H, W = shape

                # Collate → batch of 1
                batch = collate_multitask([sample])
                batch = _batch_to_device(batch, device)

                with torch.no_grad():
                    all_predictions = model.forward_multitask(
                        batch, training=False,
                    )

                if self.task_name not in all_predictions:
                    continue

                predictions = all_predictions[self.task_name]

                # [1, M, num_classes] → [M]
                preds = torch.argmax(predictions, dim=-1).squeeze(0).cpu()

                # Labels from queries col 4
                labels = (
                    batch["tasks"][self.task_name]["queries"]
                    .squeeze(0)[:, 4]
                    .long()
                    .cpu()
                )

                # Reshape to spatial grid
                n_pixels = H * W
                pred_2d = preds[:n_pixels].reshape(H, W)
                label_2d = labels[:n_pixels].reshape(H, W)

                # RGB image
                if "image" in sample:
                    image = sample["image"]
                    rgb_idx = [i for i in self.rgb_indices if i < image.shape[0]]
                    if len(rgb_idx) < 3:
                        rgb_idx = list(range(min(3, image.shape[0])))
                    rgb = _normalize_rgb(image[rgb_idx])
                else:
                    rgb = np.zeros((3, H, W))

                fig = self._make_figure(
                    rgb, label_2d.numpy(), pred_2d.numpy(),
                    idx, trainer.current_epoch,
                )
                figures.append((f"seg_{self.task_name}_{idx}", fig))

            except Exception as e:
                import traceback
                print(f"[SEG VIZ {self.task_name}] Failed on sample {idx}: {e}")
                traceback.print_exc()

        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)})
            plt.close("all")

        model.train()

    def _make_figure(self, rgb, label, pred, sample_idx, epoch):
        n_classes = (
            len(self.class_names) if self.class_names
            else int(max(label.max(), pred.max())) + 1
        )
        cmap = plt.cm.get_cmap("tab20", max(n_classes, 2))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.transpose(rgb, (1, 2, 0)))
        axes[0].set_title("RGB")
        axes[0].axis("off")

        masked_label = np.ma.masked_where(label == self.ignore_index, label)
        axes[1].imshow(
            masked_label, cmap=cmap, vmin=0, vmax=n_classes - 1,
            interpolation="nearest",
        )
        axes[1].set_title("GT Label")
        axes[1].axis("off")

        axes[2].imshow(
            pred, cmap=cmap, vmin=0, vmax=n_classes - 1,
            interpolation="nearest",
        )
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        valid = label != self.ignore_index
        if valid.sum() > 0:
            acc = (pred[valid] == label[valid]).mean() * 100
            title = (
                f"{self.task_name} — Sample {sample_idx} — "
                f"Epoch {epoch} — Acc: {acc:.1f}%"
            )
        else:
            title = f"{self.task_name} — Sample {sample_idx} — Epoch {epoch}"

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        return fig


# =============================================================================
# RECONSTRUCTION CALLBACK
# =============================================================================

class PretrainReconVizCallback(pl.Callback):
    """
    Reconstruction visualization.

    Renders RGB ground truth | predicted RGB | error map.
    Uses the encode-once path: forward_multitask(batch) → {task: preds}.

    Works with both MMEarth (single-res, reshapeable) and FLAIR-HUB
    (multi-res, may not reshape cleanly → scatter-only fallback).
    """

    def __init__(
        self,
        sample_indices=(0,),
        rgb_indices: list = None,
        log_every_n_epochs=1,
        use_wandb=True,
    ):
        super().__init__()
        self.sample_indices = sample_indices
        self.rgb_indices = rgb_indices  # None = auto-detect
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = _get_multitask_dataset(trainer)
        if dataset is None:
            return

        if hasattr(dataset, "enabled_tasks") and "reconstruction" not in dataset.enabled_tasks:
            return

        device = pl_module.device

        # ── CRITICAL: unwrap DDP to avoid NCCL deadlock ──
        model = _unwrap_model(pl_module)
        model.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                sample = dataset.get_viz_sample(idx)

                if "reconstruction" not in sample.get("tasks", {}):
                    continue

                # Determine image shape
                image_shape = sample.get("image_shape")
                if image_shape is None:
                    image_shape = list(sample["groups"].values())[0]["shape"]
                C, H, W = image_shape

                n_real = sample.get("n_real", C * H * W)

                # Auto-detect RGB indices
                rgb_idx = self.rgb_indices
                if rgb_idx is None:
                    # MMEarth merged: ≥12 bands → B04(R)=3, B03(G)=2, B02(B)=1
                    # FLAIR-HUB aerial: 4 bands → R=0, G=1, B=2
                    rgb_idx = [3, 2, 1] if C >= 12 else [0, 1, 2]

                # Collate → batch of 1
                batch = collate_multitask([sample])
                batch = _batch_to_device(batch, device)

                with torch.no_grad():
                    all_predictions = model.forward_multitask(
                        batch, training=False,
                    )

                if "reconstruction" not in all_predictions:
                    continue

                predictions = all_predictions["reconstruction"]

                # [1, M, 1] → [M]
                preds = predictions.squeeze(0).squeeze(-1).cpu()
                gt = (
                    batch["tasks"]["reconstruction"]["queries"]
                    .squeeze(0)[:, 4]
                    .cpu()
                )

                preds = preds[:n_real]
                gt = gt[:n_real]

                mse = ((gt - preds) ** 2).mean().item()
                corr_val = torch.corrcoef(torch.stack([preds, gt]))[0, 1].item()

                print(
                    f"[RECON VIZ] Sample {idx}, Epoch {trainer.current_epoch}: "
                    f"MSE={mse:.6f}, corr={corr_val:.4f}, "
                    f"shape=({C},{H},{W}), n_real={n_real}"
                )

                # Try spatial reshape: works for single-res (MMEarth) where
                # n_real == C*H*W, fails for multi-res (FLAIR-HUB)
                expected = C * H * W
                if preds.shape[0] == expected:
                    gt_img = rearrange(gt, "(C H W) -> C H W", C=C, H=H, W=W)
                    pred_img = rearrange(preds, "(C H W) -> C H W", C=C, H=H, W=W)

                    safe_rgb = [i for i in rgb_idx if i < C]
                    if len(safe_rgb) < 3:
                        safe_rgb = list(range(min(3, C)))

                    gt_rgb = _normalize_rgb(gt_img[safe_rgb].clone())
                    pred_rgb = _normalize_rgb(pred_img[safe_rgb].clone())
                    error_rgb = np.abs(pred_rgb - gt_rgb)

                    fig = self._make_spatial_figure(
                        gt_rgb, pred_rgb, error_rgb,
                        idx, trainer.current_epoch, mse,
                    )
                    figures.append((f"recon_rgb_{idx}", fig))

                    fig2 = self._make_band_figure(
                        gt_img, pred_img, gt, preds, C,
                        idx, trainer.current_epoch, corr_val, mse,
                    )
                    figures.append((f"recon_bands_{idx}", fig2))
                else:
                    # Multi-res: can't reshape cleanly → scatter only
                    fig = self._make_scatter_figure(
                        gt, preds, idx, trainer.current_epoch, corr_val, mse,
                    )
                    figures.append((f"recon_scatter_{idx}", fig))

            except Exception as e:
                import traceback
                print(f"[RECON VIZ] Failed on sample {idx}: {e}")
                traceback.print_exc()

        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)})
            plt.close("all")

        model.train()

    # ----- figure builders ---------------------------------------------------

    @staticmethod
    def _make_spatial_figure(gt_rgb, pred_rgb, error_rgb,
                             sample_idx, epoch, mse=None):
        """RGB ground-truth | prediction | error map."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.transpose(gt_rgb, (1, 2, 0)))
        axes[0].set_title("Ground Truth (RGB)")
        axes[0].axis("off")

        mse_str = f" (MSE={mse:.4f})" if mse is not None else ""
        axes[1].imshow(np.transpose(pred_rgb, (1, 2, 0)))
        axes[1].set_title(f"Predicted (RGB){mse_str}")
        axes[1].axis("off")

        error_mean = error_rgb.mean(axis=0)
        im = axes[2].imshow(error_mean, cmap="hot", vmin=0)
        axes[2].set_title("Abs Error (RGB mean)")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Reconstruction — Sample {sample_idx} — Epoch {epoch}",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    @staticmethod
    def _make_scatter_figure(gt, preds, sample_idx, epoch, corr, mse):
        """Scatter plot for multi-res cases that can't reshape spatially."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        n_scatter = min(10_000, gt.shape[0])
        scatter_idx = torch.randperm(gt.shape[0])[:n_scatter]
        ax.scatter(
            gt[scatter_idx].numpy(), preds[scatter_idx].numpy(),
            s=1, alpha=0.2,
        )
        lims = [gt.min().item(), gt.max().item()]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("GT")
        ax.set_ylabel("Predicted")
        ax.set_title(f"corr={corr:.3f}, MSE={mse:.4f}")
        ax.set_aspect("equal")

        fig.suptitle(
            f"Reconstruction (multi-res) — Sample {sample_idx} — Epoch {epoch}",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    @staticmethod
    def _make_band_figure(gt_img, pred_img, gt_flat, pred_flat, C,
                          sample_idx, epoch, corr, mse):
        """Per-band GT vs pred images + scatter plot."""
        if C <= 4:
            band_indices = list(range(C))
        else:
            band_indices = [0, C // 3, 2 * C // 3, C - 1]

        n_bands = len(band_indices)
        fig, axes = plt.subplots(2, n_bands + 1, figsize=(4 * (n_bands + 1), 8))

        for col, b in enumerate(band_indices):
            vmin = min(gt_img[b].min().item(), pred_img[b].min().item())
            vmax = max(gt_img[b].max().item(), pred_img[b].max().item())

            axes[0, col].imshow(
                gt_img[b].numpy(), cmap="gray", vmin=vmin, vmax=vmax,
            )
            axes[0, col].set_title(f"GT band {b}")
            axes[0, col].axis("off")

            axes[1, col].imshow(
                pred_img[b].numpy(), cmap="gray", vmin=vmin, vmax=vmax,
            )
            axes[1, col].set_title(f"Pred band {b}")
            axes[1, col].axis("off")

        # Scatter in the last column
        ax_scatter = fig.add_subplot(1, n_bands + 1, n_bands + 1)
        axes[0, -1].axis("off")
        axes[1, -1].axis("off")

        n_scatter = min(10_000, gt_flat.shape[0])
        scatter_idx = torch.randperm(gt_flat.shape[0])[:n_scatter]
        ax_scatter.scatter(
            gt_flat[scatter_idx].numpy(), pred_flat[scatter_idx].numpy(),
            s=1, alpha=0.2,
        )
        lims = [gt_flat.min().item(), gt_flat.max().item()]
        ax_scatter.plot(lims, lims, "r--", lw=1)
        ax_scatter.set_xlabel("GT")
        ax_scatter.set_ylabel("Predicted")
        ax_scatter.set_title(f"corr={corr:.3f}, MSE={mse:.4f}")
        ax_scatter.set_aspect("equal")

        fig.suptitle(
            f"Band diagnostics — Sample {sample_idx}, Epoch {epoch}",
            fontsize=14,
        )
        fig.tight_layout()
        return fig