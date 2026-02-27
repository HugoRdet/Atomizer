"""
Multi-Task Pre-training Visualization Callbacks
=================================================

Two callbacks for the multi-task pre-training setup:

1. PretrainSegVizCallback — Segmentation visualization for ESA + DW
2. PretrainReconVizCallback — Reconstruction visualization

Both work with Model_Pretrain which requires task_name in forward().
Both expect datasets to be stored in trainer.datamodule under known attribute names.

Usage:
    callbacks = [
        PretrainSegVizCallback(
            task_name="esa_worldcover",
            dataset_attr="train_dataset_esa",  # or however you store it
            class_names=ESA_CLASS_NAMES,
            sample_indices=(0, 1, 2),
        ),
        PretrainSegVizCallback(
            task_name="dynamic_world",
            dataset_attr="train_dataset_dw",
            class_names=DW_CLASS_NAMES,
            sample_indices=(0, 1, 2),
        ),
        PretrainReconVizCallback(
            dataset_attr="train_dataset_recon",
            sample_indices=(0,),
        ),
    ]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange

from training.utils.datasets.token_grouping import collate_grouped


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


# =============================================================================
# SEGMENTATION CALLBACK
# =============================================================================

class PretrainSegVizCallback(pl.Callback):
    """
    Segmentation visualization for one task (ESA or DW).

    Renders RGB | GT mask | Predicted mask for a few samples.
    Calls pl_module.forward(batch, task_name, training=False).
    """

    RGB_INDICES = [3, 2, 1]  # B04(Red), B03(Green), B02(Blue) in merged MMEarth order

    def __init__(
        self,
        task_name: str,
        dataset_attr: str = None,
        class_names: list = None,
        sample_indices=(0, 1, 2),
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ):
        super().__init__()
        self.task_name = task_name
        self.dataset_attr = dataset_attr
        self.class_names = class_names or []
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb
        self.ignore_index = ignore_index

    def _get_dataset(self, trainer):
        """Find the dataset, trying multiple locations."""
        dm = trainer.datamodule

        # Try explicit attribute name
        if self.dataset_attr and hasattr(dm, self.dataset_attr):
            return getattr(dm, self.dataset_attr)

        # Try datasets dict
        if hasattr(dm, "datasets") and isinstance(dm.datasets, dict):
            if self.task_name in dm.datasets:
                return dm.datasets[self.task_name]

        # Try train_datasets dict
        if hasattr(dm, "train_datasets") and isinstance(dm.train_datasets, dict):
            if self.task_name in dm.train_datasets:
                return dm.train_datasets[self.task_name]

        return None

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = self._get_dataset(trainer)
        if dataset is None or not hasattr(dataset, "get_viz_sample"):
            return

        device = pl_module.device
        pl_module.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                sample = dataset.get_viz_sample(idx)

                # Get spatial dims from tokens
                shape = list(sample["groups"].values())[0]["shape"]
                if len(shape) == 3:
                    C, H, W = shape
                else:
                    H, W = shape
                    C = sum(1 for _ in range(18))  # fallback

                batch = collate_grouped([sample])
                batch = _batch_to_device(batch, device)

                with torch.no_grad():
                    predictions = pl_module.forward(batch, self.task_name, training=False)

                # [1, M, num_classes] → [M]
                preds = torch.argmax(predictions, dim=-1).squeeze(0).cpu()

                # Labels from queries col 4
                labels = batch["queries"].squeeze(0)[:, 4].long().cpu()

                # Reshape to spatial
                n_pixels = H * W
                pred_2d = preds[:n_pixels].reshape(H, W)
                label_2d = labels[:n_pixels].reshape(H, W)

                # RGB from image if available, else from tokens
                if "image" in sample:
                    image = sample["image"]
                    rgb_idx = [i for i in self.RGB_INDICES if i < image.shape[0]]
                    if len(rgb_idx) < 3:
                        rgb_idx = [0, 0, 0]
                    rgb = self._normalize_rgb(image[rgb_idx])
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

        pl_module.train()

    @staticmethod
    def _normalize_rgb(rgb):
        rgb = rgb.numpy() if isinstance(rgb, torch.Tensor) else rgb
        for c in range(3):
            lo = np.percentile(rgb[c], 2)
            hi = np.percentile(rgb[c], 98)
            if hi - lo > 1e-6:
                rgb[c] = (rgb[c] - lo) / (hi - lo)
            else:
                rgb[c] = 0.0
        return np.clip(rgb, 0, 1)

    def _make_figure(self, rgb, label, pred, sample_idx, epoch):
        n_classes = len(self.class_names) if self.class_names else max(label.max(), pred.max()) + 1
        cmap = plt.cm.get_cmap("tab20", n_classes)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.transpose(rgb, (1, 2, 0)))
        axes[0].set_title("RGB")
        axes[0].axis("off")

        masked_label = np.ma.masked_where(label == self.ignore_index, label)
        axes[1].imshow(masked_label, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[1].set_title("GT Label")
        axes[1].axis("off")

        im = axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        valid = label != self.ignore_index
        if valid.sum() > 0:
            acc = (pred[valid] == label[valid]).mean() * 100
            title = f"{self.task_name} — Sample {sample_idx} — Epoch {epoch} — Acc: {acc:.1f}%"
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
    Calls pl_module.encoder(batch, ..., return_features=False) directly
    to use the encoder's reconstruction_head (not task-specific heads).
    """

    RGB_INDICES = [2, 1, 0]  # B04(Red=idx2), B03(Green=idx1), B02(Blue=idx0) in S2 order

    def __init__(
        self,
        dataset_attr: str = None,
        sample_indices=(0,),
        log_every_n_epochs=1,
        use_wandb=True,
    ):
        super().__init__()
        self.dataset_attr = dataset_attr
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb

    def _get_dataset(self, trainer):
        """Find the reconstruction dataset."""
        dm = trainer.datamodule

        if self.dataset_attr and hasattr(dm, self.dataset_attr):
            return getattr(dm, self.dataset_attr)

        if hasattr(dm, "datasets") and isinstance(dm.datasets, dict):
            if "reconstruction" in dm.datasets:
                return dm.datasets["reconstruction"]

        if hasattr(dm, "train_datasets") and isinstance(dm.train_datasets, dict):
            if "reconstruction" in dm.train_datasets:
                return dm.train_datasets["reconstruction"]

        return None

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = self._get_dataset(trainer)
        if dataset is None or not hasattr(dataset, "get_viz_sample"):
            return

        device = pl_module.device
        pl_module.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                sample = dataset.get_viz_sample(idx)
                C, H, W = sample["image_shape"]
                n_real = sample["n_real"]

                batch = collate_grouped([sample])
                batch = _batch_to_device(batch, device)

                with torch.no_grad():
                    # Use the reconstruction task head
                    predictions = pl_module.forward(
                        batch, "reconstruction", training=False
                    )

                # [1, M, 1] → [M]
                preds = predictions.squeeze(0).squeeze(-1).cpu()
                gt = batch["queries"].squeeze(0)[:, 4].cpu()

                preds = preds[:n_real]
                gt = gt[:n_real]

                mse = ((gt - preds) ** 2).mean().item()
                corr = torch.corrcoef(torch.stack([preds, gt]))[0, 1].item()

                print(f"[RECON VIZ] Sample {idx}, Epoch {trainer.current_epoch}: "
                      f"MSE={mse:.6f}, corr={corr:.4f}")

                # Reshape to [C, H, W]
                gt_img = rearrange(gt, "(C H W) -> C H W", C=C, H=H, W=W)
                pred_img = rearrange(preds, "(C H W) -> C H W", C=C, H=H, W=W)

                # RGB figure
                rgb_idx = [i for i in self.RGB_INDICES if i < C]
                if len(rgb_idx) < 3:
                    rgb_idx = [0, 0, 0]

                gt_rgb = self._normalize_rgb(gt_img[rgb_idx].clone())
                pred_rgb = self._normalize_rgb(pred_img[rgb_idx].clone())
                error_rgb = np.abs(pred_rgb - gt_rgb)

                fig = self._make_figure(
                    gt_rgb, pred_rgb, error_rgb,
                    idx, trainer.current_epoch, mse,
                )
                figures.append((f"recon_rgb_{idx}", fig))

                # Band diagnostics
                fig2 = self._make_band_figure(
                    gt_img, pred_img, gt, preds, C,
                    idx, trainer.current_epoch, corr, mse,
                )
                figures.append((f"recon_bands_{idx}", fig2))

            except Exception as e:
                import traceback
                print(f"[RECON VIZ] Failed on sample {idx}: {e}")
                traceback.print_exc()

        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)})
            plt.close("all")

        pl_module.train()

    @staticmethod
    def _normalize_rgb(rgb):
        rgb = rgb.numpy()
        for c in range(3):
            lo = np.percentile(rgb[c], 2)
            hi = np.percentile(rgb[c], 98)
            if hi - lo > 1e-6:
                rgb[c] = (rgb[c] - lo) / (hi - lo)
            else:
                rgb[c] = 0.0
        return np.clip(rgb, 0, 1)

    @staticmethod
    def _make_figure(gt_rgb, pred_rgb, error_rgb, sample_idx, epoch, mse=None):
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

        fig.suptitle(f"Reconstruction — Sample {sample_idx} — Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        return fig

    @staticmethod
    def _make_band_figure(gt_img, pred_img, gt_flat, pred_flat, C,
                          sample_idx, epoch, corr, mse):
        if C <= 4:
            band_indices = list(range(C))
        else:
            band_indices = [0, C // 3, 2 * C // 3, C - 1]

        n_bands = len(band_indices)
        fig, axes = plt.subplots(2, n_bands + 1, figsize=(4 * (n_bands + 1), 8))

        for col, b in enumerate(band_indices):
            vmin = min(gt_img[b].min().item(), pred_img[b].min().item())
            vmax = max(gt_img[b].max().item(), pred_img[b].max().item())

            axes[0, col].imshow(gt_img[b].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"GT band {b}")
            axes[0, col].axis("off")

            axes[1, col].imshow(pred_img[b].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f"Pred band {b}")
            axes[1, col].axis("off")

        ax_scatter = fig.add_subplot(1, n_bands + 1, n_bands + 1)
        axes[0, -1].axis("off")
        axes[1, -1].axis("off")

        n_scatter = min(10000, gt_flat.shape[0])
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

        fig.suptitle(f"Band diagnostics — Sample {sample_idx}, Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        return fig


# =============================================================================
# UTILITY
# =============================================================================

def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out
