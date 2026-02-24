"""
MMEarth Reconstruction Visualization Callback
===============================================
Plots RGB ground truth, predicted RGB, and per-pixel error.
Optionally shows per-band scatter plots for diagnostics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange

from training.utils.datasets.token_grouping import collate_grouped


class ReconstructionVizCallback(pl.Callback):
    """
    At the end of each validation epoch, run the model on a few
    viz samples and log RGB ground truth / prediction / error to wandb.
    """

    # S2 bands come first in the merged image: B02(0), B03(1), B04(2), ...
    RGB_INDICES = [2, 1, 0]  # Red=B04, Green=B03, Blue=B02

    def __init__(
        self,
        sample_indices=(0,),
        log_every_n_epochs=1,
        use_wandb=True,
    ):
        super().__init__()
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        val_dataset = trainer.datamodule.train_dataset
        if not hasattr(val_dataset, "get_viz_sample"):
            return

        device = pl_module.device
        pl_module.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(val_dataset):
                continue

            sample = val_dataset.get_viz_sample(idx)
            C, H, W = sample["image_shape"]
            n_real = sample["n_real"]

            # Collate into batch of 1
            batch = collate_grouped([sample])
            batch = _batch_to_device(batch, device)

            with torch.no_grad():
                result = pl_module(batch, training=False)

            if isinstance(result, dict):
                preds = result["predictions"]
            else:
                preds = result

            # preds: [1, total_tokens, 1] → [total_tokens]
            preds = preds.squeeze(0).squeeze(-1).cpu()
            gt = batch["queries"].squeeze(0)[:, 4].cpu()

            # Trim padding
            preds = preds[:n_real]
            gt = gt[:n_real]

            mse = ((gt - preds) ** 2).mean().item()
            corr = torch.corrcoef(torch.stack([preds, gt]))[0, 1].item()

            print(f"[VIZ] Sample {idx}, Epoch {trainer.current_epoch}: "
                  f"MSE={mse:.6f}, corr={corr:.4f}, "
                  f"pred=[{preds.min():.3f},{preds.max():.3f}], "
                  f"gt=[{gt.min():.3f},{gt.max():.3f}]")

            # Reshape to [C, H, W]
            gt_img = rearrange(gt, "(C H W) -> C H W", C=C, H=H, W=W)
            pred_img = rearrange(preds, "(C H W) -> C H W", C=C, H=H, W=W)

            # ── RGB figure ──────────────────────────────────
            rgb_idx = [i for i in self.RGB_INDICES if i < C]
            if len(rgb_idx) < 3:
                rgb_idx = [0, 0, 0]  # fallback: grayscale

            gt_rgb = self._normalize_rgb(gt_img[rgb_idx].clone())
            pred_rgb = self._normalize_rgb(pred_img[rgb_idx].clone())
            error_rgb = np.abs(pred_rgb - gt_rgb)

            fig = self._make_figure(gt_rgb, pred_rgb, error_rgb, idx, trainer.current_epoch, mse)
            figures.append((f"recon_rgb_{idx}", fig))

            # ── Per-band diagnostics (scatter + raw) ────────
            fig2 = self._make_band_figure(gt_img, pred_img, gt, preds, C,
                                          idx, trainer.current_epoch, corr, mse)
            figures.append((f"recon_bands_{idx}", fig2))

        # Log to wandb
        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)})
            plt.close("all")

        pl_module.train()

    @staticmethod
    def _normalize_rgb(rgb):
        """Normalize [3, H, W] to [0, 1] for display using percentile clipping."""
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
        """Create a 1x3 figure: GT | Predicted | Error."""
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

        fig.suptitle(f"Sample {sample_idx} — Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        return fig

    @staticmethod
    def _make_band_figure(gt_img, pred_img, gt_flat, pred_flat, C,
                          sample_idx, epoch, corr, mse):
        """
        Show raw GT vs pred for a few representative bands + overall scatter.
        
        Layout: top row = GT bands, bottom row = pred bands, last col = scatter.
        """
        # Pick up to 4 bands spread across the spectrum
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

        # Scatter plot (last column, spanning both rows)
        ax_scatter = fig.add_subplot(1, n_bands + 1, n_bands + 1)
        # Hide the two grid axes in the last column
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