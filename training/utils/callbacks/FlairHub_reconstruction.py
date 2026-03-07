"""
FlairHub Multi-Modal Reconstruction Visualization Callback
============================================================

Plots per-modality RGB composites (GT vs Predicted vs Error) for
FLAIR-HUB reconstruction across Aerial, SPOT, Sentinel-2, and Sentinel-1.

Two pieces:
    1. `get_recon_viz_sample()`  — method on FlairHubMultiTask / FlairHubBase
    2. `FlairHubReconVizCallback` — Lightning callback for W&B logging

Token format reminder:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7

Model paths:
    forward_multitask(batch) expects:
        batch["groups"]  → {res: {"tokens": [B,N,8], "mask": [B,N], "shape": ...}}
        batch["tasks"]   → {task_name: {"queries": [B,M,8], "queries_mask": [B,M]}}
    returns:
        {"reconstruction": [B, M, 1]}

    The callback wraps the viz sample into multi-task format so it goes through
    forward_multitask → encode-once path (well-tested, DDP-safe).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.utils_dataset_FLAIRHUB import (
    DATASET_NAME, IGNORE_INDEX,
    RES_AERIAL, RES_SPOT, RES_S2, RES_S1,
    N_BANDS_AERIAL, N_BANDS_SPOT, N_BANDS_S2, N_BANDS_S1,
    SIZE_AERIAL, SIZE_SPOT, SIZE_S2, SIZE_S1,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class FlairHubReconVizCallback(pl.Callback):
    """
    At the end of selected training epochs, reconstruct per-modality
    images from the FlairHub dataset and log GT / Predicted / Error
    RGB composites to Weights & Biases.

    Modality display:
        - Aerial RGBI  (0.2 m)  → RGB = R, G, B channels
        - SPOT RGBI    (1.6 m)  → RGB = R, G, B channels
        - Sentinel-2   (10  m)  → RGB = B04, B03, B02
        - Sentinel-1   (10  m)  → false-colour VV / VH / VV
    """

    MODALITY_ORDER = ["aerial", "spot", "s2", "s1_asc", "s1_des"]

    # Band indices within each modality's band list → [R, G, B] for display
    MODALITY_RGB = {
        "aerial":  [0, 1, 2],       # R, G, B
        "spot":    [0, 1, 2],       # R, G, B
        "s2":      [2, 1, 0],       # B04(Red), B03(Green), B02(Blue)
        "s1_asc":  None,            # SAR → special handling
        "s1_des":  None,
    }

    MODALITY_DISPLAY = {
        "aerial":  "Aerial (0.2 m)",
        "spot":    "SPOT (1.6 m)",
        "s2":      "S2 (10 m)",
        "s1_asc":  "S1-Asc (10 m)",
        "s1_des":  "S1-Desc (10 m)",
    }

    def __init__(
        self,
        sample_indices=(0,),
        log_every_n_epochs: int = 5,
        dataset_attr: str = None,
        use_wandb: bool = True,
    ):
        """
        Args:
            sample_indices: Which patch indices to visualise.
            log_every_n_epochs: Frequency of logging.
            dataset_attr: Attribute name on the datamodule that holds the
                FlairHub dataset with ``get_recon_viz_sample``.
                If None, the callback searches automatically.
            use_wandb: Whether to log figures to W&B.
        """
        super().__init__()
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.dataset_attr = dataset_attr
        self.use_wandb = use_wandb

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _find_dataset(self, trainer):
        """Locate the FlairHub dataset that exposes get_recon_viz_sample."""
        dm = trainer.datamodule

        # 1. Explicit attribute
        if self.dataset_attr and hasattr(dm, self.dataset_attr):
            ds = getattr(dm, self.dataset_attr)
            if hasattr(ds, "get_recon_viz_sample"):
                return ds

        # 2. Direct train_dataset (FlairHubMultiTaskDataModule)
        if hasattr(dm, "train_dataset"):
            ds = dm.train_dataset
            if hasattr(ds, "get_recon_viz_sample"):
                return ds

        # 3. CombinedMultiTaskDataModule exposes fh_train
        if hasattr(dm, "fh_train"):
            ds = dm.fh_train
            if hasattr(ds, "get_recon_viz_sample"):
                return ds

        # 4. Legacy: per-task dataset attributes
        for attr in [
            "train_dataset_cosia", "train_dataset_lpis",
            "train_dataset_flair_recon",
        ]:
            obj = getattr(dm, attr, None)
            if obj is not None and hasattr(obj, "get_recon_viz_sample"):
                return obj

        # 5. ChunkedInterleavedDataset wrappers
        if hasattr(dm, "train_dataset") and hasattr(dm.train_dataset, "datasets"):
            for name, ds in dm.train_dataset.datasets.items():
                if hasattr(ds, "get_recon_viz_sample"):
                    return ds

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Main hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = self._find_dataset(trainer)
        if dataset is None:
            return

        device = pl_module.device
        pl_module.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                fig = self._visualise_sample(
                    dataset, idx, device, pl_module, trainer
                )
                if fig is not None:
                    figures.append((f"flairhub_recon/sample_{idx}", fig))
            except Exception as e:
                print(f"[FlairHubReconViz] Error on sample {idx}: {e}")
                import traceback
                traceback.print_exc()

        # Log
        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)}, step=trainer.global_step)
            plt.close("all")

        pl_module.train()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-sample pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _visualise_sample(self, dataset, idx, device, pl_module, trainer):
        sample = dataset.get_recon_viz_sample(idx)

        # ── Extract viz extras BEFORE collating ─────────────────────
        # These contain nested dicts / non-tensor types that
        # collate_multitask can't stack into batches.
        modality_info = sample.pop("_viz_modality_info", {})
        raw_image = sample.pop("_viz_image", None)
        patch_id = sample.pop("_viz_patch_id", "")
        n_real = sample.pop("_viz_n_real", 0)

        if not modality_info:
            print(f"[FlairHubReconViz] Sample {idx}: no modality data")
            return None

        # ── Keep a CPU copy of queries for image reconstruction ─────
        queries_cpu = (
            sample["tasks"]["reconstruction"]["queries"].clone()
        )

        # ── Collate into batch-of-1 + send to device ───────────────
        batch = collate_multitask([sample])
        batch = _batch_to_device(batch, device)

        # ── Forward pass (multi-task encode-once path) ──────────────
        with torch.no_grad():
            result = pl_module.forward_multitask(batch, training=False)

        # result = {"reconstruction": [1, M, 1]}
        preds = result["reconstruction"].squeeze(0).squeeze(-1).cpu()
        gt_values = queries_cpu[:, 4]

        # ── Reconstruct per-modality images ─────────────────────────
        modality_images = {}

        for mod in self.MODALITY_ORDER:
            if mod not in modality_info:
                continue

            info = modality_info[mod]
            start = info["offset"]
            end = start + info["count"]
            n_bands, H, W = info["shape"]
            spec_idx = info["spectral_indices"]

            mod_queries = queries_cpu[start:end]
            mod_preds = preds[start:end]
            mod_gt = gt_values[start:end]

            gt_img = self._reconstruct_image(
                mod_queries, mod_gt, n_bands, H, W, spec_idx
            )
            pred_img = self._reconstruct_image(
                mod_queries, mod_preds, n_bands, H, W, spec_idx
            )

            # Per-modality metrics (skip empty modalities)
            gt_v = gt_img[~torch.isnan(gt_img)]
            pred_v = pred_img[~torch.isnan(pred_img)]

            if gt_v.numel() == 0 or pred_v.numel() == 0:
                print(
                    f"[FlairHubReconViz] Sample {idx} | {mod:8s} "
                    f"({n_bands}×{H}×{W}, {info['count']} tok): "
                    f"EMPTY — no valid pixels after scatter"
                )
                continue

            mse = ((mod_gt - mod_preds) ** 2).mean().item()
            if mod_gt.shape[0] > 2:
                corr = torch.corrcoef(
                    torch.stack([mod_preds, mod_gt])
                )[0, 1].item()
            else:
                corr = float("nan")

            modality_images[mod] = {
                "gt": gt_img,
                "pred": pred_img,
                "mse": mse,
                "corr": corr,
                "n_tokens": info["count"],
            }

            print(
                f"[FlairHubReconViz] Sample {idx} | {mod:8s} "
                f"({n_bands}×{H}×{W}, {info['count']} tok): "
                f"MSE={mse:.6f}  corr={corr:.4f}  "
                f"GT=[{gt_v.min():.4f},{gt_v.max():.4f}]  "
                f"Pred=[{pred_v.min():.4f},{pred_v.max():.4f}]"
            )

        # ── Build figure ────────────────────────────────────────────
        fig = self._make_multimodal_figure(
            modality_images,
            sample_idx=idx,
            epoch=trainer.current_epoch,
            patch_id=patch_id,
        )
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Image reconstruction from tokens
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_image(
        queries: torch.Tensor,
        values: torch.Tensor,
        n_bands: int,
        H: int,
        W: int,
        spectral_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter token values back into a [C, H, W] image.

        Token x/y coordinates are reference-grid indices (e.g., 507–516
        for a 10×10 S2 crop centered in a 1024 reference grid).  We
        convert them to local [0, H) × [0, W) pixel indices by
        subtracting the per-axis minimum.

        Args:
            queries: [N, 8] token descriptors (cols: val, x, y, spec, ...)
            values:  [N]    predicted or GT values
            n_bands: number of spectral bands for this modality
            H, W:   spatial dimensions
            spectral_indices: [n_bands] lookup-table indices for this
                modality's bands (in band order)

        Returns:
            [n_bands, H, W] tensor (NaN where no token was placed)
        """
        img = torch.full((n_bands, H, W), float("nan"))

        if queries.shape[0] == 0:
            return img

        # Map lookup-table spectral index → 0-based band position
        spec_to_band = {s.item(): i for i, s in enumerate(spectral_indices)}

        x_raw = queries[:, 1].long()
        y_raw = queries[:, 2].long()
        spec = queries[:, 3].long()

        # ── Convert reference-grid coords to local pixel indices ────
        # Reference grid coords are offset (e.g., 507 + global_offset
        # for a 10×10 crop centered in a 1024 reference grid).
        # Subtracting the min gives local [0, H) × [0, W) indices.
        x = x_raw - x_raw.min()
        y = y_raw - y_raw.min()

        bands = torch.tensor(
            [spec_to_band.get(s.item(), -1) for s in spec],
            dtype=torch.long,
        )
        valid = (
            (bands >= 0) & (bands < n_bands)
            & (x >= 0) & (x < W)
            & (y >= 0) & (y < H)
        )

        img[bands[valid], y[valid], x[valid]] = values[valid].float()
        return img

    # ─────────────────────────────────────────────────────────────────────────
    # RGB conversion
    # ─────────────────────────────────────────────────────────────────────────

    def _to_rgb(self, img: torch.Tensor, mod: str) -> np.ndarray:
        """
        Convert [C, H, W] → [3, H, W] float32 numpy in [0, 1].

        Optical modalities use natural RGB bands.
        SAR modalities use false-colour: R=VV, G=VH, B=VV.
        """
        C = img.shape[0]
        rgb_idx = self.MODALITY_RGB.get(mod)

        if rgb_idx is not None and C >= 3:
            rgb = torch.stack(
                [img[rgb_idx[0]], img[rgb_idx[1]], img[rgb_idx[2]]]
            )
        elif C >= 2:
            # SAR false-colour
            rgb = torch.stack([img[0], img[1], img[0]])
        else:
            rgb = img[0:1].expand(3, -1, -1)

        return self._percentile_stretch(rgb.numpy())

    @staticmethod
    def _percentile_stretch(
        rgb: np.ndarray, lo_pct=2, hi_pct=98
    ) -> np.ndarray:
        """Per-channel percentile stretch to [0, 1], ignoring NaNs."""
        out = rgb.copy()
        for c in range(3):
            ch = out[c]
            valid = ~np.isnan(ch)
            if valid.sum() < 10:
                out[c] = 0.0
                continue
            lo = np.nanpercentile(ch, lo_pct)
            hi = np.nanpercentile(ch, hi_pct)
            if hi - lo > 1e-6:
                out[c] = (ch - lo) / (hi - lo)
            else:
                out[c] = 0.0
        return np.clip(np.nan_to_num(out, nan=0.0), 0, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure layout
    # ─────────────────────────────────────────────────────────────────────────

    def _make_multimodal_figure(
        self, modality_images, sample_idx, epoch, patch_id=""
    ):
        """
        Build a (3 rows × N columns) figure:
            Row 0: Ground truth RGB per modality
            Row 1: Predicted RGB per modality
            Row 2: Absolute error heat-map per modality
        """
        n_mods = len(modality_images)
        if n_mods == 0:
            return None

        fig, axes = plt.subplots(
            3, n_mods,
            figsize=(4.5 * n_mods, 12),
            squeeze=False,
        )

        for col, (mod, data) in enumerate(modality_images.items()):
            gt_rgb = self._to_rgb(data["gt"], mod)
            pred_rgb = self._to_rgb(data["pred"], mod)
            error = np.abs(pred_rgb - gt_rgb).mean(axis=0)  # [H, W]

            display_name = self.MODALITY_DISPLAY.get(mod, mod)
            n_bands = data["gt"].shape[0]

            # GT
            axes[0, col].imshow(
                np.transpose(gt_rgb, (1, 2, 0)), interpolation="nearest"
            )
            axes[0, col].set_title(
                f"GT: {display_name}\n{n_bands} bands, "
                f"{data['gt'].shape[1]}×{data['gt'].shape[2]} px",
                fontsize=9,
            )
            axes[0, col].axis("off")

            # Prediction
            axes[1, col].imshow(
                np.transpose(pred_rgb, (1, 2, 0)), interpolation="nearest"
            )
            axes[1, col].set_title(
                f"Pred  MSE={data['mse']:.5f}\n"
                f"corr={data['corr']:.3f}  ({data['n_tokens']} tok)",
                fontsize=9,
            )
            axes[1, col].axis("off")

            # Error
            im = axes[2, col].imshow(
                error, cmap="hot", vmin=0, interpolation="nearest"
            )
            axes[2, col].set_title("Abs Error (RGB mean)", fontsize=9)
            axes[2, col].axis("off")
            fig.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"FlairHub Reconstruction — {patch_id} — Epoch {epoch}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_to_device(batch, device):
    """Recursively move tensors in a nested dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out