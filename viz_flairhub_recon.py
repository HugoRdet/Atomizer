#!/usr/bin/env python3
"""
Standalone FlairHub Reconstruction Visualizer
==============================================

Loads a trained checkpoint, runs reconstruction on a few FlairHub samples,
and saves per-modality GT / Predicted / Error figures to disk (+ optional W&B).

Completely decoupled from training — no callbacks, no DDP, no risk of
blocking the trainer.

Usage
-----
    python visualize_flairhub_recon.py \
        --ckpt_path  checkpoints/epoch=19-step=12000.ckpt \
        --config_model  model_flairhub_recon.yaml \
        --sample_indices 0 1 5 \
        --output_dir  viz_recon/

    # With W&B logging:
    python visualize_flairhub_recon.py \
        --ckpt_path ... --config_model ... --log_wandb \
        --wandb_run_id abc123   # attach to existing run (optional)
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Project imports ──────────────────────────────────────────────────────────
from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_FLAIRHUB import (
    FlairHubMultiTask,
    DATASET_NAME,
    RES_AERIAL, RES_SPOT, RES_S2, RES_S1,
    N_BANDS_AERIAL, N_BANDS_SPOT, N_BANDS_S2, N_BANDS_S1,
    SIZE_AERIAL, SIZE_SPOT, SIZE_S2, SIZE_S1,
)
from training.utils.datasets.token_grouping import collate_multitask


# =============================================================================
# RECONSTRUCTION UTILITIES  (extracted from the old callback, standalone)
# =============================================================================

MODALITY_ORDER = ["aerial", "spot", "s2", "s1_asc", "s1_des"]

MODALITY_RGB = {
    "aerial": [0, 1, 2],       # R, G, B
    "spot":   [0, 1, 2],       # R, G, B
    "s2":     [2, 1, 0],       # B04(Red), B03(Green), B02(Blue)
    "s1_asc": None,            # SAR → false-colour
    "s1_des": None,
}

MODALITY_DISPLAY = {
    "aerial": "Aerial (0.2 m)",
    "spot":   "SPOT (1.6 m)",
    "s2":     "S2 (10 m)",
    "s1_asc": "S1-Asc (10 m)",
    "s1_des": "S1-Desc (10 m)",
}


def reconstruct_image(
    queries: torch.Tensor,
    values: torch.Tensor,
    n_bands: int,
    H: int,
    W: int,
    spectral_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Scatter token values back into a [C, H, W] image (NaN = unfilled).

    Token x/y coordinates are reference-grid indices (e.g., 507–516
    for a 10×10 S2 crop centered in a 1024 reference grid).  We
    convert them to local [0, H) × [0, W) pixel indices by
    subtracting the per-axis minimum.
    """
    img = torch.full((n_bands, H, W), float("nan"))

    if queries.shape[0] == 0:
        return img

    spec_to_band = {s.item(): i for i, s in enumerate(spectral_indices)}

    x_raw = queries[:, 1].long()
    y_raw = queries[:, 2].long()
    spec = queries[:, 3].long()

    # ── Convert reference-grid coords to local pixel indices ────────
    x = x_raw - x_raw.min()
    y = y_raw - y_raw.min()

    bands = torch.tensor(
        [spec_to_band.get(s.item(), -1) for s in spec], dtype=torch.long,
    )
    valid = (
        (bands >= 0) & (bands < n_bands)
        & (x >= 0) & (x < W)
        & (y >= 0) & (y < H)
    )

    n_total = queries.shape[0]
    n_valid = valid.sum().item()
    n_nan = n_total - n_valid
    if n_nan > 0:
        print(
            f"    scatter: {n_valid}/{n_total} placed "
            f"(x∈[{x.min()},{x.max()}] y∈[{y.min()},{y.max()}] "
            f"target=[{n_bands},{H},{W}])"
        )

    img[bands[valid], y[valid], x[valid]] = values[valid].float()
    return img


def to_rgb(img: torch.Tensor, mod: str) -> np.ndarray:
    """[C, H, W] → [3, H, W] float32 numpy in [0, 1]."""
    C = img.shape[0]
    rgb_idx = MODALITY_RGB.get(mod)

    if rgb_idx is not None and C >= 3:
        rgb = torch.stack([img[rgb_idx[0]], img[rgb_idx[1]], img[rgb_idx[2]]])
    elif C >= 2:
        rgb = torch.stack([img[0], img[1], img[0]])  # SAR false-colour
    else:
        rgb = img[0:1].expand(3, -1, -1)

    return percentile_stretch(rgb.numpy())


def percentile_stretch(rgb: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
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


# =============================================================================
# SINGLE-SAMPLE PIPELINE
# =============================================================================

@torch.no_grad()
def visualise_sample(
    dataset: FlairHubMultiTask,
    idx: int,
    model: Model_Pretrain,
    device: torch.device,
) -> tuple:
    """
    Run reconstruction on one sample.

    Returns:
        (modality_images dict, patch_id str)   or  (None, "")
    """
    sample = dataset.get_recon_viz_sample(idx)

    # ── Pop viz extras before collating ──────────────────────────────────
    modality_info = sample.pop("_viz_modality_info", {})
    raw_image     = sample.pop("_viz_image", None)
    patch_id      = sample.pop("_viz_patch_id", "")
    n_real        = sample.pop("_viz_n_real", 0)

    if not modality_info:
        print(f"  [sample {idx}] no modality data — skipping")
        return None, ""

    # ── CPU copy of queries for image reconstruction ─────────────────────
    queries_cpu = sample["tasks"]["reconstruction"]["queries"].clone()

    # ── Collate into batch-of-1 → device ─────────────────────────────────
    batch = collate_multitask([sample])
    batch = _to_device(batch, device)

    # ── Forward ──────────────────────────────────────────────────────────
    result = model.forward_multitask(batch, training=False)

    preds = result["reconstruction"].squeeze(0).squeeze(-1).cpu()
    gt_values = queries_cpu[:, 4]

    # ── Per-modality reconstruction ──────────────────────────────────────
    modality_images = {}

    for mod in MODALITY_ORDER:
        if mod not in modality_info:
            continue

        info  = modality_info[mod]
        start = info["offset"]
        end   = start + info["count"]
        n_bands, H, W = info["shape"]
        spec_idx = info["spectral_indices"]

        mod_queries = queries_cpu[start:end]
        mod_preds   = preds[start:end]
        mod_gt      = gt_values[start:end]

        gt_img   = reconstruct_image(mod_queries, mod_gt,   n_bands, H, W, spec_idx)
        pred_img = reconstruct_image(mod_queries, mod_preds, n_bands, H, W, spec_idx)

        # Diagnostic — guard against empty images
        gt_v = gt_img[~torch.isnan(gt_img)]
        pred_v = pred_img[~torch.isnan(pred_img)]
        nan_pct = torch.isnan(gt_img).float().mean().item() * 100

        if gt_v.numel() == 0 or pred_v.numel() == 0:
            print(
                f"  {mod:8s} ({n_bands}×{H}×{W}, {info['count']} tok): "
                f"EMPTY — no valid pixels after scatter  NaN%={nan_pct:.1f}"
            )
            continue

        # Metrics
        mse = ((mod_gt - mod_preds) ** 2).mean().item()
        if mod_gt.shape[0] > 2:
            corr = torch.corrcoef(torch.stack([mod_preds, mod_gt]))[0, 1].item()
        else:
            corr = float("nan")

        modality_images[mod] = {
            "gt": gt_img, "pred": pred_img,
            "mse": mse, "corr": corr, "n_tokens": info["count"],
        }

        print(
            f"  {mod:8s} ({n_bands}×{H}×{W}, {info['count']} tok): "
            f"MSE={mse:.6f}  corr={corr:.4f}  "
            f"GT=[{gt_v.min():.4f},{gt_v.max():.4f}]  "
            f"Pred=[{pred_v.min():.4f},{pred_v.max():.4f}]  "
            f"NaN%={nan_pct:.1f}"
        )

    return modality_images, patch_id


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(modality_images: dict, sample_idx: int, patch_id: str, title_extra: str = ""):
    """3-row (GT / Pred / Error) × N-modality figure."""
    n_mods = len(modality_images)
    if n_mods == 0:
        return None

    fig, axes = plt.subplots(3, n_mods, figsize=(4.5 * n_mods, 12), squeeze=False)

    for col, (mod, data) in enumerate(modality_images.items()):
        gt_rgb   = to_rgb(data["gt"], mod)
        pred_rgb = to_rgb(data["pred"], mod)
        error    = np.abs(pred_rgb - gt_rgb).mean(axis=0)

        display = MODALITY_DISPLAY.get(mod, mod)
        n_bands = data["gt"].shape[0]

        axes[0, col].imshow(np.transpose(gt_rgb, (1, 2, 0)), interpolation="nearest")
        axes[0, col].set_title(
            f"GT: {display}\n{n_bands} bands, "
            f"{data['gt'].shape[1]}×{data['gt'].shape[2]} px", fontsize=9,
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(np.transpose(pred_rgb, (1, 2, 0)), interpolation="nearest")
        axes[1, col].set_title(
            f"Pred  MSE={data['mse']:.5f}\n"
            f"corr={data['corr']:.3f}  ({data['n_tokens']} tok)", fontsize=9,
        )
        axes[1, col].axis("off")

        im = axes[2, col].imshow(error, cmap="hot", vmin=0, interpolation="nearest")
        axes[2, col].set_title("Abs Error (RGB mean)", fontsize=9)
        axes[2, col].axis("off")
        fig.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

    suptitle = f"FlairHub Reconstruction — {patch_id}"
    if title_extra:
        suptitle += f" — {title_extra}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# HELPERS
# =============================================================================

def _to_device(batch, device):
    """Recursively move tensors in a nested dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _to_device(v, device)
        else:
            out[k] = v
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone FlairHub reconstruction visualization"
    )
    # ── Required ─────────────────────────────────────────────────────────
    parser.add_argument("--ckpt_path",     type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--config_model",  type=str, required=True,
                        help="Model config yaml file (in training/configs/)")

    # ── Dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--flairhub_path",    type=str,
                        default="./data/FLAIR-HUB/extracted")
    parser.add_argument("--flairhub_csv_dir", type=str,
                        default="./data/FLAIR-HUB")
    parser.add_argument("--mode",             type=str, default="validation",
                        choices=["train", "validation"],
                        help="Which split to visualise from")

    # ── Samples ──────────────────────────────────────────────────────────
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1],
                        help="Patch indices to visualise")

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="viz_recon",
                        help="Directory to save figures")
    parser.add_argument("--log_wandb",  action="store_true",
                        help="Also log figures to W&B")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Attach to existing W&B run (optional)")

    # ── Device ───────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config + lookup ──────────────────────────────────────────────────
    config_model   = read_yaml("./training/configs/" + args.config_model)
    configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
    bands_yaml      = "./data/bands_info/bands.yaml"
    lookup_table    = Lookup_encoding(
        read_yaml(configs_dataset), read_yaml(bands_yaml), config_model
    )
    dataset_config  = read_yaml(bands_yaml)

    # ── Model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt_path}")
    model = Model_Pretrain.load_from_checkpoint(
        args.ckpt_path,
        config=config_model,
        wand=False,
        name="viz",
        transform=None,
        lookup_table=lookup_table,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f"Loading FlairHub ({args.mode}) …")
    dataset = FlairHubMultiTask(
        mode=args.mode,
        root_path=args.flairhub_path,
        dataset_config=dataset_config,
        config_model=config_model,
        look_up=lookup_table,
        tasks=["reconstruction"],
        max_queries_recon=200_000,
        csv_dir=args.flairhub_csv_dir,
    )
    print(f"Dataset: {len(dataset)} samples\n")

    # ── W&B (optional) ───────────────────────────────────────────────────
    wandb_run = None
    if args.log_wandb:
        import wandb
        if args.wandb_run_id:
            wandb_run = wandb.init(
                project="Atomizer_Pretrain", id=args.wandb_run_id, resume="allow"
            )
        else:
            wandb_run = wandb.init(
                project="Atomizer_Pretrain",
                name=f"viz_recon_{os.path.basename(args.ckpt_path)}",
                job_type="visualization",
            )

    # ── Run ──────────────────────────────────────────────────────────────
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]

    for idx in args.sample_indices:
        if idx >= len(dataset):
            print(f"[SKIP] index {idx} >= dataset size {len(dataset)}")
            continue

        print(f"── Sample {idx} ──────────────────────────────────────")
        modality_images, patch_id = visualise_sample(dataset, idx, model, device)

        if modality_images is None:
            continue

        fig = make_figure(modality_images, idx, patch_id, title_extra=ckpt_name)
        if fig is None:
            continue

        # Save to disk
        fname = f"recon_{ckpt_name}_sample{idx}.png"
        fpath = os.path.join(args.output_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  → saved: {fpath}")

        # W&B
        if wandb_run is not None:
            import wandb
            wandb.log({f"flairhub_recon/sample_{idx}": wandb.Image(fig)})

        plt.close(fig)

    print("\nDone.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()