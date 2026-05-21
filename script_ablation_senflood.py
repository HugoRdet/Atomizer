"""
Sen1Floods11 Sliding-Window Ablation Script
=============================================

Standalone evaluation script for the "Generalization to unseen input
geometries" ablation. Runs sliding-window inference at a fixed token-set
fraction, treating each window as a standalone smaller image (re-tokenized
via TokenBuilder so coordinates are centered in the reference grid).

Per-window predictions are logit-averaged back onto the full 512x512 grid
via the existing stitch_predictions utility.

Each launch evaluates one fraction value; run 5x for the full table:

    --fraction 1.00   (sanity: matches standard full-image test)
    --fraction 0.75
    --fraction 0.50
    --fraction 0.25
    --fraction 0.10

Example
-------
    python script_ablation_senflood.py \
        --xp_name senflood_v1_frac_0.50 \
        --ckpt_path ./pth_files/atomiser_senflood_v1-best.ckpt \
        --fraction 0.50 \
        --num_workers 4

Design notes
------------
- We bypass Lightning's .test() flow entirely. The trainer's existing
  sliding-window path expects a specific batch structure that's awkward
  to construct outside of training. A manual inference loop is simpler,
  easier to debug, and isolates the ablation logic from production code.

- Window size: round(512 * sqrt(fraction)), then rounded DOWN to nearest
  even integer for compatibility. Stride = window_size // 2 (50% overlap).
  Edge windows are shifted inward to end at the image boundary (handled
  by compute_crop_positions).

- Each window is treated as an independent smaller image: tokens are built
  for the crop alone, TokenBuilder centers their coordinates in the 512
  reference grid. The model has never seen these token counts/layouts at
  training time — that's the point of the ablation.

- Aggregation: logit averaging across all windows covering each pixel
  (standard sliding-window-inference convention, matches what production
  trainer does for full-image-mode sliding window).

- Metrics: torchmetrics directly (no Lightning), computed across the full
  test set, aggregated at the end. Same metric definitions as the trainer.
"""

import os
import argparse
import math

import numpy as np
import torch
import torchmetrics
from tqdm import tqdm
from torch.utils.data import DataLoader

# ── Project imports ─────────────────────────────────────────────────────
from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.utils_dataset_SENFLOOD import Sen1Floods11Dataset
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.sliding_window import (
    compute_crop_positions,
    stitch_predictions,
)
from training.trainer_SENFLOOD import Model_SenFlood

# ── Restrict TokenBuilder.REFERENCE_SIZES to Sen1Floods11's resolution ─
# The class-level dict may have accumulated resolutions from other datasets
# (PASTIS / FLAIR-HUB / etc.) which would balloon the lookup table sizes
# at construction time and cause checkpoint shape mismatches.
# Sen1Floods11 training only ever registered the 10m resolution, so we
# restrict here to match that exact state.
TokenBuilder.REFERENCE_SIZES = {10.0: 512}


# =============================================================================
# UTILITIES
# =============================================================================

def round_to_even(x: float) -> int:
    """Round to nearest integer, then round down to nearest even number."""
    n = int(round(x))
    if n % 2 != 0:
        n -= 1
    return max(2, n)


def fraction_to_window_size(fraction: float, full_size: int) -> int:
    """
    Convert a token-fraction to a window pixel size.

    fraction = (window_size / full_size) ** 2 (token count is proportional
    to area), so window_size = full_size * sqrt(fraction).
    Rounded down to nearest even integer.
    """
    raw = full_size * math.sqrt(fraction)
    return round_to_even(raw)


def build_window_batch(
    image_full: torch.Tensor,           # [C, H, W]
    label_full: torch.Tensor,           # [H, W]
    top: int,
    left: int,
    window_size: int,
    token_builder: TokenBuilder,
    spectral_indices: torch.Tensor,
    resolution: float,
    resolution_idx: int,
    time_idx: int,
) -> dict:
    """
    Build a single-window batch in the format the Atomizer encoder expects.

    The window is re-tokenized as if it were a standalone smaller image:
    TokenBuilder centers its coordinates in the 512 reference grid.
    """
    # Crop image and label.
    crop_img = image_full[:, top:top + window_size, left:left + window_size]
    crop_lbl = label_full[top:top + window_size, left:left + window_size]
    C, h, w = crop_img.shape

    # Build tokens for the crop (treated as a standalone image of size h x w).
    tokens = token_builder.build_tokens(
        image=crop_img,
        label=crop_lbl,
        resolution=resolution,
        spectral_indices=spectral_indices,
        resolution_idx=resolution_idx,
        time_idx=time_idx,
    )

    # Build queries (per-pixel) for the crop.
    queries = token_builder.build_queries(
        label=crop_lbl,
        resolution=resolution,
        first_spectral_idx=spectral_indices[0],
        resolution_idx=resolution_idx,
        time_idx=time_idx,
    )

    # Batch the single window (batch dim = 1).
    return {
        "groups": {
            resolution: {
                "tokens": tokens.unsqueeze(0),                        # [1, N, 8]
                "mask":   torch.zeros(1, tokens.shape[0]),            # [1, N]
                "shape":  (C, h, w),
            }
        },
        "queries":      queries.unsqueeze(0),                         # [1, M, 8]
        "queries_mask": torch.zeros(1, queries.shape[0]),             # [1, M]
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Sen1Floods11 Sliding-Window Ablation"
)
parser.add_argument("--xp_name",    type=str, required=True,
                    help="Run name (used for logging only).")
parser.add_argument("--ckpt_path",  type=str, required=True,
                    help="Path to Atomizer checkpoint (.ckpt or .pth).")
parser.add_argument("--config_model", type=str,
                    default="config_test-SENFLOOD.yaml",
                    help="Atomizer config YAML under training/configs/")
parser.add_argument("--dataset_config", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml",
                    help="Dataset-level resolution config YAML (absolute or "
                         "relative path).")
parser.add_argument("--bands_yaml", type=str,
                    default="./data/bands_info/bands.yaml",
                    help="Canonical band metadata YAML (used as both the "
                         "Lookup_encoding bands source and the dataset's "
                         "dataset_config — matches the production script).")
parser.add_argument("--root_path",  type=str, default="./data/SENFLOOD")
parser.add_argument("--fraction",   type=float, required=True,
                    help="Fraction of tokens per window (= (win/512)^2). "
                         "1.0 = no sliding window. Smaller = more windows.")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--device",     type=str, default="cuda")
parser.add_argument("--wandb",      action="store_true",
                    help="Log results to wandb.")
args = parser.parse_args()


# =============================================================================
# SETUP
# =============================================================================

print(f"\n{'='*70}")
print(f"  Sen1Floods11 Sliding-Window Ablation")
print(f"{'='*70}")
print(f"  XP name:       {args.xp_name}")
print(f"  Checkpoint:    {args.ckpt_path}")
print(f"  Fraction:      {args.fraction}")

FULL_SIZE = 512
window_size = fraction_to_window_size(args.fraction, FULL_SIZE)
stride      = max(2, window_size // 2)
print(f"  Window size:   {window_size}×{window_size} (rounded to even)")
print(f"  Stride:        {stride} (50% overlap)")

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"  Device:        {device}")


# ── Load configs ──────────────────────────────────────────────────────
config_model = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset = read_yaml(args.dataset_config)

# Number of classes & ignore index from the trainer config (match production).
num_classes  = config_model["trainer"]["num_classes"]
ignore_index = 255


# ── Lookup table + bands ──────────────────────────────────────────────
# Lookup_encoding takes two separate YAML files:
#   1. configs_dataset: dataset-level config (resolutions, etc.)
#   2. bands: the canonical band metadata YAML (./data/bands_info/bands.yaml)
# This matches the production launch script's pattern.
bands = read_yaml(args.bands_yaml)
lookup_table = Lookup_encoding(configs_dataset, bands, config_model)

# Pre-register Sen1Floods11 resolution (10m) so TokenBuilder is happy.
lookup_table.get_or_register_modality(10.0, FULL_SIZE)
lookup_table.get_resolution_idx(10.0)


# ── Dataset (test split, full images at 512x512) ──────────────────────
# Note: Sen1Floods11Dataset's `dataset_config` is the bands YAML (not the
# resolution config), because the dataset accesses dataset_config["bands_senflood"].
# This matches the production launch script's pattern.
print(f"\n[Ablation] Building test dataset...")
test_ds = Sen1Floods11Dataset(
    root_path=args.root_path,
    mode="test",
    dataset_config=bands,
    config_model=config_model,
    look_up=lookup_table,
)
print(f"[Ablation] Test set: {len(test_ds)} samples")

# Plain DataLoader, batch_size=1. We don't use the dataset's tokens directly;
# we re-tokenize per window in the loop below.
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=lambda x: x[0],   # single-sample collate, return dict directly
)


# ── Token builder (shared with dataset's lookup table) ────────────────
token_builder = TokenBuilder(lookup_table)

# Cache modality-level info that doesn't change per sample.
spectral_indices = test_ds.spectral_indices
resolution       = test_ds.OPTICAL_RESOLUTION
resolution_idx   = test_ds.resolution_idx
time_idx         = test_ds.TIME_IDX_NA


# ── Model: load from checkpoint ────────────────────────────────────────
print(f"\n[Ablation] Building model and loading checkpoint...")
model = Model_SenFlood(
    config=config_model,
    wand=False,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
result = model.load_state_dict(state, strict=False)
print(f"[Ablation] missing keys: {len(result.missing_keys)}, "
      f"unexpected keys: {len(result.unexpected_keys)}")

model = model.to(device).eval()


# ── Precompute window positions (same for every test image, all 512x512) ──
positions = compute_crop_positions(
    full_h=FULL_SIZE, full_w=FULL_SIZE,
    crop_h=window_size, crop_w=window_size,
    stride_h=stride, stride_w=stride,
)
print(f"[Ablation] Windows per image: {len(positions)}")


# ── Metrics (torchmetrics, instantiated directly, no Lightning) ──────────
metric_iou_macro = torchmetrics.JaccardIndex(
    task="multiclass", num_classes=num_classes,
    average="macro", ignore_index=ignore_index,
).to(device)
metric_iou_per_class = torchmetrics.JaccardIndex(
    task="multiclass", num_classes=num_classes,
    average=None, ignore_index=ignore_index,
).to(device)
metric_acc = torchmetrics.Accuracy(
    task="multiclass", num_classes=num_classes,
    average="macro", ignore_index=ignore_index,
).to(device)


# =============================================================================
# INFERENCE LOOP
# =============================================================================

print(f"\n[Ablation] Running inference over {len(test_ds)} test images "
      f"x {len(positions)} windows each "
      f"= {len(test_ds) * len(positions)} forward passes...")

with torch.no_grad():
    for sample in tqdm(test_loader, total=len(test_ds), desc="Test"):
        # Sample from the dataset's __getitem__ — CPU tensors:
        #   sample["image"]  : [C, H, W]  (already normalized)
        #   sample["label"]  : [H, W]
        # We keep them on CPU for tokenization (TokenBuilder builds coordinate
        # tensors on CPU; mixing devices in torch.cat would error). The final
        # batch is moved to device just before the forward pass.
        image_full = sample["image"]                       # CPU [C, 512, 512]
        label_full = sample["label"]                       # CPU [512, 512]
        C, H, W = image_full.shape

        # Sanity: image must match expected full size for the position grid.
        assert (H, W) == (FULL_SIZE, FULL_SIZE), (
            f"Expected {FULL_SIZE}x{FULL_SIZE} image, got {H}x{W}"
        )

        # Run inference per window.
        crop_logits_list = []
        for (top, left) in positions:
            batch = build_window_batch(
                image_full=image_full,
                label_full=label_full,
                top=top, left=left,
                window_size=window_size,
                token_builder=token_builder,
                spectral_indices=spectral_indices,
                resolution=resolution,
                resolution_idx=resolution_idx,
                time_idx=time_idx,
            )
            # Move the assembled batch to the model's device.
            batch["groups"][resolution]["tokens"] = batch["groups"][resolution]["tokens"].to(device)
            batch["groups"][resolution]["mask"]   = batch["groups"][resolution]["mask"].to(device)
            batch["queries"]      = batch["queries"].to(device)
            batch["queries_mask"] = batch["queries_mask"].to(device)

            # Forward. The encoder returns either logits [B, M, C] directly
            # or a dict with "predictions". Match the trainer's handling.
            out = model.forward(batch, training=False, return_for_error=False)
            logits = out["predictions"] if isinstance(out, dict) else out  # [1, M, num_classes]
            logits = logits.squeeze(0)                                      # [M, num_classes]
            crop_logits_list.append(logits)

        # Stitch all window predictions onto the full 512x512 grid.
        preds_full, _ = stitch_predictions(
            crop_logits_list=crop_logits_list,
            crop_positions=positions,
            crop_h=window_size, crop_w=window_size,
            full_h=FULL_SIZE, full_w=FULL_SIZE,
            num_classes=num_classes,
        )

        # Update metrics over valid pixels only. Move label to device for
        # the metric update (preds are already on device from stitching).
        label_full_dev = label_full.to(device)
        valid = (label_full_dev != ignore_index)
        if valid.sum() > 0:
            preds_valid  = preds_full[valid]
            labels_valid = label_full_dev[valid]
            metric_iou_macro.update(preds_valid, labels_valid)
            metric_iou_per_class.update(preds_valid, labels_valid)
            metric_acc.update(preds_valid, labels_valid)


# =============================================================================
# REPORT
# =============================================================================

miou_macro    = metric_iou_macro.compute().item()
iou_per_class = metric_iou_per_class.compute().cpu().numpy()
acc_macro     = metric_acc.compute().item()

class_names = (config_model["trainer"].get("class_names")
               or [f"class_{i}" for i in range(num_classes)])

print(f"\n{'='*70}")
print(f"  ABLATION RESULTS — fraction={args.fraction}")
print(f"  Window: {window_size}x{window_size}, stride: {stride}, "
      f"windows/image: {len(positions)}")
print(f"{'='*70}")
print(f"  mIoU (macro):   {miou_macro:.4f}")
print(f"  Accuracy:       {acc_macro:.4f}")
for i, name in enumerate(class_names):
    if i < len(iou_per_class):
        print(f"  IoU {name}: {iou_per_class[i]:.4f}")
print(f"{'='*70}\n")


# ── Optional wandb logging ────────────────────────────────────────────
if args.wandb:
    try:
        import wandb
        wandb.init(
            project="Atomizer-Senflood-Ablation",
            name=f"ablation_{args.xp_name}_frac{args.fraction}",
            config={
                "fraction":    args.fraction,
                "window_size": window_size,
                "stride":      stride,
                "n_windows":   len(positions),
                "ckpt_path":   args.ckpt_path,
            },
        )
        log_dict = {
            "test_mIoU":     miou_macro,
            "test_accuracy": acc_macro,
            "fraction":      args.fraction,
            "window_size":   window_size,
        }
        for i, name in enumerate(class_names):
            if i < len(iou_per_class):
                log_dict[f"test_IoU_{name}"] = float(iou_per_class[i])
        wandb.log(log_dict)
        wandb.finish()
    except Exception as e:
        print(f"[Ablation] WARNING: wandb logging failed: {e}")


# ── Plain-text result file (one line per launch — easy to grep later) ─
os.makedirs("./ablation_results", exist_ok=True)
out_path = f"./ablation_results/senflood_ablation_{args.xp_name}_frac{args.fraction}.txt"
with open(out_path, "w") as f:
    f.write(f"xp_name={args.xp_name}\n")
    f.write(f"ckpt_path={args.ckpt_path}\n")
    f.write(f"fraction={args.fraction}\n")
    f.write(f"window_size={window_size}\n")
    f.write(f"stride={stride}\n")
    f.write(f"n_windows={len(positions)}\n")
    f.write(f"mIoU={miou_macro:.6f}\n")
    f.write(f"accuracy={acc_macro:.6f}\n")
    for i, name in enumerate(class_names):
        if i < len(iou_per_class):
            f.write(f"IoU_{name}={iou_per_class[i]:.6f}\n")
print(f"[Ablation] Results saved to: {out_path}")