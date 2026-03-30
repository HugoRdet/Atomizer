#!/usr/bin/env python3
"""
MDAS Baseline Evaluation — Sliding Window Inference + Visualization
====================================================================

Loads a trained baseline checkpoint, runs sliding window inference over
the full test image, and produces:

  1. Full-image prediction map (PNG)
  2. Ground truth label map (PNG)
  3. Per-class IoU table + bar chart (PNG)
  4. Confusion matrix (PNG)
  5. Overlay: prediction vs GT side-by-side (PNG)
  6. Metrics JSON

Supports:
  - Native eval (same sensor as training)
  - Cross-sensor eval (e.g. HySpex-trained model → S2 test data)
    with spectral interpolation + spatial resize

Sliding window strategy:
  - Window size = sensor crop size (64×64 @ 2.2m, 14×14 @ 10m, etc.)
  - Stride = window_size // stride_divisor (default: 50% overlap)
  - Overlapping regions: accumulate raw logits, argmax at the end

Usage:
    # Native eval: UNet trained on HySpex, tested on HySpex
    python eval_baseline_mdas.py \\
        --ckpt_path ./checkpoints/baselines/unet_hyspex/best.ckpt \\
        --model unet --train_sensor hyspex --test_sensor hyspex \\
        --sub_area 3 --output_dir ./results/unet_hyspex_to_hyspex

    # Cross-sensor eval: UNet trained on HySpex, tested on S2
    python eval_baseline_mdas.py \\
        --ckpt_path ./checkpoints/baselines/unet_hyspex/best.ckpt \\
        --model unet --train_sensor hyspex --test_sensor sentinel2 \\
        --sub_area 3 --output_dir ./results/unet_hyspex_to_s2
"""

import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    raise ImportError("rasterio required: pip install rasterio")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")


from training.trainer_baselines import BaselineTrainer

from training.utils.datasets_baselines.utils_dataset_MDAS import MDASBaselineDataset
from training.utils.datasets_baselines.collate import spectral_interpolate


from training.trainer_baselines import BaselineTrainer
from training.unet.model_unet import UNet
from training.VIT.vit_upernet import ViTUPerNet


# =============================================================================
# CONSTANTS
# =============================================================================
 
NUM_CLASSES = 6
IGNORE_INDEX = 255
GSD_REF = 2.2
CROP_SIZE_REF = 64
 
CLASS_NAMES = ["Pavement", "Soil", "Roof", "Low vegetation", "Tree", "Water"]
 
CLASS_COLORS = [
    [128, 128, 128],  # Pavement — grey
    [139,  90,  43],  # Soil — brown
    [255,   0,   0],  # Roof — red
    [144, 238, 144],  # Low vegetation — light green
    [  0, 100,   0],  # Tree — dark green
    [  0,   0, 255],  # Water — blue
]
 
SENSOR_FILES = {
    "hyspex":    "sub_area_{n}/HySpex_sub_area{n}.tif",
    "enmap_10m": "sub_area_{n}/EeteS_EnMAP_10m_sub_area{n}.tif",
    "enmap_30m": "sub_area_{n}/EeteS_EnMAP_30m_sub_area{n}.tif",
    "sentinel2": "sub_area_{n}/Sentinel_2_sub_area{n}.tif",
}
 
LABEL_FILES = {
    2.2:  "GT_labels/2_sub_area{n}.tif",
    10.0: "GT_labels/label_6class_10m_sub_area{n}.tif",
}
 
SENSOR_LABEL_RES = {
    "hyspex":    2.2,
    "enmap_10m": 10.0,
    "enmap_30m": 10.0,
    "sentinel2": 10.0,
}
 
SENSOR_CHANNELS = {
    "hyspex": 368,
    "enmap_10m": 242,
    "enmap_30m": 242,
    "sentinel2": 12,
}
 
 
# =============================================================================
# MODEL FACTORY
# =============================================================================
 
def build_model(model_name: str, in_channels: int, num_classes: int = 6) -> nn.Module:
    if model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes, base_dim=64)
    elif model_name == "vit_upernet":
        return ViTUPerNet(
            in_channels=in_channels, num_classes=num_classes,
            img_size=64, patch_size=4, embed_dim=384, depth=12, num_heads=6,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: unet, vit_upernet")
 
 
# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================
 
class SlidingWindowInference:
    """
    Runs baseline model inference over a full image using a sliding window.
 
    Supports both native and cross-sensor evaluation.
    For cross-sensor: spectral interpolation + spatial resize are applied
    per window to match the training sensor's input format.
    """
 
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_sensor: str,
        train_sensor: str,
        mdas_root: str,
        stats: dict,
        spectral_meta: dict,
        stride_divisor: int = 2,
    ):
        self.model = model
        self.device = device
        self.test_sensor = test_sensor
        self.train_sensor = train_sensor
        self.mdas_root = mdas_root
        self.stats = stats
        self.spectral_meta = spectral_meta
 
        # Test sensor properties
        self.test_gsd = spectral_meta[test_sensor]["gsd"]
        self.test_crop_size = max(1, int(CROP_SIZE_REF * GSD_REF / self.test_gsd))
 
        # Train sensor properties (for cross-sensor adaptation)
        self.train_gsd = spectral_meta[train_sensor]["gsd"]
        self.train_crop_size = max(1, int(CROP_SIZE_REF * GSD_REF / self.train_gsd))
        self.train_n_bands = spectral_meta[train_sensor]["n_bands"]
 
        self.stride = max(1, self.test_crop_size // stride_divisor)
 
        # Cross-sensor mode?
        self.cross_sensor = (test_sensor != train_sensor)
 
        # Normalization stats
        self.test_mean = torch.tensor(stats[test_sensor]["mean"], dtype=torch.float32)
        self.test_std = torch.tensor(stats[test_sensor]["std"], dtype=torch.float32)
 
        if self.cross_sensor:
            self.test_wavelengths = torch.tensor(
                spectral_meta[test_sensor]["wavelengths"], dtype=torch.float32
            )
            self.train_wavelengths = torch.tensor(
                spectral_meta[train_sensor]["wavelengths"], dtype=torch.float32
            )
 
        # Label resolution
        self.label_res = SENSOR_LABEL_RES[test_sensor]
 
        mode = "CROSS-SENSOR" if self.cross_sensor else "NATIVE"
        print(f"[Eval] Mode: {mode}")
        print(f"[Eval] Test sensor: {test_sensor} ({self.test_crop_size}×{self.test_crop_size} @ {self.test_gsd}m)")
        if self.cross_sensor:
            print(f"[Eval] Train sensor: {train_sensor} ({self.train_crop_size}×{self.train_crop_size} @ {self.train_gsd}m)")
            print(f"[Eval] Adapting: {spectral_meta[test_sensor]['n_bands']}ch → {self.train_n_bands}ch, "
                  f"{self.test_crop_size}px → {self.train_crop_size}px")
 
    def _adapt_cross_sensor(
        self, image: torch.Tensor
    ) -> torch.Tensor:
        """
        Adapt test sensor image to training sensor format.
 
        The image stays normalized with test sensor stats — no re-normalization.
        Only spectral interpolation (to match channel count) and spatial
        resize (to match crop size) are applied.
 
        Steps:
          1. Spectral interpolation to training sensor wavelength grid
          2. Spatial resize to training sensor crop size
        """
        # 1. Spectral interpolation (on already-normalized data)
        adapted = spectral_interpolate(image, self.test_wavelengths, self.train_wavelengths)
 
        # 2. Spatial resize
        if adapted.shape[1] != self.train_crop_size or adapted.shape[2] != self.train_crop_size:
            adapted = F.interpolate(
                adapted.unsqueeze(0),
                size=(self.train_crop_size, self.train_crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
 
        return adapted
 
    def run(self, sub_area: int) -> dict:
        """
        Run sliding window inference on a full sub_area.
 
        Returns dict with prediction, label, logits, counts arrays.
        """
        sensor_path = os.path.join(
            self.mdas_root, SENSOR_FILES[self.test_sensor].format(n=sub_area)
        )
        label_path = os.path.join(
            self.mdas_root, LABEL_FILES[self.label_res].format(n=sub_area)
        )
 
        with rasterio.open(sensor_path) as sensor_src, \
             rasterio.open(label_path) as label_src:
 
            sensor_h, sensor_w = sensor_src.height, sensor_src.width
            label_h, label_w = label_src.height, label_src.width
 
            print(f"[Eval] Sensor: {sensor_h}×{sensor_w}×{sensor_src.count} @ {self.test_gsd}m")
            print(f"[Eval] Labels: {label_h}×{label_w} @ {self.label_res}m")
 
            # Load full label
            full_label = label_src.read(1).astype(np.int64)
            full_label[(full_label < 0) | (full_label >= NUM_CLASSES)] = IGNORE_INDEX
 
            # Accumulation buffers
            logit_accum = np.zeros((label_h, label_w, NUM_CLASSES), dtype=np.float64)
            count_accum = np.zeros((label_h, label_w), dtype=np.int32)
 
            # Build window grid
            cs = self.test_crop_size
            stride = self.stride
 
            windows = set()
            for r0 in range(0, max(1, sensor_h - cs + 1), stride):
                for c0 in range(0, max(1, sensor_w - cs + 1), stride):
                    windows.add((r0, c0))
 
            # Edge coverage
            if sensor_h > cs:
                for c0 in range(0, max(1, sensor_w - cs + 1), stride):
                    windows.add((sensor_h - cs, c0))
            if sensor_w > cs:
                for r0 in range(0, max(1, sensor_h - cs + 1), stride):
                    windows.add((r0, sensor_w - cs))
            if sensor_h > cs and sensor_w > cs:
                windows.add((sensor_h - cs, sensor_w - cs))
 
            windows = sorted(windows)
            print(f"[Eval] {len(windows)} windows (crop={cs}×{cs}, stride={stride})")
 
            # Determine accumulation grid
            # For native eval: accumulate at label resolution (same as sensor)
            # For cross-sensor: accumulate at label resolution, upsample logits to label grid
            # Metrics are always computed at label resolution against native labels
            # No downsampling of logits — we upsample labels if needed instead
 
            # Label crop size
            if self.label_res == GSD_REF:
                label_cs = CROP_SIZE_REF
            else:
                label_cs = cs
 
            # Inference loop
            self.model.eval()
            n_windows = len(windows)
            t_start = time.perf_counter()
 
            for i, (r0_sensor, c0_sensor) in enumerate(windows):
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  Window {i+1}/{n_windows}...", flush=True)
 
                # Read sensor crop
                window_sensor = Window(c0_sensor, r0_sensor, cs, cs)
                sensor_data = sensor_src.read(window=window_sensor).astype(np.float32)
                image = torch.from_numpy(sensor_data)
 
                # Normalize with test sensor stats
                image = (image - self.test_mean[:, None, None]) / self.test_std[:, None, None]
 
                # Cross-sensor adaptation (spectral interp + spatial resize)
                if self.cross_sensor:
                    image = self._adapt_cross_sensor(image)
 
                # Forward pass
                with torch.no_grad():
                    logits = self.model(image.unsqueeze(0).to(self.device))  # [1, C, H_model, W_model]
 
                # Resize logits to label crop size for accumulation
                # This upsamples if model output < label (never happens for native)
                # or downsamples if model output > label (shouldn't happen)
                if logits.shape[2] != label_cs or logits.shape[3] != label_cs:
                    logits = F.interpolate(
                        logits, size=(label_cs, label_cs),
                        mode="bilinear", align_corners=False,
                    )
 
                logits = logits[0].cpu().numpy()  # [C, label_cs, label_cs]
                logits = logits.transpose(1, 2, 0)  # [label_cs, label_cs, C]
 
                # Map to label coordinates
                if self.label_res == GSD_REF:
                    r0_label = int(r0_sensor * self.test_gsd / GSD_REF)
                    c0_label = int(c0_sensor * self.test_gsd / GSD_REF)
                else:
                    r0_label = r0_sensor
                    c0_label = c0_sensor
 
                r0_label = min(r0_label, max(0, label_h - label_cs))
                c0_label = min(c0_label, max(0, label_w - label_cs))
 
                r_end = min(r0_label + label_cs, label_h)
                c_end = min(c0_label + label_cs, label_w)
                h_actual = r_end - r0_label
                w_actual = c_end - c0_label
 
                logit_accum[r0_label:r_end, c0_label:c_end] += logits[:h_actual, :w_actual]
                count_accum[r0_label:r_end, c0_label:c_end] += 1
 
            t_elapsed = time.perf_counter() - t_start
            print(f"[Eval] Inference done in {t_elapsed:.1f}s "
                  f"({t_elapsed/n_windows*1000:.1f} ms/window)")
 
            # Finalize predictions
            covered = count_accum > 0
            prediction = np.full((label_h, label_w), IGNORE_INDEX, dtype=np.int64)
            prediction[covered] = logit_accum[covered].argmax(axis=-1)
 
            total_pixels = label_h * label_w
            covered_pixels = covered.sum()
            print(f"[Eval] Coverage: {covered_pixels}/{total_pixels} "
                  f"({covered_pixels/total_pixels*100:.1f}%)")
 
        return {
            "prediction": prediction,
            "label": full_label,
            "logits": logit_accum,
            "counts": count_accum,
            "sensor_shape": (sensor_h, sensor_w),
            "label_shape": (label_h, label_w),
        }
 
 
# =============================================================================
# METRICS
# =============================================================================
 
def compute_metrics(prediction: np.ndarray, label: np.ndarray) -> dict:
    valid = label != IGNORE_INDEX
    pred_valid = prediction[valid]
    label_valid = label[valid]
 
    overall_acc = (pred_valid == label_valid).sum() / max(len(label_valid), 1)
 
    per_class = {}
    ious = []
 
    for cls_id in range(NUM_CLASSES):
        pred_cls = pred_valid == cls_id
        label_cls = label_valid == cls_id
 
        intersection = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
 
        iou = float(intersection / union) if union > 0 else float("nan")
 
        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "iou": iou,
            "precision": float(intersection / max(pred_cls.sum(), 1)),
            "recall": float(intersection / max(label_cls.sum(), 1)),
            "support": int(label_cls.sum()),
        }
 
        if not np.isnan(iou):
            ious.append(iou)
 
    miou = float(np.mean(ious)) if ious else 0.0
 
    return {
        "mIoU": miou,
        "overall_accuracy": float(overall_acc),
        "per_class": per_class,
        "n_valid_pixels": int(valid.sum()),
    }
 
 
def compute_confusion_matrix(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    valid = label != IGNORE_INDEX
    pred_valid = prediction[valid]
    label_valid = label[valid]
 
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            cm[true_cls, pred_cls] = (
                (label_valid == true_cls) & (pred_valid == pred_cls)
            ).sum()
 
    return cm
 
 
# =============================================================================
# VISUALIZATION
# =============================================================================
 
def label_to_rgb(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        mask = label_map == cls_id
        rgb[mask] = CLASS_COLORS[cls_id]
    return rgb
 
 
def plot_prediction_vs_gt(prediction, label, output_path, title="", metrics=None):
    pred_rgb = label_to_rgb(prediction)
    gt_rgb = label_to_rgb(label)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")
 
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    patches.append(mpatches.Patch(color="white", edgecolor="black", label="Nodata"))
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=10)
 
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if metrics:
        info = f"mIoU: {metrics['mIoU']:.4f}  |  OA: {metrics['overall_accuracy']:.4f}"
        fig.text(0.5, 0.02, info, ha="center", fontsize=12)
 
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")
 
 
def plot_confusion_matrix(cm, output_path, title="Confusion Matrix"):
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
 
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
 
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
 
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")
 
 
def plot_per_class_iou(metrics, output_path, title="Per-Class IoU"):
    names = []
    ious = []
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        names.append(info["name"])
        ious.append(info["iou"] if not np.isnan(info["iou"]) else 0.0)
 
    colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(NUM_CLASSES)]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ious, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title(f"{title}  (mIoU = {metrics['mIoU']:.4f})", fontsize=14)
    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f}")
    ax.legend(fontsize=10)
 
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{iou:.3f}", ha="center", fontsize=9)
 
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(description="MDAS Baseline Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "vit_upernet"])
    parser.add_argument("--train_sensor", type=str, required=True,
                        choices=["hyspex", "sentinel2", "enmap_10m", "enmap_30m"],
                        help="Sensor the model was trained on")
    parser.add_argument("--test_sensor", type=str, required=True,
                        choices=["hyspex", "sentinel2", "enmap_10m", "enmap_30m"],
                        help="Sensor to evaluate on")
    parser.add_argument("--sub_area", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results/baselines")
 
    # Paths
    parser.add_argument("--mdas_root", type=str,
                        default="./data/MDAS/Augsburg_data_4_publication")
    parser.add_argument("--stats_json", type=str, default=None)
    parser.add_argument("--spectral_meta", type=str, default=None)
 
    # Inference
    parser.add_argument("--stride_divisor", type=int, default=2)
 
    args = parser.parse_args()
 
    if args.stats_json is None:
        args.stats_json = os.path.join(args.mdas_root, "mdas_norm_stats.json")
    if args.spectral_meta is None:
        args.spectral_meta = os.path.join(args.mdas_root, "mdas_spectral_meta.json")
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    # ── Load metadata ───────────────────────────────────────────────
    with open(args.stats_json, "r") as f:
        stats = json.load(f)
    with open(args.spectral_meta, "r") as f:
        spectral_meta = json.load(f)
 
    # ── Build model ─────────────────────────────────────────────────
    train_channels = SENSOR_CHANNELS[args.train_sensor]
    print(f"\n{'='*60}")
    print(f"  MDAS Baseline Evaluation")
    print(f"  Model: {args.model} ({train_channels}ch)")
    print(f"  Train sensor: {args.train_sensor}")
    print(f"  Test sensor:  {args.test_sensor}")
    print(f"  Sub area:     {args.sub_area}")
    print(f"{'='*60}\n")
 
    model = build_model(args.model, in_channels=train_channels, num_classes=NUM_CLASSES)
 
    # ── Load checkpoint ─────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
 
    # Handle PL checkpoint format
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # Strip "model." prefix from BaselineTrainer wrapping
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        model.load_state_dict(cleaned, strict=True)
        print(f"[Eval] Loaded state_dict from PL checkpoint")
    else:
        model.load_state_dict(ckpt, strict=True)
        print(f"[Eval] Loaded state_dict from raw checkpoint")
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
 
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Eval] Parameters: {param_count:,}")
 
    # ── Run inference ───────────────────────────────────────────────
    engine = SlidingWindowInference(
        model=model,
        device=device,
        test_sensor=args.test_sensor,
        train_sensor=args.train_sensor,
        mdas_root=args.mdas_root,
        stats=stats,
        spectral_meta=spectral_meta,
        stride_divisor=args.stride_divisor,
    )
 
    result = engine.run(sub_area=args.sub_area)
 
    # ── Compute metrics ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Metrics")
    print(f"{'='*60}")
 
    metrics = compute_metrics(result["prediction"], result["label"])
 
    print(f"\n  mIoU:             {metrics['mIoU']:.4f}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Valid Pixels:     {metrics['n_valid_pixels']:,}")
    print()
    print(f"  {'Class':<20s} {'IoU':>8s} {'Precision':>10s} {'Recall':>8s} {'Support':>10s}")
    print(f"  {'-'*56}")
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        iou_str = f"{info['iou']:.4f}" if not np.isnan(info["iou"]) else "  N/A"
        print(f"  {info['name']:<20s} {iou_str:>8s} "
              f"{info['precision']:>10.4f} {info['recall']:>8.4f} "
              f"{info['support']:>10,d}")
 
    # ── Save metrics JSON ───────────────────────────────────────────
    metrics["config"] = {
        "model": args.model,
        "train_sensor": args.train_sensor,
        "test_sensor": args.test_sensor,
        "sub_area": args.sub_area,
        "ckpt_path": args.ckpt_path,
        "params": param_count,
    }
 
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → {metrics_path}")
 
    # ── Visualizations ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating visualizations")
    print(f"{'='*60}")
 
    title = (f"{args.model.upper()} — {args.train_sensor} → {args.test_sensor} "
             f"(SA{args.sub_area})")
 
    plot_prediction_vs_gt(
        result["prediction"], result["label"],
        os.path.join(args.output_dir, "prediction_vs_gt.png"),
        title=title, metrics=metrics,
    )
 
    plot_per_class_iou(
        metrics,
        os.path.join(args.output_dir, "per_class_iou.png"),
        title=title,
    )
 
    cm = compute_confusion_matrix(result["prediction"], result["label"])
    plot_confusion_matrix(
        cm,
        os.path.join(args.output_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — {args.model.upper()} {args.train_sensor}→{args.test_sensor}",
    )
 
    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()
 