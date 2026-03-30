#!/usr/bin/env python3
"""
MDAS Evaluation — Sliding Window Inference + Visualization
============================================================

Standalone script that loads a trained checkpoint, runs sliding window
inference over the full test image, and produces:

  1. Full-image prediction map (PNG)
  2. Ground truth label map (PNG)
  3. RGB composite of the sensor data (PNG)
  4. Per-class IoU table
  5. Confusion matrix (PNG)
  6. Overlay: prediction vs GT side-by-side (PNG)

Sliding window strategy:
  - Window size matches training crop (64×64 @ 2.2m, 14×14 @ 10m, etc.)
  - Stride = window_size // 2 (50% overlap)
  - Overlapping regions: accumulate raw logits, argmax at the end

Usage:
    # Exp 1: in-distribution
    python eval_mdas.py \\
        --ckpt_path ./checkpoints/mdas/best.ckpt \\
        --config_model atomiser.yaml \\
        --sensor hyspex --sub_area 3 \\
        --output_dir ./results/exp1

    # Exp 2: cross-sensor transfer
    python eval_mdas.py \\
        --ckpt_path ./checkpoints/mdas/best.ckpt \\
        --config_model atomiser.yaml \\
        --sensor sentinel2 --sub_area 3 \\
        --output_dir ./results/exp2
"""

import os
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
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
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")

from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_MDAS import (
    MDASSegmentation, create_mdas_bands_info, register_mdas_bands,
    SENSOR_FILES, LABEL_FILES, SENSOR_LABEL_RES,
    NUM_CLASSES, IGNORE_INDEX, GSD_REF,
)
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# RESOLUTION REGISTRATION (shared across train/eval)
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048,    # HySpex
    4.78: 2048,   # Planet (MuRA-T)
    10.0: 2048,   # Sentinel-2 / EnMAP
    20.0: 2048,   # Sentinel-2 20m
    30.0: 2048,   # Landsat-8 (MuRA-T)
    60.0: 2048,   # Sentinel-2 60m
}


def register_all_resolutions(lookup_table):
    """
    Pre-register all known resolutions into the lookup table.

    Must be called BEFORE Model_Pretrain so that geometry buffers
    (token_centers_lookup, token_gsd_lookup) and the resolution encoder
    (gsd_values) are built at the correct size.

    This ensures checkpoints trained with MuRA-T resolutions can be
    loaded in MDAS eval (and vice versa) without size mismatches.
    """
    # Update TokenBuilder's reference sizes
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size

    # Register modalities + resolution indices
    n_new = 0
    for res, ref_size in sorted(ALL_KNOWN_RESOLUTIONS.items()):
        # Register modality (position offset + query offset)
        lookup_table.get_or_register_modality(res, ref_size)
        # Register resolution index
        lookup_table.get_resolution_idx(res)
        n_new += 1

    print(f"[Eval] Pre-registered {n_new} resolutions: "
          f"{sorted(ALL_KNOWN_RESOLUTIONS.keys())}")


def register_murat_bands(lookup_table):
    """
    Pre-register MuRA-T sensor bands into the lookup table.
    Ensures the spectral codebook is large enough for checkpoints
    trained on MuRA-T data.
    """
    from training.utils.datasets.utils_dataset_MURAT import (
        PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO,
    )

    n_new = 0
    for bands_info in [PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO]:
        for band_name, data in bands_info.items():
            bw = int(data["bandwidth"])
            wl = int(data["central_wavelength"])
            key = (bw, wl)
            if key not in lookup_table.table_wave:
                lookup_table.table_wave[key] = len(lookup_table.table_wave)
                n_new += 1

    if n_new > 0:
        print(f"[Eval] Pre-registered {n_new} MuRA-T bands "
              f"(total: {len(lookup_table.table_wave)})")
    return n_new


# =============================================================================
# CHECKPOINT COMPATIBILITY
# =============================================================================

def align_model_to_checkpoint(model, ckpt_state):
    """
    Resize model buffers/parameters to match checkpoint sizes.

    Handles mismatches in:
      - geometry.token_centers_lookup / token_gsd_lookup
      - resolution_encoder.gsd_values
      - spectral_encoder.physics_codebook
    """
    model_state = model.state_dict()
    resized = []

    for key, ckpt_tensor in ckpt_state.items():
        if key not in model_state:
            continue

        model_tensor = model_state[key]

        if ckpt_tensor.shape != model_tensor.shape:
            # Try to resize: expand or truncate along dim 0
            if ckpt_tensor.dim() == 1 and model_tensor.dim() == 1:
                ckpt_size = ckpt_tensor.shape[0]
                model_size = model_tensor.shape[0]

                if ckpt_size > model_size:
                    # Checkpoint is larger → expand model buffer
                    new_tensor = torch.zeros(ckpt_size, dtype=model_tensor.dtype)
                    new_tensor[:model_size] = model_tensor
                else:
                    # Checkpoint is smaller → truncate model buffer
                    new_tensor = model_tensor[:ckpt_size].clone()

                # Set the buffer/param on the model
                _set_nested_attr(model, key, new_tensor)
                resized.append(f"  {key}: {model_size} → {ckpt_size}")

            elif ckpt_tensor.dim() == 2 and model_tensor.dim() == 2:
                ckpt_rows, ckpt_cols = ckpt_tensor.shape
                model_rows, model_cols = model_tensor.shape

                if ckpt_cols == model_cols:
                    # Only row count differs
                    if ckpt_rows > model_rows:
                        new_tensor = torch.zeros(
                            ckpt_rows, ckpt_cols, dtype=model_tensor.dtype
                        )
                        new_tensor[:model_rows] = model_tensor
                    else:
                        new_tensor = model_tensor[:ckpt_rows].clone()

                    _set_nested_attr(model, key, new_tensor)
                    resized.append(
                        f"  {key}: [{model_rows},{model_cols}] → [{ckpt_rows},{ckpt_cols}]"
                    )

    if resized:
        print(f"[Eval] Resized {len(resized)} buffers to match checkpoint:")
        for msg in resized:
            print(msg)

    return model


def _set_nested_attr(model, key, tensor):
    """Set a nested attribute on a model from a dot-separated key."""
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)

    attr_name = parts[-1]

    # Check if it's a buffer or parameter
    if attr_name in dict(obj.named_buffers(recurse=False)):
        obj.register_buffer(attr_name, tensor)
    elif attr_name in dict(obj.named_parameters(recurse=False)):
        obj.register_parameter(attr_name, torch.nn.Parameter(tensor))
    else:
        setattr(obj, attr_name, tensor)


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = [
    "Pavement", "Soil", "Roof", "Low vegetation", "Tree", "Water",
]

# Class colors (matching common land cover palettes)
CLASS_COLORS = [
    [128, 128, 128],  # Pavement — grey
    [139,  90,  43],  # Soil — brown
    [255,   0,   0],  # Roof — red
    [144, 238, 144],  # Low vegetation — light green
    [  0, 100,   0],  # Tree — dark green
    [  0,   0, 255],  # Water — blue
]

NODATA_COLOR = [255, 255, 255]  # white


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

class SlidingWindowInference:
    """
    Runs Atomizer inference over a full image using a sliding window.

    Accumulates logits across overlapping windows, then takes argmax.
    """

    def __init__(
        self,
        model: Model_Pretrain,
        dataset: MDASSegmentation,
        device: torch.device,
        stride_divisor: int = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.sensor_gsd = dataset.sensor_gsd
        self.crop_size_ref = dataset.crop_size_ref
        self.sensor_crop_size = dataset.sensor_crop_size
        self.stride = max(1, self.sensor_crop_size // stride_divisor)

        self.label_res = dataset.label_res
        self.label_crop_size = dataset.label_crop_size

    def run(self, sub_area: int) -> dict:
        """
        Run sliding window inference on a full sub_area image.

        Returns:
            {
                "prediction": [H, W] int array (class indices),
                "label": [H, W] int array (ground truth),
                "logits": [H, W, C] float array (raw accumulated logits),
                "counts": [H, W] int array (number of overlapping windows),
                "sensor_shape": (H_sensor, W_sensor),
                "label_shape": (H_label, W_label),
            }
        """
        # ── Open sensor and label files ─────────────────────────────
        sensor_path = os.path.join(
            self.dataset.root,
            SENSOR_FILES[self.dataset.sensor].format(n=sub_area),
        )
        label_path = os.path.join(
            self.dataset.root,
            LABEL_FILES[self.label_res].format(n=sub_area),
        )

        with rasterio.open(sensor_path) as sensor_src, \
             rasterio.open(label_path) as label_src:

            sensor_h, sensor_w = sensor_src.height, sensor_src.width
            label_h, label_w = label_src.height, label_src.width

            print(f"[Eval] Sensor: {sensor_h}×{sensor_w}×{sensor_src.count} "
                  f"@ {self.sensor_gsd}m")
            print(f"[Eval] Labels: {label_h}×{label_w} @ {self.label_res}m")

            # ── Load full label ──────────────────────────────────────
            full_label = label_src.read(1).astype(np.int64)
            full_label[(full_label < 0) | (full_label >= NUM_CLASSES)] = IGNORE_INDEX

            # ── Allocate accumulation buffers ────────────────────────
            logit_accum = np.zeros(
                (label_h, label_w, NUM_CLASSES), dtype=np.float64,
            )
            count_accum = np.zeros((label_h, label_w), dtype=np.int32)

            # ── Build window grid ────────────────────────────────────
            cs = self.sensor_crop_size
            stride = self.stride

            windows = []
            for r0 in range(0, sensor_h - cs + 1, stride):
                for c0 in range(0, sensor_w - cs + 1, stride):
                    windows.append((r0, c0))

            # Handle edges
            for r0 in range(0, sensor_h - cs + 1, stride):
                if (sensor_w - cs, ) not in [(c,) for _, c in windows if _ == r0]:
                    windows.append((r0, sensor_w - cs))
            for c0 in range(0, sensor_w - cs + 1, stride):
                if (sensor_h - cs,) not in [(r,) for r, _ in windows if _ == c0]:
                    windows.append((sensor_h - cs, c0))
            windows.append((sensor_h - cs, sensor_w - cs))

            # Deduplicate
            windows = sorted(set(windows))

            print(f"[Eval] {len(windows)} windows "
                  f"(crop={cs}×{cs}, stride={stride})")

            # ── Inference loop ───────────────────────────────────────
            self.model.eval()
            n_windows = len(windows)

            for i, (r0_sensor, c0_sensor) in enumerate(windows):
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  Window {i+1}/{n_windows}...", flush=True)

                r0_ref = int(r0_sensor * self.sensor_gsd / GSD_REF)
                c0_ref = int(c0_sensor * self.sensor_gsd / GSD_REF)

                # Read sensor crop
                window_sensor = Window(c0_sensor, r0_sensor, cs, cs)
                sensor_data = sensor_src.read(window=window_sensor).astype(np.float32)
                sensor_data = torch.from_numpy(sensor_data)

                # Normalize
                sensor_data = (
                    (sensor_data - self.dataset.norm_mean[:, None, None])
                    / self.dataset.norm_std[:, None, None]
                )

                # Read label crop
                if self.label_res == GSD_REF:
                    r0_label, c0_label = r0_ref, c0_ref
                    label_cs = self.crop_size_ref
                else:
                    r0_label, c0_label = r0_sensor, c0_sensor
                    label_cs = cs

                r0_label = min(r0_label, label_h - label_cs)
                c0_label = min(c0_label, label_w - label_cs)

                window_label = Window(c0_label, r0_label, label_cs, label_cs)
                label_data = label_src.read(1, window=window_label).astype(np.int64)
                label_data = torch.from_numpy(label_data)
                label_data[(label_data < 0) | (label_data >= NUM_CLASSES)] = IGNORE_INDEX

                # Build tokens + queries
                tokens = self.dataset.token_builder.build_tokens(
                    image=sensor_data,
                    label=label_data,
                    resolution=self.sensor_gsd,
                    spectral_indices=self.dataset.spectral_indices,
                    resolution_idx=self.dataset.resolution_idx,
                    time_idx=-1,
                )

                token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

                queries = self.dataset.token_builder.build_queries(
                    label=label_data,
                    resolution=self.sensor_gsd,
                    first_spectral_idx=self.dataset.spectral_indices[0].item(),
                    resolution_idx=self.dataset.resolution_idx,
                    time_idx=-1,
                )

                queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

                # Build batch
                sample = {
                    "groups": {
                        self.sensor_gsd: {
                            "tokens": tokens,
                            "mask": token_mask,
                            "shape": (sensor_data.shape[0], cs, cs),
                        }
                    },
                    "tasks": {
                        "mdas_segmentation": {
                            "queries": queries,
                            "queries_mask": queries_mask,
                        }
                    },
                    "target_resolution": self.sensor_gsd,
                    "dataset_name": "MDAS",
                }

                batch = collate_multitask([sample])
                batch = _batch_to_device(batch, self.device)

                # Forward
                with torch.no_grad():
                    preds = self.model.forward_multitask(batch, training=False)

                logits = preds["mdas_segmentation"]  # [1, M, C]
                logits = logits[0].cpu().numpy()  # [M, C]

                # Scatter logits back to spatial grid
                h_label, w_label = label_data.shape
                n_pixels = h_label * w_label

                if logits.shape[0] == n_pixels:
                    logits_2d = logits.reshape(h_label, w_label, NUM_CLASSES)

                    r_end = min(r0_label + label_cs, label_h)
                    c_end = min(c0_label + label_cs, label_w)
                    h_actual = r_end - r0_label
                    w_actual = c_end - c0_label

                    logit_accum[
                        r0_label:r_end, c0_label:c_end
                    ] += logits_2d[:h_actual, :w_actual]
                    count_accum[r0_label:r_end, c0_label:c_end] += 1

            # ── Finalize predictions ─────────────────────────────────
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

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union

        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "iou": float(iou),
            "precision": float(intersection / max(pred_cls.sum(), 1)),
            "recall": float(intersection / max(label_cls.sum(), 1)),
            "support": int(label_cls.sum()),
        }

        if not np.isnan(iou):
            ious.append(iou)

    miou = np.mean(ious) if ious else 0.0

    return {
        "mIoU": float(miou),
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
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=10, frameon=True)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if metrics:
        info = f"mIoU: {metrics['mIoU']:.4f}  |  OA: {metrics['overall_accuracy']:.4f}"
        fig.text(0.5, 0.02, info, ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {output_path}")


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
    print(f"  → Saved {output_path}")


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
    print(f"  → Saved {output_path}")


def save_prediction_map(prediction, output_path):
    rgb = label_to_rgb(prediction)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {output_path}")


# =============================================================================
# HELPERS
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


# =============================================================================
# FLOPS / LATENCY PROFILING
# =============================================================================

def profile_inference(model, dataset, device, sub_area, n_warmup=3, n_measure=10):
    model.eval()

    if len(dataset) == 0:
        print("[Profile] No samples available, skipping profiling.")
        return {"gflops": -1, "latency_ms": -1, "n_tokens": 0, "n_queries": 0}

    sample = dataset[0]
    batch = collate_multitask([sample])
    batch = _batch_to_device(batch, device)

    first_res = next(iter(batch["groups"]))
    n_tokens = batch["groups"][first_res]["tokens"].shape[1]
    n_queries = batch["tasks"]["mdas_segmentation"]["queries"].shape[1]

    print(f"[Profile] Sample: {n_tokens:,} tokens, {n_queries:,} queries")

    print(f"[Profile] Warming up ({n_warmup} forwards)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.forward_multitask(batch, training=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print("[Profile] Measuring GFLOPs...")
    gflops = -1.0
    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU] + (
                    [torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []
                ),
                with_flops=True,
            ) as prof:
                _ = model.forward_multitask(batch, training=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        total_flops = sum(e.flops for e in prof.key_averages() if e.flops and e.flops > 0)
        gflops = total_flops / 1e9
        if gflops < 0.01:
            print(f"[Profile] Warning: profiler reported {gflops:.4f} GFLOPs")
    except Exception as e:
        print(f"[Profile] Profiler failed: {e}")

    print(f"[Profile] Measuring latency ({n_measure} forwards)...")
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(n_measure):
            _ = model.forward_multitask(batch, training=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    latency_ms = (t_end - t_start) / n_measure * 1000

    result = {
        "gflops": round(gflops, 2),
        "latency_ms": round(latency_ms, 2),
        "n_tokens": n_tokens,
        "n_queries": n_queries,
    }

    print(f"\n  {'Metric':<20s} {'Value':>12s}")
    print(f"  {'-'*32}")
    print(f"  {'GFLOPs':<20s} {gflops:>12.2f}")
    print(f"  {'Latency (ms)':<20s} {latency_ms:>12.2f}")
    print(f"  {'Tokens':<20s} {n_tokens:>12,d}")
    print(f"  {'Queries':<20s} {n_queries:>12,d}")

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MDAS Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_model", type=str, default="atomiser.yaml")
    parser.add_argument("--sensor", type=str, required=True,
                        choices=["hyspex", "enmap_10m", "enmap_30m", "sentinel2"])
    parser.add_argument("--sub_area", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results/mdas")

    # Paths
    parser.add_argument("--mdas_root", type=str,
                        default="./data/MDAS/Augsburg_data_4_publication")
    parser.add_argument("--crop_index", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--spectral_meta", type=str, default=None)

    # Inference
    parser.add_argument("--stride_divisor", type=int, default=2,
                        help="Stride = crop_size // stride_divisor")
    parser.add_argument("--crop_size_ref", type=int, default=64,
                        help="Crop size on 2.2m reference grid")

    args = parser.parse_args()

    if args.crop_index is None:
        args.crop_index = os.path.join(args.mdas_root, "mdas_crop_index.csv")
    if args.stats is None:
        args.stats = os.path.join(args.mdas_root, "mdas_norm_stats.json")
    if args.spectral_meta is None:
        args.spectral_meta = os.path.join(args.mdas_root, "mdas_spectral_meta.json")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ──────────────────────────────────────────────────────
    config_model = read_yaml("./training/configs/" + args.config_model)
    bands_yaml_path = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    # ── Lookup table ────────────────────────────────────────────────
    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )

    # ── Dataset config with MDAS bands ──────────────────────────────
    dataset_config = read_yaml(bands_yaml_path)
    mdas_bands = create_mdas_bands_info(args.spectral_meta)
    dataset_config.update(mdas_bands)

    # ── Pre-register ALL known resolutions and bands ─────────────────
    # This ensures geometry buffers and resolution encoder match any
    # checkpoint, whether trained on MDAS, MuRA-T, or both.
    register_all_resolutions(lookup_table)
    register_mdas_bands(lookup_table, dataset_config)
    register_murat_bands(lookup_table)

    # ── Peek at checkpoint ──────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("state_dict", ckpt)

    codebook_key = "encoder.input_processor.spectral_encoder.physics_codebook"
    if codebook_key in ckpt_state:
        ckpt_codebook_size = ckpt_state[codebook_key].shape[0]
        print(f"[Eval] Checkpoint codebook size: {ckpt_codebook_size}")
        print(f"[Eval] Current lookup table size: {len(lookup_table.table_wave)}")

    # ── Build model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Loading model")
    print("=" * 60)

    model = Model_Pretrain(
        config=config_model,
        wand=False,
        name="eval",
        transform=None,
        lookup_table=lookup_table,
    )

    # ── Align model buffers to checkpoint sizes ─────────────────────
    align_model_to_checkpoint(model, ckpt_state)

    # ── Load checkpoint ─────────────────────────────────────────────
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  Loaded state_dict from {args.ckpt_path}")
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ── Build dataset ───────────────────────────────────────────────
    dataset = MDASSegmentation(
        root=args.mdas_root,
        sensor=args.sensor,
        sub_areas=[args.sub_area],
        crop_index_path=args.crop_index,
        stats_path=args.stats,
        spectral_meta_path=args.spectral_meta,
        look_up=lookup_table,
        dataset_config=dataset_config,
        config_model=config_model,
        mode="test",
        augment=False,
        crop_size_ref=args.crop_size_ref,
    )

    # ── Run inference ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Inference: sensor={args.sensor}, sub_area={args.sub_area}")
    print("=" * 60)

    engine = SlidingWindowInference(
        model=model,
        dataset=dataset,
        device=device,
        stride_divisor=args.stride_divisor,
    )

    result = engine.run(sub_area=args.sub_area)

    # ── Compute metrics ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Metrics")
    print("=" * 60)

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
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → Saved {metrics_path}")

    # ── GFLOPs / Latency profiling ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Profiling inference cost")
    print("=" * 60)

    profile_result = profile_inference(
        model=model, dataset=dataset, device=device, sub_area=args.sub_area,
    )

    metrics["profiling"] = profile_result
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → Updated {metrics_path} with profiling data")

    # ── Visualizations ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating visualizations")
    print("=" * 60)

    title = f"MDAS — {args.sensor} → sub_area_{args.sub_area}"

    plot_prediction_vs_gt(
        result["prediction"], result["label"],
        os.path.join(args.output_dir, "prediction_vs_gt.png"),
        title=title, metrics=metrics,
    )

    save_prediction_map(
        result["prediction"],
        os.path.join(args.output_dir, "prediction_map.png"),
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
        title=f"Confusion Matrix — {args.sensor}",
    )

    print("\n" + "=" * 60)
    print(f"  Done. Results in {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()