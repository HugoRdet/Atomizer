#!/usr/bin/env python3
"""
C2Seg Evaluation — Sliding Window Inference + Visualization
=============================================================

Loads a trained checkpoint, runs sliding window inference over the
full test image, and produces:

  1. Prediction vs GT side-by-side (PNG)
  2. Per-class IoU bar chart (PNG)
  3. Confusion matrix (PNG)
  4. Full prediction map (PNG)
  5. Metrics JSON (mIoU, OA, per-class IoU, profiling)

mIoU is computed ONLY over classes present in the test ground truth.
Classes with 0 support are excluded from the mean but reported as N/A.

Usage:
    # Germany: eval on Berlin with HSI
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi \
        --output_dir ./results/c2seg/germany_hsi

    # SANITY CHECK: eval on Augsburg (train city) to verify pipeline
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi --eval_on_train \
        --output_dir ./results/c2seg/germany_hsi_train_sanity

    # Germany: eval on Berlin with SAR (cross-sensor)
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor sar \
        --output_dir ./results/c2seg/germany_sar

    # Fusion eval: all sensors at once
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi msi sar --fusion \
        --output_dir ./results/c2seg/germany_fusion

    # Cross-continent: Germany checkpoint on China
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset china --sensor hsi \
        --output_dir ./results/c2seg/cross_de_cn_hsi
"""

import os
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")

from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_C2SEG import (
    C2SegDataset, create_c2seg_bands_info, register_c2seg_bands,
    NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, CHINA_LABEL_REMAP,
    SENSOR_META_KEY, SENSOR_GSD, AXIS_ORDER, MAT_KEYS,
)
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# SUBSET CONFIG (duplicated from train script for standalone use)
# =============================================================================

SUBSET_CONFIG = {
    "germany": {
        "train_city": "augsburg",
        "test_city": "berlin",
        "train_mat": "augsburg_multimodal.mat",
        "test_mat": "berlin_multimodal.mat",
    },
    "china": {
        "train_city": "beijing",
        "test_city": "wuhan",
        "train_mat": "beijing.mat",
        "test_mat": "wuhan.mat",
    },
}


# =============================================================================
# COLORS — 14 classes
# =============================================================================

CLASS_COLORS = [
    [255, 255, 255],  # 0  Background — white
    [255,   0,   0],  # 1  Urban Fabric — red
    [204,   0, 230],  # 2  Industrial/Commercial — purple
    [  0,   0,   0],  # 3  Street Network — black
    [166,  77,   0],  # 4  Mine/Dump/Construction — brown
    [255, 170, 255],  # 5  Artificially Vegetated — pink
    [255, 255,   0],  # 6  Arable Land — yellow
    [255, 170,   0],  # 7  Permanent Crops — orange
    [190, 255,   0],  # 8  Pastures — yellow-green
    [  0, 120,   0],  # 9  Forests — dark green
    [170, 210,  90],  # 10 Shrub — olive
    [210, 200, 160],  # 11 Open Spaces — beige
    [  0, 200, 200],  # 12 Inland Wetlands — teal
    [  0,   0, 255],  # 13 Surface Water — blue
]


# =============================================================================
# RESOLUTION REGISTRATION
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048, 4.78: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048, 60.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# CHECKPOINT COMPATIBILITY
# =============================================================================

def align_model_to_checkpoint(model, ckpt_state):
    """Resize model buffers/parameters to match checkpoint sizes."""
    model_state = model.state_dict()
    resized = []

    for key, ckpt_tensor in ckpt_state.items():
        if key not in model_state:
            continue

        model_tensor = model_state[key]
        if ckpt_tensor.shape == model_tensor.shape:
            continue

        if ckpt_tensor.dim() == 1 and model_tensor.dim() == 1:
            new_size = ckpt_tensor.shape[0]
            new_tensor = torch.zeros(new_size, dtype=model_tensor.dtype)
            copy_size = min(new_size, model_tensor.shape[0])
            new_tensor[:copy_size] = model_tensor[:copy_size]
            _set_nested_attr(model, key, new_tensor)
            resized.append(f"  {key}: {model_tensor.shape[0]} → {new_size}")

        elif ckpt_tensor.dim() == 2 and model_tensor.dim() == 2:
            if ckpt_tensor.shape[1] == model_tensor.shape[1]:
                new_rows = ckpt_tensor.shape[0]
                new_tensor = torch.zeros(new_rows, ckpt_tensor.shape[1],
                                         dtype=model_tensor.dtype)
                copy_rows = min(new_rows, model_tensor.shape[0])
                new_tensor[:copy_rows] = model_tensor[:copy_rows]
                _set_nested_attr(model, key, new_tensor)
                resized.append(f"  {key}: {model_tensor.shape} → {ckpt_tensor.shape}")

    if resized:
        print(f"[Eval] Resized {len(resized)} buffers:")
        for msg in resized:
            print(msg)


def _set_nested_attr(model, key, tensor):
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    attr_name = parts[-1]
    if attr_name in dict(obj.named_buffers(recurse=False)):
        obj.register_buffer(attr_name, tensor)
    elif attr_name in dict(obj.named_parameters(recurse=False)):
        obj.register_parameter(attr_name, torch.nn.Parameter(tensor))
    else:
        setattr(obj, attr_name, tensor)


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

class SlidingWindowInference:
    """
    Runs Atomizer inference over a full .mat image using a sliding window.
    Accumulates logits across overlapping windows, then takes argmax.

    Supports single-sensor and multi-sensor (fusion) evaluation.
    """

    def __init__(
        self,
        model: Model_Pretrain,
        dataset: C2SegDataset,
        device: torch.device,
        stride_divisor: int = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.stride_divisor = stride_divisor

    def run(self) -> dict:
        """
        Run sliding window inference on the full test image.

        Returns dict with prediction, label, logits, counts.
        """
        ds = self.dataset

        # Determine label dimensions
        label_h, label_w = ds.label_dims
        print(f"[Eval] Label grid: {label_h}×{label_w} at 10m")

        # Read full label
        ds.reader._open()
        full_label = ds.reader.read_label_crop(0, 0, label_h, label_w)
        full_label = full_label.astype(np.int64)

        # Remap China labels
        if ds.needs_label_remap:
            remapped = np.full_like(full_label, IGNORE_INDEX)
            for raw_val, new_val in CHINA_LABEL_REMAP.items():
                remapped[full_label == raw_val] = new_val
            full_label = remapped

        # Get crop size from first crop in the index
        if len(ds.crops) == 0:
            raise ValueError("No crops in dataset")

        crop_h = ds.crops[0]["crop_h"]
        crop_w = ds.crops[0]["crop_w"]
        stride_h = max(1, crop_h // self.stride_divisor)
        stride_w = max(1, crop_w // self.stride_divisor)

        print(f"[Eval] Crop: {crop_h}×{crop_w}, stride: {stride_h}×{stride_w}")

        # Allocate accumulators
        logit_accum = np.zeros((label_h, label_w, NUM_CLASSES), dtype=np.float64)
        count_accum = np.zeros((label_h, label_w), dtype=np.int32)

        # Build window grid
        windows = []
        for r0 in range(0, label_h - crop_h + 1, stride_h):
            for c0 in range(0, label_w - crop_w + 1, stride_w):
                windows.append((r0, c0))

        # Edge windows
        edge_set = set(windows)
        for r0 in range(0, label_h - crop_h + 1, stride_h):
            w = (r0, label_w - crop_w)
            if w not in edge_set:
                windows.append(w)
                edge_set.add(w)
        for c0 in range(0, label_w - crop_w + 1, stride_w):
            w = (label_h - crop_h, c0)
            if w not in edge_set:
                windows.append(w)
                edge_set.add(w)
        corner = (label_h - crop_h, label_w - crop_w)
        if corner not in edge_set:
            windows.append(corner)

        windows = sorted(set(windows))
        n_windows = len(windows)
        print(f"[Eval] {n_windows} windows")

        # Inference loop
        self.model.eval()
        t_start = time.perf_counter()

        for i, (r0, c0) in enumerate(windows):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (n_windows - i - 1)
                print(f"  Window {i+1}/{n_windows} "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

            # Build a crop dict matching the dataset format
            crop = {
                "row_10m": r0,
                "col_10m": c0,
                "crop_h": crop_h,
                "crop_w": crop_w,
                "hsi_row": r0 // 3,
                "hsi_col": c0 // 3,
                "hsi_crop_h": int(np.ceil(crop_h / 3)),
                "hsi_crop_w": int(np.ceil(crop_w / 3)),
            }

            # Use dataset's __getitem__ logic but without augmentation
            try:
                label = ds._read_label_crop(crop)
            except Exception as e:
                print(f"  Window ({r0},{c0}) label error: {e}")
                continue

            groups = {}
            for sensor in ds.sensors:
                try:
                    image = ds._read_sensor_crop(sensor, crop)
                except Exception as e:
                    print(f"  Window ({r0},{c0}) {sensor} error: {e}")
                    continue

                info = ds.sensor_info[sensor]
                gsd = info["gsd"]
                spectral_indices = info["spectral_indices"]
                res_idx = info["resolution_idx"]

                C, H, W = image.shape

                if gsd > 10.0:
                    token_label = torch.full((H, W), IGNORE_INDEX, dtype=torch.int64)
                else:
                    token_label = label

                tokens = ds.token_builder.build_tokens(
                    image=image,
                    label=token_label,
                    resolution=gsd,
                    spectral_indices=spectral_indices,
                    resolution_idx=res_idx,
                    time_idx=-1,
                )
                token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

                if gsd in groups:
                    groups[gsd]["tokens"] = torch.cat(
                        [groups[gsd]["tokens"], tokens], dim=0)
                    groups[gsd]["mask"] = torch.cat(
                        [groups[gsd]["mask"], token_mask], dim=0)
                    old_c = groups[gsd]["shape"][0]
                    groups[gsd]["shape"] = (old_c + C, H, W)
                else:
                    groups[gsd] = {
                        "tokens": tokens,
                        "mask": token_mask,
                        "shape": (C, H, W),
                    }

            if not groups:
                continue

            # Build queries at 10m
            query_sensor = None
            for s in ds.sensors:
                if ds.sensor_info[s]["gsd"] <= 10.0:
                    query_sensor = s
                    break
            if query_sensor is None:
                query_sensor = ds.sensors[0]

            query_info = ds.sensor_info[query_sensor]
            queries = ds.token_builder.build_queries(
                label=label,
                resolution=10.0,
                first_spectral_idx=query_info["spectral_indices"][0].item(),
                resolution_idx=ds.look_up.get_resolution_idx(10.0),
                time_idx=-1,
            )
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

            sample = {
                "groups": groups,
                "tasks": {
                    "c2seg_segmentation": {
                        "queries": queries,
                        "queries_mask": queries_mask,
                    }
                },
                "target_resolution": 10.0,
                "dataset_name": "C2Seg",
            }

            batch = collate_multitask([sample])
            batch = _batch_to_device(batch, self.device)

            with torch.no_grad():
                preds = self.model.forward_multitask(batch, training=False)

            logits = preds["c2seg_segmentation"][0].cpu().numpy()  # [M, C]

            # Scatter logits to spatial grid
            n_pixels = crop_h * crop_w
            if logits.shape[0] == n_pixels:
                logits_2d = logits.reshape(crop_h, crop_w, NUM_CLASSES)
                r_end = min(r0 + crop_h, label_h)
                c_end = min(c0 + crop_w, label_w)
                h_actual = r_end - r0
                w_actual = c_end - c0

                logit_accum[r0:r_end, c0:c_end] += logits_2d[:h_actual, :w_actual]
                count_accum[r0:r_end, c0:c_end] += 1

        # Finalize
        covered = count_accum > 0
        prediction = np.full((label_h, label_w), IGNORE_INDEX, dtype=np.int64)
        prediction[covered] = logit_accum[covered].argmax(axis=-1)

        coverage = covered.sum() / (label_h * label_w) * 100
        elapsed = time.perf_counter() - t_start
        print(f"[Eval] Done: {elapsed:.1f}s, coverage: {coverage:.1f}%")

        return {
            "prediction": prediction,
            "label": full_label,
            "logits": logit_accum,
            "counts": count_accum,
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(prediction, label, exclude_background=True):
    """
    Compute mIoU, mF1, OA, per-class IoU and F1.

    Matching the original C2Seg paper's evaluation protocol:
        mask = (y_true > 0) & (y_true <= 13)
    Background (class 0) is EXCLUDED from all metrics by default.

    Also reports metrics WITH Background for completeness (as _with_bg).
    """
    # ── Pixel validity mask ──────────────────────────────────────────
    valid = (label != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
    if exclude_background:
        # Match original paper: exclude class 0
        valid = valid & (label > 0)

    pred_valid = prediction[valid]
    label_valid = label[valid]

    overall_acc = float((pred_valid == label_valid).sum() / max(len(label_valid), 1))

    # ── Per-class metrics ────────────────────────────────────────────
    per_class = {}
    ious = []
    f1s = []

    start_cls = 1 if exclude_background else 0

    for cls_id in range(NUM_CLASSES):
        pred_cls = pred_valid == cls_id
        label_cls = label_valid == cls_id

        tp = int((pred_cls & label_cls).sum())
        fp = int((pred_cls & ~label_cls).sum())
        fn = int((~pred_cls & label_cls).sum())

        union = tp + fp + fn
        support = tp + fn  # = label_cls.sum()

        # IoU
        if support == 0 or union == 0:
            iou = float("nan")
        else:
            iou = tp / union

        # Precision, Recall, F1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": support,
            "in_test": support > 0,
        }

        # Accumulate for mean metrics (skip Background if excluded)
        if cls_id >= start_cls and not np.isnan(iou) and support > 0:
            ious.append(iou)
            f1s.append(f1)

    miou = float(np.mean(ious)) if ious else 0.0
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    n_classes_evaluated = len(ious)

    return {
        "mIoU": miou,
        "mF1": mf1,
        "overall_accuracy": overall_acc,
        "n_classes_evaluated": n_classes_evaluated,
        "n_classes_total": NUM_CLASSES,
        "exclude_background": exclude_background,
        "per_class": per_class,
        "n_valid_pixels": int(valid.sum()),
    }


def compute_confusion_matrix(prediction, label, exclude_background=True):
    """
    Confusion matrix matching original C2Seg paper protocol.
    Excludes Background (class 0) from valid pixels if exclude_background=True.
    """
    valid = (label != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
    if exclude_background:
        valid = valid & (label > 0)

    pred_valid = prediction[valid]
    label_valid = label[valid]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            cm[t, p] = ((label_valid == t) & (pred_valid == p)).sum()

    return cm


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_to_rgb(label_map):
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        mask = label_map == cls_id
        rgb[mask] = CLASS_COLORS[cls_id]
    return rgb


def get_class_names_list():
    return [CLASS_NAMES[i] for i in range(NUM_CLASSES)]


def plot_prediction_vs_gt(prediction, label, output_path, title="", metrics=None):
    pred_rgb = label_to_rgb(prediction)
    gt_rgb = label_to_rgb(label)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")

    # Legend: only classes present in GT or prediction
    present = set(np.unique(label[label != IGNORE_INDEX])) | \
              set(np.unique(prediction[prediction != IGNORE_INDEX]))
    patches = []
    for i in sorted(present):
        if i < NUM_CLASSES:
            patches.append(mpatches.Patch(
                color=np.array(CLASS_COLORS[i]) / 255.0,
                label=CLASS_NAMES[i],
            ))
    patches.append(mpatches.Patch(color="white", edgecolor="black", label="Ignore/Nodata"))
    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(patches), 7), fontsize=9, frameon=True)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if metrics:
        mf1 = metrics.get("mF1", 0.0)
        bg_note = " (no BG)" if metrics.get("exclude_background", True) else ""
        info = (f"mIoU: {metrics['mIoU']:.4f}  |  "
                f"mF1: {mf1:.4f}  |  "
                f"OA: {metrics['overall_accuracy']:.4f}  |  "
                f"{metrics['n_classes_evaluated']}/{metrics['n_classes_total']-1} classes{bg_note}")
        fig.text(0.5, 0.02, info, ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_per_class_iou(metrics, output_path, title="Per-Class IoU"):
    names = []
    ious = []
    colors = []
    hatches = []

    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        names.append(info["name"])
        iou = info["iou"] if not np.isnan(info["iou"]) else 0.0
        ious.append(iou)
        colors.append(np.array(CLASS_COLORS[cls_id]) / 255.0)
        # Hatch: classes not in test set OR Background (excluded from means)
        if not info["in_test"] or cls_id == 0:
            hatches.append("//")
        else:
            hatches.append("")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(NUM_CLASSES), ious, color=colors, edgecolor="black",
                  linewidth=0.5)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    n_eval = metrics["n_classes_evaluated"]
    exclude_bg = metrics.get("exclude_background", True)
    bg_label = ", no BG" if exclude_bg else ""
    mf1 = metrics.get("mF1", 0.0)
    ax.set_title(f"{title}  (mIoU={metrics['mIoU']:.4f}, mF1={mf1:.4f}, "
                 f"{n_eval}/{NUM_CLASSES - (1 if exclude_bg else 0)} classes{bg_label})",
                 fontsize=13)

    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f} (no BG)")
    ax.legend(fontsize=10)

    for bar, iou, cls_id in zip(bars, ious, range(NUM_CLASSES)):
        info = metrics["per_class"][cls_id]
        if cls_id == 0 and info["in_test"]:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{iou:.3f}*", ha="center", fontsize=7, color="gray")
        elif info["in_test"]:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{iou:.3f}", ha="center", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05,
                    "N/A", ha="center", fontsize=8, color="gray")

    # Footnote
    ax.text(0.01, -0.15, "* Background excluded from mIoU/mF1 (hatched = excluded from means)",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_confusion_matrix(cm, output_path, title="Confusion Matrix",
                          classes_present=None):
    """Plot confusion matrix, optionally filtering to only present classes."""
    if classes_present is not None:
        idx = sorted(classes_present)
        cm_sub = cm[np.ix_(idx, idx)]
        names_sub = [CLASS_NAMES[i] for i in idx]
    else:
        cm_sub = cm
        names_sub = get_class_names_list()

    n = cm_sub.shape[0]
    cm_norm = cm_sub.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names_sub, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names_sub, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def save_prediction_map(prediction, output_path):
    rgb = label_to_rgb(prediction)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


# =============================================================================
# DIAGNOSTIC: Token/Normalization inspector
# =============================================================================

def run_diagnostics(dataset, n_samples=5):
    """
    Print diagnostic info about the first few crops to catch normalization
    and encoding issues early, BEFORE running full inference.
    """
    ds = dataset
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS — {ds.city} ({ds.subset})")
    print(f"{'='*60}")

    # 1. Report normalization stats being used
    print(f"\n  Normalization stats loaded for:")
    for sensor in ds.sensors:
        info = ds.sensor_info[sensor]
        stats_key = f"{ds.subset}_{sensor}_{ds.city}"
        mean_vals = info.get("mean", None)
        std_vals = info.get("std", None)
        if mean_vals is not None:
            mean_arr = np.array(mean_vals)
            std_arr = np.array(std_vals)
            print(f"    {sensor} ({info['gsd']}m): "
                  f"mean range [{mean_arr.min():.4f}, {mean_arr.max():.4f}], "
                  f"std range [{std_arr.min():.4f}, {std_arr.max():.4f}]")
        else:
            print(f"    {sensor}: stats not found in sensor_info — CHECK THIS")

    # 2. Sample a few crops and check value ranges after normalization
    print(f"\n  Sampling {n_samples} crops to check post-normalization ranges:")
    for i in range(min(n_samples, len(ds.crops))):
        crop = ds.crops[i]
        for sensor in ds.sensors:
            try:
                image = ds._read_sensor_crop(sensor, crop)  # [C, H, W]
                print(f"    Crop {i}, {sensor}: "
                      f"shape={image.shape}, "
                      f"dtype={image.dtype}, "
                      f"range=[{image.min():.4f}, {image.max():.4f}], "
                      f"mean={image.mean():.4f}, std={image.std():.4f}, "
                      f"nan={np.isnan(image).sum()}, inf={np.isinf(image).sum()}")
            except Exception as e:
                print(f"    Crop {i}, {sensor}: ERROR — {e}")

    # 3. Check a single token build
    print(f"\n  Token construction check (crop 0):")
    crop = ds.crops[0]
    for sensor in ds.sensors:
        try:
            image = ds._read_sensor_crop(sensor, crop)
            info = ds.sensor_info[sensor]
            label = ds._read_label_crop(crop)

            tokens = ds.token_builder.build_tokens(
                image=image,
                label=label,
                resolution=info["gsd"],
                spectral_indices=info["spectral_indices"],
                resolution_idx=info["resolution_idx"],
                time_idx=-1,
            )
            # Token format: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
            values = tokens[:, 0]  # reflectance/value column
            x_coords = tokens[:, 1]
            y_coords = tokens[:, 2]

            print(f"    {sensor}: {tokens.shape[0]} tokens, dim={tokens.shape[1]}")
            print(f"      values:  range=[{values.min():.4f}, {values.max():.4f}], "
                  f"mean={values.mean():.4f}, std={values.std():.4f}")
            print(f"      x_coord: range=[{x_coords.min():.4f}, {x_coords.max():.4f}]")
            print(f"      y_coord: range=[{y_coords.min():.4f}, {y_coords.max():.4f}]")

            # Check for degenerate cases
            n_unique_values = len(torch.unique(values))
            if n_unique_values < 10:
                print(f"      ⚠️  WARNING: only {n_unique_values} unique token values — "
                      f"possible normalization collapse!")
        except Exception as e:
            print(f"    {sensor}: ERROR — {e}")

    # 4. Check label distribution
    print(f"\n  Label distribution (full image):")
    ds.reader._open()
    label_h, label_w = ds.label_dims
    full_label = ds.reader.read_label_crop(0, 0, label_h, label_w).astype(np.int64)
    if ds.needs_label_remap:
        remapped = np.full_like(full_label, IGNORE_INDEX)
        for raw_val, new_val in CHINA_LABEL_REMAP.items():
            remapped[full_label == raw_val] = new_val
        full_label = remapped

    unique, counts = np.unique(full_label, return_counts=True)
    total_valid = sum(c for u, c in zip(unique, counts) if u != IGNORE_INDEX)
    for u, c in zip(unique, counts):
        if u == IGNORE_INDEX:
            name = "IGNORE"
        elif u < NUM_CLASSES:
            name = CLASS_NAMES[u]
        else:
            name = f"UNKNOWN({u})"
        pct = c / total_valid * 100 if u != IGNORE_INDEX else 0
        print(f"    {u:>3d} {name:<30s} {c:>10,d} ({pct:>5.1f}%)")

    print(f"{'='*60}\n")


# =============================================================================
# HELPERS
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="C2Seg Evaluation")
    parser.add_argument("--ckpt_path",     type=str, required=True)
    parser.add_argument("--config_model",  type=str,
                        default="config_test-Atomiser_Atos_One.yaml")
    parser.add_argument("--subset",        type=str, required=True,
                        choices=["germany", "china"])
    parser.add_argument("--sensor",        type=str, nargs="+", required=True,
                        help="One or more sensors (e.g., --sensor hsi or --sensor hsi msi sar)")
    parser.add_argument("--fusion",        action="store_true",
                        help="Load all specified sensors per window (fusion mode)")
    parser.add_argument("--output_dir",    type=str, default="./results/c2seg")

    # ── NEW: evaluate on training city ──────────────────────────────
    parser.add_argument("--eval_on_train", action="store_true",
                        help="Evaluate on the TRAINING city (sanity check). "
                             "Germany: Augsburg, China: Beijing.")

    # Paths
    parser.add_argument("--data_dir",      type=str, default=None)
    parser.add_argument("--processed_dir", type=str,
                        default="./data/CrossCity/c2seg_processed")

    # Inference
    parser.add_argument("--stride_divisor", type=int, default=2)

    # Diagnostics
    parser.add_argument("--diagnostics_only", action="store_true",
                        help="Run diagnostics (normalization, tokens, labels) then exit. "
                             "No model loading, no inference.")
    parser.add_argument("--skip_diagnostics", action="store_true",
                        help="Skip diagnostics and go straight to inference.")

    args = parser.parse_args()

    if args.data_dir is None:
        subset_dir = "Germany" if args.subset == "germany" else "China"
        args.data_dir = f"./data/CrossCity/{subset_dir}"

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SUBSET_CONFIG[args.subset]

    # ── Select city based on --eval_on_train ────────────────────────
    if args.eval_on_train:
        eval_city = cfg["train_city"]
        eval_mat = os.path.join(args.data_dir, cfg["train_mat"])
        eval_split = "train"   # crop index filter
    else:
        eval_city = cfg["test_city"]
        eval_mat = os.path.join(args.data_dir, cfg["test_mat"])
        eval_split = "test"

    sensors = args.sensor
    if args.fusion and len(sensors) > 1:
        sensors_label = "+".join(sensors)
    else:
        sensors_label = sensors[0]
        # Single sensor: don't use fusion
        args.fusion = False

    mode_label = "TRAIN-SANITY" if args.eval_on_train else "TEST"

    print(f"\n{'='*60}")
    print(f"  C2Seg Evaluation ({mode_label})")
    print(f"  Subset:  {args.subset}")
    print(f"  City:    {eval_city}")
    print(f"  Split:   {eval_split}")
    print(f"  Sensors: {sensors_label} ({'fusion' if args.fusion else 'single'})")
    if not args.diagnostics_only:
        print(f"  Ckpt:    {args.ckpt_path}")
    print(f"{'='*60}\n")

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

    # ── Dataset config ──────────────────────────────────────────────
    spectral_meta_path = os.path.join(args.processed_dir, "c2seg_spectral_meta.json")
    crop_index_path = os.path.join(args.processed_dir, "c2seg_crop_index.csv")
    stats_path = os.path.join(args.processed_dir, "c2seg_norm_stats.json")

    dataset_config = read_yaml(bands_yaml_path)
    c2seg_bands = create_c2seg_bands_info(spectral_meta_path)
    dataset_config.update(c2seg_bands)

    register_all_resolutions(lookup_table)
    register_c2seg_bands(lookup_table, dataset_config)

    # ── Build dataset ───────────────────────────────────────────────
    if args.fusion:
        eval_sensors = sensors
    else:
        eval_sensors = [sensors[0]]

    dataset = C2SegDataset(
        mat_path=eval_mat,
        subset=args.subset,
        city=eval_city,
        split=eval_split,
        sensors=eval_sensors,
        crop_index_path=crop_index_path,
        stats_path=stats_path,
        spectral_meta_path=spectral_meta_path,
        look_up=lookup_table,
        dataset_config=dataset_config,
        mode="test",       # always "test" mode = no augmentation
        augment=False,
    )

    # ── Diagnostics ─────────────────────────────────────────────────
    if not args.skip_diagnostics:
        run_diagnostics(dataset)

    if args.diagnostics_only:
        print("Diagnostics complete. Exiting (--diagnostics_only).")
        return

    # ── Load checkpoint ─────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("state_dict", ckpt)

    # ── Build model ─────────────────────────────────────────────────
    model = Model_Pretrain(
        config=config_model,
        wand=False,
        name="eval",
        transform=None,
        lookup_table=lookup_table,
    )

    align_model_to_checkpoint(model, ckpt_state)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ── Run inference ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running inference: {sensors_label} on {eval_city} ({mode_label})")
    print(f"{'='*60}\n")

    engine = SlidingWindowInference(
        model=model,
        dataset=dataset,
        device=device,
        stride_divisor=args.stride_divisor,
    )

    result = engine.run()

    # ── Compute metrics ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Metrics ({mode_label})")
    print(f"  Protocol: Background (class 0) excluded — matching original C2Seg paper")
    print(f"{'='*60}\n")

    # Primary metrics: exclude Background (matching original paper)
    metrics = compute_metrics(result["prediction"], result["label"],
                              exclude_background=True)

    # Secondary metrics: include Background (for reference)
    metrics_with_bg = compute_metrics(result["prediction"], result["label"],
                                      exclude_background=False)

    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  mIoU (no BG):    {metrics['mIoU']:>7.4f}  "
          f"({metrics['n_classes_evaluated']}/{metrics['n_classes_total']-1} classes) │")
    print(f"  │  mF1  (no BG):    {metrics['mF1']:>7.4f}                      │")
    print(f"  │  OA   (no BG):    {metrics['overall_accuracy']:>7.4f}                      │")
    print(f"  │                                             │")
    print(f"  │  mIoU (with BG):  {metrics_with_bg['mIoU']:>7.4f}  (for reference)   │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Valid Pixels: {metrics['n_valid_pixels']:,}")

    print()
    print(f"  {'Class':<30s} {'IoU':>8s} {'F1':>8s} {'Prec':>8s} {'Recall':>8s} {'Support':>10s}")
    print(f"  {'-'*74}")
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        if cls_id == 0:
            # Show Background for reference but mark as excluded
            iou_str = f"{info['iou']:.4f}" if info["in_test"] else "   N/A"
            f1_str = f"{info['f1']:.4f}" if info["in_test"] else "   N/A"
            print(f"  {info['name']:<30s} {iou_str:>8s} {f1_str:>8s} "
                  f"{info['precision']:>8.4f} {info['recall']:>8.4f} "
                  f"{info['support']:>10,d}  ← excluded from means")
        elif info["in_test"]:
            print(f"  {info['name']:<30s} {info['iou']:>8.4f} {info['f1']:>8.4f} "
                  f"{info['precision']:>8.4f} {info['recall']:>8.4f} "
                  f"{info['support']:>10,d}")
        else:
            print(f"  {info['name']:<30s} {'N/A':>8s} {'N/A':>8s} "
                  f"{'N/A':>8s} {'N/A':>8s} "
                  f"{info['support']:>10,d}")

    # Save metrics (both protocols)
    metrics["metrics_with_background"] = {
        "mIoU": metrics_with_bg["mIoU"],
        "mF1": metrics_with_bg["mF1"],
        "overall_accuracy": metrics_with_bg["overall_accuracy"],
        "n_classes_evaluated": metrics_with_bg["n_classes_evaluated"],
    }
    metrics["config"] = {
        "subset": args.subset,
        "city": eval_city,
        "split": eval_split,
        "eval_on_train": args.eval_on_train,
        "sensors": sensors,
        "fusion": args.fusion,
        "ckpt_path": args.ckpt_path,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → {metrics_path}")

    # ── Visualizations ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Visualizations")
    print(f"{'='*60}\n")

    title = f"C2Seg — {sensors_label} on {eval_city} ({args.subset}, {mode_label})"

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

    # Confusion matrix: only for classes present in GT
    cm = compute_confusion_matrix(result["prediction"], result["label"])
    present_classes = [
        cls_id for cls_id in range(NUM_CLASSES)
        if metrics["per_class"][cls_id]["in_test"]
    ]

    plot_confusion_matrix(
        cm,
        os.path.join(args.output_dir, "confusion_matrix_full.png"),
        title=f"Confusion Matrix (all classes) — {sensors_label} ({mode_label})",
    )

    if len(present_classes) < NUM_CLASSES:
        plot_confusion_matrix(
            cm,
            os.path.join(args.output_dir, "confusion_matrix_present.png"),
            title=f"Confusion Matrix (present classes) — {sensors_label} ({mode_label})",
            classes_present=present_classes,
        )

    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()