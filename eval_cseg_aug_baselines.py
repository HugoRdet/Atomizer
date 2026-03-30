#!/usr/bin/env python3
"""
C2Seg Baseline Evaluation — Sliding Window Inference + Visualization
=====================================================================

Loads a trained baseline checkpoint (UNet/ViT), runs sliding window
inference over the full image, and produces the same outputs as the
Atomizer eval script for direct comparison.

Supports:
  - Same-sensor eval (trained on HSI, eval on HSI)
  - Cross-sensor eval via spectral interpolation (trained on HSI, eval on MSI)

Usage:
    # Same sensor: UNet trained on HSI, eval on HSI test split
    python eval_c2seg_baseline.py \
        --ckpt_path ./checkpoints/c2seg_baselines/germany/bl_unet_hsi_best.pth \
        --model unet --train_sensor hsi --eval_sensor hsi \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg_bl/unet_hsi_test

    # Cross-sensor: UNet trained on HSI, eval on MSI (interpolated)
    python eval_c2seg_baseline.py \
        --ckpt_path ./checkpoints/c2seg_baselines/germany/bl_unet_hsi_best.pth \
        --model unet --train_sensor hsi --eval_sensor msi \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg_bl/unet_hsi_eval_msi

    # Dedicated: UNet trained on MSI, eval on MSI
    python eval_c2seg_baseline.py \
        --ckpt_path ./checkpoints/c2seg_baselines/germany/bl_unet_msi_best.pth \
        --model unet --train_sensor msi --eval_sensor msi \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg_bl/unet_msi_test
"""

import os
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")

try:
    import scipy.io as sio
except ImportError:
    sio = None

from training.unet.model_unet import UNet
from training.threeDunet.unet3d import UNet3D
from training.VIT.vit_upernet import ViTUPerNet
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.senpa_seg.senpa import SenPaSeg
from training.utils.datasets_baselines.utils_dataset_Cseg import (
    NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, CHINA_LABEL_REMAP,
    SENSOR_META_KEY, SENSOR_GSD, AXIS_ORDER, MAT_KEYS,
    MatFileReader,
)
from training.utils.datasets_baselines.collate import spectral_interpolate



# =============================================================================
# CONFIG
# =============================================================================

CITY_MAT = {
    "augsburg": "augsburg_multimodal.mat",
    "berlin": "berlin_multimodal.mat",
    "beijing": "beijing.mat",
    "wuhan": "wuhan.mat",
}

CITY_SUBSET = {
    "augsburg": "germany", "berlin": "germany",
    "beijing": "china", "wuhan": "china",
}

# =============================================================================
# COLORS
# =============================================================================

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

CLASS_COLORS_HEX = {
    0:  "#1A1A1A", 1:  "#1E90FF", 2:  "#4D4D4D", 3:  "#E60000",
    4:  "#A020F0", 5:  "#CC6600", 6:  "#FF99CC", 7:  "#FFD700",
    8:  "#D2B48C", 9:  "#BFFF00", 10: "#006400", 11: "#8DB360",
    12: "#F5DEB3", 13: "#00CED1",
}

CLASS_COLORS = [hex_to_rgb(CLASS_COLORS_HEX[i]) for i in range(14)]


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, img_size=128,
                wavelengths=None, bandwidths=None, gsd=10.0):
    if model_name == "unet":
        return UNet(in_channels, num_classes)
    elif model_name == "unet3d":
        return UNet3D(in_channels, num_classes, base_features=32)
    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=4,
        )
    elif model_name == "perceiver":
        return PerceiverSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            num_latents=256,
            latent_dim=256,
            depth=6,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            self_per_cross_attn=1,
            weight_tie_layers=True,
            num_freq_bands=6,
            max_freq=10.0,
            attn_dropout=0.0,
            ff_dropout=0.0,
        )
    elif model_name == "senpa":
        return SenPaSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=16 if in_channels > 50 else 8,
            emb_dim=256,
            num_layers=4 if in_channels > 50 else 6,
            num_heads=8,
            wavelengths=wavelengths,
            bandwidths=bandwidths,
            gsd=gsd,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# SPLIT MASK
# =============================================================================

def build_split_mask(crop_index_path, city, subset, split, label_shape):
    H, W = label_shape
    mask = np.zeros((H, W), dtype=bool)

    with open(crop_index_path) as f:
        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])
        if not has_split:
            return np.ones((H, W), dtype=bool)

        n_crops = 0
        for row in reader:
            if row["city"] != city or row["subset"] != subset:
                continue
            if row.get("split", "") != split:
                continue
            r0 = int(row["row_10m"])
            c0 = int(row["col_10m"])
            h = int(row["crop_h"])
            w = int(row["crop_w"])
            mask[r0:min(r0+h, H), c0:min(c0+w, W)] = True
            n_crops += 1

    coverage = mask.sum() / (H * W) * 100
    print(f"[SplitMask] {split}: {n_crops} crops, {mask.sum():,} px ({coverage:.1f}%)")
    return mask


def build_all_split_masks(crop_index_path, city, subset, label_shape):
    return {s: build_split_mask(crop_index_path, city, subset, s, label_shape)
            for s in ["train", "val", "test"]}


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_crop(data, norm_min, norm_range):
    """Per-band min-max normalization. data: [C, H, W]."""
    n = min(data.shape[0], len(norm_min))
    data[:n] = (data[:n] - norm_min[:n, None, None]) / norm_range[:n, None, None]
    data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    data = torch.clamp(data, -0.5, 1.5)
    return data


def normalize_crop_zscore(data, norm_mean, norm_std):
    """Per-band z-score normalization, rescaled to [0,1]. data: [C, H, W]."""
    n = min(data.shape[0], len(norm_mean))
    data[:n] = (data[:n] - norm_mean[:n, None, None]) / norm_std[:n, None, None]
    data = (data + 3.0) / 6.0  # map ±3σ → [0, 1]
    data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    data = torch.clamp(data, -0.5, 1.5)
    return data


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

class BaselineSlidingWindow:
    """
    Sliding window inference for baseline models (UNet, ViT).

    Handles:
      - Same-sensor eval: read eval_sensor, normalize, forward
      - Cross-sensor eval: read eval_sensor, normalize with eval stats,
        interpolate spectrally to train_sensor's band count
    """

    def __init__(
        self, model, device,
        mat_path, subset, city,
        eval_sensor, train_sensor,
        spectral_meta, stats,
        crop_size=128, stride_divisor=2,
        # Cross-sensor interpolation
        cross_sensor=False,
        eval_wavelengths=None,
        train_wavelengths=None,
        norm_mode="band_minmax",
        zscore_stats=None,
    ):
        self.model = model
        self.device = device
        self.reader = MatFileReader(mat_path)
        self.subset = subset
        self.city = city
        self.eval_sensor = eval_sensor
        self.train_sensor = train_sensor
        self.crop_size = crop_size
        self.stride_divisor = stride_divisor
        self.cross_sensor = cross_sensor
        self.axis_order = AXIS_ORDER[subset]
        self.needs_label_remap = (subset == "china")
        self.norm_mode = norm_mode

        # Eval sensor info
        eval_meta_key = SENSOR_META_KEY[(subset, eval_sensor)]
        self.eval_n_bands = spectral_meta[eval_meta_key]["n_bands"]
        self.eval_gsd = SENSOR_GSD[(subset, eval_sensor)]
        self.eval_mat_key = MAT_KEYS.get(eval_sensor, None)

        # Check if eval sensor uses NPY
        from training.utils.datasets_baselines.utils_dataset_Cseg import NPY_SENSORS
        npy_key = (subset, eval_sensor)
        self.is_npy = npy_key in NPY_SENSORS
        self._npy_data = None

        if self.is_npy:
            data_dir = os.path.dirname(mat_path)
            npy_path = os.path.join(data_dir, NPY_SENSORS[npy_key])
            self._npy_data = np.load(npy_path, mmap_mode="r")
            print(f"[Eval-BL] Eval sensor '{eval_sensor}': NPY source ({npy_path})")

        # Normalization for eval sensor
        eval_stat_key = f"{subset}_{eval_sensor}_{city}"

        if norm_mode == "zscore" and zscore_stats is not None:
            # Z-score normalization
            for key in [eval_stat_key, f"{subset}_{eval_sensor}"]:
                if key in zscore_stats and "band_mean" in zscore_stats[key]:
                    entry = zscore_stats[key]
                    self.eval_norm_mean = torch.tensor(
                        entry["band_mean"], dtype=torch.float32)
                    self.eval_norm_std = torch.tensor(
                        entry["band_std"], dtype=torch.float32)
                    print(f"[Eval-BL] Zscore normalization for '{eval_sensor}' ({key})")
                    break
            else:
                print(f"[Eval-BL] WARNING: no zscore stats for '{eval_sensor}', "
                      f"falling back to band_minmax")
                self.norm_mode = "band_minmax"

        if self.norm_mode == "band_minmax":
            if eval_stat_key in stats and "band_min" in stats[eval_stat_key]:
                entry = stats[eval_stat_key]
                self.eval_norm_min = torch.tensor(entry["band_min"], dtype=torch.float32)
                self.eval_norm_range = torch.clamp(
                    torch.tensor(entry["band_max"], dtype=torch.float32) - self.eval_norm_min,
                    min=1e-6)
            else:
                print(f"[Eval-BL] WARNING: no stats for '{eval_stat_key}'")
                self.eval_norm_min = torch.zeros(self.eval_n_bands)
                self.eval_norm_range = torch.ones(self.eval_n_bands)

        # Cross-sensor wavelengths
        if cross_sensor:
            self.eval_wl = torch.tensor(eval_wavelengths, dtype=torch.float32)
            self.train_wl = torch.tensor(train_wavelengths, dtype=torch.float32)

    def _read_sensor_crop(self, r0, c0, h, w):
        """Read eval sensor crop, normalize, optionally interpolate.
        Supports both mat file and aligned NPY sources.
        Handles any resolution and resizes to crop_size for the model."""

        # Scale coordinates from 10m reference to sensor resolution
        if self.eval_gsd != 10.0:
            scale = 10.0 / self.eval_gsd  # >1 for finer, <1 for coarser
            sr0 = int(r0 * scale)
            sc0 = int(c0 * scale)
            sh = max(1, int(h * scale))
            sw = max(1, int(w * scale))
        else:
            sr0, sc0, sh, sw = r0, c0, h, w

        if self.is_npy and self._npy_data is not None:
            npy_arr = self._npy_data
            r1 = min(sr0 + sh, npy_arr.shape[1])
            c1 = min(sc0 + sw, npy_arr.shape[2])
            sr0 = max(0, sr0)
            sc0 = max(0, sc0)
            data = np.array(npy_arr[:, sr0:r1, sc0:c1], dtype=np.float32)
            data = torch.from_numpy(data)
        else:
            data = self.reader.read_crop(
                self.eval_mat_key, sr0, sc0, sh, sw, axis_order=self.axis_order)
            data = torch.from_numpy(data)

        if data.shape[0] > self.eval_n_bands:
            data = data[:self.eval_n_bands]

        # Normalize with eval sensor stats
        if self.norm_mode == "zscore":
            data = normalize_crop_zscore(data, self.eval_norm_mean, self.eval_norm_std)
        else:
            data = normalize_crop(data, self.eval_norm_min, self.eval_norm_range)

        # Cross-sensor: interpolate to training sensor's spectral grid
        if self.cross_sensor:
            data = spectral_interpolate(data, self.eval_wl, self.train_wl)

        # Resize to model's expected spatial size if needed
        if data.shape[1] != self.crop_size or data.shape[2] != self.crop_size:
            if self.eval_gsd > 10.0:
                # Coarser resolution: zero-pad to crop_size (don't interpolate —
                # that would just recreate the upsampled training data)
                C_data = data.shape[0]
                padded = torch.zeros(C_data, self.crop_size, self.crop_size,
                                     dtype=data.dtype)
                h_actual = min(data.shape[1], self.crop_size)
                w_actual = min(data.shape[2], self.crop_size)
                padded[:, :h_actual, :w_actual] = data[:, :h_actual, :w_actual]
                data = padded
            else:
                # Finer resolution: bilinear downsample to crop_size
                data = torch.nn.functional.interpolate(
                    data.unsqueeze(0),
                    size=(self.crop_size, self.crop_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        return data

    def _read_label_crop(self, r0, c0, h, w):
        data = self.reader.read_label_crop(r0, c0, h, w)
        label = torch.from_numpy(data.astype(np.int64))
        if self.needs_label_remap:
            remapped = torch.full_like(label, IGNORE_INDEX)
            for raw_val, new_val in CHINA_LABEL_REMAP.items():
                remapped[label == raw_val] = new_val
            label = remapped
        return label

    def run(self):
        self.reader._open()

        # Label dimensions
        label_shape = self.reader.get_shape("label")
        label_h, label_w = label_shape[0], label_shape[1]
        print(f"[Eval-BL] Label: {label_h}×{label_w}")

        # Full label
        full_label = self.reader.read_label_crop(0, 0, label_h, label_w).astype(np.int64)
        if self.needs_label_remap:
            remapped = np.full_like(full_label, IGNORE_INDEX)
            for raw_val, new_val in CHINA_LABEL_REMAP.items():
                remapped[full_label == raw_val] = new_val
            full_label = remapped

        crop_h = crop_w = self.crop_size
        stride_h = max(1, crop_h // self.stride_divisor)
        stride_w = max(1, crop_w // self.stride_divisor)

        print(f"[Eval-BL] Crop: {crop_h}×{crop_w}, stride: {stride_h}×{stride_w}")

        logit_accum = np.zeros((label_h, label_w, NUM_CLASSES), dtype=np.float64)
        count_accum = np.zeros((label_h, label_w), dtype=np.int32)

        # Window grid
        windows = set()
        for r0 in range(0, label_h - crop_h + 1, stride_h):
            for c0 in range(0, label_w - crop_w + 1, stride_w):
                windows.add((r0, c0))
        # Edges
        for r0 in range(0, label_h - crop_h + 1, stride_h):
            windows.add((r0, label_w - crop_w))
        for c0 in range(0, label_w - crop_w + 1, stride_w):
            windows.add((label_h - crop_h, c0))
        windows.add((label_h - crop_h, label_w - crop_w))
        windows = sorted(windows)

        n_windows = len(windows)
        print(f"[Eval-BL] {n_windows} windows")

        self.model.eval()
        t_start = time.perf_counter()

        for i, (r0, c0) in enumerate(windows):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (n_windows - i - 1)
                print(f"  Window {i+1}/{n_windows} "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

            try:
                image = self._read_sensor_crop(r0, c0, crop_h, crop_w)
            except Exception:
                continue

            # [C, H, W] → [1, C, H, W]
            image_batch = image.unsqueeze(0).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(image_batch)  # [1, num_classes, H, W]

            logits = logits[0].cpu().float().numpy()  # [num_classes, H, W]
            logits = logits.transpose(1, 2, 0)  # [H, W, num_classes]

            r_end = min(r0 + crop_h, label_h)
            c_end = min(c0 + crop_w, label_w)
            h_actual = r_end - r0
            w_actual = c_end - c0

            logit_accum[r0:r_end, c0:c_end] += logits[:h_actual, :w_actual]
            count_accum[r0:r_end, c0:c_end] += 1

        covered = count_accum > 0
        prediction = np.full((label_h, label_w), IGNORE_INDEX, dtype=np.int64)
        prediction[covered] = logit_accum[covered].argmax(axis=-1)

        elapsed = time.perf_counter() - t_start
        coverage = covered.sum() / (label_h * label_w) * 100
        print(f"[Eval-BL] Done: {elapsed:.1f}s, coverage: {coverage:.1f}%")

        return {
            "prediction": prediction,
            "label": full_label,
            "logits": logit_accum,
            "counts": count_accum,
        }


# =============================================================================
# METRICS (same as Atomizer eval)
# =============================================================================

def compute_metrics(prediction, label, split_mask=None, exclude_background=True):
    valid = (label != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
    if exclude_background:
        valid = valid & (label > 0)
    if split_mask is not None:
        valid = valid & split_mask

    pred_valid = prediction[valid]
    label_valid = label[valid]

    overall_acc = float((pred_valid == label_valid).sum() / max(len(label_valid), 1))

    per_class = {}
    ious, f1s = [], []

    for cls_id in range(NUM_CLASSES):
        pred_cls = pred_valid == cls_id
        label_cls = label_valid == cls_id
        tp = int((pred_cls & label_cls).sum())
        fp = int((pred_cls & ~label_cls).sum())
        fn = int((~pred_cls & label_cls).sum())
        union = tp + fp + fn
        support = tp + fn
        iou = tp / union if (support > 0 and union > 0) else float("nan")
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id], "iou": float(iou), "f1": float(f1),
            "precision": float(precision), "recall": float(recall),
            "support": support, "in_test": support > 0,
        }
        if cls_id >= 1 and not np.isnan(iou) and support > 0:
            ious.append(iou)
            f1s.append(f1)

    return {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mF1": float(np.mean(f1s)) if f1s else 0.0,
        "overall_accuracy": overall_acc,
        "n_classes_evaluated": len(ious),
        "n_classes_total": NUM_CLASSES,
        "exclude_background": exclude_background,
        "per_class": per_class,
        "n_valid_pixels": int(valid.sum()),
    }


def compute_confusion_matrix(prediction, label, split_mask=None, exclude_background=True):
    valid = (label != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
    if exclude_background:
        valid = valid & (label > 0)
    if split_mask is not None:
        valid = valid & split_mask
    pred_valid = prediction[valid]
    label_valid = label[valid]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            cm[t, p] = ((label_valid == t) & (pred_valid == p)).sum()
    return cm


# =============================================================================
# VISUALIZATION (same as Atomizer eval)
# =============================================================================

def label_to_rgb(label_map):
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        rgb[label_map == cls_id] = CLASS_COLORS[cls_id]
    return rgb


def plot_prediction_vs_gt(prediction, label, output_path, title="",
                          metrics=None, split_mask=None):
    pred_rgb = label_to_rgb(prediction)
    gt_rgb = label_to_rgb(label)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(gt_rgb); axes[0].set_title("Ground Truth", fontsize=14); axes[0].axis("off")
    axes[1].imshow(pred_rgb); axes[1].set_title("Prediction", fontsize=14); axes[1].axis("off")

    if split_mask is not None:
        for ax in axes:
            edges = np.zeros_like(split_mask)
            edges[1:] |= split_mask[1:] != split_mask[:-1]
            edges[:, 1:] |= split_mask[:, 1:] != split_mask[:, :-1]
            ey, ex = np.where(edges)
            if len(ey) > 0:
                ax.scatter(ex, ey, c="red", s=0.2, alpha=0.8, zorder=10)

    present = set(np.unique(label[label != IGNORE_INDEX])) | \
              set(np.unique(prediction[prediction != IGNORE_INDEX]))
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
               for i in sorted(present) if i < NUM_CLASSES]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 7), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if metrics:
        info = (f"mIoU: {metrics['mIoU']:.4f}  |  mF1: {metrics['mF1']:.4f}  |  "
                f"OA: {metrics['overall_accuracy']:.4f}  |  "
                f"{metrics['n_classes_evaluated']}/{metrics['n_classes_total']-1} classes")
        fig.text(0.5, 0.02, info, ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_per_class_iou(metrics, output_path, title="Per-Class IoU"):
    names, ious, colors, hatches = [], [], [], []
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        names.append(info["name"])
        ious.append(info["iou"] if not np.isnan(info["iou"]) else 0.0)
        colors.append(np.array(CLASS_COLORS[cls_id]) / 255.0)
        hatches.append("//" if (not info["in_test"] or cls_id == 0) else "")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(NUM_CLASSES), ious, color=colors, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{title}  (mIoU={metrics['mIoU']:.4f})", fontsize=13)
    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f}")
    ax.legend()

    for bar, iou, cls_id in zip(bars, ious, range(NUM_CLASSES)):
        info = metrics["per_class"][cls_id]
        if info["in_test"] and cls_id > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{iou:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_confusion_matrix(cm, output_path, title="Confusion Matrix", classes_present=None):
    if classes_present is not None:
        idx = sorted(classes_present)
        cm_sub = cm[np.ix_(idx, idx)]
        names_sub = [CLASS_NAMES[i] for i in idx]
    else:
        cm_sub = cm
        names_sub = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    n = cm_sub.shape[0]
    cm_norm = cm_sub.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    fig, ax = plt.subplots(figsize=(max(8, n*0.8), max(7, n*0.7)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names_sub, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names_sub, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="C2Seg Baseline Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "unet3d", "vit", "perceiver", "senpa"])
    parser.add_argument("--train_sensor", type=str, required=True,
                        choices=["hsi", "msi", "msi12", "sar", "enmap_30m", "hyspex", "hyspex_10m"],
                        help="Sensor the model was trained on")
    parser.add_argument("--eval_sensor", type=str, required=True,
                        choices=["hsi", "msi", "msi12", "sar", "enmap_30m", "hyspex", "hyspex_10m"],
                        help="Sensor to evaluate on")
    parser.add_argument("--output_dir", type=str, default="./results/c2seg_bl")

    # City and split
    parser.add_argument("--subset", type=str, default="germany",
                        choices=["germany", "china"])
    parser.add_argument("--eval_city", type=str, default="augsburg")
    parser.add_argument("--eval_split", type=str, default=None,
                        choices=["train", "val", "test"])

    # Paths
    parser.add_argument("--processed_dir", type=str,
                        default="./data/CrossCity/c2seg_processed")
    parser.add_argument("--crop_index", type=str, default=None)

    # Inference
    parser.add_argument("--stride_divisor", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--norm_mode", type=str, default="band_minmax",
                        choices=["band_minmax", "zscore", "identity"],
                        help="Must match training normalization mode")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eval_subset = CITY_SUBSET.get(args.eval_city, args.subset)
    subset_dir = "Germany" if eval_subset == "germany" else "China"
    eval_mat = os.path.join(f"./data/CrossCity/{subset_dir}",
                            CITY_MAT.get(args.eval_city, f"{args.eval_city}.mat"))

    cross_sensor = (args.train_sensor != args.eval_sensor)
    mode_label = "cross-sensor" if cross_sensor else "same-sensor"

    split_label = f" [{args.eval_split}]" if args.eval_split else " [full]"

    print(f"\n{'='*60}")
    print(f"  C2Seg Baseline Evaluation ({mode_label})")
    print(f"  Model:       {args.model}")
    print(f"  Train sensor: {args.train_sensor}")
    print(f"  Eval sensor:  {args.eval_sensor}")
    print(f"  City:        {args.eval_city} ({eval_subset})")
    print(f"  Split:       {args.eval_split or 'full'}")
    print(f"  Ckpt:        {args.ckpt_path}")
    print(f"{'='*60}\n")

    # ── Spectral metadata ───────────────────────────────────────────
    spectral_meta_path = os.path.join(args.processed_dir, "c2seg_spectral_meta.json")
    with open(spectral_meta_path) as f:
        spectral_meta = json.load(f)

    stats_path = os.path.join(args.processed_dir, "c2seg_norm_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    # Load zscore stats if needed
    zscore_stats = None
    if args.norm_mode == "zscore":
        zscore_path = os.path.join(args.processed_dir, "c2seg_zscore_stats.json")
        if os.path.exists(zscore_path):
            with open(zscore_path) as f:
                zscore_stats = json.load(f)
            print(f"[Eval-BL] Zscore stats loaded from {zscore_path}")
        else:
            print(f"[Eval-BL] WARNING: {zscore_path} not found, falling back to band_minmax")
            args.norm_mode = "band_minmax"

    crop_index_path = args.crop_index or os.path.join(
        args.processed_dir, "c2seg_crop_index_split.csv")

    # ── Determine input channels ────────────────────────────────────
    train_meta_key = SENSOR_META_KEY[(eval_subset, args.train_sensor)]
    train_n_bands = spectral_meta[train_meta_key]["n_bands"]
    train_wavelengths = spectral_meta[train_meta_key]["wavelengths"]
    train_bandwidths = spectral_meta[train_meta_key].get("bandwidths", None)
    train_gsd = SENSOR_GSD.get((eval_subset, args.train_sensor), 10.0)

    eval_meta_key = SENSOR_META_KEY[(eval_subset, args.eval_sensor)]
    eval_wavelengths = spectral_meta[eval_meta_key]["wavelengths"]

    # ── Load model ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        args.model, train_n_bands, NUM_CLASSES, img_size=args.crop_size,
        wavelengths=train_wavelengths, bandwidths=train_bandwidths,
        gsd=train_gsd,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        # Lightning checkpoint
        state = {k.replace("model.", "", 1): v
                 for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state)
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Run inference ───────────────────────────────────────────────
    engine = BaselineSlidingWindow(
        model=model, device=device,
        mat_path=eval_mat, subset=eval_subset, city=args.eval_city,
        eval_sensor=args.eval_sensor, train_sensor=args.train_sensor,
        spectral_meta=spectral_meta, stats=stats,
        crop_size=args.crop_size, stride_divisor=args.stride_divisor,
        cross_sensor=cross_sensor,
        eval_wavelengths=eval_wavelengths if cross_sensor else None,
        train_wavelengths=train_wavelengths if cross_sensor else None,
        norm_mode=args.norm_mode,
        zscore_stats=zscore_stats,
    )

    result = engine.run()

    # ── Split mask ──────────────────────────────────────────────────
    label_shape = result["label"].shape
    split_mask = None
    all_masks = None

    if args.eval_split:
        split_mask = build_split_mask(
            crop_index_path, args.eval_city, eval_subset,
            args.eval_split, label_shape)
        all_masks = build_all_split_masks(
            crop_index_path, args.eval_city, eval_subset, label_shape)

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = compute_metrics(result["prediction"], result["label"],
                              split_mask=split_mask, exclude_background=True)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  mIoU:  {metrics['mIoU']:>7.4f}  "
          f"({metrics['n_classes_evaluated']} classes, no BG)    │")
    print(f"  │  mF1:   {metrics['mF1']:>7.4f}                            │")
    print(f"  │  OA:    {metrics['overall_accuracy']:>7.4f}                            │")
    print(f"  └─────────────────────────────────────────────┘")

    print()
    print(f"  {'Class':<35s} {'IoU':>8s} {'F1':>8s} {'Support':>10s}")
    print(f"  {'-'*63}")
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        if cls_id == 0:
            print(f"  {info['name']:<35s} {'excl':>8s} {'':>8s} "
                  f"{info['support']:>10,d}")
        elif info["in_test"]:
            print(f"  {info['name']:<35s} {info['iou']:>8.4f} {info['f1']:>8.4f} "
                  f"{info['support']:>10,d}")
        else:
            print(f"  {info['name']:<35s} {'N/A':>8s} {'N/A':>8s} "
                  f"{info['support']:>10,d}")

    # ── Save metrics ────────────────────────────────────────────────
    metrics["config"] = {
        "model": args.model, "train_sensor": args.train_sensor,
        "eval_sensor": args.eval_sensor, "cross_sensor": cross_sensor,
        "city": args.eval_city, "subset": eval_subset,
        "split": args.eval_split, "ckpt_path": args.ckpt_path,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → {os.path.join(args.output_dir, 'metrics.json')}")

    # ── Visualizations ──────────────────────────────────────────────
    sensor_str = (f"{args.train_sensor}→{args.eval_sensor}"
                  if cross_sensor else args.eval_sensor)
    title = f"BL {args.model} — {sensor_str} on {args.eval_city}{split_label}"

    plot_prediction_vs_gt(
        result["prediction"], result["label"],
        os.path.join(args.output_dir, "prediction_vs_gt.png"),
        title=title, metrics=metrics, split_mask=split_mask)

    plot_per_class_iou(
        metrics, os.path.join(args.output_dir, "per_class_iou.png"), title=title)

    cm = compute_confusion_matrix(result["prediction"], result["label"],
                                  split_mask=split_mask)
    present = [c for c in range(NUM_CLASSES) if metrics["per_class"][c]["in_test"]]
    plot_confusion_matrix(
        cm, os.path.join(args.output_dir, "confusion_matrix.png"),
        title=f"CM — {args.model} {sensor_str}{split_label}",
        classes_present=present if len(present) < NUM_CLASSES else None)

    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()