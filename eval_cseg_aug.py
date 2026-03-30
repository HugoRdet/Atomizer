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
  5. Split overlay visualization (PNG)
  6. Metrics JSON (mIoU, OA, per-class IoU, profiling)

mIoU is computed ONLY over classes present in the test ground truth
AND only over pixels in the specified split region.

Usage:
    # Augsburg same-city: eval test split with HSI (same sensor)
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg/aug_hsi

    # Augsburg cross-sensor: eval test split with MSI
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor msi \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg/aug_msi

    # Augsburg cross-sensor: eval test split with SAR
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor sar \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg/aug_sar

    # Cross-city: eval on Berlin
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi \
        --eval_city berlin \
        --output_dir ./results/c2seg/berlin_hsi

    # Fusion eval
    python eval_c2seg.py \
        --ckpt_path ./checkpoints/c2seg/germany/best.ckpt \
        --subset germany --sensor hsi msi sar --fusion \
        --eval_city augsburg --eval_split test \
        --output_dir ./results/c2seg/aug_fusion
"""

import os
import argparse
import csv
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
# SUBSET CONFIG
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

# City → mat file name
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
# COLORS — 14 classes
# =============================================================================

CLASS_COLORS = [
    [255, 255, 255],  # 0  Background
    [255,   0,   0],  # 1  Urban Fabric
    [204,   0, 230],  # 2  Industrial/Commercial
    [  0,   0,   0],  # 3  Street Network
    [166,  77,   0],  # 4  Mine/Dump/Construction
    [255, 170, 255],  # 5  Artificially Vegetated
    [255, 255,   0],  # 6  Arable Land
    [255, 170,   0],  # 7  Permanent Crops
    [190, 255,   0],  # 8  Pastures
    [  0, 120,   0],  # 9  Forests
    [170, 210,  90],  # 10 Shrub
    [210, 200, 160],  # 11 Open Spaces
    [  0, 200, 200],  # 12 Inland Wetlands
    [  0,   0, 255],  # 13 Surface Water
]


# =============================================================================
# RESOLUTION REGISTRATION
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048, 2.5: 2048, 4.78: 2048, 5.0: 2048,
    10.0: 2048, 20.0: 2048, 30.0: 2048, 60.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# SPLIT MASK
# =============================================================================

def build_split_mask(crop_index_path, city, subset, split, label_shape):
    """
    Build boolean mask [H, W] marking pixels belonging to a split.
    Returns all-True if no split column in CSV.
    """
    H, W = label_shape
    mask = np.zeros((H, W), dtype=bool)

    with open(crop_index_path) as f:
        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])

        if not has_split:
            print(f"[SplitMask] No 'split' column → returning full mask")
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
    print(f"[SplitMask] {split}: {n_crops} crops, "
          f"{mask.sum():,} pixels ({coverage:.1f}%)")
    return mask


def build_all_split_masks(crop_index_path, city, subset, label_shape):
    """Build masks for train, val, test splits."""
    return {
        s: build_split_mask(crop_index_path, city, subset, s, label_shape)
        for s in ["train", "val", "test"]
    }


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
    """

    def __init__(self, model, dataset, device, stride_divisor=2,
                 crop_size_override=None, band_norm=False, force_10m=False,
                 spectral_interp=None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.stride_divisor = stride_divisor
        self.crop_size_override = crop_size_override
        self.band_norm = band_norm
        self.force_10m = force_10m
        self.spectral_interp = spectral_interp  # dict with target sensor info or None
        self._band_norm_stats = None

        if band_norm:
            print("[Eval] Band normalization enabled — computing per-band stats...")
            self._band_norm_stats = self._precompute_band_stats()

    def _precompute_band_stats(self):
        """
        Compute per-band p2/p98 from a sample of the full image.
        Uses the raw data BEFORE the ÷10000 normalization.
        Returns dict: sensor → (low [C], range [C]) as tensors.
        """
        ds = self.dataset
        stats = {}

        for sensor in ds.sensors:
            info = ds.sensor_info[sensor]

            if info.get("is_npy", False) and sensor in ds._npy_data:
                # Sample from NPY: take every 10th row/col for speed
                npy_arr = ds._npy_data[sensor]
                sample = np.array(npy_arr[:, ::10, ::10], dtype=np.float32)
            else:
                # Sample a few crops from mat
                ds.reader._open()
                sample = ds.reader.read_crop(
                    info["mat_key"], 0, 0, 200, 200,
                    axis_order=ds.axis_order)

            C = sample.shape[0]
            low = np.zeros(C, dtype=np.float32)
            high = np.zeros(C, dtype=np.float32)

            for b in range(C):
                vals = sample[b].flatten()
                vals = vals[vals > 0]  # exclude zeros/padding
                if len(vals) > 0:
                    low[b] = np.percentile(vals, 2)
                    high[b] = np.percentile(vals, 98)
                else:
                    low[b] = 0.0
                    high[b] = 1.0

            rng = np.clip(high - low, 1e-6, None)
            stats[sensor] = (
                torch.tensor(low, dtype=torch.float32),
                torch.tensor(rng, dtype=torch.float32),
            )
            print(f"[Eval] Band norm stats for '{sensor}': "
                  f"p2=[{low[0]:.0f}..{low[-1]:.0f}], "
                  f"p98=[{high[0]:.0f}..{high[-1]:.0f}]")

        return stats

    def _apply_band_norm(self, image, sensor):
        """
        Normalize each band to ~[0, 1] using precomputed p2/p98.
        Applied on the raw-reflectance image (already ÷10000, clamped).
        Re-normalizes using per-band statistics so each band has
        similar value distribution to what the model saw during training.
        """
        if self._band_norm_stats is None or sensor not in self._band_norm_stats:
            return image

        low, rng = self._band_norm_stats[sensor]
        C = min(image.shape[0], low.shape[0])

        # Stats are in raw DN space; image is already ÷10000 and clamped
        # Convert stats to reflectance space
        low_r = low[:C] / 10000.0
        rng_r = rng[:C] / 10000.0

        image[:C] = (image[:C] - low_r[:, None, None]) / rng_r[:, None, None]
        image = torch.clamp(image, 0.0, 1.0)
        return image

    def run(self):
        ds = self.dataset
        label_h, label_w = ds.label_dims
        print(f"[Eval] Label grid: {label_h}×{label_w} at 10m")

        # Read full label
        ds.reader._open()
        full_label = ds.reader.read_label_crop(0, 0, label_h, label_w).astype(np.int64)

        if ds.needs_label_remap:
            remapped = np.full_like(full_label, IGNORE_INDEX)
            for raw_val, new_val in CHINA_LABEL_REMAP.items():
                remapped[full_label == raw_val] = new_val
            full_label = remapped

        # Crop size: override if specified (useful for HySpex — smaller 10m crops
        # → manageable native-resolution crops, e.g., 28×28 at 10m → 127×127 at 2.2m)
        if self.crop_size_override:
            crop_h = crop_w = self.crop_size_override
        else:
            crop_h = ds.crops[0]["crop_h"] if ds.crops else 128
            crop_w = ds.crops[0]["crop_w"] if ds.crops else 128
        stride_h = max(1, crop_h // self.stride_divisor)
        stride_w = max(1, crop_w // self.stride_divisor)

        print(f"[Eval] Crop: {crop_h}×{crop_w}, stride: {stride_h}×{stride_w}")

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

        self.model.eval()
        t_start = time.perf_counter()

        for i, (r0, c0) in enumerate(windows):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (n_windows - i - 1)
                print(f"  Window {i+1}/{n_windows} "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

            crop = {
                "row_10m": r0, "col_10m": c0,
                "crop_h": crop_h, "crop_w": crop_w,
                "hsi_row": r0 // 3, "hsi_col": c0 // 3,
                "hsi_crop_h": int(np.ceil(crop_h / 3)),
                "hsi_crop_w": int(np.ceil(crop_w / 3)),
            }

            try:
                label = ds._read_label_crop(crop)
            except Exception as e:
                continue

            groups = {}
            for sensor in ds.sensors:
                try:
                    if self.spectral_interp is not None:
                        # Read raw DN (no normalization) for interpolation
                        image = ds._read_sensor_crop(sensor, crop, raw_dn=True)
                    else:
                        image = ds._read_sensor_crop(sensor, crop)
                except Exception:
                    continue

                # Optional per-band normalization (diagnostic)
                # Normalizes each band to [0, 1] using p2/p98 percentiles
                # from the crop. Tests whether value distribution mismatch
                # is causing poor cross-sensor transfer.
                if self.band_norm:
                    image = self._apply_band_norm(image, sensor)

                info = ds.sensor_info[sensor]
                gsd = info["gsd"]

                # Spectral interpolation: map eval sensor to training
                # sensor's spectral grid (same preprocessing as baselines).
                #
                # When active, _read_sensor_crop was called with skip_norm=True
                # so image is raw (÷10000 only). We interpolate on raw values,
                # then normalize with training sensor's stats.
                if self.spectral_interp is not None:
                    from training.utils.datasets_baselines.collate import spectral_interpolate
                    source_wl = self.spectral_interp["source_wl"]
                    target_wl = self.spectral_interp["target_wl"]
                    target_info = self.spectral_interp["target_info"]

                    # Interpolate on raw reflectance values
                    image = spectral_interpolate(image, source_wl, target_wl)

                    # Now normalize with training sensor's stats
                    if "train_zscore" in self.spectral_interp:
                        train_mean, train_std = self.spectral_interp["train_zscore"]
                        n = min(image.shape[0], len(train_mean))
                        image[:n] = (image[:n] - train_mean[:n, None, None]) / train_std[:n, None, None]
                        image = (image + 3.0) / 6.0
                    else:
                        # Fallback: just clamp raw reflectance
                        pass

                    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                    image = torch.clamp(image, 0.0, 1.0)
                    info = target_info

                C, H, W = image.shape

                token_label = (torch.full((H, W), IGNORE_INDEX, dtype=torch.int64)
                               if gsd != 10.0 else label)

                tokens = ds.token_builder.build_tokens(
                    image=image, label=token_label,
                    resolution=gsd,
                    spectral_indices=info["spectral_indices"],
                    resolution_idx=info["resolution_idx"],
                    time_idx=-1,
                )
                token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

                if gsd in groups:
                    groups[gsd]["tokens"] = torch.cat([groups[gsd]["tokens"], tokens], 0)
                    groups[gsd]["mask"] = torch.cat([groups[gsd]["mask"], token_mask], 0)
                    old_c = groups[gsd]["shape"][0]
                    groups[gsd]["shape"] = (old_c + C, H, W)
                else:
                    groups[gsd] = {"tokens": tokens, "mask": token_mask, "shape": (C, H, W)}

            if not groups:
                continue

            # Pad missing resolution groups (same as training)
            for res in ds.all_resolutions:
                if res not in groups:
                    groups[res] = {
                        "tokens": torch.zeros(1, 8),
                        "mask": torch.ones(1, dtype=torch.bool),
                        "shape": (1, 1, 1),
                    }

            # Build queries
            query_sensor = None
            for s in ds.sensors:
                if ds.sensor_info[s]["gsd"] <= 10.0:
                    query_sensor = s
                    break
            if query_sensor is None:
                query_sensor = ds.sensors[0]

            query_info = ds.sensor_info[query_sensor]
            queries = ds.token_builder.build_queries(
                label=label, resolution=10.0,
                first_spectral_idx=query_info["spectral_indices"][0].item(),
                resolution_idx=ds.look_up.get_resolution_idx(10.0),
                time_idx=-1,
            )

            sample = {
                "groups": groups,
                "tasks": {"c2seg_segmentation": {
                    "queries": queries,
                    "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
                }},
                "target_resolution": 10.0,
                "dataset_name": "C2Seg",
            }

            batch = collate_multitask([sample])
            batch = _batch_to_device(batch, self.device)

            with torch.no_grad():
                preds = self.model.forward_multitask(batch, training=False)

            logits = preds["c2seg_segmentation"][0].cpu().numpy()

            n_pixels = crop_h * crop_w
            if logits.shape[0] == n_pixels:
                logits_2d = logits.reshape(crop_h, crop_w, NUM_CLASSES)
                r_end = min(r0 + crop_h, label_h)
                c_end = min(c0 + crop_w, label_w)
                h_actual = r_end - r0
                w_actual = c_end - c0
                logit_accum[r0:r_end, c0:c_end] += logits_2d[:h_actual, :w_actual]
                count_accum[r0:r_end, c0:c_end] += 1

        covered = count_accum > 0
        prediction = np.full((label_h, label_w), IGNORE_INDEX, dtype=np.int64)
        prediction[covered] = logit_accum[covered].argmax(axis=-1)

        elapsed = time.perf_counter() - t_start
        coverage = covered.sum() / (label_h * label_w) * 100
        print(f"[Eval] Done: {elapsed:.1f}s, coverage: {coverage:.1f}%")

        return {
            "prediction": prediction,
            "label": full_label,
            "logits": logit_accum,
            "counts": count_accum,
        }

    def profile_flops(self, n_warmup=5, n_active=3):
        """
        Measure GFLOPS per forward pass using PyTorch profiler.

        Builds one representative batch from the first crop, runs warmup
        forwards to fill caches (geo pruning, grid configs), then profiles
        active runs.

        Returns dict with flops stats or None if profiling fails.
        """
        ds = self.dataset
        if not ds.crops:
            print("[Profile] No crops available, skipping.")
            return None

        print(f"\n[Profile] Measuring FLOPS ({n_warmup} warmup + {n_active} active)...")

        # ── Build a representative batch from first crop ──────────────
        crop_entry = ds.crops[0]
        r0, c0 = crop_entry["row_10m"], crop_entry["col_10m"]
        if self.crop_size_override:
            crop_h = crop_w = self.crop_size_override
        else:
            crop_h, crop_w = crop_entry["crop_h"], crop_entry["crop_w"]

        crop = {
            "row_10m": r0, "col_10m": c0,
            "crop_h": crop_h, "crop_w": crop_w,
            "hsi_row": r0 // 3, "hsi_col": c0 // 3,
            "hsi_crop_h": int(np.ceil(crop_h / 3)),
            "hsi_crop_w": int(np.ceil(crop_w / 3)),
        }

        try:
            label = ds._read_label_crop(crop)
        except Exception as e:
            print(f"[Profile] Failed to read label: {e}")
            return None

        groups = {}
        for sensor in ds.sensors:
            try:
                image = ds._read_sensor_crop(sensor, crop)
            except Exception:
                continue

            info = ds.sensor_info[sensor]
            gsd = info["gsd"]
            C, H, W = image.shape

            token_label = (torch.full((H, W), IGNORE_INDEX, dtype=torch.int64)
                           if gsd != 10.0 else label)

            tokens = ds.token_builder.build_tokens(
                image=image, label=token_label,
                resolution=gsd,
                spectral_indices=info["spectral_indices"],
                resolution_idx=info["resolution_idx"],
                time_idx=-1,
            )
            token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

            if gsd in groups:
                groups[gsd]["tokens"] = torch.cat([groups[gsd]["tokens"], tokens], 0)
                groups[gsd]["mask"] = torch.cat([groups[gsd]["mask"], token_mask], 0)
                old_c = groups[gsd]["shape"][0]
                groups[gsd]["shape"] = (old_c + C, H, W)
            else:
                groups[gsd] = {"tokens": tokens, "mask": token_mask, "shape": (C, H, W)}

        if not groups:
            print("[Profile] No valid groups, skipping.")
            return None

        for res in ds.all_resolutions:
            if res not in groups:
                groups[res] = {
                    "tokens": torch.zeros(1, 8),
                    "mask": torch.ones(1, dtype=torch.bool),
                    "shape": (1, 1, 1),
                }

        query_sensor = None
        for s in ds.sensors:
            if ds.sensor_info[s]["gsd"] <= 10.0:
                query_sensor = s
                break
        if query_sensor is None:
            query_sensor = ds.sensors[0]

        query_info = ds.sensor_info[query_sensor]
        queries = ds.token_builder.build_queries(
            label=label, resolution=10.0,
            first_spectral_idx=query_info["spectral_indices"][0].item(),
            resolution_idx=ds.look_up.get_resolution_idx(10.0),
            time_idx=-1,
        )

        sample = {
            "groups": groups,
            "tasks": {"c2seg_segmentation": {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }},
            "target_resolution": 10.0,
            "dataset_name": "C2Seg",
        }

        batch = collate_multitask([sample])
        batch = _batch_to_device(batch, self.device)

        # ── Token count info ──────────────────────────────────────────
        total_tokens = sum(
            groups[res]["tokens"].shape[0] for res in groups
            if not groups[res]["mask"].all()
        )
        n_queries = queries.shape[0]
        print(f"[Profile] Input: {total_tokens:,} tokens, {n_queries:,} queries")

        # ── Warmup (fills caches, JIT compilation, etc.) ──────────────
        self.model.eval()
        for i in range(n_warmup):
            with torch.no_grad():
                _ = self.model.forward_multitask(batch, training=False)
        torch.cuda.synchronize()
        print(f"[Profile] Warmup complete ({n_warmup} runs)")

        # ── Profile with torch.profiler ───────────────────────────────
        from torch.profiler import profile, ProfilerActivity

        flops_list = []
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            for i in range(n_active):
                with torch.no_grad():
                    _ = self.model.forward_multitask(batch, training=False)
                torch.cuda.synchronize()

        # Extract total FLOPS from profiler events
        total_flops = sum(
            evt.flops for evt in prof.key_averages()
            if evt.flops is not None and evt.flops > 0
        )
        # Profiler reports total across all n_active runs
        flops_per_forward = total_flops / n_active
        gflops = flops_per_forward / 1e9

        # ── Timing (separate from profiler for clean measurement) ─────
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        n_timing = 10
        for _ in range(n_timing):
            with torch.no_grad():
                _ = self.model.forward_multitask(batch, training=False)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t_start) / n_timing * 1000

        # ── Report ────────────────────────────────────────────────────
        result = {
            "gflops_per_forward": round(gflops, 2),
            "flops_per_forward": int(flops_per_forward),
            "avg_latency_ms": round(avg_ms, 1),
            "throughput_crops_per_sec": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0,
            "input_tokens": total_tokens,
            "n_queries": n_queries,
            "sensors": ds.sensors,
            "n_warmup": n_warmup,
            "n_active": n_active,
        }

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  GFLOPS:     {gflops:>10.2f}                    │")
        print(f"  │  Latency:    {avg_ms:>10.1f} ms                 │")
        print(f"  │  Throughput: {result['throughput_crops_per_sec']:>10.2f} crops/s           │")
        print(f"  │  Tokens:     {total_tokens:>10,}                    │")
        print(f"  │  Queries:    {n_queries:>10,}                    │")
        print(f"  └─────────────────────────────────────────────┘")

        return result


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(prediction, label, split_mask=None, exclude_background=True):
    """
    Compute mIoU, mF1, OA, per-class IoU and F1.

    If split_mask is provided, only pixels where split_mask==True are evaluated.
    Background (class 0) excluded from all metrics by default.
    """
    valid = (label != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
    if exclude_background:
        valid = valid & (label > 0)
    if split_mask is not None:
        valid = valid & split_mask

    pred_valid = prediction[valid]
    label_valid = label[valid]

    overall_acc = float((pred_valid == label_valid).sum() / max(len(label_valid), 1))

    per_class = {}
    ious = []
    f1s = []

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
            "name": CLASS_NAMES[cls_id],
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": support,
            "in_test": support > 0,
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


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_to_rgb(label_map):
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        rgb[label_map == cls_id] = CLASS_COLORS[cls_id]
    return rgb


def plot_prediction_vs_gt(prediction, label, output_path, title="",
                          metrics=None, split_mask=None):
    """Side-by-side GT and prediction. Optionally overlay split boundary."""
    pred_rgb = label_to_rgb(prediction)
    gt_rgb = label_to_rgb(label)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")

    # Overlay split boundary on both panels
    if split_mask is not None:
        for ax in axes:
            # Draw red boundary around evaluated region
            edges = np.zeros_like(split_mask)
            edges[1:] |= split_mask[1:] != split_mask[:-1]
            edges[:, 1:] |= split_mask[:, 1:] != split_mask[:, :-1]
            ey, ex = np.where(edges)
            if len(ey) > 0:
                ax.scatter(ex, ey, c="red", s=0.2, alpha=0.8, zorder=10)

    # Legend
    present = set(np.unique(label[label != IGNORE_INDEX])) | \
              set(np.unique(prediction[prediction != IGNORE_INDEX]))
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0,
                              label=CLASS_NAMES[i])
               for i in sorted(present) if i < NUM_CLASSES]
    patches.append(mpatches.Patch(color=[0.78]*3, edgecolor="black", label="Ignore"))
    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(patches), 7), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if metrics:
        info = (f"mIoU: {metrics['mIoU']:.4f}  |  "
                f"mF1: {metrics['mF1']:.4f}  |  "
                f"OA: {metrics['overall_accuracy']:.4f}  |  "
                f"{metrics['n_classes_evaluated']}/{metrics['n_classes_total']-1} classes (no BG)")
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
        iou = info["iou"] if not np.isnan(info["iou"]) else 0.0
        ious.append(iou)
        colors.append(np.array(CLASS_COLORS[cls_id]) / 255.0)
        hatches.append("//" if (not info["in_test"] or cls_id == 0) else "")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(NUM_CLASSES), ious, color=colors, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{title}  (mIoU={metrics['mIoU']:.4f}, mF1={metrics['mF1']:.4f}, "
                 f"{metrics['n_classes_evaluated']} classes)", fontsize=13)
    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f}")
    ax.legend(fontsize=10)

    for bar, iou, cls_id in zip(bars, ious, range(NUM_CLASSES)):
        info = metrics["per_class"][cls_id]
        if info["in_test"] and cls_id > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{iou:.3f}", ha="center", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                    "N/A" if not info["in_test"] else f"{iou:.3f}*",
                    ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_confusion_matrix(cm, output_path, title="Confusion Matrix",
                          classes_present=None):
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
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n*0.8), max(7, n*0.7)))
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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_split_overlay(label, masks, output_path, title="Train/Val/Test Split"):
    """Label map with colored overlays showing split regions."""
    H, W = label.shape
    rgb = label_to_rgb(label).astype(np.float32)

    split_colors = {
        "train": np.array([0, 100, 255], dtype=np.float32),
        "val":   np.array([255, 165, 0], dtype=np.float32),
        "test":  np.array([255, 0, 0], dtype=np.float32),
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: raw labels
    axes[0].imshow(rgb.astype(np.uint8))
    axes[0].set_title("Ground Truth Labels", fontsize=13)
    axes[0].axis("off")

    # Panel 2: labels + split overlay
    overlay = rgb.copy()
    alpha = 0.35
    for split_name, mask in masks.items():
        if split_name in split_colors:
            color = split_colors[split_name]
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("Split Regions", fontsize=13)
    axes[1].axis("off")

    patches = []
    for s in ["train", "val", "test"]:
        if s in masks:
            n_px = masks[s].sum()
            patches.append(mpatches.Patch(
                color=split_colors[s] / 255, label=f"{s} ({n_px:,} px)"))
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


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


def compute_confusion_matrix(prediction, label, split_mask=None,
                             exclude_background=True):
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="C2Seg Evaluation")
    parser.add_argument("--ckpt_path",     type=str, required=True)
    parser.add_argument("--config_model",  type=str,
                        default="config_test-Atomiser_Atos_One.yaml")
    parser.add_argument("--subset",        type=str, required=True,
                        choices=["germany", "china"])
    parser.add_argument("--sensor",        type=str, nargs="+", required=True)
    parser.add_argument("--fusion",        action="store_true")
    parser.add_argument("--output_dir",    type=str, default="./results/c2seg")

    # City and split overrides
    parser.add_argument("--eval_city",     type=str, default=None,
                        help="City to evaluate on (e.g., augsburg, berlin)")
    parser.add_argument("--eval_mat",      type=str, default=None,
                        help="Override mat file path")
    parser.add_argument("--eval_split",    type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Restrict metrics to this split region only")
    parser.add_argument("--eval_on_train", action="store_true",
                        help="Evaluate on training city (legacy)")

    # Paths
    parser.add_argument("--data_dir",      type=str, default=None)
    parser.add_argument("--processed_dir", type=str,
                        default="./data/CrossCity/c2seg_processed")
    parser.add_argument("--crop_index",    type=str, default=None,
                        help="Override crop index CSV (e.g., with split column)")

    # Inference
    parser.add_argument("--stride_divisor", type=int, default=2)
    parser.add_argument("--eval_crop_size", type=int, default=None,
                        help="Override 10m crop size for sliding window. "
                             "For HySpex: use 28 → 28×28 at 10m → ~127×127 at 2.2m")
    parser.add_argument("--band_norm", action="store_true",
                        help="Apply per-band percentile normalization at inference (diagnostic)")
    parser.add_argument("--norm_mode", type=str, default="raw",
                        choices=["raw", "band_minmax", "zscore"],
                        help="Must match training normalization mode")
    parser.add_argument("--profile", action="store_true",
                        help="Measure GFLOPS per forward pass using PyTorch profiler")
    parser.add_argument("--remap_spectral", type=str, default=None,
                        help="Remap eval sensor spectral indices to nearest bands of "
                             "this reference sensor (e.g., 'hsi'). Diagnostic: tests "
                             "whether spectral encoding mismatch is the bottleneck.")
    parser.add_argument("--spectral_interp_to", type=str, default=None,
                        help="Interpolate eval sensor bands to this training sensor's "
                             "spectral grid before tokenizing (e.g., 'hsi'). "
                             "Same preprocessing as baselines, for fair comparison.")

    args = parser.parse_args()

    if args.data_dir is None:
        subset_dir = "Germany" if args.subset == "germany" else "China"
        args.data_dir = f"./data/CrossCity/{subset_dir}"

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SUBSET_CONFIG[args.subset]

    # ── Resolve eval city and mat ───────────────────────────────────
    if args.eval_city:
        eval_city = args.eval_city
        eval_subset = CITY_SUBSET.get(eval_city, args.subset)
        if args.eval_mat:
            eval_mat = args.eval_mat
        else:
            subset_dir = "Germany" if eval_subset == "germany" else "China"
            eval_mat = os.path.join(f"./data/CrossCity/{subset_dir}",
                                    CITY_MAT.get(eval_city, f"{eval_city}.mat"))
    elif args.eval_on_train:
        eval_city = cfg["train_city"]
        eval_mat = os.path.join(args.data_dir, cfg["train_mat"])
        eval_subset = args.subset
    else:
        eval_city = cfg["test_city"]
        eval_mat = os.path.join(args.data_dir, cfg["test_mat"])
        eval_subset = args.subset

    eval_split = args.eval_split  # None if not specified

    sensors = args.sensor
    if args.fusion and len(sensors) > 1:
        sensors_label = "+".join(sensors)
    else:
        sensors_label = sensors[0]
        args.fusion = False

    split_label = f" [{eval_split} split]" if eval_split else " [full image]"

    print(f"\n{'='*60}")
    print(f"  C2Seg Evaluation")
    print(f"  City:    {eval_city} ({eval_subset})")
    print(f"  Sensors: {sensors_label}")
    print(f"  Split:   {eval_split or 'full image'}")
    print(f"  Ckpt:    {args.ckpt_path}")
    print(f"{'='*60}\n")

    # ── Config ──────────────────────────────────────────────────────
    config_model = read_yaml("./training/configs/" + args.config_model)
    bands_yaml_path = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )

    spectral_meta_path = os.path.join(args.processed_dir, "c2seg_spectral_meta.json")
    crop_index_path = args.crop_index or os.path.join(args.processed_dir, "c2seg_crop_index.csv")
    stats_path = os.path.join(args.processed_dir, "c2seg_norm_stats.json")

    dataset_config = read_yaml(bands_yaml_path)
    c2seg_bands = create_c2seg_bands_info(spectral_meta_path)
    dataset_config.update(c2seg_bands)

    register_all_resolutions(lookup_table)
    register_c2seg_bands(lookup_table, dataset_config)

    # ── Dataset ─────────────────────────────────────────────────────
    eval_sensors = sensors if args.fusion else [sensors[0]]

    dataset = C2SegDataset(
        mat_path=eval_mat,
        subset=eval_subset,
        city=eval_city,
        split=eval_split or "test",
        sensors=eval_sensors,
        crop_index_path=crop_index_path,
        stats_path=stats_path,
        spectral_meta_path=spectral_meta_path,
        look_up=lookup_table,
        dataset_config=dataset_config,
        mode="test",
        augment=False,
        norm_mode=args.norm_mode,
    )

    # ── Spectral index remapping (diagnostic) ───────────────────────
    if args.remap_spectral:
        import json as _json
        with open(spectral_meta_path) as _f:
            _smeta = _json.load(_f)

        ref_key = SENSOR_META_KEY.get((eval_subset, args.remap_spectral))
        if ref_key and ref_key in _smeta:
            ref_wl = np.array(_smeta[ref_key]["wavelengths"])
            ref_info = dataset.sensor_info.get(args.remap_spectral)

            # Get reference sensor's spectral indices
            if ref_info is not None:
                ref_spec_idx = ref_info["spectral_indices"]  # [n_ref_bands]
            else:
                # Build from lookup table
                ref_bw = _smeta[ref_key]["bandwidths"]
                ref_spec_idx = torch.tensor([
                    lookup_table.table_wave.get(
                        (int(round(ref_bw[i])), int(round(ref_wl[i]))), 0)
                    for i in range(len(ref_wl))
                ])

            # For each eval sensor, remap spectral indices
            for sensor in dataset.sensors:
                s_info = dataset.sensor_info[sensor]
                s_key = SENSOR_META_KEY.get((eval_subset, sensor))
                if s_key and s_key in _smeta:
                    s_wl = np.array(_smeta[s_key]["wavelengths"])
                    # Map each eval band to nearest reference band
                    remapped = torch.zeros(len(s_wl), dtype=torch.long)
                    for i, wl in enumerate(s_wl):
                        nearest_idx = np.argmin(np.abs(ref_wl - wl))
                        remapped[i] = ref_spec_idx[nearest_idx]
                    
                    old_unique = len(torch.unique(s_info["spectral_indices"]))
                    new_unique = len(torch.unique(remapped))
                    s_info["spectral_indices"] = remapped
                    print(f"[Eval] Remapped '{sensor}' spectral indices to nearest "
                          f"'{args.remap_spectral}' bands ({old_unique}→{new_unique} unique)")
        else:
            print(f"[Eval] WARNING: reference sensor '{args.remap_spectral}' not found, "
                  f"skipping remap")

    # ── Load model ──────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("state_dict", ckpt)

    model = Model_Pretrain(
        config=config_model, wand=False, name="eval",
        transform=None, lookup_table=lookup_table,
    )
    align_model_to_checkpoint(model, ckpt_state)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ── Spectral interpolation config (fair comparison with baselines) ─
    spectral_interp = None
    if args.spectral_interp_to:
        import json as _json
        with open(spectral_meta_path) as _f:
            _smeta = _json.load(_f)

        target_meta_key = SENSOR_META_KEY.get((eval_subset, args.spectral_interp_to))
        if target_meta_key and target_meta_key in _smeta:
            target_wl = torch.tensor(_smeta[target_meta_key]["wavelengths"],
                                     dtype=torch.float32)
            # Get source sensor wavelengths
            eval_sensors_list = sensors if args.fusion else [sensors[0]]
            src_sensor = eval_sensors_list[0]
            src_meta_key = SENSOR_META_KEY.get((eval_subset, src_sensor))
            source_wl = torch.tensor(_smeta[src_meta_key]["wavelengths"],
                                     dtype=torch.float32)

            # Get target sensor info (spectral indices, resolution, etc.)
            # Build a temporary dataset entry or reuse if available
            target_info = None
            if args.spectral_interp_to in dataset.sensor_info:
                target_info = dataset.sensor_info[args.spectral_interp_to]
            else:
                # Build target info from metadata
                target_bw = _smeta[target_meta_key]["bandwidths"]
                target_spec_idx = torch.tensor([
                    lookup_table.table_wave.get(
                        (int(round(target_bw[i])), int(round(float(target_wl[i])))), 0)
                    for i in range(len(target_wl))
                ])
                target_info = dict(dataset.sensor_info[src_sensor])  # copy base
                target_info["spectral_indices"] = target_spec_idx
                target_info["n_bands"] = len(target_wl)

            spectral_interp = {
                "source_wl": source_wl,
                "target_wl": target_wl,
                "target_info": target_info,
            }

            # Add training sensor's zscore stats for re-normalization
            if args.norm_mode == "zscore":
                train_sensor = args.spectral_interp_to
                if train_sensor in dataset._band_zscore:
                    spectral_interp["train_zscore"] = dataset._band_zscore[train_sensor]
                else:
                    # Load from zscore stats file
                    zscore_path = os.path.join(args.processed_dir,
                                               "c2seg_zscore_stats.json")
                    if os.path.exists(zscore_path):
                        import json as _j2
                        with open(zscore_path) as _f2:
                            _zs = _j2.load(_f2)
                        for zkey in [f"{eval_subset}_{train_sensor}_{eval_city}",
                                     f"{eval_subset}_{train_sensor}"]:
                            if zkey in _zs and "band_mean" in _zs[zkey]:
                                spectral_interp["train_zscore"] = (
                                    torch.tensor(_zs[zkey]["band_mean"], dtype=torch.float32),
                                    torch.tensor(_zs[zkey]["band_std"], dtype=torch.float32),
                                )
                                print(f"[Eval] Training sensor zscore stats loaded ({zkey})")
                                break
            print(f"[Eval] Spectral interpolation: {src_sensor} "
                  f"({len(source_wl)} bands) → {args.spectral_interp_to} "
                  f"({len(target_wl)} bands)")
        else:
            print(f"[Eval] WARNING: target sensor '{args.spectral_interp_to}' "
                  f"not found, skipping interpolation")

    # ── Inference ───────────────────────────────────────────────────
    engine = SlidingWindowInference(
        model=model, dataset=dataset,
        device=device, stride_divisor=args.stride_divisor,
        crop_size_override=args.eval_crop_size,
        band_norm=args.band_norm,
        spectral_interp=spectral_interp,
    )
    result = engine.run()

    # ── FLOPS profiling ─────────────────────────────────────────────
    profile_result = None
    if args.profile:
        profile_result = engine.profile_flops(n_warmup=5, n_active=3)

    # ── Build split mask ────────────────────────────────────────────
    label_shape = result["label"].shape
    split_mask = None
    all_masks = None

    if eval_split:
        split_mask = build_split_mask(
            crop_index_path, eval_city, eval_subset,
            eval_split, label_shape,
        )
        all_masks = build_all_split_masks(
            crop_index_path, eval_city, eval_subset, label_shape,
        )

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = compute_metrics(result["prediction"], result["label"],
                              split_mask=split_mask, exclude_background=True)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  mIoU:  {metrics['mIoU']:>7.4f}  "
          f"({metrics['n_classes_evaluated']} classes, no BG)    │")
    print(f"  │  mF1:   {metrics['mF1']:>7.4f}                            │")
    print(f"  │  OA:    {metrics['overall_accuracy']:>7.4f}                            │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Valid pixels: {metrics['n_valid_pixels']:,}{split_label}")

    print()
    print(f"  {'Class':<30s} {'IoU':>8s} {'F1':>8s} {'Support':>10s}")
    print(f"  {'-'*56}")
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        if cls_id == 0:
            iou_str = f"{info['iou']:.4f}" if info["in_test"] else "   N/A"
            print(f"  {info['name']:<30s} {iou_str:>8s} {'':>8s} "
                  f"{info['support']:>10,d}  ← excluded")
        elif info["in_test"]:
            print(f"  {info['name']:<30s} {info['iou']:>8.4f} {info['f1']:>8.4f} "
                  f"{info['support']:>10,d}")
        else:
            print(f"  {info['name']:<30s} {'N/A':>8s} {'N/A':>8s} "
                  f"{info['support']:>10,d}")

    # ── Save metrics ────────────────────────────────────────────────
    metrics["config"] = {
        "subset": args.subset, "city": eval_city, "split": eval_split,
        "sensors": sensors, "fusion": args.fusion, "ckpt_path": args.ckpt_path,
    }
    if profile_result is not None:
        metrics["profiling"] = profile_result
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → {metrics_path}")

    # ── Visualizations ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Visualizations")
    print(f"{'='*60}\n")

    title = f"C2Seg — {sensors_label} on {eval_city}{split_label}"

    plot_prediction_vs_gt(
        result["prediction"], result["label"],
        os.path.join(args.output_dir, "prediction_vs_gt.png"),
        title=title, metrics=metrics, split_mask=split_mask,
    )

    plot_per_class_iou(
        metrics,
        os.path.join(args.output_dir, "per_class_iou.png"),
        title=title,
    )

    # Confusion matrix
    cm = compute_confusion_matrix(
        result["prediction"], result["label"],
        split_mask=split_mask,
    )
    present = [c for c in range(NUM_CLASSES)
               if metrics["per_class"][c]["in_test"]]

    plot_confusion_matrix(
        cm, os.path.join(args.output_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — {sensors_label}{split_label}",
        classes_present=present if len(present) < NUM_CLASSES else None,
    )

    # Split overlay
    if all_masks:
        plot_split_overlay(
            result["label"], all_masks,
            os.path.join(args.output_dir, "split_overlay.png"),
            title=f"Augsburg Split — Evaluating: {eval_split}",
        )

    # Prediction map
    pred_rgb = label_to_rgb(result["prediction"])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(pred_rgb)
    ax.axis("off")
    plt.savefig(os.path.join(args.output_dir, "prediction_map.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {os.path.join(args.output_dir, 'prediction_map.png')}")

    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()