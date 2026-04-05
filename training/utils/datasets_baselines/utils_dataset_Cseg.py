"""
C2Seg Baseline Dataset
=======================

Fixed-format dataset for baseline models (UNet, ViT+UperNet, ScaleMAE).
Returns {"image": {sensor: [C, H, W]}, "target": [H, W], "metadata": {...}}

No tokenization, no metadata encoding — just normalized imagery.

Each sample loads a single sensor crop + aligned label at 10m resolution.

Normalization:
  Per-band min-max (matching the original C2Seg paper protocol).
  Each band independently mapped to [0, 1] using per-city statistics.
  This is the standard baseline preprocessing — Atomizer-IO uses raw
  reflectance instead, which is its architectural advantage.

Augmentation:
  - D4 (rotation + flip)
  - Spectral band averaging (optional, same as Atomizer-IO for fairness)

Prerequisites:
    - c2seg_crop_index_split.csv (with split column)
    - c2seg_norm_stats.json
    - c2seg_spectral_meta.json
"""

import csv
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 10
IGNORE_INDEX = 255

CLASS_NAMES = {
    0: "Background",
    1: "Urban Fabric",
    2: "Industrial/Commercial",
    3: "Street Network",
    4: "Mine/Dump/Construction",
    5: "Artif. Vegetated",
    6: "Arable Land",
    7: "Low Vegetation",
    8: "Forests",
    9: "Water",
}

# Remap original 14 classes → merged 10 classes
C2SEG_CLASS_REMAP = {
    0: 0,    # Background
    1: 1,    # Urban Fabric
    2: 2,    # Industrial/Commercial
    3: 3,    # Street Network
    4: 4,    # Mine/Dump/Construction
    5: 5,    # Artif. Vegetated
    6: 6,    # Arable Land
    7: 6,    # Permanent Crops → Arable Land
    8: 7,    # Pastures → Low Vegetation
    9: 8,    # Forests
    10: 7,   # Shrub → Low Vegetation
    11: 7,   # Open Spaces → Low Vegetation
    12: 9,   # Inland Wetlands → Water
    13: 9,   # Surface Water → Water
}

CHINA_LABEL_REMAP = {
    -1: IGNORE_INDEX,
    5: 1, 6: 2, 11: 3, 12: 4, 13: 5,
    14: 6, 21: 7, 22: 8, 23: 9,
    31: 10, 32: 11, 33: 12, 41: 13,
}

MAT_KEYS = {"hsi": "HSI", "msi": "MSI", "sar": "SAR", "label": "label"}

SENSOR_META_KEY = {
    ("germany", "hsi"): "germany_hsi",
    ("germany", "msi"): "germany_msi",
    ("germany", "msi12"): "germany_msi12",
    ("germany", "sar"): "germany_sar",
    ("germany", "hyspex"): "germany_hyspex",
    ("germany", "hyspex_10m"): "germany_hyspex_10m",
    ("germany", "enmap_30m"): "germany_enmap_30m",
    ("china", "hsi"): "china_hsi",
    ("china", "msi"): "china_msi",
    ("china", "sar"): "china_sar",
}

AXIS_ORDER = {"germany": "HWC", "china": "CHW"}

SENSOR_GSD = {
    ("germany", "hsi"): 10.0, ("germany", "msi"): 10.0,
    ("germany", "msi12"): 10.0,
    ("germany", "sar"): 10.0, ("germany", "hyspex"): 2.2,
    ("germany", "hyspex_10m"): 10.0,
    ("germany", "enmap_30m"): 30.0,
    ("china", "hsi"): 30.0, ("china", "msi"): 10.0, ("china", "sar"): 10.0,
}

# Sensors that load from aligned .npy files instead of the mat file
NPY_SENSORS = {
    ("germany", "msi12"):     "aligned/sentinel2_aligned.npy",
    ("germany", "enmap_30m"): "aligned/enmap_30m_aligned.npy",
    ("germany", "hyspex"):      "aligned/hyspex_aligned.npy",
    ("germany", "hyspex_10m"):  "aligned/hyspex_10m_aligned.npy",
}


# ═══════════════════════════════════════════════════════════════════════
# MAT FILE READER (shared with Atomizer dataset)
# ═══════════════════════════════════════════════════════════════════════

class MatFileReader:
    """Lazy reader for .mat files. Supports v5/v7 and HDF5."""

    def __init__(self, path: str):
        self.path = path
        self._data = None
        self._format = None
        self._shapes = {}

    def _open(self):
        if self._data is not None:
            return
        try:
            self._data = sio.loadmat(self.path)
            self._format = "scipy"
            for k, v in self._data.items():
                if not k.startswith("_") and hasattr(v, "shape"):
                    self._shapes[k] = v.shape
        except NotImplementedError:
            self._data = h5py.File(self.path, "r")
            self._format = "h5py"
            for k in self._data.keys():
                self._shapes[k] = self._data[k].shape

    def get_shape(self, key: str) -> tuple:
        self._open()
        return self._shapes[key]

    def read_crop(self, key: str, r0: int, c0: int, h: int, w: int,
                  axis_order: str = "auto") -> np.ndarray:
        self._open()
        arr = self._data[key]
        shape = arr.shape

        if axis_order == "auto":
            if len(shape) == 3:
                axis_order = "HWC" if shape[2] < shape[0] and shape[2] < shape[1] else "CHW"
            elif len(shape) == 2:
                axis_order = "HW"

        if axis_order == "HWC":
            crop = arr[r0:r0 + h, c0:c0 + w, :]
            crop = np.array(crop) if not isinstance(crop, np.ndarray) else crop
            crop = crop.transpose(2, 0, 1)  # → CHW
        elif axis_order == "CHW":
            crop = arr[:, r0:r0 + h, c0:c0 + w]
            crop = np.array(crop) if not isinstance(crop, np.ndarray) else crop
        elif axis_order == "HW":
            crop = arr[r0:r0 + h, c0:c0 + w]
            crop = np.array(crop) if not isinstance(crop, np.ndarray) else crop
            return crop.astype(np.float32 if crop.dtype != np.int8 else np.int64)
        else:
            raise ValueError(f"Unknown axis_order: {axis_order}")

        return crop.astype(np.float32)

    def read_label_crop(self, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        self._open()
        crop = self._data["label"][r0:r0 + h, c0:c0 + w]
        return np.array(crop) if not isinstance(crop, np.ndarray) else crop

    def close(self):
        if self._data is not None and self._format == "h5py":
            self._data.close()
        self._data = None

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._data = None
        self._format = None
        self._shapes = {}

    def __del__(self):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

def augment_d4(image: torch.Tensor, label: torch.Tensor):
    """Random D4 symmetry group (4 rotations × 2 flips)."""
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    if random.random() > 0.5:
        image = torch.flip(image, dims=(-1,))
        label = torch.flip(label, dims=(-1,))
    return image, label


# ═══════════════════════════════════════════════════════════════════════
# BASELINE DATASET
# ═══════════════════════════════════════════════════════════════════════

class C2SegBaselineDataset(Dataset):
    """
    C2Seg dataset for baseline segmentation models (UNet, ViT+UperNet).

    Returns:
        {
            "image": {sensor_name: [C, H, W]},
            "target": [H, W],
            "metadata": {
                "sensor": str,
                "subset": str,
                "city": str,
                "gsd": float,
                "n_bands": int,
                "wavelengths": list,
                "bandwidths": list,
            }
        }

    Normalization: per-band min-max (original C2Seg protocol).
    Augmentation: D4 + spectral band averaging.
    """

    def __init__(
        self,
        mat_path: str,
        subset: str,
        city: str,
        split: str,
        sensor: str,
        crop_index_path: str,
        stats_path: str,
        spectral_meta_path: str,
        mode: str = "train",
        augment: bool = True,
        crop_size: int = 128,
        norm_mode: str = "band_minmax",
    ):
        super().__init__()

        self.mat_path = mat_path
        self.subset = subset
        self.city = city
        self.split = split
        self.sensor = sensor
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.crop_size = crop_size

        self.axis_order = AXIS_ORDER[subset]
        self.needs_label_remap = (subset == "china")

        # ── Mat file reader ─────────────────────────────────────────
        self.reader = MatFileReader(mat_path)
        self._detect_label_dims()

        # ── Spectral metadata ───────────────────────────────────────
        with open(spectral_meta_path, "r") as f:
            all_meta = json.load(f)

        meta_key = SENSOR_META_KEY[(subset, sensor)]
        self.sensor_meta = all_meta[meta_key]
        self.gsd = SENSOR_GSD[(subset, sensor)]
        self.n_bands = self.sensor_meta["n_bands"]
        self.wavelengths = self.sensor_meta["wavelengths"]
        self.bandwidths = self.sensor_meta["bandwidths"]
        self.mat_key = MAT_KEYS.get(sensor, None)

        # Check if sensor loads from NPY instead of mat
        npy_key = (subset, sensor)
        self.is_npy = npy_key in NPY_SENSORS
        self._npy_data = None

        if self.is_npy:
            data_dir = os.path.dirname(mat_path)
            npy_path = os.path.join(data_dir, NPY_SENSORS[npy_key])
            self._npy_data = np.load(npy_path, mmap_mode="r")
            print(f"[C2Seg-BL] Sensor '{sensor}' ({meta_key}): "
                  f"{self.n_bands} bands, {self.gsd}m [NPY: {npy_path}]")
        else:
            print(f"[C2Seg-BL] Sensor '{sensor}' ({meta_key}): "
                  f"{self.n_bands} bands, {self.gsd}m")

        # ── Normalization ──────────────────────────────────────────────
        # "band_minmax": per-band min-max (original C2Seg protocol)
        # "zscore":      per-band z-score → rescale to [0,1]
        # "identity":    no normalization (raw values)
        self.requested_norm_mode = norm_mode
        self.norm_mode = "identity"  # fallback

        if norm_mode == "zscore":
            zscore_path = os.path.join(
                os.path.dirname(stats_path), "c2seg_zscore_stats.json")
            if os.path.exists(zscore_path):
                with open(zscore_path, "r") as f:
                    all_zstats = json.load(f)

                for stat_key in [f"{subset}_{sensor}_{city}",
                                 f"{subset}_{sensor}"]:
                    if stat_key in all_zstats and "band_mean" in all_zstats[stat_key]:
                        entry = all_zstats[stat_key]
                        self.norm_mean = torch.tensor(
                            entry["band_mean"], dtype=torch.float32)[:self.n_bands]
                        self.norm_std = torch.tensor(
                            entry["band_std"], dtype=torch.float32)[:self.n_bands]
                        self.norm_mode = "zscore"
                        print(f"[C2Seg-BL] Zscore normalization ({stat_key})")
                        break
                else:
                    print(f"[C2Seg-BL] WARNING: no zscore stats for '{sensor}', "
                          f"falling back to identity")
            else:
                print(f"[C2Seg-BL] WARNING: {zscore_path} not found, "
                      f"falling back to identity")

        elif norm_mode == "band_minmax":
            with open(stats_path, "r") as f:
                all_stats = json.load(f)

            stat_key = f"{subset}_{sensor}_{city}"
            if stat_key in all_stats:
                entry = all_stats[stat_key]
                if "band_min" in entry and "band_max" in entry:
                    band_min = torch.tensor(entry["band_min"], dtype=torch.float32)
                    band_max = torch.tensor(entry["band_max"], dtype=torch.float32)
                    self.norm_min = band_min[:self.n_bands]
                    self.norm_range = torch.clamp(
                        band_max[:self.n_bands] - band_min[:self.n_bands], min=1e-6)
                    self.norm_mode = "band_minmax"
                    print(f"[C2Seg-BL] Per-band min-max normalization ({stat_key})")
                else:
                    print(f"[C2Seg-BL] WARNING: no band_min/max for '{stat_key}', "
                          f"using identity")
            else:
                print(f"[C2Seg-BL] WARNING: no stats for '{stat_key}', using identity")

        else:
            print(f"[C2Seg-BL] Normalization: identity (raw values)")

        # ── Crop index ──────────────────────────────────────────────
        self.crops = self._load_crop_index(crop_index_path)
        print(f"[C2Seg-BL] {len(self.crops)} crops for {subset}/{city} "
              f"(split={split}, sensor={sensor})")

    # ═════════════════════════════════════════════════════════════════
    # INIT HELPERS
    # ═════════════════════════════════════════════════════════════════

    def _detect_label_dims(self):
        try:
            shape = self.reader.get_shape("label")
            self.label_dims = (shape[0], shape[1])
        except Exception:
            self.label_dims = None
        self.reader.close()

    def _load_crop_index(self, csv_path: str) -> List[dict]:
        crops = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            has_split = "split" in (reader.fieldnames or [])
            for row in reader:
                if row["city"] != self.city or row["subset"] != self.subset:
                    continue
                if has_split and row.get("split", "") != self.split:
                    continue
                crops.append({
                    "row_10m": int(row["row_10m"]),
                    "col_10m": int(row["col_10m"]),
                    "crop_h": int(row["crop_h"]),
                    "crop_w": int(row["crop_w"]),
                    "hsi_row": int(row["hsi_row"]),
                    "hsi_col": int(row["hsi_col"]),
                    "hsi_crop_h": int(row["hsi_crop_h"]),
                    "hsi_crop_w": int(row["hsi_crop_w"]),
                })
        return crops

    # ═════════════════════════════════════════════════════════════════
    # SUB-CROP (for China 256→128)
    # ═════════════════════════════════════════════════════════════════

    def _subcrop(self, crop: dict) -> dict:
        target = self.crop_size
        crop_h, crop_w = crop["crop_h"], crop["crop_w"]

        if crop_h <= target and crop_w <= target:
            return crop

        max_dr = max(0, crop_h - target)
        max_dc = max(0, crop_w - target)

        if self.augment:
            dr = random.randint(0, max_dr) if max_dr > 0 else 0
            dc = random.randint(0, max_dc) if max_dc > 0 else 0
        else:
            dr = max_dr // 2
            dc = max_dc // 2

        hsi_h, hsi_w = crop["hsi_crop_h"], crop["hsi_crop_w"]
        scale_h = hsi_h / crop_h
        scale_w = hsi_w / crop_w

        return {
            "row_10m": crop["row_10m"] + dr,
            "col_10m": crop["col_10m"] + dc,
            "crop_h": target,
            "crop_w": target,
            "hsi_row": crop["hsi_row"] + int(dr * scale_h),
            "hsi_col": crop["hsi_col"] + int(dc * scale_w),
            "hsi_crop_h": max(1, int(target * scale_h)),
            "hsi_crop_w": max(1, int(target * scale_w)),
        }

    # ═════════════════════════════════════════════════════════════════
    # READING
    # ═════════════════════════════════════════════════════════════════

    def _read_sensor_crop(self, crop: dict) -> torch.Tensor:
        """Read sensor crop and apply per-band min-max normalization.
        Supports both mat file and aligned NPY sources.
        Handles any resolution by scaling from 10m coordinates."""

        # Compute crop coordinates by scaling from 10m reference
        if self.gsd != 10.0:
            scale = 10.0 / self.gsd  # >1 for finer (2.2m), <1 for coarser (30m)
            r0 = int(crop["row_10m"] * scale)
            c0 = int(crop["col_10m"] * scale)
            h = max(1, int(crop["crop_h"] * scale))
            w = max(1, int(crop["crop_w"] * scale))
        else:
            r0, c0 = crop["row_10m"], crop["col_10m"]
            h, w = crop["crop_h"], crop["crop_w"]

        # Read from NPY or mat
        if self.is_npy and self._npy_data is not None:
            npy_arr = self._npy_data  # [C, H_full, W_full]
            r1 = min(r0 + h, npy_arr.shape[1])
            c1 = min(c0 + w, npy_arr.shape[2])
            r0 = max(0, r0)
            c0 = max(0, c0)
            data = np.array(npy_arr[:, r0:r1, c0:c1], dtype=np.float32)
            data = torch.from_numpy(data)
        else:
            data = self.reader.read_crop(self.mat_key, r0, c0, h, w,
                                         axis_order=self.axis_order)
            data = torch.from_numpy(data)

        if data.shape[0] > self.n_bands:
            data = data[:self.n_bands]

        # Per-band normalization
        if self.norm_mode == "band_minmax":
            n = min(data.shape[0], len(self.norm_min))
            data[:n] = (data[:n] - self.norm_min[:n, None, None]) / self.norm_range[:n, None, None]
        elif self.norm_mode == "zscore":
            n = min(data.shape[0], len(self.norm_mean))
            data[:n] = (data[:n] - self.norm_mean[:n, None, None]) / self.norm_std[:n, None, None]
            data = (data + 3.0) / 6.0  # map ±3σ → [0, 1]

        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        data = torch.clamp(data, min=-0.5, max=1.5)

        return data

    def _read_label_crop(self, crop: dict) -> torch.Tensor:
        r0, c0 = crop["row_10m"], crop["col_10m"]
        h, w = crop["crop_h"], crop["crop_w"]
        data = self.reader.read_label_crop(r0, c0, h, w)
        label = torch.from_numpy(data.astype(np.int64))
        if self.needs_label_remap:
            remapped = torch.full_like(label, IGNORE_INDEX)
            for raw_val, new_val in CHINA_LABEL_REMAP.items():
                remapped[label == raw_val] = new_val
            label = remapped
        # Merge classes (14 → 10)
        merged = torch.full_like(label, IGNORE_INDEX)
        for old_val, new_val in C2SEG_CLASS_REMAP.items():
            merged[label == old_val] = new_val
        label = merged
        return label

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index: int) -> dict:
        crop = self.crops[index]
        crop = self._subcrop(crop)

        try:
            image = self._read_sensor_crop(crop)
            label = self._read_label_crop(crop)
        except Exception as e:
            print(f"[C2Seg-BL] Error at crop {index}: {e}")
            image = torch.zeros(self.n_bands, self.crop_size, self.crop_size)
            label = torch.full((self.crop_size, self.crop_size),
                               IGNORE_INDEX, dtype=torch.long)

        # ── D4 augmentation ─────────────────────────────────────────
        if self.augment:
            image, label = augment_d4(image, label)

        # NOTE: Spectral augmentation (band merging) is handled in the
        # collate function (get_augmented_collate_fn), not here.
        # This keeps batch_size > 1 working without shape mismatches.

        return {
            "image": {self.sensor: image},
            "target": label,
            "metadata": {
                "sensor": self.sensor,
                "subset": self.subset,
                "city": self.city,
                "gsd": self.gsd,
                "n_bands": self.n_bands,
                "wavelengths": self.wavelengths,
                "bandwidths": self.bandwidths,
            },
        }