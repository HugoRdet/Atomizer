"""
C2Seg Segmentation Dataset for Atomizer-IO
============================================

Cross-city, cross-sensor, cross-continent semantic segmentation on C2Seg.

Query Augmentation:
  During training, queries are split between two resolutions:
    - Standard queries at 10m (label resolution)
    - Upsampled boundary-focused queries at 2.5m (4× finer)
  The upsampled queries are sampled preferentially from class boundaries,
  teaching the decoder sub-pixel precision at transitions. The total
  number of queries is always fixed (max_queries) for batch consistency.

Normalization:
  RAW REFLECTANCE — no per-city or per-band normalization.
  - If raw data max > 100, it's raw DN → divide by 10000 to get reflectance.
  - If raw data max <= 100, it's already reflectance → no conversion.
  - Clamp to [0, 1].
  
  This preserves physical comparability across cities and sensors:
  the same material at the same wavelength produces the same input value
  regardless of city or satellite, enabling cross-city/cross-sensor transfer
  without domain adaptation.

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7
"""

import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

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

from .token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 10
IGNORE_INDEX = 255
DATASET_NAME = "C2Seg"

# Raw DN detection: if max pixel value > this, data is raw DN (not reflectance)
RAW_DN_THRESHOLD = 100.0
REFLECTANCE_SCALE = 10000.0

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
# Applied to labels when loading from .mat
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
STATIC_TIME_IDX = -1


# ═══════════════════════════════════════════════════════════════════════
# BOUNDARY DETECTION & UPSAMPLED QUERY SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def detect_boundaries(label: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    Detect boundary pixels where at least one 4-connected neighbor
    has a different class.

    Returns [H, W] bool mask.
    """
    H, W = label.shape
    valid = (label != ignore_index)

    padded = F.pad(label.unsqueeze(0).float(), (1, 1, 1, 1), mode="replicate").squeeze(0).long()

    center = label
    up = padded[0:H, 1:W+1]
    down = padded[2:H+2, 1:W+1]
    left = padded[1:H+1, 0:W]
    right = padded[1:H+1, 2:W+2]

    boundary = (
        ((center != up) & valid & (up != ignore_index)) |
        ((center != down) & valid & (down != ignore_index)) |
        ((center != left) & valid & (left != ignore_index)) |
        ((center != right) & valid & (right != ignore_index))
    )
    return boundary


def dilate_boundary(boundary: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """Dilate boundary mask to include a context band around transitions."""
    if dilation <= 0:
        return boundary
    kernel = 2 * dilation + 1
    dilated = F.max_pool2d(
        boundary.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel, stride=1, padding=dilation,
    ).squeeze(0).squeeze(0)
    return dilated > 0


def sample_boundary_focused_indices(
    label: torch.Tensor,
    n_queries: int,
    boundary_fraction: float = 0.7,
    dilation: int = 2,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Sample pixel indices focused on class boundaries.

    Returns [n_queries, 2] tensor of (row, col) indices.
    """
    valid = (label != ignore_index)
    boundary = detect_boundaries(label, ignore_index)
    boundary_zone = dilate_boundary(boundary, dilation)

    boundary_valid = boundary_zone & valid
    interior_valid = (~boundary_zone) & valid

    boundary_indices = boundary_valid.nonzero(as_tuple=False)
    interior_indices = interior_valid.nonzero(as_tuple=False)

    n_boundary = boundary_indices.shape[0]
    n_interior = interior_indices.shape[0]

    n_boundary_target = int(n_queries * boundary_fraction)
    n_interior_target = n_queries - n_boundary_target

    # Adjust if not enough pixels in either pool
    if n_boundary < n_boundary_target:
        n_boundary_target = n_boundary
        n_interior_target = n_queries - n_boundary_target
    if n_interior < n_interior_target:
        n_interior_target = n_interior
        n_boundary_target = min(n_boundary, n_queries - n_interior_target)

    total_available = n_boundary_target + n_interior_target
    if total_available == 0:
        all_valid = valid.nonzero(as_tuple=False)
        if all_valid.shape[0] == 0:
            return torch.zeros(n_queries, 2, dtype=torch.long)
        perm = torch.randperm(all_valid.shape[0])[:n_queries]
        if perm.shape[0] < n_queries:
            # Repeat if not enough valid pixels
            repeats = (n_queries // perm.shape[0]) + 1
            perm = perm.repeat(repeats)[:n_queries]
        return all_valid[perm]

    selected = []
    if n_boundary_target > 0:
        perm = torch.randperm(n_boundary)[:n_boundary_target]
        selected.append(boundary_indices[perm])
    if n_interior_target > 0:
        perm = torch.randperm(n_interior)[:n_interior_target]
        selected.append(interior_indices[perm])

    indices = torch.cat(selected, dim=0)

    # Fill any deficit with random valid pixels
    if indices.shape[0] < n_queries:
        all_valid = valid.nonzero(as_tuple=False)
        deficit = n_queries - indices.shape[0]
        if all_valid.shape[0] > 0:
            perm = torch.randperm(all_valid.shape[0])[:deficit]
            # If still not enough valid pixels, sample with replacement
            if perm.shape[0] < deficit:
                perm = torch.randint(0, all_valid.shape[0], (deficit,))
            indices = torch.cat([indices, all_valid[perm]], dim=0)
        else:
            # No valid pixels at all — fill with zeros
            indices = torch.cat([indices, torch.zeros(deficit, 2, dtype=torch.long)], dim=0)

    # Shuffle
    perm = torch.randperm(indices.shape[0])[:n_queries]
    return indices[perm]


# ═══════════════════════════════════════════════════════════════════════
# MAT FILE READER
# ═══════════════════════════════════════════════════════════════════════

class MatFileReader:
    """
    Lazy reader for .mat files. Opens file on first read, caches handle.
    Supports both v5/v7 (scipy) and v7.3/HDF5 (h5py) formats.
    """

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
            crop = crop.transpose(2, 0, 1)
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
# D4 AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════

def augment_d4(image, label):
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    if random.random() > 0.5:
        image = torch.flip(image, dims=(-1,))
        label = torch.flip(label, dims=(-1,))
    return image, label


# ═══════════════════════════════════════════════════════════════════════
# RESOLUTION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════

def downsample_image(image: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return image
    return F.avg_pool2d(image.unsqueeze(0), kernel_size=factor, stride=factor).squeeze(0)


def downsample_label_majority(label: torch.Tensor, factor: int,
                               n_classes: int = NUM_CLASSES) -> torch.Tensor:
    if factor <= 1:
        return label
    H, W = label.shape
    new_h, new_w = H // factor, W // factor
    cropped = label[:new_h * factor, :new_w * factor]
    blocked = cropped.reshape(new_h, factor, new_w, factor)
    blocked = blocked.permute(0, 2, 1, 3).reshape(new_h * new_w, factor * factor)
    valid_mask = (blocked != IGNORE_INDEX)
    safe = blocked.clone()
    safe[~valid_mask] = 0
    counts = torch.zeros(new_h * new_w, n_classes + 1, dtype=torch.long)
    counts.scatter_add_(1, safe.long(), valid_mask.long())
    counts = counts[:, :n_classes]
    has_valid = counts.sum(dim=1) > 0
    result = torch.full((new_h * new_w,), IGNORE_INDEX, dtype=label.dtype)
    result[has_valid] = counts[has_valid].argmax(dim=1).to(label.dtype)
    return result.reshape(new_h, new_w)


# ═══════════════════════════════════════════════════════════════════════
# BANDS INFO FACTORY
# ═══════════════════════════════════════════════════════════════════════

def create_c2seg_bands_info(spectral_meta_path: str) -> dict:
    with open(spectral_meta_path, "r") as f:
        meta = json.load(f)
    bands_info = {}
    for sensor_key, sensor_meta in meta.items():
        config_key = f"bands_c2seg_{sensor_key}"
        bands = {}
        wavelengths = sensor_meta["wavelengths"]
        bandwidths = sensor_meta["bandwidths"]
        for i, (wl, bw) in enumerate(zip(wavelengths, bandwidths)):
            bands[f"band_{i:03d}"] = {
                "central_wavelength": int(round(wl)),
                "bandwidth": int(round(bw)),
                "idx": i,
            }
        bands_info[config_key] = bands
    return bands_info


def register_c2seg_bands(look_up, dataset_config: dict):
    n_new = 0
    for key, bands_info in dataset_config.items():
        if not key.startswith("bands_c2seg_"):
            continue
        for band_name, data in bands_info.items():
            if not all(k in data for k in ("bandwidth", "central_wavelength")):
                continue
            bw = int(data["bandwidth"])
            wl = int(data["central_wavelength"])
            wave_key = (bw, wl)
            if wave_key not in look_up.table_wave:
                look_up.table_wave[wave_key] = len(look_up.table_wave)
                n_new += 1
    print(f"[C2Seg] Pre-registered {n_new} new bands into lookup table "
          f"(total: {len(look_up.table_wave)})")
    return n_new


def preregister_spectral_merges(
    look_up,
    spectral_meta_path: str,
    groups: List[int] = None,
    sensors: List[str] = None,
    n_random_samples: int = 500,
):
    """
    Pre-register merged (bandwidth, wavelength) pairs that spectral
    augmentation can produce. Must be called BEFORE model creation
    so the embedding table is large enough.

    Builds the same pool used during training and registers all
    unique (bw, wl) pairs encountered.
    """
    pool = build_spectral_aug_pool(n_random=n_random_samples)

    n_new = 0
    for config in pool:
        for sim_wl, sim_bw in config:
            wave_key = (int(round(sim_bw)), int(round(sim_wl)))
            if wave_key not in look_up.table_wave:
                look_up.table_wave[wave_key] = len(look_up.table_wave)
                n_new += 1

    print(f"[C2Seg] Pre-registered {n_new} spectral entries from "
          f"{len(pool)} pooled configs "
          f"(total: {len(look_up.table_wave)})")
    return pool  # Return pool so dataset uses the SAME configs


# ═══════════════════════════════════════════════════════════════════════
# SUBSET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

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

# Sensors that load from aligned .npy files instead of the mat file.
# Path is relative to data_dir (e.g., ./data/CrossCity/Germany/).
# Format: [C, H, W] float32, aligned to label grid.
NPY_SENSORS = {
    ("germany", "msi12"):     "aligned/sentinel2_aligned.npy",
    ("germany", "enmap_30m"): "aligned/enmap_30m_aligned.npy",
    ("germany", "hyspex"):      "aligned/hyspex_aligned.npy",
    ("germany", "hyspex_10m"):  "aligned/hyspex_10m_aligned.npy",
}

# Corresponding validity masks (True = real pixel, False = padding)
NPY_MASKS = {
    ("germany", "msi12"):     "aligned/sentinel2_mask.npy",
    ("germany", "enmap_30m"): "aligned/enmap_30m_mask.npy",
    ("germany", "hyspex"):      "aligned/hyspex_mask.npy",
    ("germany", "hyspex_10m"):  "aligned/hyspex_10m_mask.npy",
}

MAT_KEYS = {"hsi": "HSI", "msi": "MSI", "sar": "SAR", "label": "label"}


# ═══════════════════════════════════════════════════════════════════════
# GENERATIVE SPECTRAL AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════
# Instead of hardcoding specific sensor configurations (which would let
# reviewers argue we "targeted" test sensors), we define spectral ANCHOR
# REGIONS where real satellites typically place bands — absorption features,
# vegetation edges, atmospheric windows — and randomly sample from them.
#
# Each augmented sample gets a unique, never-before-seen sensor config
# that is physically plausible but never exactly S2, Landsat, or MODIS.
#
# Anchor format: (center_nm, spread_nm, bw_min_nm, bw_max_nm, weight)
#   center:  typical wavelength for this feature
#   spread:  how much to jitter the center (±spread)
#   bw_min/max: bandwidth range (narrow vs broad)
#   weight:  relative probability of including this anchor

SPECTRAL_ANCHORS = [
    # VNIR region — densely sampled by most sensors
    {"name": "coastal",    "center": 443,  "spread": 15, "bw_min": 10, "bw_max": 30,  "weight": 0.6},
    {"name": "blue",       "center": 490,  "spread": 20, "bw_min": 20, "bw_max": 80,  "weight": 0.9},
    {"name": "green",      "center": 560,  "spread": 20, "bw_min": 15, "bw_max": 60,  "weight": 0.9},
    {"name": "yellow",     "center": 610,  "spread": 15, "bw_min": 15, "bw_max": 50,  "weight": 0.4},
    {"name": "red",        "center": 665,  "spread": 15, "bw_min": 15, "bw_max": 50,  "weight": 0.9},
    # Red edge — critical for vegetation, narrow bands
    {"name": "red_edge_1", "center": 705,  "spread": 10, "bw_min": 8,  "bw_max": 20,  "weight": 0.7},
    {"name": "red_edge_2", "center": 740,  "spread": 10, "bw_min": 8,  "bw_max": 25,  "weight": 0.6},
    {"name": "red_edge_3", "center": 783,  "spread": 10, "bw_min": 10, "bw_max": 30,  "weight": 0.5},
    # NIR plateau
    {"name": "nir_broad",  "center": 842,  "spread": 30, "bw_min": 20, "bw_max": 150, "weight": 0.9},
    {"name": "nir_narrow", "center": 865,  "spread": 15, "bw_min": 10, "bw_max": 30,  "weight": 0.5},
    # Water vapour / O2 absorption
    {"name": "wv_945",     "center": 945,  "spread": 20, "bw_min": 10, "bw_max": 30,  "weight": 0.3},
    {"name": "nir_edge",   "center": 1020, "spread": 30, "bw_min": 20, "bw_max": 60,  "weight": 0.3},
    # SWIR — important for water, minerals, soil
    {"name": "swir_1200",  "center": 1240, "spread": 40, "bw_min": 15, "bw_max": 40,  "weight": 0.4},
    {"name": "swir_1",     "center": 1610, "spread": 50, "bw_min": 40, "bw_max": 120, "weight": 0.7},
    {"name": "swir_2",     "center": 2190, "spread": 60, "bw_min": 50, "bw_max": 200, "weight": 0.6},
]

# How many bands to sample: (min_bands, max_bands)
RANDOM_SENSOR_N_BANDS = (4, 25)


def generate_random_sensor_config() -> List[Tuple[float, float]]:
    """
    Generate a random, physically plausible sensor configuration by
    sampling from spectral anchor regions.

    Returns list of (center_wavelength_nm, bandwidth_nm) tuples, sorted
    by wavelength. Each call produces a unique config.
    """
    n_min, n_max = RANDOM_SENSOR_N_BANDS
    n_bands = random.randint(n_min, n_max)

    # Sample anchors weighted by their probability
    weights = [a["weight"] for a in SPECTRAL_ANCHORS]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Sample with replacement allowed (sensor might have 2 NIR bands)
    chosen_indices = random.choices(range(len(SPECTRAL_ANCHORS)),
                                    weights=probs, k=n_bands)

    bands = []
    for idx in chosen_indices:
        anchor = SPECTRAL_ANCHORS[idx]

        # Jitter center wavelength
        center = anchor["center"] + random.gauss(0, anchor["spread"] * 0.5)
        center = max(380, min(2500, center))

        # Sample bandwidth (log-uniform between min and max)
        log_bw_min = np.log(anchor["bw_min"])
        log_bw_max = np.log(anchor["bw_max"])
        bw = np.exp(random.uniform(log_bw_min, log_bw_max))
        bw = max(5, bw)

        bands.append((round(center, 1), round(bw, 1)))

    # Sort by wavelength, remove near-duplicates (within 5nm)
    bands.sort(key=lambda b: b[0])
    filtered = [bands[0]]
    for b in bands[1:]:
        if abs(b[0] - filtered[-1][0]) > 5:
            filtered.append(b)

    return filtered


# Named sensor templates (used as seeds for jittered variants, NOT exact copies)
_NAMED_SENSOR_TEMPLATES = {
    "s2_like": [
        (443, 20), (490, 65), (560, 35), (665, 30),
        (705, 15), (740, 15), (783, 20),
        (842, 115), (865, 20), (945, 20),
        (1610, 90), (2190, 180),
    ],
    "landsat_like": [
        (443, 16), (482, 60), (561, 57), (655, 37),
        (865, 28), (1609, 85), (2201, 187),
    ],
    "modis_like": [
        (645, 50), (858, 35), (469, 20), (555, 20),
        (1240, 20), (1640, 24), (2130, 50),
    ],
    "olci_like": [
        (400, 15), (443, 10), (490, 10), (560, 10),
        (665, 10), (709, 10), (779, 15), (865, 20),
        (940, 20), (1020, 40),
    ],
    "wv3_like": [
        (425, 50), (480, 60), (545, 70), (605, 40),
        (660, 60), (725, 40), (832, 125), (950, 180),
    ],
    "planet_like": [
        (443, 20), (490, 65), (531, 36), (565, 36),
        (665, 31), (705, 15), (740, 18), (842, 115),
    ],
}


def _jitter_sensor_config(
    bands: List[Tuple[float, float]],
    wl_noise: float = 10.0,
    bw_noise_frac: float = 0.2,
) -> List[Tuple[float, float]]:
    """Jitter a named sensor config so it's never an exact copy."""
    jittered = []
    for wl, bw in bands:
        new_wl = wl + random.gauss(0, wl_noise)
        new_bw = bw * (1.0 + random.gauss(0, bw_noise_frac))
        new_wl = max(380, min(2500, new_wl))
        new_bw = max(5, new_bw)
        jittered.append((round(new_wl, 1), round(new_bw, 1)))
    return jittered


def build_spectral_aug_pool(n_random: int = 500) -> List[List[Tuple[float, float]]]:
    """
    Build a large pool of diverse sensor configurations for augmentation.

    Sources:
      1. Jittered named sensors (S2-like, Landsat-like, etc.) — 10 variants each
      2. Random anchor-sampled configs — n_random unique configs
      3. Uniform group configs (4-128 equal groups from 242 EnMAP bands)

    Returns list of configs, each a list of (wavelength, bandwidth) tuples.
    """
    pool = []

    # ── 1. Jittered named sensors (never exact copies) ──────────────
    for name, template in _NAMED_SENSOR_TEMPLATES.items():
        for _ in range(10):
            pool.append(_jitter_sensor_config(template))

    # ── 2. Random anchor-sampled configs ────────────────────────────
    for _ in range(n_random):
        pool.append(generate_random_sensor_config())

    # ── 3. Uniform group configs (as band lists) ───────────────────
    # These represent "split spectrum into N equal parts" — still useful
    # for teaching the model about arbitrary groupings
    enmap_range = (420, 2450)  # approximate EnMAP range
    for n_groups in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        span = (enmap_range[1] - enmap_range[0]) / n_groups
        bands = []
        for g in range(n_groups):
            center = enmap_range[0] + span * (g + 0.5)
            # Add some jitter so not perfectly uniform
            center += random.gauss(0, span * 0.1)
            bw = span * (0.8 + random.random() * 0.4)  # 80-120% of span
            bands.append((round(center, 1), round(bw, 1)))
        pool.append(bands)

    # Shuffle for good measure
    random.shuffle(pool)
    return pool


# ═══════════════════════════════════════════════════════════════════════
# RAW DN DETECTION (per mat file, cached)
# ═══════════════════════════════════════════════════════════════════════

# Cache: mat_path → {sensor_key: bool}
_IS_RAW_DN_CACHE: Dict[str, Dict[str, bool]] = {}


def _detect_raw_dn(reader: MatFileReader, sensor_key: str, mat_path: str,
                    axis_order: str) -> bool:
    """
    Detect whether a sensor in a mat file contains raw DN values (max > 100)
    or reflectance (max <= 1). Reads a small crop to check efficiently.
    Caches result per (mat_path, sensor_key).
    """
    cache_key = mat_path
    if cache_key in _IS_RAW_DN_CACHE and sensor_key in _IS_RAW_DN_CACHE[cache_key]:
        return _IS_RAW_DN_CACHE[cache_key][sensor_key]

    # Read a small crop to check scale
    try:
        crop = reader.read_crop(MAT_KEYS[sensor_key], 0, 0, 64, 64,
                                axis_order=axis_order)
        is_raw = float(crop.max()) > RAW_DN_THRESHOLD
    except Exception:
        is_raw = False

    if cache_key not in _IS_RAW_DN_CACHE:
        _IS_RAW_DN_CACHE[cache_key] = {}
    _IS_RAW_DN_CACHE[cache_key][sensor_key] = is_raw
    return is_raw


# ═══════════════════════════════════════════════════════════════════════
# MAIN DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════

class C2SegDataset(Dataset):
    """
    C2Seg segmentation dataset for Atomizer-IO.

    Normalization: Raw reflectance only.
      - Raw DN (max > 100) → ÷10000 → reflectance
      - Already reflectance → no-op
      - Clamp [0, 1]
    No per-city, no per-band normalization stats needed.

    Query budget (always exactly max_queries total):
      - Training with query augmentation:
          n_standard  = max_queries * (1 - query_upsample_fraction)  at 10m
          n_upsampled = max_queries * query_upsample_fraction        at 2.5m (boundary-focused)
        Applied stochastically with query_upsample_prob.
      - Training without augmentation or eval:
          All max_queries at 10m.
    """

    TASK_NAME = "c2seg_segmentation"

    def __init__(
        self,
        mat_path: str,
        subset: str,
        city: str,
        split: str,
        sensors: List[str],
        crop_index_path: str,
        stats_path: str,
        spectral_meta_path: str,
        look_up,
        dataset_config: dict,
        mode: str = "train",
        augment: bool = True,
        max_queries: int = 65_536,
        crop_size: int = 128,
        resolution_augment_factors: List[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.mat_path = mat_path
        self.subset = subset
        self.city = city
        self.split = split
        self.sensors = sensors
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.max_queries = max_queries
        self.crop_size = crop_size
        self.look_up = look_up
        self.dataset_config = dataset_config

        self.axis_order = AXIS_ORDER[subset]
        self.needs_label_remap = (subset == "china")

        self.token_builder = TokenBuilder(look_up)

        # ── Mat file reader ─────────────────────────────────────────
        self.reader = MatFileReader(mat_path)
        self._detect_dimensions()

        # ── Load spectral metadata ──────────────────────────────────
        with open(spectral_meta_path, "r") as f:
            self.spectral_meta = json.load(f)

        # ── Setup per-sensor metadata ───────────────────────────────
        self.sensor_info = {}
        self._npy_data = {}  # Loaded NPY arrays (memory-mapped)

        for sensor in sensors:
            meta_key = SENSOR_META_KEY[(subset, sensor)]
            sensor_meta = self.spectral_meta[meta_key]
            gsd = SENSOR_GSD[(subset, sensor)]
            config_key = f"bands_c2seg_{meta_key}"
            spectral_indices = self._get_spectral_indices(config_key)
            res_idx = self.look_up.get_resolution_idx(gsd)

            # Determine source: NPY file or mat file
            npy_key = (subset, sensor)
            is_npy = npy_key in NPY_SENSORS
            mat_key = MAT_KEYS.get(sensor, None)  # None for NPY-only sensors

            self.sensor_info[sensor] = {
                "meta_key": meta_key, "gsd": gsd,
                "n_bands": sensor_meta["n_bands"],
                "spectral_indices": spectral_indices,
                "resolution_idx": res_idx,
                "mat_key": mat_key,
                "wavelengths": sensor_meta["wavelengths"],
                "bandwidths": sensor_meta["bandwidths"],
                "is_npy": is_npy,
            }

            # Load NPY data (memory-mapped for efficiency)
            if is_npy:
                data_dir = os.path.dirname(mat_path)
                npy_path = os.path.join(data_dir, NPY_SENSORS[npy_key])
                self._npy_data[sensor] = np.load(npy_path, mmap_mode="r")
                # Load mask if available
                mask_key = npy_key
                if mask_key in NPY_MASKS:
                    mask_path = os.path.join(data_dir, NPY_MASKS[mask_key])
                    self._npy_data[f"{sensor}_mask"] = np.load(mask_path, mmap_mode="r")
                print(f"[C2Seg] Sensor '{sensor}' ({meta_key}): "
                      f"{sensor_meta['n_bands']} bands, {gsd}m, res_idx={res_idx} "
                      f"[NPY: {npy_path}]")
            else:
                print(f"[C2Seg] Sensor '{sensor}' ({meta_key}): "
                      f"{sensor_meta['n_bands']} bands, {gsd}m, res_idx={res_idx}")

        # Collect all resolutions that can appear in groups across ALL C2Seg subsets
        # for the sensors we're using. Must be global (not per-subset) so DDP has
        # consistent groups across ranks when using ConcatDataset.
        self.all_resolutions = sorted(set(
            gsd for (subset_key, sensor_key), gsd in SENSOR_GSD.items()
            if sensor_key in sensors  # only sensors we actually train with
        ))

        # ── Resolution augmentation ─────────────────────────────────
        self.resolution_augment_factors = (
            resolution_augment_factors if (resolution_augment_factors and self.augment)
            else None
        )
        if self.resolution_augment_factors:
            self.augment_gsd_map = {}
            for sensor in sensors:
                gsd = self.sensor_info[sensor]["gsd"]
                sensor_map = {}
                for factor in self.resolution_augment_factors:
                    aug_gsd = gsd * factor
                    aug_res_idx = self.look_up.get_resolution_idx(aug_gsd)
                    sensor_map[factor] = (aug_gsd, aug_res_idx)
                self.augment_gsd_map[sensor] = sensor_map
            print(f"[C2Seg] Resolution augmentation factors: {self.resolution_augment_factors}")
        else:
            self.augment_gsd_map = None

        # ── Spectral band merge augmentation ────────────────────────
        # Pre-generates a large pool of diverse sensor configs at init:
        #   - Named real sensors (S2-like, Landsat-like, etc.)
        #   - Hundreds of random anchor-sampled configs
        #   - Uniform group configs
        # During training: random.choice(pool). Maximum diversity.
        self.spectral_aug = None
        spectral_aug_prob = kwargs.get("spectral_aug_prob", 0.0)
        if self.augment and spectral_aug_prob > 0:
            # Use pre-built pool if provided (ensures same configs as preregistration)
            pool = kwargs.get("spectral_aug_pool", None)
            if pool is None:
                pool = build_spectral_aug_pool(
                    n_random=kwargs.get("spectral_aug_pool_size", 500),
                )
            self.spectral_aug = {
                "prob": spectral_aug_prob,
                "pool": pool,
            }
            # Stats
            band_counts = [len(cfg) for cfg in pool]
            print(f"[C2Seg] Spectral augmentation: prob={spectral_aug_prob}, "
                  f"pool={len(pool)} configs, "
                  f"bands={min(band_counts)}-{max(band_counts)} "
                  f"(mean={np.mean(band_counts):.1f})")

        # ── Query augmentation config ───────────────────────────────
        self.query_aug = None
        query_upsample_factor = kwargs.get("query_upsample_factor", 0)
        if self.augment and query_upsample_factor > 1:
            target_gsd = 10.0 / query_upsample_factor
            fraction = kwargs.get("query_upsample_fraction", 0.5)
            self.query_aug = {
                "upsample_factor": query_upsample_factor,
                "target_gsd": target_gsd,
                "fraction": fraction,
                "n_upsampled": int(max_queries * fraction),
                "n_standard": max_queries - int(max_queries * fraction),
                "boundary_fraction": kwargs.get("query_boundary_fraction", 0.7),
                "boundary_dilation": kwargs.get("query_boundary_dilation", 2),
                "prob": kwargs.get("query_upsample_prob", 0.5),
            }
            TokenBuilder.REFERENCE_SIZES[target_gsd] = 2048
            look_up.get_or_register_modality(target_gsd, 2048)
            look_up.get_resolution_idx(target_gsd)
            print(f"[C2Seg] Query augmentation: {target_gsd}m (factor={query_upsample_factor}), "
                  f"fraction={fraction:.0%} upsampled + {1-fraction:.0%} standard, "
                  f"boundary_fraction={self.query_aug['boundary_fraction']}, "
                  f"prob={self.query_aug['prob']}")
            print(f"[C2Seg]   Per sample: {self.query_aug['n_standard']} queries at 10m "
                  f"+ {self.query_aug['n_upsampled']} queries at {target_gsd}m "
                  f"= {max_queries} total")

        # ── Detect raw DN vs reflectance per sensor ─────────────────
        self.reader._open()
        self._is_raw_dn = {}
        for sensor in sensors:
            info = self.sensor_info[sensor]
            if info.get("is_npy", False) and sensor in self._npy_data:
                # Detect from NPY array
                npy_arr = self._npy_data[sensor]
                sample = np.array(npy_arr[:, :64, :64], dtype=np.float32)
                is_raw = float(sample.max()) > RAW_DN_THRESHOLD
            else:
                is_raw = _detect_raw_dn(self.reader, sensor, mat_path, self.axis_order)

            self._is_raw_dn[sensor] = is_raw
            src = "NPY" if info.get("is_npy", False) else "MAT"
            if is_raw:
                print(f"[C2Seg] Sensor '{sensor}' ({city}, {src}): "
                      f"raw DN detected → will ÷{REFLECTANCE_SCALE:.0f}")
            else:
                print(f"[C2Seg] Sensor '{sensor}' ({city}, {src}): "
                      f"already reflectance → no conversion")
        self.reader.close()

        # ── Normalization mode ──────────────────────────────────────
        # "raw":         ÷10000 + clamp [0,1] (default, physics-preserving)
        # "band_minmax": per-band min-max from stats JSON (like baselines)
        # "zscore":      per-band z-score computed from data, rescaled to [0,1]
        #                Makes value encoding sensor-agnostic: same relative
        #                brightness → same Fourier features regardless of sensor.
        self.norm_mode = kwargs.get("norm_mode", "raw")
        self._band_norm = {}
        self._band_zscore = {}

        if self.norm_mode == "band_minmax":
            with open(stats_path, "r") as f:
                all_stats = json.load(f)

            for sensor in sensors:
                stat_key = f"{subset}_{sensor}_{city}"
                if stat_key in all_stats and "band_min" in all_stats[stat_key]:
                    entry = all_stats[stat_key]
                    band_min = torch.tensor(entry["band_min"], dtype=torch.float32)
                    band_max = torch.tensor(entry["band_max"], dtype=torch.float32)
                    band_range = torch.clamp(band_max - band_min, min=1e-6)
                    self._band_norm[sensor] = (band_min, band_range)
                    print(f"[C2Seg] Sensor '{sensor}': band_minmax normalization loaded")
                else:
                    print(f"[C2Seg] WARNING: no band stats for '{stat_key}', "
                          f"falling back to raw for this sensor")
            print(f"[C2Seg] Normalization mode: {self.norm_mode}")

        elif self.norm_mode == "zscore":
            zscore_path = os.path.join(
                os.path.dirname(stats_path), "c2seg_zscore_stats.json")
            if os.path.exists(zscore_path):
                with open(zscore_path, "r") as f:
                    all_zstats = json.load(f)

                for sensor in sensors:
                    # Try multiple key formats
                    for stat_key in [f"{subset}_{sensor}_{city}",
                                     f"{subset}_{sensor}"]:
                        if stat_key in all_zstats and "band_mean" in all_zstats[stat_key]:
                            entry = all_zstats[stat_key]
                            band_mean = torch.tensor(entry["band_mean"], dtype=torch.float32)
                            band_std = torch.tensor(entry["band_std"], dtype=torch.float32)
                            self._band_zscore[sensor] = (band_mean, band_std)
                            print(f"[C2Seg] Sensor '{sensor}': zscore stats loaded "
                                  f"({len(entry['band_mean'])} bands, key={stat_key})")
                            break
                    else:
                        print(f"[C2Seg] WARNING: no zscore stats for '{sensor}', "
                              f"falling back to raw")
            else:
                print(f"[C2Seg] WARNING: {zscore_path} not found. "
                      f"Run compute_zscore_stats.py first. Falling back to raw.")
                self.norm_mode = "raw"
            print(f"[C2Seg] Normalization mode: zscore (per-band mean/std → [0,1])")

        else:
            print(f"[C2Seg] Normalization mode: raw reflectance")

        # ── Load and filter crop index ──────────────────────────────
        self.crops = self._load_crop_index(crop_index_path)
        print(f"[C2Seg] {len(self.crops)} crops for {subset}/{city} ({split}), "
              f"sensors={sensors}")

    # ═════════════════════════════════════════════════════════════════
    # INITIALIZATION HELPERS
    # ═════════════════════════════════════════════════════════════════

    def _compute_zscore_stats(self, sensors):
        """
        Compute per-band mean and std from the actual sensor data.

        Uses subsampled data (every 10th row/col for NPY, or a 200×200
        patch for mat) to compute statistics quickly.

        Stats are in RAW DN space (before ÷10000). The normalization
        in _read_sensor_crop converts to z-score then rescales to [0,1]:
            z = (x - mean) / std
            x_norm = clamp((z + 3) / 6, 0, 1)

        This maps ±3σ to [0, 1], making the Fourier encoding
        sensor-agnostic while preserving the [0,1] contract.
        """
        self.reader._open()

        for sensor in sensors:
            info = self.sensor_info[sensor]

            if info.get("is_npy", False) and sensor in self._npy_data:
                npy_arr = self._npy_data[sensor]
                # Subsample for speed: every 10th pixel
                sample = np.array(npy_arr[:, ::10, ::10], dtype=np.float32)
            else:
                # Read a large patch from mat
                mat_key = info["mat_key"]
                sample = self.reader.read_crop(mat_key, 0, 0, 300, 300,
                                                axis_order=self.axis_order)

            C = sample.shape[0]
            band_mean = np.zeros(C, dtype=np.float32)
            band_std = np.zeros(C, dtype=np.float32)

            for b in range(C):
                vals = sample[b].flatten()
                vals = vals[vals > 0]  # exclude zeros/nodata
                if len(vals) > 10:
                    band_mean[b] = vals.mean()
                    band_std[b] = vals.std()
                else:
                    band_mean[b] = 0.0
                    band_std[b] = 1.0

            # Clamp std to avoid division by zero
            band_std = np.clip(band_std, 1.0, None)

            self._band_zscore[sensor] = (
                torch.tensor(band_mean, dtype=torch.float32),
                torch.tensor(band_std, dtype=torch.float32),
            )
            print(f"[C2Seg] Sensor '{sensor}': zscore stats computed "
                  f"(mean=[{band_mean[0]:.0f}..{band_mean[-1]:.0f}], "
                  f"std=[{band_std[0]:.0f}..{band_std[-1]:.0f}])")

        self.reader.close()

    def _detect_dimensions(self):
        self.dims = {}
        for modality, key in MAT_KEYS.items():
            if modality == "label":
                continue
            try:
                shape = self.reader.get_shape(key)
                if self.axis_order == "HWC":
                    self.dims[modality] = (shape[0], shape[1])
                else:
                    self.dims[modality] = (shape[1], shape[2])
            except (KeyError, Exception):
                pass
        try:
            label_shape = self.reader.get_shape("label")
            self.label_dims = (label_shape[0], label_shape[1])
        except (KeyError, Exception):
            self.label_dims = None
        print(f"[C2Seg] Dimensions: {self.dims}")
        print(f"[C2Seg] Label dims: {self.label_dims}")
        self.reader.close()

    def _get_spectral_indices(self, config_key: str) -> torch.Tensor:
        if config_key not in self.dataset_config:
            raise KeyError(f"Band config '{config_key}' not in dataset_config.")
        bands_info = self.dataset_config[config_key]
        all_bands = []
        for name, data in bands_info.items():
            if all(k in data for k in ("bandwidth", "central_wavelength", "idx")):
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                })
        all_bands.sort(key=lambda b: b["idx"])
        indices = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                self.look_up.table_wave[key] = len(self.look_up.table_wave)
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    def _load_crop_index(self, csv_path: str) -> List[dict]:
        crops = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            has_split = "split" in (reader.fieldnames or [])
            for row in reader:
                if row["city"] != self.city or row["subset"] != self.subset:
                    continue
                # Filter by split column if present
                if has_split and "split" in row:
                    if row["split"] != self.split:
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

    def _subcrop(self, crop: dict) -> dict:
        """
        Sub-crop to self.crop_size if the crop is larger.

        Handles both 10m and HSI coordinates consistently.
        China crops are 256×256 at 10m → sub-cropped to 128×128.
        HSI at 30m scales proportionally (128/3 ≈ 43×43).

        Training: random offset within the larger crop.
        Eval: center crop.
        """
        target = self.crop_size
        crop_h, crop_w = crop["crop_h"], crop["crop_w"]

        if crop_h <= target and crop_w <= target:
            return crop

        # Compute offset at 10m
        max_dr = max(0, crop_h - target)
        max_dc = max(0, crop_w - target)

        if self.augment:
            dr = random.randint(0, max_dr) if max_dr > 0 else 0
            dc = random.randint(0, max_dc) if max_dc > 0 else 0
        else:
            dr = max_dr // 2
            dc = max_dc // 2

        # Scale factor for HSI (may be at coarser resolution)
        hsi_h, hsi_w = crop["hsi_crop_h"], crop["hsi_crop_w"]
        scale_h = hsi_h / crop_h
        scale_w = hsi_w / crop_w

        hsi_dr = int(dr * scale_h)
        hsi_dc = int(dc * scale_w)
        hsi_target_h = max(1, int(target * scale_h))
        hsi_target_w = max(1, int(target * scale_w))

        return {
            "row_10m": crop["row_10m"] + dr,
            "col_10m": crop["col_10m"] + dc,
            "crop_h": target,
            "crop_w": target,
            "hsi_row": crop["hsi_row"] + hsi_dr,
            "hsi_col": crop["hsi_col"] + hsi_dc,
            "hsi_crop_h": hsi_target_h,
            "hsi_crop_w": hsi_target_w,
        }

    # ═════════════════════════════════════════════════════════════════
    # SPECTRAL AUGMENTATION (partial masking — physically correct)
    # ═════════════════════════════════════════════════════════════════

    def _apply_spectral_merge_masked(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        C: int,
        H: int,
        W: int,
        wavelengths: list,
        bandwidths: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Physically correct spectral augmentation: simulate broadband
        sensors by replacing narrow-band regions with broad measurements.

        Picks 1-4 random sensor configs. For each virtual band:
          1. Find overlapping EnMAP bands (±0.75× bandwidth)
          2. MASK all overlapping bands (narrow measurements disappear)
          3. UNMASK one representative with Gaussian-weighted average
             reflectance and merged spectral index

        Bands outside any virtual band's range stay native and unmasked.

        Example with 3 configs totaling 25 virtual bands:
          242 total → 25 regions affected
          → ~70 overlapping EnMAP bands masked
          → 25 representatives unmasked with merged values
          → 217 native bands untouched
          Result: 242 - 70 + 25 = ~197 unmasked tokens
          (217 native + 25 merged, ~45 masked)

        This teaches the model that some spectral regions are measured
        at broad resolution (like a real multispectral sensor) while
        others retain full hyperspectral detail.

        Args:
            tokens: [C*H*W, 8] — token tensor
            token_mask: [C*H*W] — bool mask (True = masked/ignored)
            C, H, W: spectral and spatial dimensions
            wavelengths, bandwidths: per-band metadata

        Returns:
            tokens: [C*H*W, 8] — some values modified
            token_mask: [C*H*W] — some bands masked within virtual band regions
        """
        cfg = self.spectral_aug
        if cfg is None or random.random() > cfg["prob"]:
            return tokens, token_mask

        # Pick 1-8 configs: mostly gentle (1-3), sometimes moderate (5), rarely aggressive (8)
        n_configs = random.choices([1, 2, 3, 5, 8], weights=[3, 3, 3, 2, 1], k=1)[0]
        selected_configs = random.choices(cfg["pool"], k=n_configs)

        n_pixels = H * W
        tokens_3d = tokens.reshape(C, n_pixels, 8)
        mask_2d = token_mask.reshape(C, n_pixels)
        wl_arr = np.array(wavelengths)

        # Track which bands are claimed by a virtual band
        claimed = set()

        for sensor_bands in selected_configs:
            if len(sensor_bands) < 2:
                continue

            for sim_wl, sim_bw in sensor_bands:
                # Find overlapping EnMAP bands
                lo = sim_wl - sim_bw * 0.75
                hi = sim_wl + sim_bw * 0.75
                band_mask = (wl_arr >= lo) & (wl_arr <= hi)
                band_indices = np.where(band_mask)[0]

                if len(band_indices) == 0:
                    continue

                # Skip if any band in this region already claimed
                if any(idx in claimed for idx in band_indices):
                    continue

                # Representative = middle band
                rep_idx = band_indices[len(band_indices) // 2]

                # Gaussian-weighted average of overlapping bands
                group_values = tokens_3d[band_indices, :, 0]
                group_wl_vals = wl_arr[band_indices]

                sigma = max(sim_bw / 4.0, 1.0)
                wl_t = torch.tensor(group_wl_vals, dtype=torch.float32)
                weights = torch.exp(-0.5 * ((wl_t - sim_wl) / sigma) ** 2)
                weights = weights / weights.sum()

                weighted_val = (weights[:, None] * group_values).sum(dim=0)

                # Register merged spectral entry
                wave_key = (int(round(sim_bw)), int(round(sim_wl)))
                if wave_key not in self.look_up.table_wave:
                    self.look_up.table_wave[wave_key] = len(self.look_up.table_wave)

                # MASK all overlapping bands (narrow bands disappear)
                for idx in band_indices:
                    mask_2d[idx, :] = True
                    claimed.add(idx)

                # UNMASK representative with merged value
                tokens_3d[rep_idx, :, 0] = weighted_val
                tokens_3d[rep_idx, :, 3] = float(self.look_up.table_wave[wave_key])
                mask_2d[rep_idx, :] = False

        tokens = tokens_3d.reshape(C * n_pixels, 8)
        token_mask = mask_2d.reshape(C * n_pixels)
        return tokens, token_mask

    # ═════════════════════════════════════════════════════════════════
    # CROP READING
    # ═════════════════════════════════════════════════════════════════

    def _read_sensor_crop(self, sensor: str, crop: dict, raw_dn: bool = False) -> torch.Tensor:
        """
        Read a sensor crop and convert to reflectance.

        Supports two sources:
          - Mat file (default): reads via MatFileReader
          - NPY file (aligned MDAS data): reads from memory-mapped array

        Handles any resolution by scaling 10m crop coordinates:
          - 10m sensor: use coordinates directly
          - 30m sensor (coarser): scale down by 10/30
          - 2.2m sensor (finer): scale up by 10/2.2

        Raw DN (max > 100) → ÷10000.
        Already reflectance → no-op.
        Clamp to [0, 1]. No per-band or per-city normalization.
        """
        info = self.sensor_info[sensor]
        gsd = info["gsd"]

        # Compute crop coordinates by scaling from 10m reference
        if gsd != 10.0:
            scale = 10.0 / gsd  # >1 for finer (2.2m), <1 for coarser (30m)
            r0 = int(crop["row_10m"] * scale)
            c0 = int(crop["col_10m"] * scale)
            h = max(1, int(crop["crop_h"] * scale))
            w = max(1, int(crop["crop_w"] * scale))
        else:
            r0, c0 = crop["row_10m"], crop["col_10m"]
            h, w = crop["crop_h"], crop["crop_w"]

        # Read from NPY or mat
        if info.get("is_npy", False) and sensor in self._npy_data:
            npy_arr = self._npy_data[sensor]  # [C, H_full, W_full]
            # Clamp to array bounds
            r1 = min(r0 + h, npy_arr.shape[1])
            c1 = min(c0 + w, npy_arr.shape[2])
            r0 = max(0, r0)
            c0 = max(0, c0)
            data = np.array(npy_arr[:, r0:r1, c0:c1], dtype=np.float32)
            data = torch.from_numpy(data)
        else:
            mat_key = info["mat_key"]
            data = self.reader.read_crop(mat_key, r0, c0, h, w,
                                         axis_order=self.axis_order)
            data = torch.from_numpy(data)

        expected_bands = info["n_bands"]
        if data.shape[0] > expected_bands:
            data = data[:expected_bands]

        # ── Normalization ───────────────────────────────────────────
        if raw_dn:
            # Return raw DN values (no normalization, no ÷10000, no clamp)
            # Used by spectral interpolation: interpolate on raw, normalize after
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data

        if self.norm_mode == "band_minmax" and sensor in self._band_norm:
            # Per-band min-max (same as baselines)
            band_min, band_range = self._band_norm[sensor]
            n = min(data.shape[0], band_min.shape[0])
            data[:n] = (data[:n] - band_min[:n, None, None]) / band_range[:n, None, None]
        elif self.norm_mode == "zscore" and sensor in self._band_zscore:
            # Per-band z-score → rescale to [0, 1]
            # z = (x - mean) / std, then (z + 3) / 6 maps ±3σ → [0, 1]
            band_mean, band_std = self._band_zscore[sensor]
            n = min(data.shape[0], band_mean.shape[0])
            data[:n] = (data[:n] - band_mean[:n, None, None]) / band_std[:n, None, None]
            data = (data + 3.0) / 6.0
        else:
            # Raw reflectance: ÷10000 if raw DN
            if self._is_raw_dn.get(sensor, False):
                data = data / REFLECTANCE_SCALE

        # Clean up NaN/Inf, clamp to [0, 1]
        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        data = torch.clamp(data, min=0.0, max=1.0)

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
    # QUERY BUILDING
    # ═════════════════════════════════════════════════════════════════

    def _build_upsampled_queries(
        self, label_10m: torch.Tensor, first_spectral_idx: int,
    ) -> torch.Tensor:
        """
        Build boundary-focused queries at upsampled resolution (e.g., 2.5m).

        Upsamples label via nearest-neighbor, detects boundaries, samples
        preferentially from boundary zones.

        Returns [n_upsampled, 8] query tensor.
        """
        cfg = self.query_aug
        factor = cfg["upsample_factor"]
        target_gsd = cfg["target_gsd"]
        n_queries = cfg["n_upsampled"]

        # Upsample label: 128×128 → 512×512 at factor=4
        label_up = F.interpolate(
            label_10m.float().unsqueeze(0).unsqueeze(0),
            scale_factor=factor, mode="nearest",
        ).squeeze(0).squeeze(0).long()

        # Sample boundary-focused indices in upsampled grid
        indices = sample_boundary_focused_indices(
            label=label_up,
            n_queries=n_queries,
            boundary_fraction=cfg["boundary_fraction"],
            dilation=cfg["boundary_dilation"],
        )

        rows = indices[:, 0]
        cols = indices[:, 1]

        # Get resolution index and position offset for target GSD
        res_idx = self.look_up.get_resolution_idx(target_gsd)

        # Position offset from lookup table's reference grid
        ref_size = TokenBuilder.REFERENCE_SIZES.get(target_gsd, TokenBuilder.DEFAULT_REF_SIZE)
        offset = self.look_up.get_or_register_modality(target_gsd, ref_size)
        query_offset = self.look_up.get_query_offset(target_gsd, ref_size)

        # Calculate centered window in reference grid (same logic as TokenBuilder)
        H_up, W_up = label_up.shape
        ref_center = ref_size // 2
        y_start = ref_center - H_up // 2
        x_start = ref_center - W_up // 2

        # Build query tokens: [value, x, y, spectral_idx, label, query_idx, res_idx, time_idx]
        queries = torch.zeros(n_queries, 8)
        queries[:, 0] = 0.0                                              # value (unused)
        queries[:, 1] = (cols + x_start + offset).float()                # x position in reference grid
        queries[:, 2] = (rows + y_start + offset).float()                # y position in reference grid
        queries[:, 3] = float(first_spectral_idx)                        # spectral index
        queries[:, 4] = label_up[rows, cols].float()                     # label
        queries[:, 5] = float(query_offset)                              # query offset
        queries[:, 6] = float(res_idx)                                   # resolution index
        queries[:, 7] = float(STATIC_TIME_IDX)                           # time index

        return queries

    def _build_standard_queries(
        self, label: torch.Tensor, first_spectral_idx: int,
        n_queries: int, gsd: float = 10.0,
    ) -> torch.Tensor:
        """
        Build standard queries at label resolution.

        Returns exactly n_queries query tokens.
        """
        res_idx = self.look_up.get_resolution_idx(gsd)

        queries = self.token_builder.build_queries(
            label=label,
            resolution=gsd,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=res_idx,
            time_idx=STATIC_TIME_IDX,
        )

        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=n_queries,
            ignore_index=IGNORE_INDEX,
            prioritize_valid=True,
        )

        # Pad if not enough (e.g., very small valid area)
        if queries.shape[0] < n_queries:
            deficit = n_queries - queries.shape[0]
            if queries.shape[0] > 0:
                repeats = (deficit // queries.shape[0]) + 1
                extra = queries.repeat(repeats, 1)[:deficit]
                queries = torch.cat([queries, extra], dim=0)
            else:
                queries = torch.zeros(n_queries, 8)
                queries[:, 4] = IGNORE_INDEX

        return queries[:n_queries]

    # ═════════════════════════════════════════════════════════════════
    # DUMMY SAMPLE
    # ═════════════════════════════════════════════════════════════════

    def _make_dummy_sample(self) -> dict:
        groups = {}
        for res in self.all_resolutions:
            dummy_tokens = torch.zeros(1, 8)
            dummy_mask = torch.ones(1, dtype=torch.bool)
            groups[res] = {"tokens": dummy_tokens, "mask": dummy_mask, "shape": (1, 1, 1)}
        queries = torch.zeros(self.max_queries, 8)
        queries[:, 4] = IGNORE_INDEX
        return {
            "groups": groups,
            "tasks": {self.TASK_NAME: {
                "queries": queries,
                "queries_mask": torch.zeros(self.max_queries, dtype=torch.bool),
            }},
            "target_resolution": 10.0,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index: int) -> dict:
        crop = self.crops[index]

        # ── Sub-crop to target size (128×128) if needed ─────────────
        crop = self._subcrop(crop)

        # ── Read label ──────────────────────────────────────────────
        try:
            label = self._read_label_crop(crop)
        except Exception as e:
            print(f"[C2Seg] Error reading label at crop {index}: {e}")
            return self._make_dummy_sample()

        # ── D4 augmentation ─────────────────────────────────────────
        if self.augment:
            d4_k = random.randint(0, 3)
            d4_flip = random.random() > 0.5
        else:
            d4_k = 0
            d4_flip = False

        if d4_k > 0:
            label = torch.rot90(label, d4_k, dims=(-2, -1))
        if d4_flip:
            label = torch.flip(label, dims=(-1,))

        # ── Resolution augmentation ─────────────────────────────────
        res_factor = (random.choice(self.resolution_augment_factors)
                      if self.augment_gsd_map is not None else 1)

        # ── Process each sensor ─────────────────────────────────────
        groups = {}

        for sensor in self.sensors:
            try:
                image = self._read_sensor_crop(sensor, crop)
            except Exception as e:
                print(f"[C2Seg] Error reading {sensor} at crop {index}: {e}")
                return self._make_dummy_sample()

            info = self.sensor_info[sensor]
            gsd = info["gsd"]
            spectral_indices = info["spectral_indices"]
            res_idx = info["resolution_idx"]

            if d4_k > 0:
                image = torch.rot90(image, d4_k, dims=(-2, -1))
            if d4_flip:
                image = torch.flip(image, dims=(-1,))

            current_gsd, current_res_idx = gsd, res_idx
            if res_factor > 1 and self.augment_gsd_map is not None:
                image = downsample_image(image, res_factor)
                current_gsd, current_res_idx = self.augment_gsd_map[sensor][res_factor]

            image = image.contiguous()
            C, H, W = image.shape

            sensor_label = label.clone()
            if res_factor > 1 and gsd == 10.0:
                sensor_label = downsample_label_majority(sensor_label, res_factor)
            token_label = (torch.full((H, W), IGNORE_INDEX, dtype=torch.int64)
                           if gsd != 10.0 else sensor_label)

            tokens = self.token_builder.build_tokens(
                image=image, label=token_label,
                resolution=current_gsd,
                spectral_indices=spectral_indices,
                resolution_idx=current_res_idx,
                time_idx=STATIC_TIME_IDX,
            )
            token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

            # ── Spectral band merge augmentation (post-tokenization) ─
            # Masks redundant tokens, keeps one representative per group.
            # Token count unchanged — DDP safe.
            if self.spectral_aug is not None and sensor != "sar":
                tokens, token_mask = self._apply_spectral_merge_masked(
                    tokens, token_mask, C, H, W,
                    info["wavelengths"], info["bandwidths"],
                )

            if current_gsd in groups:
                groups[current_gsd]["tokens"] = torch.cat(
                    [groups[current_gsd]["tokens"], tokens], dim=0)
                groups[current_gsd]["mask"] = torch.cat(
                    [groups[current_gsd]["mask"], token_mask], dim=0)
                old_c = groups[current_gsd]["shape"][0]
                groups[current_gsd]["shape"] = (old_c + C, H, W)
            else:
                groups[current_gsd] = {
                    "tokens": tokens, "mask": token_mask, "shape": (C, H, W),
                }

        # ── Pad missing resolution groups for DDP consistency ────────
        # All ranks must have the same set of resolution keys so the
        # encoder follows the same code path on every rank.
        for res in self.all_resolutions:
            if res not in groups:
                dummy_token = torch.zeros(1, 8)
                dummy_mask = torch.ones(1, dtype=torch.bool)  # masked = ignored
                groups[res] = {
                    "tokens": dummy_token,
                    "mask": dummy_mask,
                    "shape": (1, 1, 1),
                }

        # ── Build queries ───────────────────────────────────────────
        query_sensor = None
        for s in self.sensors:
            if self.sensor_info[s]["gsd"] <= 10.0:
                query_sensor = s
                break
        if query_sensor is None:
            query_sensor = self.sensors[0]

        query_info = self.sensor_info[query_sensor]
        first_spectral_idx = query_info["spectral_indices"][0].item()

        # Decide: augmented (mixed 10m + 2.5m) or standard (all 10m)
        use_query_aug = (
            self.query_aug is not None
            and random.random() < self.query_aug["prob"]
        )

        if use_query_aug:
            # Mixed queries: n_standard at 10m + n_upsampled at 2.5m
            cfg = self.query_aug

            # Standard queries at 10m
            query_label_10m = label
            if res_factor > 1 and self.augment_gsd_map is not None:
                query_label_10m = downsample_label_majority(label, res_factor)

            standard_queries = self._build_standard_queries(
                label=query_label_10m,
                first_spectral_idx=first_spectral_idx,
                n_queries=cfg["n_standard"],
                gsd=10.0 * res_factor if (res_factor > 1 and self.augment_gsd_map) else 10.0,
            )

            # Upsampled boundary-focused queries at target_gsd
            upsampled_queries = self._build_upsampled_queries(
                label_10m=label,
                first_spectral_idx=first_spectral_idx,
            )

            # Concatenate: always exactly max_queries total
            queries = torch.cat([standard_queries, upsampled_queries], dim=0)

        else:
            # All queries at 10m (standard path)
            query_label = label
            query_gsd = 10.0

            if res_factor > 1 and self.augment_gsd_map is not None:
                query_label = downsample_label_majority(label, res_factor)
                query_gsd = 10.0 * res_factor

            queries = self._build_standard_queries(
                label=query_label,
                first_spectral_idx=first_spectral_idx,
                n_queries=self.max_queries,
                gsd=query_gsd,
            )

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": groups,
            "tasks": {
                self.TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                },
            },
            "target_resolution": 10.0,
            "dataset_name": DATASET_NAME,
        }