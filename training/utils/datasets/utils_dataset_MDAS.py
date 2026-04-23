"""
MDAS Segmentation Dataset for Atomizer-IO
==========================================

Semantic segmentation on the Multi-modal Data Augsburg dataset.
Designed for cross-sensor zero-shot transfer experiments.

Follows the same token format and return structure as FLAIR-HUB datasets
so that the same model, collation, and training loop work without changes.

Experiments (all use the same trained model):
  Exp 1: HySpex 2.2m  → HySpex 2.2m   (in-distribution)
  Exp 2: HySpex 2.2m  → Sentinel-2 10m (cross-sensor + cross-res)
  Exp 3: HySpex 2.2m  → EnMAP 10m      (cross-res, similar spectra)
  Exp 4: EnMAP 10m    → HySpex 2.2m    (inverse transfer)

Token format (same as FLAIR-HUB):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7

Prerequisites (produced by prepare_mdas.py):
    - mdas_crop_index.csv
    - mdas_norm_stats.json
    - mdas_spectral_meta.json
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
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None

from .token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 6
IGNORE_INDEX = 255
DATASET_NAME = "MDAS"

CLASS_NAMES = {
    0: "Pavement",
    1: "Soil",
    2: "Roof",
    3: "Low vegetation",
    4: "Tree",
    5: "Water",
}

GSD_REF = 2.2  # Reference grid (HySpex / 2.2m labels)
CROP_SIZE_REF = 64  # Crop size on the reference grid

# Sensor file templates
SENSOR_FILES = {
    "hyspex":    "sub_area_{n}/HySpex_sub_area{n}.tif",
    "enmap_10m": "sub_area_{n}/EeteS_EnMAP_10m_sub_area{n}.tif",
    "enmap_30m": "sub_area_{n}/EeteS_EnMAP_30m_sub_area{n}.tif",
    "sentinel2": "sub_area_{n}/Sentinel_2_sub_area{n}.tif",
}

# Label file templates (keyed by label resolution)
LABEL_FILES = {
    2.2:  "GT_labels/2_sub_area{n}.tif",
    10.0: "GT_labels/label_6class_10m_sub_area{n}.tif",
}

# Sensor → label resolution mapping
SENSOR_LABEL_RES = {
    "hyspex":    2.2,
    "enmap_10m": 10.0,
    "enmap_30m": 10.0,
    "sentinel2": 10.0,
}

# Fixed time index for static acquisitions (no temporal dimension)
STATIC_TIME_IDX = -1


# ═══════════════════════════════════════════════════════════════════════
# SPATIAL MAPPING
# ═══════════════════════════════════════════════════════════════════════

def crop_to_sensor_window(
    r0_ref: int,
    c0_ref: int,
    crop_size_ref: int,
    sensor_gsd: float,
    sensor_h: int,
    sensor_w: int,
) -> Tuple[int, int, int, int]:
    """
    Map a crop position from the 2.2m reference grid to sensor pixel coords.

    Returns (r0, c0, h, w) in sensor pixels.
    Crop size at sensor resolution = floor(crop_size_ref * GSD_REF / sensor_gsd).
    """
    scale = GSD_REF / sensor_gsd

    r0 = int(r0_ref * scale)
    c0 = int(c0_ref * scale)
    size = max(1, int(crop_size_ref * scale))

    # Clamp to image bounds
    r0 = min(r0, max(0, sensor_h - size))
    c0 = min(c0, max(0, sensor_w - size))
    h = min(size, sensor_h - r0)
    w = min(size, sensor_w - c0)

    return r0, c0, h, w


# ═══════════════════════════════════════════════════════════════════════
# D4 AUGMENTATION (rotation + flip)
# ═══════════════════════════════════════════════════════════════════════

def augment_d4(
    image: torch.Tensor,
    label: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random D4 symmetry transform (4 rotations × 2 flip states).
    Both image [C, H, W] and label [H, W] are transformed consistently.
    """
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
    """
    Downsample image by integer factor using average pooling.

    Args:
        image: [C, H, W] float tensor.
        factor: integer downsampling factor (2, 3, 4, ...).

    Returns:
        [C, H//factor, W//factor] float tensor.
    """
    if factor <= 1:
        return image
    return F.avg_pool2d(image.unsqueeze(0), kernel_size=factor, stride=factor).squeeze(0)


def downsample_label_majority(label: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample label map by integer factor using majority voting.

    For each factor×factor block, the most frequent valid class wins.
    If all pixels in a block are IGNORE_INDEX, the output is IGNORE_INDEX.

    Args:
        label: [H, W] int64 tensor with class indices.
        factor: integer downsampling factor.

    Returns:
        [H//factor, W//factor] int64 tensor.
    """
    if factor <= 1:
        return label

    H, W = label.shape
    new_h = H // factor
    new_w = W // factor

    # Crop to exact multiple
    cropped = label[:new_h * factor, :new_w * factor]

    # Reshape into blocks
    blocked = cropped.reshape(new_h, factor, new_w, factor)
    blocked = blocked.permute(0, 2, 1, 3).reshape(new_h * new_w, factor * factor)

    # Majority vote per block
    valid_mask = (blocked != IGNORE_INDEX)
    safe = blocked.clone()
    safe[~valid_mask] = 0

    counts = torch.zeros(new_h * new_w, NUM_CLASSES + 1, dtype=torch.long)
    counts.scatter_add_(1, safe.long(), valid_mask.long())

    # Only consider actual classes (0..NUM_CLASSES-1)
    counts = counts[:, :NUM_CLASSES]

    has_valid = counts.sum(dim=1) > 0
    result = torch.full((new_h * new_w,), IGNORE_INDEX, dtype=label.dtype)
    result[has_valid] = counts[has_valid].argmax(dim=1).to(label.dtype)

    return result.reshape(new_h, new_w)


# ═══════════════════════════════════════════════════════════════════════
# BANDS INFO FACTORY (for Lookup_encoding initialization)
# ═══════════════════════════════════════════════════════════════════════

def create_mdas_bands_info(spectral_meta_path: str) -> dict:
    """
    Generate band configuration dicts for all MDAS sensors.

    Returns a dict compatible with dataset_config, e.g.:
        {
            "bands_mdas_hyspex": {
                "band_000": {"central_wavelength": 400, "bandwidth": 5, "idx": 0},
                ...
            },
            "bands_mdas_sentinel2": { ... },
            ...
        }
    """
    with open(spectral_meta_path, "r") as f:
        meta = json.load(f)

    bands_info = {}
    for sensor_name, sensor_meta in meta.items():
        config_key = f"bands_mdas_{sensor_name}"
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


def register_mdas_bands(look_up, dataset_config: dict):
    """
    Pre-register all MDAS bands into the lookup table.

    Must be called BEFORE model construction so that the spectral
    encoder codebook is built with the correct size. Otherwise,
    checkpoint loading will fail with a size mismatch.

    Args:
        look_up: Lookup_encoding instance with table_wave dict.
        dataset_config: dict containing bands_mdas_* keys
            (output of create_mdas_bands_info merged into base config).

    Returns:
        Number of newly registered bands.
    """
    n_new = 0
    for key, bands_info in dataset_config.items():
        if not key.startswith("bands_mdas_"):
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

    print(f"[MDAS] Pre-registered {n_new} new bands into lookup table "
          f"(total: {len(look_up.table_wave)})")
    return n_new


# ═══════════════════════════════════════════════════════════════════════
# IMAGE DIMENSION CACHE
# ═══════════════════════════════════════════════════════════════════════

class _ImageDimCache:
    """
    Cache image dimensions (height, width) per file path.
    
    Avoids reopening TIFs just to get dimensions for crop_to_sensor_window.
    Populated once in the main process, survives pickling to workers.
    """
    
    def __init__(self):
        self._cache = {}
    
    def get_dims(self, path: str) -> Tuple[int, int]:
        """Return (height, width) for a TIF, caching the result."""
        if path not in self._cache:
            with rasterio.open(path) as src:
                self._cache[path] = (src.height, src.width)
        return self._cache[path]


# ═══════════════════════════════════════════════════════════════════════
# MAIN DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════

class MDASSegmentation(Dataset):
    """
    MDAS segmentation dataset for Atomizer-IO.

    Each sample loads a single sensor crop + aligned label, tokenizes with
    Atomizer's metadata-enriched token format, and returns a dict compatible
    with FLAIR-HUB's collation and training loop.

    File I/O: Uses open-read-close per crop (no persistent handles).
    This is robust with DataLoader num_workers > 0 — no stale file
    descriptors across fork boundaries. GDAL's internal block cache
    minimizes the overhead of reopening.

    Parameters
    ----------
    root : str
        Path to Augsburg_data_4_publication/.
    sensor : str
        Sensor to load: "hyspex", "enmap_10m", "enmap_30m", "sentinel2".
    sub_areas : list of int
        Which sub-areas to include (e.g., [1, 2] for train, [3] for test).
    crop_index_path : str
        Path to mdas_crop_index.csv.
    stats_path : str
        Path to mdas_norm_stats.json.
    spectral_meta_path : str
        Path to mdas_spectral_meta.json.
    look_up : Lookup_encoding
        Shared lookup table for spectral/resolution/time indices.
    dataset_config : dict
        Band configs (output of create_mdas_bands_info merged into main config).
    mode : str
        "train" or "test". Controls augmentation and split filtering.
    augment : bool
        Whether to apply D4 augmentation (only used if mode == "train").
    max_queries : int
        Maximum number of segmentation query tokens per sample.
    crop_size_ref : int
        Crop size on the 2.2m reference grid (default: 64).
    """

    TASK_NAME = "mdas_segmentation"

    def __init__(
        self,
        root: str,
        sensor: str,
        sub_areas: List[int],
        crop_index_path: str,
        stats_path: str,
        spectral_meta_path: str,
        look_up,
        dataset_config: dict,
        mode: str = "train",
        augment: bool = True,
        max_queries: int = 65_536,
        crop_size_ref: int = CROP_SIZE_REF,
        config_model: dict = None,
        resolution_augment_factors: List[int] = None,
        spectral_configs: List[int] = None,
        max_spectral_group_size: int = 35,
        **kwargs,
    ):
        super().__init__()

        self.root = root
        self.sensor = sensor
        self.sub_areas = sub_areas
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.max_queries = max_queries
        self.crop_size_ref = crop_size_ref
        self.look_up = look_up
        self.dataset_config = dataset_config
        self.config_model = config_model

        # Resolution augmentation: list of integer pool factors.
        self.resolution_augment_factors = (
            resolution_augment_factors if (resolution_augment_factors and self.augment)
            else None
        )

        # Spectral configs: list of fixed output band counts.
        self.spectral_configs = (
            spectral_configs if (spectral_configs and self.augment)
            else None
        )
        self.max_spectral_group_size = max_spectral_group_size

        self.token_builder = TokenBuilder(look_up)

        # ── Load spectral metadata ──────────────────────────────────
        with open(spectral_meta_path, "r") as f:
            self.spectral_meta = json.load(f)

        if sensor not in self.spectral_meta:
            raise ValueError(
                f"Sensor '{sensor}' not in spectral metadata. "
                f"Available: {list(self.spectral_meta.keys())}"
            )
        self.sensor_meta = self.spectral_meta[sensor]
        self.sensor_gsd = self.sensor_meta["gsd"]
        self.n_bands = self.sensor_meta["n_bands"]

        # ── Sensor crop size (pixels at sensor resolution) ──────────
        self.sensor_crop_size = max(
            1, int(crop_size_ref * GSD_REF / self.sensor_gsd)
        )
        print(f"[MDAS] Sensor crop: {self.sensor_crop_size}×{self.sensor_crop_size} "
              f"px at {self.sensor_gsd}m "
              f"(from {crop_size_ref}×{crop_size_ref} @ {GSD_REF}m)")

        # ── Label resolution ────────────────────────────────────────
        self.label_res = SENSOR_LABEL_RES[sensor]
        self.label_crop_size = max(
            1, int(crop_size_ref * GSD_REF / self.label_res)
        )

        # ── Setup band indices via lookup table ─────────────────────
        self._setup_band_indices()

        # ── Resolution index ────────────────────────────────────────
        self.resolution_idx = self.look_up.get_resolution_idx(self.sensor_gsd)

        # ── Pre-register augmented resolutions ──────────────────────
        if self.resolution_augment_factors:
            self.augment_gsd_map = {}
            for factor in self.resolution_augment_factors:
                aug_gsd = self.sensor_gsd * factor
                aug_res_idx = self.look_up.get_resolution_idx(aug_gsd)
                self.augment_gsd_map[factor] = (aug_gsd, aug_res_idx)
                self.token_builder._ensure_resolution_registered(aug_gsd)

            gsds_str = ", ".join(
                f"{f}× → {g:.1f}m" for f, (g, _) in self.augment_gsd_map.items()
            )
            print(f"[MDAS] Resolution augmentation: {gsds_str}")
        else:
            self.augment_gsd_map = None

        # ── Precompute fixed spectral configs ────────────────────────
        if self.spectral_configs:
            self._precompute_spectral_configs()
        else:
            self._spectral_merge_configs = None

        # ── Load normalization stats ────────────────────────────────
        with open(stats_path, "r") as f:
            all_stats = json.load(f)

        if sensor in all_stats:
            self.norm_mean = torch.tensor(
                all_stats[sensor]["mean"], dtype=torch.float32
            )
            self.norm_std = torch.tensor(
                all_stats[sensor]["std"], dtype=torch.float32
            )
            print(f"[MDAS] Loaded normalization for '{sensor}': "
                  f"{len(self.norm_mean)} bands")
        else:
            print(f"[MDAS] WARNING: no normalization stats for '{sensor}', "
                  f"using identity")
            self.norm_mean = torch.zeros(self.n_bands)
            self.norm_std = torch.ones(self.n_bands)

        # ── Load and filter crop index ──────────────────────────────
        self.crops = self._load_crop_index(crop_index_path)
        print(f"[MDAS] {len(self.crops)} crops for sensor='{sensor}', "
              f"sub_areas={sub_areas}, mode='{mode}'")

        # ── Pre-cache image dimensions ──────────────────────────────
        # Cache (height, width) for each file so workers don't need
        # to open files just to get dimensions for crop_to_sensor_window.
        self._dim_cache = _ImageDimCache()
        for sa in sub_areas:
            sensor_path = os.path.join(root, SENSOR_FILES[sensor].format(n=sa))
            label_path = os.path.join(root, LABEL_FILES[self.label_res].format(n=sa))

            if os.path.exists(sensor_path):
                self._dim_cache.get_dims(sensor_path)
                print(f"[MDAS]   sub_area_{sa} sensor: OK → {sensor_path}")
            else:
                print(f"[MDAS]   sub_area_{sa} sensor: MISSING → {sensor_path}")

            if os.path.exists(label_path):
                self._dim_cache.get_dims(label_path)
                print(f"[MDAS]   sub_area_{sa} label:  OK → {label_path}")
            else:
                print(f"[MDAS]   sub_area_{sa} label:  MISSING → {label_path}")

    # ═════════════════════════════════════════════════════════════════
    # INITIALIZATION HELPERS
    # ═════════════════════════════════════════════════════════════════

    def _setup_band_indices(self):
        """Register sensor bands with the lookup table and store spectral indices."""
        config_key = f"bands_mdas_{self.sensor}"

        if config_key not in self.dataset_config:
            raise KeyError(
                f"Band config '{config_key}' not in dataset_config. "
                f"Did you call create_mdas_bands_info() and merge?"
            )

        bands_info = self.dataset_config[config_key]

        all_bands = []
        for name, data in bands_info.items():
            if all(k in data for k in ("bandwidth", "central_wavelength", "idx")):
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        indices = []
        wavelengths = []
        bandwidths = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                new_idx = len(self.look_up.table_wave)
                self.look_up.table_wave[key] = new_idx
            indices.append(self.look_up.table_wave[key])
            wavelengths.append(band["central_wavelength"])
            bandwidths.append(band["bandwidth"])

        self.spectral_indices = torch.tensor(indices, dtype=torch.long)
        self.band_wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
        self.band_bandwidths = torch.tensor(bandwidths, dtype=torch.float32)

    def _load_crop_index(self, csv_path: str) -> List[dict]:
        """Load crop index CSV and filter by sub_areas."""
        crops = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_area = int(row["sub_area"])
                if sub_area not in self.sub_areas:
                    continue
                crops.append({
                    "crop_id": row["crop_id"],
                    "sub_area": sub_area,
                    "r0": int(row["r0"]),
                    "c0": int(row["c0"]),
                })
        return crops

    # ═════════════════════════════════════════════════════════════════
    # FIXED SPECTRAL CONFIGURATIONS
    # ═════════════════════════════════════════════════════════════════

    def _precompute_spectral_configs(self):
        """
        Precompute fixed spectral merge configurations.

        For each target band count N in self.spectral_configs, precomputes:
          - group_slices: list of (start, end) for each group
          - weights: Gaussian weights per group for averaging
          - spectral_indices: lookup table indices for merged bands

        All merge metadata is computed in the main process. Workers
        only apply precomputed configs — no registration, no shared
        state modification.
        """
        C = self.n_bands
        wl = self.band_wavelengths
        bw = self.band_bandwidths
        max_gs = self.max_spectral_group_size

        self._spectral_merge_configs = {}

        for n_target in self.spectral_configs:
            if n_target >= C:
                # Full bands — no merging needed
                self._spectral_merge_configs[n_target] = None
                print(f"[MDAS] Spectral config {n_target} bands: no merging")
                continue

            # Equal-ish partition into n_target groups
            base_size = C // n_target
            remainder = C % n_target

            group_slices = []
            cursor = 0
            for i in range(n_target):
                size = base_size + (1 if i < remainder else 0)
                group_slices.append((cursor, cursor + size))
                cursor += size

            # Enforce max group size by splitting oversized groups
            final_slices = []
            for start, end in group_slices:
                size = end - start
                if size <= max_gs:
                    final_slices.append((start, end))
                else:
                    cur = start
                    while cur < end:
                        chunk_end = min(cur + max_gs, end)
                        final_slices.append((cur, chunk_end))
                        cur = chunk_end

            # Precompute weights and spectral indices for each group
            config_groups = []
            merged_spectral_indices = []

            for start, end in final_slices:
                group_wl = wl[start:end]
                group_bw = bw[start:end]

                # Gaussian weights
                center_wavelength = group_wl.mean()
                lo_edge = (group_wl - group_bw / 2).min().item()
                hi_edge = (group_wl + group_bw / 2).max().item()
                full_bandwidth = hi_edge - lo_edge

                sigma = full_bandwidth / 4.0
                sigma = max(sigma, 1.0)

                weights = torch.exp(
                    -0.5 * ((group_wl - center_wavelength) / sigma) ** 2
                )
                weights = weights / weights.sum()

                # Quantized spectral identity
                center_wl_q = int(round(center_wavelength.item() / 20.0) * 20)
                bandwidth_q = int(round(full_bandwidth / 50.0) * 50)
                bandwidth_q = max(bandwidth_q, 50)

                wave_key = (bandwidth_q, center_wl_q)
                if wave_key not in self.look_up.table_wave:
                    self.look_up.table_wave[wave_key] = len(self.look_up.table_wave)

                merged_spectral_indices.append(self.look_up.table_wave[wave_key])

                config_groups.append({
                    "start": start,
                    "end": end,
                    "weights": weights,
                })

            spectral_idx_tensor = torch.tensor(merged_spectral_indices, dtype=torch.long)

            self._spectral_merge_configs[n_target] = {
                "groups": config_groups,
                "spectral_indices": spectral_idx_tensor,
                "n_output": len(final_slices),
            }

            print(f"[MDAS] Spectral config {n_target} bands → "
                  f"{len(final_slices)} output channels "
                  f"(max group {max_gs} bands)")

        print(f"[MDAS] Spectral configs: {list(self._spectral_merge_configs.keys())} "
              f"(lookup table total: {len(self.look_up.table_wave)})")

    def _apply_spectral_config(
        self, image: torch.Tensor, n_target: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a precomputed spectral merge config to an image.

        Merges adjacent bands using Gaussian-weighted averaging.
        Returns a new image with fewer channels and the corresponding
        spectral indices.

        Args:
            image: [C, H, W] — full spectral image.
            n_target: target band count (must be in self._spectral_merge_configs).

        Returns:
            merged_image: [N_out, H, W]
            spectral_indices: [N_out]
        """
        config = self._spectral_merge_configs[n_target]

        if config is None:
            return image, self.spectral_indices

        merged_channels = []
        for group in config["groups"]:
            start = group["start"]
            end = group["end"]
            weights = group["weights"]  # [group_size]

            # Gaussian-weighted average: [group_size, H, W] → [H, W]
            group_data = image[start:end]  # [group_size, H, W]
            merged = (group_data * weights[:, None, None]).sum(dim=0)
            merged_channels.append(merged)

        merged_image = torch.stack(merged_channels, dim=0)  # [N_out, H, W]
        return merged_image, config["spectral_indices"]

    # ═════════════════════════════════════════════════════════════════
    # FILE I/O (open-read-close per crop)
    # ═════════════════════════════════════════════════════════════════

    def _read_sensor_crop(
        self, sub_area: int, r0_ref: int, c0_ref: int,
    ) -> torch.Tensor:
        """
        Read a sensor crop aligned to the reference grid position.

        Opens the file, reads the window, closes immediately.
        Robust with DataLoader num_workers > 0.

        Returns: [C, h, w] float32 tensor, normalized.
        """
        path = os.path.join(
            self.root, SENSOR_FILES[self.sensor].format(n=sub_area)
        )

        sensor_h, sensor_w = self._dim_cache.get_dims(path)

        r0, c0, h, w = crop_to_sensor_window(
            r0_ref, c0_ref, self.crop_size_ref,
            self.sensor_gsd, sensor_h, sensor_w,
        )

        with rasterio.open(path) as handle:
            window = Window(c0, r0, w, h)
            data = handle.read(window=window).astype(np.float32)

        data = torch.from_numpy(data)

        # Normalize: (x - mean) / std
        data = (data - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]

        return data

    def _read_label_crop(
        self, sub_area: int, r0_ref: int, c0_ref: int,
    ) -> torch.Tensor:
        """
        Read the label crop at the appropriate resolution.

        Opens the file, reads the window, closes immediately.
        Robust with DataLoader num_workers > 0.

        Returns: [h, w] int64 tensor with class indices (0-5, 255=ignore).
        """
        path = os.path.join(
            self.root, LABEL_FILES[self.label_res].format(n=sub_area)
        )

        label_h, label_w = self._dim_cache.get_dims(path)

        if self.label_res == GSD_REF:
            # Labels at reference resolution: direct crop
            r0, c0 = r0_ref, c0_ref
            h = w = self.crop_size_ref
        else:
            # Labels at sensor resolution: use spatial mapping
            r0, c0, h, w = crop_to_sensor_window(
                r0_ref, c0_ref, self.crop_size_ref,
                self.label_res, label_h, label_w,
            )

        with rasterio.open(path) as handle:
            window = Window(c0, r0, w, h)
            data = handle.read(1, window=window).astype(np.int64)

        label = torch.from_numpy(data)

        # Clamp invalid classes
        label[(label < 0) | (label >= NUM_CLASSES)] = IGNORE_INDEX

        return label

    # ═════════════════════════════════════════════════════════════════
    # DUMMY SAMPLE (for error recovery)
    # ═════════════════════════════════════════════════════════════════

    def _make_dummy_sample(self) -> dict:
        dummy_tokens = torch.zeros(1, 8)
        dummy_mask = torch.ones(1, dtype=torch.bool)

        groups = {
            self.sensor_gsd: {
                "tokens": dummy_tokens,
                "mask": dummy_mask,
                "shape": (self.n_bands, self.sensor_crop_size,
                          self.sensor_crop_size),
            }
        }

        queries = torch.zeros(1, 8)
        queries[:, 4] = IGNORE_INDEX

        return {
            "groups": groups,
            "tasks": {
                self.TASK_NAME: {
                    "queries": queries,
                    "queries_mask": torch.ones(1, dtype=torch.bool),
                },
            },
            "target_resolution": self.sensor_gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index: int) -> dict:
        crop_info = self.crops[index]
        sub_area = crop_info["sub_area"]
        r0_ref = crop_info["r0"]
        c0_ref = crop_info["c0"]

        # ── Load data ───────────────────────────────────────────────
        try:
            image = self._read_sensor_crop(sub_area, r0_ref, c0_ref)
            label = self._read_label_crop(sub_area, r0_ref, c0_ref)
        except Exception as e:
            print(f"[MDAS] Error loading crop {crop_info['crop_id']}: {e}")
            return self._make_dummy_sample()

        # ── Align image and label sizes (edge crops may differ) ─────
        _, H_img, W_img = image.shape
        H_lab, W_lab = label.shape
        if H_img != H_lab or W_img != W_lab:
            H_min = min(H_img, H_lab)
            W_min = min(W_img, W_lab)
            image = image[:, :H_min, :W_min]
            label = label[:H_min, :W_min]

        # ── D4 augmentation (training only) ─────────────────────────
        if self.augment:
            image, label = augment_d4(image, label)

        # ── Spectral merging (training only) ────────────────────────
        current_spectral_indices = self.spectral_indices
        current_n_bands = self.n_bands

        if self._spectral_merge_configs is not None:
            n_target = random.choice(self.spectral_configs)
            image, current_spectral_indices = self._apply_spectral_config(
                image, n_target
            )
            current_n_bands = image.shape[0]

        # ── Resolution augmentation (training only) ─────────────────
        current_gsd = self.sensor_gsd
        current_res_idx = self.resolution_idx

        if self.augment_gsd_map is not None:
            factor = random.choice(self.resolution_augment_factors)
            if factor > 1:
                image = downsample_image(image, factor)
                label = downsample_label_majority(label, factor)
            current_gsd, current_res_idx = self.augment_gsd_map[factor]

        # Make sure image and label are contiguous after transforms
        image = image.contiguous()
        label = label.contiguous()

        H_img, W_img = image.shape[1], image.shape[2]

        # ── Build tokens from (possibly merged) image ──────────────
        tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=current_gsd,
            spectral_indices=current_spectral_indices,
            resolution_idx=current_res_idx,
            time_idx=STATIC_TIME_IDX,
        )

        token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        groups = {
            current_gsd: {
                "tokens": tokens,
                "mask": token_mask,
                "shape": (current_n_bands, H_img, W_img),
            }
        }

        # ── Build segmentation queries ──────────────────────────────
        first_spectral_idx = current_spectral_indices[0].item()

        queries = self.token_builder.build_queries(
            label=label,
            resolution=current_gsd,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=current_res_idx,
            time_idx=STATIC_TIME_IDX,
        )

        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries,
            ignore_index=IGNORE_INDEX,
            prioritize_valid=True,
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
            "target_resolution": current_gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # VISUALIZATION SAMPLE
    # ═════════════════════════════════════════════════════════════════

    def get_viz_sample(self, index: int) -> dict:
        """
        Full-resolution sample for visualization (no query subsampling,
        no augmentation).
        """
        crop_info = self.crops[index]
        sub_area = crop_info["sub_area"]
        r0_ref = crop_info["r0"]
        c0_ref = crop_info["c0"]

        image = self._read_sensor_crop(sub_area, r0_ref, c0_ref)
        label = self._read_label_crop(sub_area, r0_ref, c0_ref)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=self.sensor_gsd,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=STATIC_TIME_IDX,
        )

        token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        groups = {
            self.sensor_gsd: {
                "tokens": tokens,
                "mask": token_mask,
                "shape": (self.n_bands, image.shape[1], image.shape[2]),
            }
        }

        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.sensor_gsd,
            first_spectral_idx=self.spectral_indices[0].item(),
            resolution_idx=self.resolution_idx,
            time_idx=STATIC_TIME_IDX,
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
            "target_resolution": self.sensor_gsd,
            "dataset_name": DATASET_NAME,
            "image": image,
            "label": label,
            "crop_info": crop_info,
        }