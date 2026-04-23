"""
FLAIR-HUB Pre-training Datasets — Multi-task, Multi-resolution, Multi-temporal
================================================================================

Three dataset classes sharing a common base for FLAIR-HUB data:

1. FlairHubSegCOSIA        — COSIA land cover segmentation (18 valid classes)
2. FlairHubSegLPIS         — LPIS crop type segmentation (23 classes, hierarchical)
3. FlairHubReconstruction  — Reconstruction from latent bottleneck

Indexing supports two modes:
  - CSV-based (official splits): uses FLAIR-HUB_TRAIN.csv / VALID / TEST
    Set csv_dir to the directory containing the CSVs.
  - Directory-based (fallback): walks AERIAL_RGBI folders + random split.

Patch ID format: "D004-2021_AA-S1-32_1-1"
    → domain = "D004-2021"
    → roi    = "AA-S1-32"
    → coords = "1-1"
    → path for modality MOD: {root}/{domain}_{MOD}/{roi}/{domain}_{MOD}_{roi}_{coords}.tif

Modality structure per patch (102.4m × 102.4m):
    - Aerial RGBI:  [4, 512, 512]     at 0.2   m/px  (uint8)
    - SPOT RGBI:    [4, 64, 64]       at 1.6   m/px  (uint16)
    - S2 time series: [T×10, 10, 10]  at 10.24 m/px  (uint16)
    - S1 ASC TS:    [T×2, 10, 10]     at 10.24 m/px  (float32)
    - S1 DESC TS:   [T×2, 10, 10]     at 10.24 m/px  (float32)
    - S2 cloud mask:[T×2, 10, 10]     at 10.24 m/px  (uint16)
    - Label COSIA:  [1, 512, 512]     at 0.2   m/px  (uint8, classes 0-17)
    - Label LPIS:   [3, 512, 512]     at 0.2   m/px  (uint8, 3 hierarchy levels)

Temporal modalities are padded to max_timestamps (default 40) with mask=True
on padded tokens so the encoder ignores them.

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7
"""

import os
import json
import math
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from .token_builder import TokenBuilder


# =============================================================================
# CONSTANTS
# =============================================================================

COSIA_NUM_CLASSES = 18  # Classes 0-17 valid, class 18 (Undefined) → ignore
IGNORE_INDEX = 255

# Resolutions in m/px
RES_AERIAL = 0.2
RES_SPOT = 1.6      # 102.4m / 64px
RES_S2 = 10.0      # Sentinel-2 at ~10m
RES_S1 = 10.0      # Sentinel-1 at ~10m

# Spatial sizes in pixels
SIZE_AERIAL = 512
SIZE_SPOT = 64
SIZE_S2 = 10
SIZE_S1 = 10

# Dataset identifier — included in every sample for callback dispatch
DATASET_NAME = "FlairHub"

# Band counts
N_BANDS_AERIAL = 4   # R, G, B, NIR
N_BANDS_SPOT = 4     # R, G, B, NIR
N_BANDS_S2 = 10      # B02-B08, B8A, B11, B12
N_BANDS_S1 = 2       # VV, VH
N_BANDS_S2_MASK = 2  # snow, cloud

# Modality folder suffixes (appended to domain to form folder name)
MOD_SUFFIXES = {
    "aerial":  "AERIAL_RGBI",
    "spot":    "SPOT_RGBI",
    "s2":      "SENTINEL2_TS",
    "s2_mask": "SENTINEL2_MSK-SC",
    "s1_asc":  "SENTINEL1-ASC_TS",
    "s1_des":  "SENTINEL1-DESC_TS",
    "cosia":   "AERIAL_LABEL-COSIA",
    "lpis":    "ALL_LABEL-LPIS",
}

# GeoPackage date file patterns per modality
GPKG_DATE_PATTERNS = {
    "aerial":  "{domain}_AERIAL_MTD_DATES.gpkg",
    "spot":    "{domain}_SPOT_MTD_DATES.gpkg",
    "s2":      "{domain}_SENTINEL2_MTD_DATES.gpkg",
    "s1_asc":  "{domain}_SENTINEL1-ASC_MTD_DATES.gpkg",
    "s1_des":  "{domain}_SENTINEL1-DESC_MTD_DATES.gpkg",
}

# GeoPackage key column: aerial/spot use patch_id, S1/S2 use zone_id
GPKG_KEY_COL = {
    "aerial":  "patch_id",
    "spot":    "patch_id",
    "s2":      "zone_id",
    "s1_asc":  "zone_id",
    "s1_des":  "zone_id",
}


# LPIS: 3 hierarchy bands in TIF, Band 0 = coarsest (23 classes, values 0-22)
LPIS_NUM_CLASSES = 23
LPIS_BAND = 0


# =============================================================================
# PRE-TRAINING STRATEGY CONSTANTS
# =============================================================================

CROP_AERIAL = 256
CROP_SPOT   = 32
CROP_S2     = 5

CROP_POSITIONS = {
    "top_left": {
        "aerial": (0, 0, CROP_AERIAL, CROP_AERIAL),
        "spot":   (0, 0, CROP_SPOT, CROP_SPOT),
        "s2":     (0, 0, CROP_S2, CROP_S2),
        "labels": (0, 0, CROP_AERIAL, CROP_AERIAL),
    },
    "top_right": {
        "aerial": (0, SIZE_AERIAL - CROP_AERIAL, CROP_AERIAL, CROP_AERIAL),
        "spot":   (0, SIZE_SPOT - CROP_SPOT, CROP_SPOT, CROP_SPOT),
        "s2":     (0, SIZE_S2 - CROP_S2, CROP_S2, CROP_S2),
        "labels": (0, SIZE_AERIAL - CROP_AERIAL, CROP_AERIAL, CROP_AERIAL),
    },
    "bottom_left": {
        "aerial": (SIZE_AERIAL - CROP_AERIAL, 0, CROP_AERIAL, CROP_AERIAL),
        "spot":   (SIZE_SPOT - CROP_SPOT, 0, CROP_SPOT, CROP_SPOT),
        "s2":     (SIZE_S2 - CROP_S2, 0, CROP_S2, CROP_S2),
        "labels": (SIZE_AERIAL - CROP_AERIAL, 0, CROP_AERIAL, CROP_AERIAL),
    },
    "bottom_right": {
        "aerial": (SIZE_AERIAL - CROP_AERIAL, SIZE_AERIAL - CROP_AERIAL,
                   CROP_AERIAL, CROP_AERIAL),
        "spot":   (SIZE_SPOT - CROP_SPOT, SIZE_SPOT - CROP_SPOT,
                   CROP_SPOT, CROP_SPOT),
        "s2":     (SIZE_S2 - CROP_S2, SIZE_S2 - CROP_S2,
                   CROP_S2, CROP_S2),
        "labels": (SIZE_AERIAL - CROP_AERIAL, SIZE_AERIAL - CROP_AERIAL,
                   CROP_AERIAL, CROP_AERIAL),
    },
    "center": {
        "aerial": (128, 128, CROP_AERIAL, CROP_AERIAL),
        "spot":   (16, 16, CROP_SPOT, CROP_SPOT),
        "s2":     (2, 2, CROP_S2, CROP_S2),
        "labels": (128, 128, CROP_AERIAL, CROP_AERIAL),
    },
}

P_FULL    = 1.0
P_S2_ONLY = 0.0
P_STATIC  = 0.0

P_TEMPORAL_MAE   = 0.0
P_CROSSMODAL     = 1.0

N_TIMESTAMPS_ENCODER_A = 6
N_TIMESTAMPS_ENCODER_C = 20

SPAN_MIN_DAYS = 30
SPAN_MAX_DAYS = 90


# =============================================================================
# CROP HELPERS
# =============================================================================

def random_crop_position() -> str:
    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
    return random.choice(positions)


def apply_crop_2d(tensor: torch.Tensor, crop: Tuple[int, int, int, int]) -> torch.Tensor:
    y0, x0, h, w = crop
    return tensor[..., y0:y0+h, x0:x0+w]


def apply_crop_temporal(tensor: torch.Tensor, crop: Tuple[int, int, int, int]) -> torch.Tensor:
    y0, x0, h, w = crop
    return tensor[:, :, y0:y0+h, x0:x0+w]


# =============================================================================
# TEMPORAL SPAN MASKING
# =============================================================================

def temporal_span_mask(
    doys: List[int],
    n_encoder: int = N_TIMESTAMPS_ENCODER_C,
    n_spans: int = None,
    min_span_days: int = SPAN_MIN_DAYS,
    max_span_days: int = SPAN_MAX_DAYS,
) -> Tuple[List[int], List[int]]:
    n = len(doys)
    if n <= n_encoder:
        return list(range(n)), []

    if n_spans is None:
        n_spans = random.randint(1, 3)

    masked_ranges = []
    for _ in range(n_spans):
        start = random.randint(1, 365)
        length = random.randint(min_span_days, max_span_days)
        masked_ranges.append((start, start + length))

    def is_masked(doy):
        for s, e in masked_ranges:
            if e <= 365:
                if s <= doy <= e:
                    return True
            else:
                if doy >= s or doy <= (e - 365):
                    return True
        return False

    encoder_idx = [i for i, d in enumerate(doys) if not is_masked(d)]
    query_idx   = [i for i, d in enumerate(doys) if is_masked(d)]

    if len(encoder_idx) < n_encoder and len(query_idx) > 0:
        deficit = n_encoder - len(encoder_idx)
        stolen = query_idx[:deficit]
        query_idx = query_idx[deficit:]
        encoder_idx = sorted(encoder_idx + stolen)

    if len(encoder_idx) > n_encoder:
        step = len(encoder_idx) / n_encoder
        encoder_idx = [encoder_idx[int(i * step)] for i in range(n_encoder)]

    if len(query_idx) == 0 and n > n_encoder:
        start_idx = n // 3
        end_idx = 2 * n // 3
        query_idx = list(range(start_idx, end_idx))
        encoder_idx = [i for i in range(n) if i not in query_idx]
        if len(encoder_idx) > n_encoder:
            step = len(encoder_idx) / n_encoder
            encoder_idx = [encoder_idx[int(i * step)] for i in range(n_encoder)]

    return encoder_idx, query_idx


def select_evenly_spaced(
    valid_indices: List[int],
    doys: List[int],
    n: int,
) -> List[int]:
    if len(valid_indices) <= n:
        return valid_indices
    sorted_by_doy = sorted(valid_indices, key=lambda i: doys[i])
    step = len(sorted_by_doy) / n
    return [sorted_by_doy[int(i * step)] for i in range(n)]


# =============================================================================
# LABEL DOWNSAMPLING (for S2-only mode)
# =============================================================================

def downsample_labels_majority(
    labels: torch.Tensor,
    target_h: int,
    target_w: int,
    n_classes: int = 256,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    H, W = labels.shape
    bH = H // target_h
    bW = W // target_w

    blocked = labels[:target_h * bH, :target_w * bW].reshape(
        target_h, bH, target_w, bW
    )
    blocked = blocked.permute(0, 2, 1, 3).reshape(target_h * target_w, bH * bW)

    valid_mask = (blocked != ignore_index)
    safe = blocked.clone()
    safe[~valid_mask] = 0

    counts = torch.zeros(target_h * target_w, n_classes, dtype=torch.long)
    counts.scatter_add_(1, safe.long(), valid_mask.long())

    has_valid = counts.sum(dim=1) > 0
    result = torch.full((target_h * target_w,), ignore_index, dtype=labels.dtype)
    result[has_valid] = counts[has_valid].argmax(dim=1).to(labels.dtype)

    return result.reshape(target_h, target_w)


# =============================================================================
# STRATIFIED RECONSTRUCTION QUERY SAMPLING
# =============================================================================

def stratified_recon_sample(
    groups: Dict[float, dict],
    max_queries: int,
) -> torch.Tensor:
    valid_per_group = {}
    for res, g in groups.items():
        tokens = g["tokens"]
        mask = g["mask"]
        valid = tokens[~mask]
        if valid.shape[0] > 0:
            valid_per_group[res] = valid

    if not valid_per_group:
        return torch.zeros(1, 8)

    n_groups = len(valid_per_group)
    per_group_budget = max_queries // n_groups

    sampled_parts = []
    for res, valid in valid_per_group.items():
        n = valid.shape[0]
        k = min(n, per_group_budget)
        if n > k:
            idx = torch.randperm(n)[:k]
            sampled = valid[idx]
        else:
            sampled = valid
        sampled_parts.append(sampled)

    queries = torch.cat(sampled_parts, dim=0)
    queries[:, 4] = queries[:, 0].clone()
    return queries


# =============================================================================
# CLOUD FILTERING
# =============================================================================

def filter_time_series(
    mask_data: torch.Tensor,
    max_cloud_value: float = 1.0,
    max_snow_value: float = 1.0,
    max_fraction_covered: float = 0.05,
) -> torch.Tensor:
    T = mask_data.shape[0]
    H, W = mask_data.shape[2], mask_data.shape[3]
    num_pix = H * W
    threshold = (1 - max_fraction_covered) * num_pix

    select = (mask_data[:, 1, :, :] <= max_cloud_value) & \
             (mask_data[:, 0, :, :] <= max_snow_value)
    valid_counts = select.view(T, -1).sum(dim=1)
    selected_idx = valid_counts >= threshold

    if not selected_idx.any():
        snow_valid = (mask_data[:, 0, :, :] <= max_snow_value).view(T, -1).sum(dim=1)
        selected_idx = snow_valid >= threshold

    if not selected_idx.any():
        selected_idx = torch.ones(T, dtype=torch.bool)

    return selected_idx


# =============================================================================
# PATCH INDEX: CSV-BASED DISCOVERY (official splits) + DIRECTORY FALLBACK
# =============================================================================

CSV_SPLIT_FILES = {
    "train":      "FLAIR-HUB_TRAIN.csv",
    "val":        "FLAIR-HUB_VALID.csv",
    "validation": "FLAIR-HUB_VALID.csv",
    "test":       "FLAIR-HUB_TEST.csv",
}


def parse_patch_id(patch_id: str) -> Dict:
    match = re.match(r"^(D\d+-\w+?)_(.+)_(\d+-\d+)$", patch_id)
    if not match:
        return None
    return {
        "patch_id": patch_id,
        "domain": match.group(1),
        "roi": match.group(2),
        "coords": match.group(3),
    }


def discover_patches_from_csv(csv_path: str) -> List[Dict]:
    import csv

    patches = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            patch_id = row.get("patch_id", "").strip()
            if not patch_id:
                continue

            info = parse_patch_id(patch_id)
            if info is None:
                print(f"[FlairHub] Warning: cannot parse patch_id '{patch_id}'")
                continue

            patches.append(info)

    return patches


def discover_patches(root_path: str) -> List[Dict]:
    patches = []
    seen = set()

    for entry in sorted(os.listdir(root_path)):
        entry_path = os.path.join(root_path, entry)
        if not os.path.isdir(entry_path):
            continue
        if "AERIAL_RGBI" not in entry:
            continue

        domain = entry.replace("_AERIAL_RGBI", "")

        for roi in sorted(os.listdir(entry_path)):
            roi_path = os.path.join(entry_path, roi)
            if not os.path.isdir(roi_path):
                continue

            for fname in sorted(os.listdir(roi_path)):
                if not fname.endswith(".tif"):
                    continue

                base = fname.replace(".tif", "")
                suffix = base.replace(f"{domain}_AERIAL_RGBI_", "")
                match = re.match(r"^(.+)_(\d+-\d+)$", suffix)
                if not match:
                    print(f"[FlairHub] Warning: cannot parse filename {fname}")
                    continue

                parsed_roi = match.group(1)
                coords = match.group(2)
                patch_id = f"{domain}_{parsed_roi}_{coords}"

                if patch_id not in seen:
                    seen.add(patch_id)
                    patches.append({
                        "patch_id": patch_id,
                        "domain": domain,
                        "roi": parsed_roi,
                        "coords": coords,
                    })

    return patches


def get_modality_path(root: str, domain: str, roi: str, coords: str, mod: str) -> str:
    suffix = MOD_SUFFIXES[mod]
    folder = f"{domain}_{suffix}"
    filename = f"{domain}_{suffix}_{roi}_{coords}.tif"
    return os.path.join(root, folder, roi, filename)


# =============================================================================
# BASE CLASS
# =============================================================================

class FlairHubBase(Dataset):
    """
    Shared loading logic for FLAIR-HUB pre-training datasets.
    """

    ENCODER_MODALITIES = ["aerial", "spot", "s2", "s1_asc", "s1_des"]

    MODALITY_BAND_CONFIGS = {
        "aerial":  "bands_flairhub_aerial",
        "spot":    "bands_flairhub_spot",
        "s2":      "bands_flairhub_s2",
        "s1_asc":  "bands_flairhub_s1",
        "s1_des":  "bands_flairhub_s1",
    }

    MODALITY_SPECS = {
        "aerial":  (RES_AERIAL, SIZE_AERIAL, N_BANDS_AERIAL, False),
        "spot":    (RES_SPOT,   SIZE_SPOT,   N_BANDS_SPOT,   False),
        "s2":      (RES_S2,     SIZE_S2,     N_BANDS_S2,     True),
        "s1_asc":  (RES_S1,     SIZE_S1,     N_BANDS_S1,     True),
        "s1_des":  (RES_S1,     SIZE_S1,     N_BANDS_S1,     True),
    }

    def __init__(
        self,
        root_path: str = "./data/FLAIR-HUB/toy/FLAIR-HUB_TOY",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        partition: float = 1.0,
        temporal_dropout: float = 0.0,
        max_timestamps: int = 40,
        train_ratio: float = 0.9,
        seed: int = 42,
        csv_dir: str = None,
        transform=None,
        model=None,
        modality_mode="train",
        subset: str = None,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model
        self.dataset_config = dataset_config
        self.partition = partition
        self.temporal_dropout = temporal_dropout
        self.max_timestamps = max_timestamps

        self.csv_dir = csv_dir or root_path

        self.token_builder = TokenBuilder(look_up)

        self.resolution_indices = {}
        for mod, (res, _, _, _) in self.MODALITY_SPECS.items():
            if res not in self.resolution_indices:
                self.resolution_indices[res] = self.look_up.get_resolution_idx(res)

        self._setup_all_band_indices()

        csv_filename = CSV_SPLIT_FILES.get(mode)
        csv_path = None

        if csv_filename:
            search_dirs = [self.csv_dir, self.root_path, os.path.dirname(self.root_path)]
            for d in search_dirs:
                candidate = os.path.join(d, csv_filename)
                if os.path.exists(candidate):
                    csv_path = candidate
                    break

        if csv_path:
            self.patches = discover_patches_from_csv(csv_path)
            split_source = f"CSV ({os.path.basename(csv_path)} from {os.path.dirname(csv_path)})"
        else:
            all_patches = discover_patches(root_path)
            if not all_patches:
                raise FileNotFoundError(
                    f"No patches found in {root_path}. "
                    f"Expected AERIAL_RGBI folders with TIF files "
                    f"or CSV split files in {self.csv_dir}."
                )

            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(all_patches))
            split_idx = int(len(all_patches) * train_ratio)

            if mode == "train":
                selected = indices[:split_idx]
            elif mode in ("validation", "val"):
                selected = indices[split_idx:]
            else:
                selected = indices

            self.patches = [all_patches[i] for i in selected]
            split_source = f"directory ({train_ratio:.0%}/{1-train_ratio:.0%} split)"

        if partition < 1.0:
            n = max(1, int(len(self.patches) * partition))
            self.patches = self.patches[:n]

        if self.split == "train" and len(self.patches) > 100_000:
            self.patches = self.patches[:100_000]

        self._load_all_date_metadata()
        self._load_norm_stats()

        print(f"[FlairHub] {len(self.patches)} patches for split='{mode}' "
              f"via {split_source}")

        if self.patches:
            p = self.patches[0]
            print(f"[FlairHub] Path diagnostic for first patch: {p['patch_id']}")
            for mod in self.ENCODER_MODALITIES:
                path = get_modality_path(self.root_path, p["domain"], p["roi"], p["coords"], mod)
                exists = os.path.exists(path)
                print(f"  {mod:10s}: {'OK' if exists else 'MISSING'} → {path}")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _setup_all_band_indices(self):
        self.modality_band_info = {}
        self.modality_spectral_idx = {}
        self.modality_num_bands = {}

        for mod in self.ENCODER_MODALITIES:
            yaml_key = self.MODALITY_BAND_CONFIGS[mod]
            bands_info = self.dataset_config[yaml_key]

            parsed, spectral_indices = self._parse_and_build_indices(
                bands_info, mod
            )

            self.modality_band_info[mod] = parsed
            self.modality_spectral_idx[mod] = spectral_indices
            self.modality_num_bands[mod] = len(parsed)

    def _parse_and_build_indices(self, bands_info: dict, modality_name: str):
        all_bands = []
        for name, data in bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        indices = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                if band["bandwidth"] < 0:
                    new_idx = len(self.look_up.table_wave)
                    self.look_up.table_wave[key] = new_idx
                else:
                    raise KeyError(
                        f"Band {band['name']} key={key} not in lookup table "
                        f"for modality '{modality_name}'."
                    )
            indices.append(self.look_up.table_wave[key])

        return all_bands, torch.tensor(indices, dtype=torch.long)

    def _load_all_date_metadata(self):
        self.date_cache = {}

        domains = set(p["domain"] for p in self.patches)

        for mod in list(self.ENCODER_MODALITIES):
            if mod not in GPKG_DATE_PATTERNS:
                continue

            self.date_cache[mod] = {}
            key_col = GPKG_KEY_COL[mod]

            for domain in domains:
                mtd_folder = f"{domain}_ALL_MTD"
                gpkg_name = GPKG_DATE_PATTERNS[mod].format(domain=domain)
                gpkg_path = os.path.join(self.root_path, mtd_folder, gpkg_name)

                if not os.path.exists(gpkg_path):
                    continue

                try:
                    gdf = gpd.read_file(gpkg_path)
                except Exception as e:
                    print(f"[FlairHub] Warning: failed to read {gpkg_path}: {e}")
                    continue

                for _, row in gdf.iterrows():
                    key = row.get(key_col)
                    if key is None:
                        continue

                    if "acquisition_dates" in row.index:
                        raw = row["acquisition_dates"]
                        try:
                            date_dict = json.loads(raw) if isinstance(raw, str) else raw
                            doys = []
                            for idx in sorted(date_dict.keys(), key=int):
                                dt = datetime.strptime(str(date_dict[idx]), "%Y%m%d")
                                doys.append(dt.timetuple().tm_yday)
                            self.date_cache[mod][key] = doys
                        except (ValueError, TypeError, KeyError):
                            pass
                    elif "date" in row.index:
                        try:
                            dt = datetime.strptime(str(row["date"]), "%Y%m%d")
                            self.date_cache[mod][key] = [dt.timetuple().tm_yday]
                        except (ValueError, TypeError):
                            pass

        loaded = {m: len(v) for m, v in self.date_cache.items() if v}
        print(f"[FlairHub] Loaded date metadata: {loaded}")

    # =========================================================================
    # DATE HANDLING
    # =========================================================================

    def _get_doys(self, mod: str, patch_info: dict) -> List[int]:
        if mod not in self.date_cache:
            return [-1]

        key_col = GPKG_KEY_COL.get(mod, "patch_id")
        if key_col == "zone_id":
            key = f"{patch_info['domain']}_{patch_info['roi']}"
        else:
            key = patch_info["patch_id"]

        return self.date_cache[mod].get(key, [-1])

    def _doy_to_time_idx(self, doy: int) -> int:
        if doy < 0:
            return -1
        return self.look_up.get_or_register_time_idx(doy)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_tif(self, filepath: str) -> Optional[torch.Tensor]:
        if not os.path.exists(filepath):
            return None
        with rasterio.open(filepath) as f:
            data = f.read()
        return torch.from_numpy(data.astype(np.float32))

    @staticmethod
    def _normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            return (data - mean[:, None, None]) / std[:, None, None].clamp(min=1e-6)
        elif data.dim() == 4:
            return (data - mean[None, :, None, None]) / std[None, :, None, None].clamp(min=1e-6)
        else:
            return data

    def _load_norm_stats(self):
        stats_filename = "normalization_stats.json"
        stats_path = None
        search_dirs = [self.root_path, os.path.dirname(self.root_path)]
        if hasattr(self, 'csv_dir'):
            search_dirs.insert(0, self.csv_dir)

        for d in search_dirs:
            candidate = os.path.join(d, stats_filename)
            if os.path.exists(candidate):
                stats_path = candidate
                break

        self.norm_stats = {}
        if stats_path is None:
            print(f"[FlairHub] Warning: {stats_filename} not found, normalization disabled")
            return

        with open(stats_path, "r") as f:
            raw = json.load(f)

        for mod in self.ENCODER_MODALITIES:
            if mod in raw:
                self.norm_stats[mod] = {
                    "mean": torch.tensor(raw[mod]["mean"], dtype=torch.float32),
                    "std": torch.tensor(raw[mod]["std"], dtype=torch.float32),
                }

        print(f"[FlairHub] Loaded normalization stats from {stats_path}: "
              f"{list(self.norm_stats.keys())}")

    def _load_modality(self, patch_info: dict, mod: str):
        path = get_modality_path(
            self.root_path,
            patch_info["domain"],
            patch_info["roi"],
            patch_info["coords"],
            mod,
        )
        raw = self._load_tif(path)
        if raw is None:
            return None, [-1], None

        _, _, _, is_temporal = self.MODALITY_SPECS[mod]
        doys = self._get_doys(mod, patch_info)

        if is_temporal:
            n_bands = self.modality_num_bands[mod]
            total_bands = raw.shape[0]
            T = total_bands // n_bands

            if total_bands % n_bands != 0:
                print(f"[FlairHub] Warning: {mod} has {total_bands} bands, "
                      f"not divisible by {n_bands}")
                T = total_bands // n_bands

            H, W = raw.shape[1], raw.shape[2]
            data = raw[:T * n_bands].view(T, n_bands, H, W)

            if len(doys) > T:
                doys = doys[:T]
            elif len(doys) < T:
                doys = doys + [-1] * (T - len(doys))

            if mod == "s2":
                data, doys = self._apply_cloud_filter(patch_info, data, doys)

            T_current = data.shape[0]
            if T_current > self.max_timestamps:
                step = T_current / self.max_timestamps
                keep_idx = [int(i * step) for i in range(self.max_timestamps)]
                data = data[keep_idx]
                doys = [doys[i] for i in keep_idx]

            if (
                self.split == "train"
                and self.temporal_dropout > 0
                and data.shape[0] > 1
            ):
                T_now = data.shape[0]
                keep_n = max(1, int(T_now * (1 - self.temporal_dropout)))
                perm = torch.randperm(T_now)[:keep_n].sort().values
                data = data[perm]
                doys = [doys[i] for i in perm.tolist()]

            T_final = data.shape[0]
            valid_mask = torch.ones(self.max_timestamps, dtype=torch.bool)

            if T_final > 0 and mod in self.norm_stats:
                data = self._normalize(
                    data,
                    self.norm_stats[mod]["mean"],
                    self.norm_stats[mod]["std"],
                )

            if T_final < self.max_timestamps:
                pad_count = self.max_timestamps - T_final
                pad_data = torch.zeros(pad_count, n_bands, H, W, dtype=data.dtype)
                data = torch.cat([data, pad_data], dim=0)
                doys = doys + [-1] * pad_count
                valid_mask[T_final:] = False
            elif T_final > self.max_timestamps:
                data = data[:self.max_timestamps]
                doys = doys[:self.max_timestamps]

            return data, doys, valid_mask

        else:
            if mod in self.norm_stats:
                raw = self._normalize(
                    raw,
                    self.norm_stats[mod]["mean"],
                    self.norm_stats[mod]["std"],
                )
            return raw, doys, None

    def _apply_cloud_filter(self, patch_info, s2_data, s2_doys):
        mask_path = get_modality_path(
            self.root_path,
            patch_info["domain"],
            patch_info["roi"],
            patch_info["coords"],
            "s2_mask",
        )
        mask_raw = self._load_tif(mask_path)
        if mask_raw is None:
            return s2_data, s2_doys

        T_data = s2_data.shape[0]
        T_mask = mask_raw.shape[0] // N_BANDS_S2_MASK

        T = min(T_data, T_mask)
        mask_data = mask_raw[:T * N_BANDS_S2_MASK].view(
            T, N_BANDS_S2_MASK, mask_raw.shape[1], mask_raw.shape[2]
        )
        s2_data = s2_data[:T]
        s2_doys = s2_doys[:T]

        valid_idx = filter_time_series(
            mask_data,
            max_cloud_value=1,
            max_snow_value=1,
            max_fraction_covered=0.05,
        )

        filtered_data = s2_data[valid_idx]
        filtered_doys = [d for d, v in zip(s2_doys, valid_idx.tolist()) if v]

        return filtered_data, filtered_doys

    # =========================================================================
    # TOKEN BUILDING — Multi-resolution, Multi-temporal
    # =========================================================================

    def _build_all_tokens(
        self, patch_info: dict, label: Optional[torch.Tensor] = None
    ) -> Dict[float, dict]:
        res_tokens = {}
        res_masks = {}
        res_bands = {}
        res_size = {}

        for mod in self.ENCODER_MODALITIES:
            res, size, n_bands, is_temporal = self.MODALITY_SPECS[mod]
            res_idx = self.resolution_indices[res]
            spectral_idx = self.modality_spectral_idx[mod]

            data, doys, valid_mask = self._load_modality(patch_info, mod)

            if data is None:
                continue

            if res not in res_tokens:
                res_tokens[res] = []
                res_masks[res] = []
                res_bands[res] = 0
                res_size[res] = size

            if is_temporal:
                T = data.shape[0]
                for t in range(T):
                    time_idx = self._doy_to_time_idx(doys[t])
                    frame = data[t]
                    dummy_label = torch.full(
                        (size, size), IGNORE_INDEX, dtype=torch.long
                    )
                    tokens_t = self.token_builder.build_tokens(
                        image=frame,
                        label=dummy_label,
                        resolution=res,
                        spectral_indices=spectral_idx,
                        resolution_idx=res_idx,
                        time_idx=time_idx,
                    )
                    res_tokens[res].append(tokens_t)

                    n_tok = tokens_t.shape[0]
                    if valid_mask is not None and not valid_mask[t]:
                        res_masks[res].append(torch.ones(n_tok, dtype=torch.bool))
                    else:
                        res_masks[res].append(torch.zeros(n_tok, dtype=torch.bool))

                res_bands[res] += n_bands
            else:
                time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

                if res == RES_AERIAL and label is not None:
                    tok_label = label
                else:
                    tok_label = torch.full(
                        (size, size), IGNORE_INDEX, dtype=torch.long
                    )

                tokens = self.token_builder.build_tokens(
                    image=data,
                    label=tok_label,
                    resolution=res,
                    spectral_indices=spectral_idx,
                    resolution_idx=res_idx,
                    time_idx=time_idx,
                )
                res_tokens[res].append(tokens)
                res_masks[res].append(torch.zeros(tokens.shape[0], dtype=torch.bool))
                res_bands[res] += n_bands

        groups = {}
        for res in res_tokens:
            if not res_tokens[res]:
                continue
            all_tokens = torch.cat(res_tokens[res], dim=0)
            all_masks = torch.cat(res_masks[res], dim=0)

            groups[res] = {
                "tokens": all_tokens,
                "mask": all_masks,
                "shape": (res_bands[res], res_size[res], res_size[res]),
            }

        ALL_EXPECTED = {
            RES_AERIAL: (N_BANDS_AERIAL, SIZE_AERIAL),
            RES_SPOT:   (N_BANDS_SPOT,   SIZE_SPOT),
            RES_S2:     (N_BANDS_S2,     SIZE_S2),
        }
        for res, (n_bands_exp, size_exp) in ALL_EXPECTED.items():
            if res not in groups:
                dummy_tokens = torch.zeros(1, 8)
                dummy_mask = torch.ones(1, dtype=torch.bool)
                groups[res] = {
                    "tokens": dummy_tokens,
                    "mask": dummy_mask,
                    "shape": (n_bands_exp, size_exp, size_exp),
                }

        return groups

    def _load_cosia_label(self, patch_info: dict) -> torch.Tensor:
        path = get_modality_path(
            self.root_path,
            patch_info["domain"],
            patch_info["roi"],
            patch_info["coords"],
            "cosia",
        )
        raw = self._load_tif(path)
        if raw is None:
            return torch.full(
                (SIZE_AERIAL, SIZE_AERIAL), IGNORE_INDEX, dtype=torch.long
            )

        label = raw[0].long()
        label[label >= COSIA_NUM_CLASSES] = IGNORE_INDEX
        return label

    def _load_lpis_label(
        self, patch_info: dict, lpis_band: int = LPIS_BAND
    ) -> torch.Tensor:
        path = get_modality_path(
            self.root_path,
            patch_info["domain"],
            patch_info["roi"],
            patch_info["coords"],
            "lpis",
        )
        raw = self._load_tif(path)
        if raw is None:
            return torch.full(
                (SIZE_AERIAL, SIZE_AERIAL), IGNORE_INDEX, dtype=torch.long
            )

        label = raw[lpis_band].long()
        label[label >= LPIS_NUM_CLASSES] = IGNORE_INDEX
        return label

    # =========================================================================
    # INTERFACE
    # =========================================================================

    def _make_dummy_sample(self, task_name: str) -> dict:
        dummy_tokens = torch.zeros(1, 8)
        dummy_mask = torch.ones(1, dtype=torch.bool)

        groups = {}
        for res, (n_bands, size) in [
            (RES_AERIAL, (N_BANDS_AERIAL, SIZE_AERIAL)),
            (RES_SPOT,   (N_BANDS_SPOT,   SIZE_SPOT)),
            (RES_S2,     (N_BANDS_S2,     SIZE_S2)),
        ]:
            groups[res] = {
                "tokens": dummy_tokens.clone(),
                "mask": dummy_mask.clone(),
                "shape": (n_bands, size, size),
            }

        queries = torch.zeros(1, 8)
        queries[:, 4] = IGNORE_INDEX if task_name != "reconstruction" else 0.0

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": torch.ones(1, dtype=torch.bool),
            "target_resolution": RES_AERIAL,
            "task": task_name,
            "dataset_name": DATASET_NAME,
        }

    def __len__(self):
        return len(self.patches)

    def _load_raw_aerial_image(self, patch_info: dict) -> torch.Tensor:
        path = get_modality_path(
            self.root_path,
            patch_info["domain"],
            patch_info["roi"],
            patch_info["coords"],
            "aerial",
        )
        raw = self._load_tif(path)
        if raw is None:
            return torch.zeros(N_BANDS_AERIAL, SIZE_AERIAL, SIZE_AERIAL)
        return raw

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError("Subclasses must implement __getitem__")


# =============================================================================
# SEGMENTATION: COSIA LAND COVER (18 valid classes)
# =============================================================================

class FlairHubSegCOSIA(FlairHubBase):

    NUM_CLASSES = COSIA_NUM_CLASSES
    TASK_NAME = "flairhub_cosia"

    def __init__(self, **kwargs):
        config_model = kwargs.get("config_model", {})
        self.max_queries = config_model.get("trainer", {}).get(
            "max_tokens_reconstruction", 100_000
        )
        super().__init__(**kwargs)
        print(f"[FlairHub-SegCOSIA] {len(self.patches)} patches, "
              f"{self.NUM_CLASSES} classes, max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        patch_info = self.patches[index]

        label = self._load_cosia_label(patch_info)
        groups = self._build_all_tokens(patch_info, label)

        if not groups:
            return self._make_dummy_sample(self.TASK_NAME)

        res_idx = self.resolution_indices[RES_AERIAL]
        first_spectral_idx = self.modality_spectral_idx["aerial"][0]
        doys = self._get_doys("aerial", patch_info)
        time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

        queries = self.token_builder.build_queries(
            label=label,
            resolution=RES_AERIAL,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=res_idx,
            time_idx=time_idx,
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
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": RES_AERIAL,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }

    def get_viz_sample(self, index: int) -> dict:
        patch_info = self.patches[index]

        cosia_label = self._load_cosia_label(patch_info)
        groups = self._build_all_tokens(patch_info, label=cosia_label)

        tasks = {}
        recon_queries = {}
        for res in [0.2, 1.6, 10.0]:
            if res in groups:
                q = groups[res]["tokens"].clone()
                q[:, 4] = q[:, 0].clone()

                if res == 10.0:
                    first_time = q[:, 7].min()
                    q = q[q[:, 7] == first_time]

                recon_queries[res] = {
                    "queries": q,
                    "queries_mask": torch.zeros(q.shape[0], dtype=torch.bool)
                }

        doys = self._get_doys("aerial", patch_info)
        time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

        tasks["flairhub_cosia"] = self._build_seg_queries(
            cosia_label, time_idx, at_spot_res=False, crop_specs=None
        )

        return {
            "groups": groups,
            "tasks": tasks,
            "recon_viz_queries": recon_queries,
            "target_resolution": 0.2,
            "patch_id": patch_info["patch_id"],
            "dataset_name": DATASET_NAME,
        }


# =============================================================================
# SEGMENTATION: LPIS CROP TYPE (23 classes, hierarchical)
# =============================================================================

class FlairHubSegLPIS(FlairHubBase):

    NUM_CLASSES = LPIS_NUM_CLASSES
    TASK_NAME = "flairhub_lpis"

    def __init__(self, lpis_band: int = LPIS_BAND, **kwargs):
        config_model = kwargs.get("config_model", {})
        self.max_queries = config_model.get("trainer", {}).get(
            "max_tokens_reconstruction", 100_000
        )
        self.lpis_band = lpis_band
        super().__init__(**kwargs)
        print(f"[FlairHub-SegLPIS] {len(self.patches)} patches, "
              f"{self.NUM_CLASSES} classes, band={lpis_band}, "
              f"max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        patch_info = self.patches[index]

        label = self._load_lpis_label(patch_info, self.lpis_band)
        groups = self._build_all_tokens(patch_info, label)

        if not groups:
            return self._make_dummy_sample(self.TASK_NAME)

        res_idx = self.resolution_indices[RES_AERIAL]
        first_spectral_idx = self.modality_spectral_idx["aerial"][0]
        doys = self._get_doys("aerial", patch_info)
        time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

        queries = self.token_builder.build_queries(
            label=label,
            resolution=RES_AERIAL,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=res_idx,
            time_idx=time_idx,
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
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": RES_AERIAL,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }


# =============================================================================
# RECONSTRUCTION
# =============================================================================

class FlairHubReconstruction(FlairHubBase):

    TASK_NAME = "reconstruction"

    VIZ_GROUPS = {
        RES_AERIAL: "aerial_0.2m",
        RES_SPOT:   "spot_1.6m",
        RES_S2:     "sentinel_10m",
    }

    def __init__(self, max_queries: int = 200_000, **kwargs):
        self.max_queries = max_queries
        super().__init__(**kwargs)
        print(f"[FlairHub-Recon] {len(self.patches)} patches, "
              f"max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        patch_info = self.patches[index]

        groups = self._build_all_tokens(patch_info, label=None)

        if not groups:
            return self._make_dummy_sample(self.TASK_NAME)

        valid_parts = []
        for g in groups.values():
            tokens = g["tokens"]
            mask = g["mask"]
            valid = tokens[~mask]
            if valid.shape[0] > 0:
                valid_parts.append(valid)

        if not valid_parts:
            valid_parts = [g["tokens"] for g in groups.values() if g["tokens"].shape[0] > 0]
            if not valid_parts:
                return self._make_dummy_sample(self.TASK_NAME)

        all_valid = torch.cat(valid_parts, dim=0)

        queries = all_valid.clone()
        queries[:, 4] = queries[:, 0].clone()

        N = queries.shape[0]
        n_queries = min(N, self.max_queries)
        if N > n_queries:
            perm = torch.randperm(N)[:n_queries]
            queries = queries[perm]

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": RES_AERIAL,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }

    def get_viz_sample(self, index: int) -> dict:
        patch_info = self.patches[index]
        groups = self._build_all_tokens(patch_info, label=None)

        available_res = list(groups.keys())
        viz_res = random.choice(available_res)
        viz_name = self.VIZ_GROUPS.get(viz_res, f"{viz_res}m")

        g = groups[viz_res]
        tokens = g["tokens"]
        mask = g["mask"]

        valid_tokens = tokens[~mask]

        if viz_res == RES_S2:
            time_indices = valid_tokens[:, 7].unique()
            if len(time_indices) > 0:
                first_time = time_indices[0]
                valid_tokens = valid_tokens[valid_tokens[:, 7] == first_time]

        queries = valid_tokens.clone()
        queries[:, 4] = queries[:, 0].clone()
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        if viz_res == RES_AERIAL:
            n_bands = N_BANDS_AERIAL
            spatial = SIZE_AERIAL
        elif viz_res == RES_SPOT:
            n_bands = N_BANDS_SPOT
            spatial = SIZE_SPOT
        else:
            n_bands = N_BANDS_S2
            spatial = SIZE_S2

        image = self._load_raw_aerial_image(patch_info)

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": viz_res,
            "task": self.TASK_NAME,
            "image": image,
            "image_shape": (n_bands, spatial, spatial),
            "viz_group": viz_name,
            "dataset_name": DATASET_NAME,
        }


# =============================================================================
# BANDS INFO HELPER (for Lookup_encoding initialization)
# =============================================================================

def create_flairhub_bands_info() -> dict:
    from .lookup_encoding import ABSTRACT_CHANNELS

    return {
        "bands_flairhub_aerial": {
            "RED":   {"central_wavelength": 660, "bandwidth": 80,  "idx": 0},
            "GREEN": {"central_wavelength": 560, "bandwidth": 80,  "idx": 1},
            "BLUE":  {"central_wavelength": 490, "bandwidth": 80,  "idx": 2},
            "NIR":   {"central_wavelength": 835, "bandwidth": 130, "idx": 3},
        },
        "bands_flairhub_spot": {
            "SPOT_R":   {"central_wavelength": 660, "bandwidth": 70,  "idx": 0},
            "SPOT_G":   {"central_wavelength": 560, "bandwidth": 60,  "idx": 1},
            "SPOT_B":   {"central_wavelength": 490, "bandwidth": 70,  "idx": 2},
            "SPOT_NIR": {"central_wavelength": 825, "bandwidth": 130, "idx": 3},
        },
        "bands_flairhub_s2": {
            "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 0},
            "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 1},
            "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 2},
            "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 3},
            "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 4},
            "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 5},
            "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 6},
            "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 7},
            "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 8},
            "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
        },
        "bands_flairhub_s1": {
            "VV": {
                "bandwidth": ABSTRACT_CHANNELS["VV"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV"]["central_wavelength"],
                "idx": 0,
            },
            "VH": {
                "bandwidth": ABSTRACT_CHANNELS["VH"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH"]["central_wavelength"],
                "idx": 1,
            },
        },
    }


# =============================================================================
# MULTI-TASK WITH PRE-TRAINING STRATEGY
# =============================================================================

class FlairHubMultiTask(FlairHubBase):

    TASK_NAME = "multitask"

    def __init__(
        self,
        tasks: list = None,
        max_queries_seg: int = 65_536,
        max_queries_recon: int = 200_000,
        max_samples: int = None,
        lpis_band: int = LPIS_BAND,
        p_full: float = P_FULL,
        p_s2_only: float = P_S2_ONLY,
        p_static: float = P_STATIC,
        n_timestamps_a: int = N_TIMESTAMPS_ENCODER_A,
        n_timestamps_c: int = N_TIMESTAMPS_ENCODER_C,
        **kwargs,
    ):
        if tasks is None:
            tasks = ["flairhub_cosia", "flairhub_lpis", "reconstruction"]
        self.enabled_tasks = tasks
        self.max_queries_seg = max_queries_seg
        self.max_queries_recon = max_queries_recon
        self.lpis_band = lpis_band

        self.p_full = p_full
        self.p_s2_only = p_s2_only
        self.p_static = p_static

        self.n_timestamps_a = n_timestamps_a
        self.n_timestamps_c = n_timestamps_c

        super().__init__(**kwargs)

        if max_samples is not None and len(self.patches) > max_samples:
            self.patches = self.patches[:max_samples]

        print(f"[FlairHub-MultiTask] {len(self.patches)} patches, "
              f"tasks={self.enabled_tasks}, "
              f"dropout: full={p_full}, s2_only={p_s2_only}, static={p_static}")

    # =========================================================================
    # CONFIG SELECTION
    # =========================================================================

    def _choose_config(self) -> Dict:
        r = random.random()
        mode = "full"

        recon_type = None
        if "reconstruction" in self.enabled_tasks:
            if mode == "static":
                recon_type = "crossmodal"
            else:
                recon_type = "temporal_mae" if random.random() < P_TEMPORAL_MAE else "crossmodal"

        configs = {
            "full": {
                "mode": "full",
                "modalities": ["aerial", "spot", "s2", "s1_asc", "s1_des"],
                "crop": random_crop_position(),
                "n_timestamps": self.n_timestamps_a,
                "seg_at_spot": False,
                "recon_type": recon_type,
            },
            "s2_only": {
                "mode": "s2_only",
                "modalities": ["s2", "s1_asc", "s1_des"],
                "crop": random_crop_position(),
                "n_timestamps": self.n_timestamps_c,
                "seg_at_spot": True,
                "recon_type": recon_type,
            },
            "static": {
                "mode": "static",
                "modalities": ["aerial", "spot"],
                "crop": random_crop_position(),
                "n_timestamps": 0,
                "seg_at_spot": False,
                "recon_type": recon_type,
            },
        }
        return configs[mode]

    # =========================================================================
    # TOKEN BUILDING WITH CROP + MODALITY SUBSET
    # =========================================================================

    def _build_tokens_with_config(
        self,
        patch_info: dict,
        config: Dict,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[float, dict]:
        crop_name = config["crop"]
        crop_specs = CROP_POSITIONS[crop_name]
        n_ts = config["n_timestamps"]

        res_tokens = {}
        res_masks = {}
        res_bands = {}
        res_size = {}

        for mod in config["modalities"]:
            res, size, n_bands, is_temporal = self.MODALITY_SPECS[mod]
            res_idx = self.resolution_indices[res]
            spectral_idx = self.modality_spectral_idx[mod]

            data, doys, valid_mask = self._load_modality(patch_info, mod)
            if data is None:
                continue

            if mod in ("aerial",):
                crop_box = crop_specs["aerial"]
            elif mod in ("spot",):
                crop_box = crop_specs["spot"]
            elif mod in ("s2", "s1_asc", "s1_des"):
                crop_box = crop_specs["s2"]
            else:
                crop_box = None

            if crop_box is not None:
                if is_temporal:
                    data = apply_crop_temporal(data, crop_box)
                else:
                    data = apply_crop_2d(data, crop_box)
                _, _, ch, cw = crop_box
                size = ch

            if is_temporal and n_ts > 0 and valid_mask is not None:
                valid_ts_idx = [
                    t for t in range(valid_mask.shape[0]) if valid_mask[t]
                ]

                selected = select_evenly_spaced(
                    valid_ts_idx, doys, n=n_ts
                )

                H_curr, W_curr = data.shape[2], data.shape[3]
                new_data = torch.zeros(n_ts, n_bands, H_curr, W_curr,
                                       dtype=data.dtype)
                new_doys = [-1] * n_ts
                new_mask = torch.zeros(n_ts, dtype=torch.bool)

                for i, t in enumerate(selected):
                    if i >= n_ts:
                        break
                    new_data[i] = data[t]
                    new_doys[i] = doys[t]
                    new_mask[i] = True

                data = new_data
                doys = new_doys
                valid_mask = new_mask

            elif is_temporal and n_ts == 0:
                continue

            if res not in res_tokens:
                res_tokens[res] = []
                res_masks[res] = []
                res_bands[res] = 0
                res_size[res] = size

            if is_temporal:
                T = data.shape[0]
                for t in range(T):
                    time_idx = self._doy_to_time_idx(doys[t])
                    frame = data[t]
                    dummy_label = torch.full(
                        (size, size), IGNORE_INDEX, dtype=torch.long
                    )
                    tokens_t = self.token_builder.build_tokens(
                        image=frame,
                        label=dummy_label,
                        resolution=res,
                        spectral_indices=spectral_idx,
                        resolution_idx=res_idx,
                        time_idx=time_idx,
                    )
                    res_tokens[res].append(tokens_t)

                    n_tok = tokens_t.shape[0]
                    is_padded = valid_mask is not None and not valid_mask[t]
                    mask_val = torch.ones if is_padded else torch.zeros
                    res_masks[res].append(mask_val(n_tok, dtype=torch.bool))

                res_bands[res] += n_bands
            else:
                time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

                if res == RES_AERIAL and label is not None:
                    tok_label = apply_crop_2d(label, crop_specs["labels"])
                else:
                    tok_label = torch.full(
                        (size, size), IGNORE_INDEX, dtype=torch.long
                    )

                tokens = self.token_builder.build_tokens(
                    image=data,
                    label=tok_label,
                    resolution=res,
                    spectral_indices=spectral_idx,
                    resolution_idx=res_idx,
                    time_idx=time_idx,
                )
                res_tokens[res].append(tokens)
                res_masks[res].append(
                    torch.zeros(tokens.shape[0], dtype=torch.bool)
                )
                res_bands[res] += n_bands

        groups = {}
        for res in res_tokens:
            if not res_tokens[res]:
                continue
            all_tokens = torch.cat(res_tokens[res], dim=0)
            all_masks = torch.cat(res_masks[res], dim=0)

            groups[res] = {
                "tokens": all_tokens,
                "mask": all_masks,
                "shape": (res_bands[res], res_size[res], res_size[res]),
            }

        EXPECTED_GROUPS = {
            "full":    {RES_AERIAL: (N_BANDS_AERIAL, CROP_AERIAL),
                        RES_SPOT:   (N_BANDS_SPOT,   CROP_SPOT),
                        RES_S2:     (N_BANDS_S2,     CROP_S2)},
            "s2_only": {RES_S2:     (N_BANDS_S2,     CROP_S2)},
            "static":  {RES_AERIAL: (N_BANDS_AERIAL, CROP_AERIAL),
                        RES_SPOT:   (N_BANDS_SPOT,   CROP_SPOT)},
        }
        expected = EXPECTED_GROUPS.get(config["mode"], {})
        for res, (n_bands_exp, size_exp) in expected.items():
            if res not in groups:
                dummy_tokens = torch.zeros(1, 8)
                dummy_mask = torch.ones(1, dtype=torch.bool)
                groups[res] = {
                    "tokens": dummy_tokens,
                    "mask": dummy_mask,
                    "shape": (n_bands_exp, size_exp, size_exp),
                }

        return groups

    def _build_seg_queries(
        self,
        label: torch.Tensor,
        time_idx: int,
        at_spot_res: bool = False,
        crop_specs: Optional[dict] = None,
    ) -> dict:
        if crop_specs is not None:
            label = apply_crop_2d(label, crop_specs["labels"])

        if at_spot_res:
            H, W = label.shape
            target_h = H // 8
            target_w = W // 8
            n_cls = max(COSIA_NUM_CLASSES, LPIS_NUM_CLASSES)
            label = downsample_labels_majority(
                label, target_h, target_w,
                n_classes=n_cls, ignore_index=IGNORE_INDEX,
            )
            query_res = RES_SPOT
        else:
            query_res = RES_AERIAL

        res_idx = self.resolution_indices[query_res]
        first_spectral_idx = self.modality_spectral_idx["aerial"][0]

        queries = self.token_builder.build_queries(
            label=label,
            resolution=query_res,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=res_idx,
            time_idx=time_idx,
        )
        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries_seg,
            ignore_index=IGNORE_INDEX,
            prioritize_valid=True,
        )

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": queries_mask}

    def _build_recon_queries(
        self,
        groups: dict,
        recon_type: str,
        patch_info: dict,
        config: Dict,
        self_recon_fraction: float = 0.5,
    ) -> dict:
        total_budget = self.max_queries_recon
        self_budget = int(total_budget * self_recon_fraction)
        spec_budget = total_budget - self_budget

        self_queries = stratified_recon_sample(groups, self_budget)

        spec_result = None

        if recon_type == "temporal_mae":
            spec_result = self._build_temporal_mae_recon(patch_info, config)
            if spec_result["queries_mask"].all():
                spec_result = None

        if spec_result is None and recon_type in ("crossmodal", "temporal_mae"):
            spec_result = self._build_crossmodal_recon(patch_info, config)
            if spec_result["queries_mask"].all():
                spec_result = None

        if spec_result is not None:
            spec_queries = spec_result["queries"]
            if spec_queries.shape[0] > spec_budget:
                idx = torch.randperm(spec_queries.shape[0])[:spec_budget]
                spec_queries = spec_queries[idx]
            queries = torch.cat([self_queries, spec_queries], dim=0)
        else:
            queries = stratified_recon_sample(groups, total_budget)

        mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": mask}

    def _build_crossmodal_recon(self, patch_info: dict, config: Dict) -> dict:
        crop_specs = CROP_POSITIONS[config["crop"]]

        data, doys, _ = self._load_modality(patch_info, "spot")
        if data is None:
            return self._dummy_recon_queries()

        data = apply_crop_2d(data, crop_specs["spot"])

        C, H, W = data.shape
        res_idx = self.resolution_indices[RES_SPOT]
        spectral_idx = self.modality_spectral_idx["spot"]
        time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

        dummy_label = torch.full((H, W), IGNORE_INDEX, dtype=torch.long)
        tokens = self.token_builder.build_tokens(
            image=data,
            label=dummy_label,
            resolution=RES_SPOT,
            spectral_indices=spectral_idx,
            resolution_idx=res_idx,
            time_idx=time_idx,
        )

        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()

        N = queries.shape[0]
        if N > self.max_queries_recon:
            idx = torch.randperm(N)[:self.max_queries_recon]
            queries = queries[idx]

        mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": mask}

    def _build_temporal_mae_recon(self, patch_info: dict, config: Dict) -> dict:
        data, doys, valid_mask = self._load_modality(patch_info, "s2")
        if data is None or valid_mask is None:
            return self._dummy_recon_queries()

        crop_specs = CROP_POSITIONS[config["crop"]]

        valid_ts = [t for t in range(valid_mask.shape[0]) if valid_mask[t]]
        if len(valid_ts) < 5:
            return self._dummy_recon_queries()

        valid_doys = [doys[t] for t in valid_ts]

        enc_idx, qry_idx = temporal_span_mask(
            valid_doys, n_encoder=self.n_timestamps_c
        )

        if len(qry_idx) == 0:
            return self._dummy_recon_queries()

        query_abs_idx = [valid_ts[i] for i in qry_idx]

        res_idx = self.resolution_indices[RES_S2]
        spectral_idx = self.modality_spectral_idx["s2"]

        data = apply_crop_temporal(data, crop_specs["s2"])
        _, _, H, W = data.shape
        size = H

        all_queries = []
        for t_abs in query_abs_idx:
            frame = data[t_abs]
            time_idx = self._doy_to_time_idx(doys[t_abs])
            dummy_label = torch.full((size, size), IGNORE_INDEX, dtype=torch.long)

            tokens = self.token_builder.build_tokens(
                image=frame,
                label=dummy_label,
                resolution=RES_S2,
                spectral_indices=spectral_idx,
                resolution_idx=res_idx,
                time_idx=time_idx,
            )
            queries = tokens.clone()
            queries[:, 4] = queries[:, 0].clone()
            all_queries.append(queries)

        queries = torch.cat(all_queries, dim=0)

        N = queries.shape[0]
        if N > self.max_queries_recon:
            idx = torch.randperm(N)[:self.max_queries_recon]
            queries = queries[idx]

        mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": mask}

    @staticmethod
    def _dummy_recon_queries() -> dict:
        dummy = torch.zeros(1, 8)
        return {"queries": dummy, "queries_mask": torch.ones(1, dtype=torch.bool)}

    # =========================================================================
    # __getitem__
    # =========================================================================

    def __getitem__(self, index: int) -> dict:
        patch_info = self.patches[index]

        if self.split == "train":
            config = self._choose_config()
        else:
            config = {
                "mode": "full",
                "modalities": ["aerial", "spot", "s2", "s1_asc", "s1_des"],
                "crop": "center",
                "n_timestamps": self.n_timestamps_a,
                "seg_at_spot": False,
                "recon_type": "crossmodal" if "reconstruction" in self.enabled_tasks else None,
            }

        crop_specs = CROP_POSITIONS[config["crop"]]

        doys = self._get_doys("aerial", patch_info)
        time_idx = self._doy_to_time_idx(doys[0] if doys else -1)

        cosia_label = None
        lpis_label = None
        label_for_tokens = None

        if "flairhub_cosia" in self.enabled_tasks:
            cosia_label = self._load_cosia_label(patch_info)
            label_for_tokens = cosia_label

        if "flairhub_lpis" in self.enabled_tasks:
            lpis_label = self._load_lpis_label(patch_info, self.lpis_band)
            if label_for_tokens is None:
                label_for_tokens = lpis_label

        groups = self._build_tokens_with_config(
            patch_info, config, label=label_for_tokens
        )

        if not groups:
            return self._make_dummy_multitask()

        tasks = {}

        if "flairhub_cosia" in self.enabled_tasks and cosia_label is not None:
            tasks["flairhub_cosia"] = self._build_seg_queries(
                cosia_label, time_idx,
                at_spot_res=config["seg_at_spot"],
                crop_specs=crop_specs,
            )

        if "flairhub_lpis" in self.enabled_tasks and lpis_label is not None:
            tasks["flairhub_lpis"] = self._build_seg_queries(
                lpis_label, time_idx,
                at_spot_res=config["seg_at_spot"],
                crop_specs=crop_specs,
            )

        if "reconstruction" in self.enabled_tasks and config["recon_type"]:
            tasks["reconstruction"] = self._build_recon_queries(
                groups, config["recon_type"], patch_info, config,
            )

        return {
            "groups": groups,
            "tasks": tasks,
            "target_resolution": RES_AERIAL,
            "dataset_name": DATASET_NAME,
        }

    # =========================================================================
    # DUMMY SAMPLE (for missing modalities)
    # =========================================================================

    def _make_dummy_multitask(self) -> dict:
        dummy_tokens = torch.zeros(1, 8)
        dummy_mask = torch.ones(1, dtype=torch.bool)

        groups = {}
        for res, (n_bands, size) in [
            (RES_AERIAL, (N_BANDS_AERIAL, SIZE_AERIAL)),
            (RES_SPOT,   (N_BANDS_SPOT,   SIZE_SPOT)),
            (RES_S2,     (N_BANDS_S2,     SIZE_S2)),
        ]:
            groups[res] = {
                "tokens": dummy_tokens.clone(),
                "mask": dummy_mask.clone(),
                "shape": (n_bands, size, size),
            }

        tasks = {}
        for task in self.enabled_tasks:
            q = torch.zeros(1, 8)
            if task != "reconstruction":
                q[:, 4] = IGNORE_INDEX
            tasks[task] = {
                "queries": q,
                "queries_mask": torch.ones(1, dtype=torch.bool),
            }

        return {
            "groups": groups,
            "tasks": tasks,
            "target_resolution": RES_AERIAL,
            "dataset_name": DATASET_NAME,
        }

    # =========================================================================
    # VIZ SAMPLE (full queries, no subsampling, no dropout, center crop)
    # =========================================================================

    def get_recon_viz_sample(self, index: int) -> dict:
        """
        Build a reconstruction-only viz sample with per-modality metadata.

        Returns a multi-task-format sample (with batch["tasks"] dict) so it
        goes through forward_multitask cleanly. The extra viz keys are
        prefixed with ``_viz_`` and must be popped BEFORE collating (they
        contain nested dicts / non-tensor types that collate_grouped
        can't stack).

        For temporal modalities (S2, S1) only the first valid timestamp
        is included to produce a clean single-frame reconstruction.
        """
        patch_info = self.patches[index]

        # Full tokens — no crop, no dropout
        groups = self._build_all_tokens(patch_info, label=None)

        modality_info = {}
        all_queries_list = []
        offset = 0

        for mod in self.ENCODER_MODALITIES:
            res, size, n_bands, is_temporal = self.MODALITY_SPECS[mod]
            if res not in groups:
                continue

            g = groups[res]
            tokens = g["tokens"]
            mask = g["mask"]
            valid_tokens = tokens[~mask]

            if valid_tokens.shape[0] == 0:
                continue

            # ── Filter tokens belonging to this modality ────────────────
            spectral_idx_set = set(self.modality_spectral_idx[mod].tolist())
            spec_mask = torch.tensor(
                [int(t[3].item()) in spectral_idx_set for t in valid_tokens],
                dtype=torch.bool,
            )
            mod_tokens = valid_tokens[spec_mask]

            if mod_tokens.shape[0] == 0:
                continue

            # ── Temporal: keep first valid timestamp only ───────────────
            if is_temporal:
                time_vals = mod_tokens[:, 7].unique(sorted=True)
                valid_times = time_vals[time_vals >= 0]
                if len(valid_times) > 0:
                    mod_tokens = mod_tokens[mod_tokens[:, 7] == valid_times[0]]
                else:
                    continue

            if mod_tokens.shape[0] == 0:
                continue

            # ── Build reconstruction queries ────────────────────────────
            queries = mod_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()   # GT value → label slot

            n_q = queries.shape[0]
            modality_info[mod] = {
                "offset": offset,
                "count": n_q,
                "shape": (n_bands, size, size),
                "spectral_indices": self.modality_spectral_idx[mod],
            }
            all_queries_list.append(queries)
            offset += n_q

        if all_queries_list:
            combined_queries = torch.cat(all_queries_list, dim=0)
        else:
            combined_queries = torch.zeros(1, 8)

        # ── Return multi-task format for forward_multitask ──────────────
        return {
            # Keys that collate_grouped knows how to handle:
            "groups": groups,
            "tasks": {
                "reconstruction": {
                    "queries": combined_queries,
                    "queries_mask": torch.zeros(
                        combined_queries.shape[0], dtype=torch.bool
                    ),
                },
            },
            "target_resolution": RES_AERIAL,
            "dataset_name": DATASET_NAME,
            # ── Viz extras: pop() these BEFORE calling collate_grouped ──
            "_viz_modality_info": modality_info,
            "_viz_image": self._load_raw_aerial_image(patch_info),
            "_viz_patch_id": patch_info["patch_id"],
            "_viz_n_real": combined_queries.shape[0],
        }