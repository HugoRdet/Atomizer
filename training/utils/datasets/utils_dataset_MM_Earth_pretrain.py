"""
MMEarth Pre-training Datasets — Multi-task
============================================

Four dataset classes sharing a common base for MMEarth HDF5 data:

1. MMEarthSegESA          — ESA WorldCover segmentation (11 classes)
2. MMEarthSegDW           — Dynamic World segmentation (9 classes)
3. MMEarthReconstruction  — Reconstruction from latent bottleneck
4. MMEarthMultiTask       — Unified encode-once multi-task dataset

All classes share:
    - HDF5 loading (sentinel2, sentinel1, aster, canopy_height_eth)
    - Band configuration and normalization
    - Temporal encoding via S2_DATE → DOY → time_idx
    - TokenBuilder for token construction

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7
"""

import os
import json
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .token_builder import TokenBuilder


# =============================================================================
# CONSTANTS
# =============================================================================

# Dataset identifier — included in every sample for callback dispatch
DATASET_NAME = "MMEarth"

# ESA WORLDCOVER CLASS REMAPPING: ESA codes → contiguous 0-10, anything else → 255
ESA_CLASS_MAP = {
    10: 0,   # Tree cover
    20: 1,   # Shrubland
    30: 2,   # Grassland
    40: 3,   # Cropland
    50: 4,   # Built-up
    60: 5,   # Bare / sparse vegetation
    70: 6,   # Snow and ice
    80: 7,   # Permanent water bodies
    90: 8,   # Herbaceous wetland
    95: 9,   # Mangroves
    100: 10, # Moss and lichen
}
ESA_NUM_CLASSES = 11

# Dynamic World: 0-8 already contiguous
DW_NUM_CLASSES = 9


# =============================================================================
# BASE CLASS
# =============================================================================

class MMEarthBase(Dataset):
    """
    Shared loading logic for MMEarth HDF5 pre-training datasets.

    Handles:
        - HDF5 data loading (multi-modal)
        - Band configuration and spectral index setup
        - Normalization from band_stats.json
        - Temporal encoding from tile_info.json (S2_DATE → DOY → time_idx)
        - Token construction via TokenBuilder
    """

    RESOLUTION = 10.0
    IMAGE_SIZE = 128
    IGNORE_INDEX = 255

    # Modality → YAML key mapping
    MODALITY_BAND_CONFIGS = {
        "sentinel2":         "bands_mmearth_s2",
        "sentinel1_asc":     "bands_mmearth_s1",
        "sentinel1_desc":    "bands_mmearth_s1",
        "aster":             "bands_mmearth_aster",
        "canopy_height_eth": "bands_mmearth_canopy",
    }

    # Subset → filename mapping
    SUBSET_FILENAMES = {
        "MMEarth":     "data_1M_v001",
        "MMEarth64":   "data_1M_v001_64",
        "MMEarth100k": "data_100k_v001",
    }

    # Band indices within HDF5 arrays
    S2_HDF5_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # 12 bands (skip B10)
    S1_ASC_INDICES  = [0, 1]
    S1_DESC_INDICES = [4, 5]

    def __init__(
        self,
        root_path: str = "./data/MM-Earth",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        subset: str = "MMEarth",
        # Unused but kept for interface compatibility
        transform=None,
        model=None,
        modality_mode="train",
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model
        self.dataset_config = dataset_config
        self.subset = subset

        # Token builder
        self.token_builder = TokenBuilder(look_up)

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Which modalities to use
        self.modalities = list(self.MODALITY_BAND_CONFIGS.keys())

        # Parse band info from YAML and build spectral indices
        self._setup_all_band_indices()

        # Load HDF5 dataset
        self._load_h5_dataset()

        # Load normalization stats
        self._load_band_stats()

        # Load temporal info (S2_DATE per tile)
        self._load_temporal_info()

        # Compute sizes
        self._compute_sizes()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _setup_all_band_indices(self):
        """Parse band configs and build spectral index mappings."""
        self.modality_band_info = {}
        self.modality_spectral_idx = {}
        self.modality_num_bands = {}

        all_spectral = []

        for modality in self.modalities:
            yaml_key = self.MODALITY_BAND_CONFIGS[modality]
            bands_info = self.dataset_config[yaml_key]

            parsed, spectral_indices = self._parse_and_build_indices(
                bands_info, modality
            )

            self.modality_band_info[modality] = parsed
            self.modality_spectral_idx[modality] = spectral_indices
            self.modality_num_bands[modality] = len(parsed)

            all_spectral.append(spectral_indices)

        self.all_spectral_indices = torch.cat(all_spectral, dim=0)

    def _parse_and_build_indices(self, bands_info: dict, modality_name: str):
        """Parse band info dict and register spectral indices in lookup table."""
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

    def _load_h5_dataset(self):
        """Load HDF5 path and split indices."""
        filename = self.SUBSET_FILENAMES.get(self.subset, self.subset)
        self.h5_path = os.path.join(self.root_path, f"{filename}.h5")
        splits_path = os.path.join(self.root_path, f"{filename}_splits.json")

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        split_map = {"train": "train", "validation": "val", "test": "test"}
        split_key = split_map.get(self.split, self.split)

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self.tile_indices = splits[split_key]
        self.h5 = None

    def _load_band_stats(self):
        """Load normalization statistics."""
        filename = self.SUBSET_FILENAMES.get(self.subset, self.subset)
        stats_path = os.path.join(self.root_path, f"{filename}_band_stats.json")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.band_stats = json.load(f)
        else:
            print(f"[MMEarth] Warning: band stats not found at {stats_path}")
            self.band_stats = None

    def _load_temporal_info(self):
        """
        Load tile_info.json and build tile_idx → time_idx mapping.

        Extracts S2_DATE per tile → converts to DOY → registers in lookup table.
        Tiles without dates get TIME_IDX = -1.
        """
        filename = self.SUBSET_FILENAMES.get(self.subset, self.subset)
        tile_info_path = os.path.join(self.root_path, f"{filename}_tile_info.json")

        self.tile_time_idx = {}

        if not os.path.exists(tile_info_path):
            print(f"[MMEarth] Warning: tile_info not found at {tile_info_path}. "
                  f"Using time_idx=-1 for all tiles.")
            return

        with open(tile_info_path, "r") as f:
            tile_info = json.load(f)

        registered = 0
        for tile_key, info in tile_info.items():
            s2_date = info.get("S2_DATE")
            if s2_date is None:
                continue

            try:
                dt = datetime.strptime(s2_date, "%Y-%m-%d")
                doy = dt.timetuple().tm_yday
                time_idx = self.look_up.get_or_register_time_idx(doy)
                self.tile_time_idx[tile_key] = time_idx
                registered += 1
            except (ValueError, TypeError):
                continue

        print(f"[MMEarth] Registered temporal info for {registered} tiles "
              f"({self.look_up.num_time_indices - 1} unique DOY values)")

    def _get_time_idx(self, tile_idx) -> int:
        """Get time_idx for a tile. Returns -1 if no temporal info."""
        tile_key = str(tile_idx)
        return self.tile_time_idx.get(tile_key, -1)

    def _compute_sizes(self):
        """Compute total band/token counts."""
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE
        self.total_bands = sum(self.modality_num_bands.values())
        self.total_tokens = self.total_bands * H * W

    def _get_h5(self):
        """Lazy HDF5 file handle (for multiprocessing compatibility)."""
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_and_merge(self, tile_idx: int) -> torch.Tensor:
        """Load all modalities for a tile and merge into [C, H, W]."""
        h5 = self._get_h5()
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        merged_bands = []

        for modality in self.modalities:
            data = self._extract_modality(h5, tile_idx, modality)
            n_bands = self.modality_num_bands[modality]

            if data is None:
                merged_bands.append(torch.zeros(n_bands, H, W))
                continue

            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = torch.clamp(data, -10, 10)

            if data.shape[1] != H or data.shape[2] != W:
                data = F.interpolate(
                    data.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False
                ).squeeze(0)

            merged_bands.append(data)

        return torch.cat(merged_bands, dim=0)

    def _extract_modality(self, h5, tile_idx: int, modality: str) -> torch.Tensor:
        """Extract and normalize a single modality from HDF5."""
        h5_key_map = {
            "sentinel2":         "sentinel2",
            "sentinel1_asc":     "sentinel1",
            "sentinel1_desc":    "sentinel1",
            "aster":             "aster",
            "canopy_height_eth": "canopy_height_eth",
        }

        h5_key = h5_key_map.get(modality)
        if h5_key is None or h5_key not in h5:
            return None

        raw = torch.from_numpy(h5[h5_key][tile_idx]).float()

        if modality == "sentinel2":
            data = raw[self.S2_HDF5_INDICES]
        elif modality == "sentinel1_asc":
            data = raw[self.S1_ASC_INDICES]
        elif modality == "sentinel1_desc":
            data = raw[self.S1_DESC_INDICES]
        elif modality == "aster":
            data = raw[0:1]
        elif modality == "canopy_height_eth":
            data = raw[0:1]
        else:
            data = raw

        if torch.isnan(data).all() or (data == 0).all():
            return None

        # Normalize
        stats_key_map = {
            "sentinel2":         "sentinel2_l2a",
            "sentinel1":         "sentinel1",
            "aster":             "aster",
            "canopy_height_eth": "canopy_height_eth",
        }
        stats_key = stats_key_map.get(h5_key, h5_key)

        if self.band_stats is not None and stats_key in self.band_stats:
            stats = self.band_stats[stats_key]
            mean = torch.tensor(stats["mean"], dtype=torch.float32)
            std = torch.tensor(stats["std"], dtype=torch.float32)
            std = std.clamp(min=1e-6)

            if modality == "sentinel2":
                mean = mean[self.S2_HDF5_INDICES]
                std = std[self.S2_HDF5_INDICES]
            elif modality == "sentinel1_asc":
                mean = mean[self.S1_ASC_INDICES]
                std = std[self.S1_ASC_INDICES]
            elif modality == "sentinel1_desc":
                mean = mean[self.S1_DESC_INDICES]
                std = std[self.S1_DESC_INDICES]
            elif modality in ("aster", "canopy_height_eth"):
                mean = mean[0:1]
                std = std[0:1]

            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
            data = (data - mean) / std

        return data

    # =========================================================================
    # LABEL LOADING
    # =========================================================================

    def _load_esa_worldcover(self, tile_idx: int) -> torch.Tensor:
        """
        Load ESA WorldCover label and remap to contiguous 0-10.

        Returns:
            label: [H, W] long tensor, unmapped values → 255
        """
        h5 = self._get_h5()
        raw = torch.from_numpy(h5["esa_worldcover"][tile_idx][0]).long()

        label = torch.full_like(raw, self.IGNORE_INDEX)
        for esa_code, new_idx in ESA_CLASS_MAP.items():
            label[raw == esa_code] = new_idx

        return label

    def _load_dynamic_world(self, tile_idx: int) -> torch.Tensor:
        """
        Load Dynamic World label (already 0-8 contiguous).

        Returns:
            label: [H, W] long tensor, values ≥ 9 → 255
        """
        h5 = self._get_h5()
        raw = torch.from_numpy(h5["dynamic_world"][tile_idx][0]).long()

        label = raw.clone()
        label[raw >= DW_NUM_CLASSES] = self.IGNORE_INDEX

        return label

    # =========================================================================
    # INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError("Subclasses must implement __getitem__")


# =============================================================================
# SEGMENTATION: ESA WORLDCOVER (11 classes)
# =============================================================================

class MMEarthSegESA(MMEarthBase):
    """
    ESA WorldCover segmentation (11 classes).

    Classes: Tree cover, Shrubland, Grassland, Cropland, Built-up,
             Bare/sparse, Snow/ice, Water, Wetland, Mangroves, Moss/lichen.
    """

    NUM_CLASSES = ESA_NUM_CLASSES
    TASK_NAME = "esa_worldcover"

    def __init__(self, **kwargs):
        config_model = kwargs.get("config_model", {})
        self.max_queries = config_model.get("trainer", {}).get(
            "max_tokens_reconstruction", 100_000
        )
        super().__init__(**kwargs)

        print(f"[MMEarth-SegESA] {len(self.tile_indices)} samples, "
              f"{self.NUM_CLASSES} classes, max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        label = self._load_esa_worldcover(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]

        tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        first_spectral_idx = self.all_spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )

        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX,
            prioritize_valid=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }

    def get_viz_sample(self, index: int) -> dict:
        """
        Visualization sample: full queries (no subsampling) + raw image/label.
        """
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        label = self._load_esa_worldcover(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]

        tokens = self.token_builder.build_tokens(
            image=image, label=label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        first_spectral_idx = self.all_spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label, resolution=self.RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
            # Viz extras
            "image": image,
            "label": label,
            "image_shape": (C, H, W),
        }


# =============================================================================
# SEGMENTATION: DYNAMIC WORLD (9 classes)
# =============================================================================

class MMEarthSegDW(MMEarthBase):
    """
    Dynamic World segmentation (9 classes).

    Classes: Water, Trees, Grass, Flooded vegetation, Crops,
             Shrub/scrub, Built, Bare, Snow/ice.
    """

    NUM_CLASSES = DW_NUM_CLASSES
    TASK_NAME = "dynamic_world"

    def __init__(self, **kwargs):
        config_model = kwargs.get("config_model", {})
        self.max_queries = config_model.get("trainer", {}).get(
            "max_tokens_reconstruction", 100_000
        )
        super().__init__(**kwargs)

        print(f"[MMEarth-SegDW] {len(self.tile_indices)} samples, "
              f"{self.NUM_CLASSES} classes, max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        label = self._load_dynamic_world(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]

        tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        first_spectral_idx = self.all_spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )

        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX,
            prioritize_valid=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }

    def get_viz_sample(self, index: int) -> dict:
        """
        Visualization sample: full queries (no subsampling) + raw image/label.
        """
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        label = self._load_dynamic_world(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]

        tokens = self.token_builder.build_tokens(
            image=image, label=label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        first_spectral_idx = self.all_spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label, resolution=self.RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
            # Viz extras
            "image": image,
            "label": label,
            "image_shape": (C, H, W),
        }


# =============================================================================
# RECONSTRUCTION
# =============================================================================

class MMEarthReconstruction(MMEarthBase):
    """
    Reconstruction from latent bottleneck.

    All tokens go to the encoder. A random subset of tokens become
    queries with col 4 = target reflectance (from col 0).
    The compression bottleneck (millions of tokens → hundreds of latents)
    is the challenge, not masking.
    """

    TASK_NAME = "reconstruction"

    def __init__(self, max_queries: int = 200_000, **kwargs):
        self.max_queries = max_queries
        super().__init__(**kwargs)

        print(f"[MMEarth-Recon] {len(self.tile_indices)} samples, "
              f"max_queries={self.max_queries}")

    def __getitem__(self, index: int) -> dict:
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()

        N = queries.shape[0]
        n_queries = min(N, self.max_queries)
        if N > n_queries:
            perm = torch.randperm(N)[:n_queries]
            queries = queries[perm]

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
        }

    def get_viz_sample(self, index: int) -> dict:
        """
        Visualization sample: ALL tokens as queries (no subsampling).

        Queries are ordered band-major: [pixel0_band0, pixel0_band1, ..., pixelN_bandC]
        so the callback can reshape to [C, H, W] for image reconstruction.
        """
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        time_idx = self._get_time_idx(tile_idx)

        C = image.shape[0]
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()
        n_real = queries.shape[0]

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "task": self.TASK_NAME,
            "dataset_name": DATASET_NAME,
            # Viz extras
            "image_shape": (C, H, W),
            "n_real": n_real,
        }


# =============================================================================
# MULTI-TASK: ENCODE-ONCE
# =============================================================================

class MMEarthMultiTask(MMEarthBase):
    """
    Unified multi-task dataset for MMEarth.

    Loads each sample ONCE and returns queries for all enabled tasks.
    The trainer encodes the input tokens once, then decodes per-task.

    Args:
        tasks: List of tasks to enable. Default: all three.
            Options: "esa_worldcover", "dynamic_world", "reconstruction"
        max_queries_seg: Max queries per segmentation task.
        max_queries_recon: Max queries for reconstruction.
        max_samples: Cap on number of training samples (None = use all).
        **kwargs: Passed to MMEarthBase.
    """

    TASK_NAME = "multitask"

    def __init__(
        self,
        tasks: list = None,
        max_queries_seg: int = 100_000,
        max_queries_recon: int = 200_000,
        max_samples: int = None,
        **kwargs,
    ):
        if tasks is None:
            tasks = ["esa_worldcover", "dynamic_world", "reconstruction"]
        self.enabled_tasks = tasks
        self.max_queries_seg = max_queries_seg
        self.max_queries_recon = max_queries_recon

        super().__init__(**kwargs)

        # Cap dataset size (deterministic: takes first N after split ordering)
        if max_samples is not None and len(self.tile_indices) > max_samples:
            self.tile_indices = self.tile_indices[:max_samples]

        print(f"[MMEarth-MultiTask] {len(self.tile_indices)} samples, "
              f"tasks={self.enabled_tasks}, "
              f"max_queries_seg={max_queries_seg}, "
              f"max_queries_recon={max_queries_recon}")

    # =========================================================================
    # QUERY BUILDERS (per task)
    # =========================================================================

    def _build_seg_queries(self, label: torch.Tensor, time_idx: int) -> dict:
        """Build segmentation queries from a label map."""
        first_spectral_idx = self.all_spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries_seg,
            ignore_index=self.IGNORE_INDEX,
            prioritize_valid=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": queries_mask}

    def _build_recon_queries(self, tokens: torch.Tensor) -> dict:
        """Build reconstruction queries by cloning tokens."""
        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()

        N = queries.shape[0]
        n_queries = min(N, self.max_queries_recon)
        if N > n_queries:
            perm = torch.randperm(N)[:n_queries]
            queries = queries[perm]

        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        return {"queries": queries, "queries_mask": queries_mask}

    # =========================================================================
    # __getitem__
    # =========================================================================

    def __getitem__(self, index: int) -> dict:
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        time_idx = self._get_time_idx(tile_idx)
        C = image.shape[0]

        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        tasks = {}

        if "esa_worldcover" in self.enabled_tasks:
            esa_label = self._load_esa_worldcover(tile_idx)
            tasks["esa_worldcover"] = self._build_seg_queries(esa_label, time_idx)

        if "dynamic_world" in self.enabled_tasks:
            dw_label = self._load_dynamic_world(tile_idx)
            tasks["dynamic_world"] = self._build_seg_queries(dw_label, time_idx)

        if "reconstruction" in self.enabled_tasks:
            tasks["reconstruction"] = self._build_recon_queries(tokens)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "tasks": tasks,
            "target_resolution": self.RESOLUTION,
            "dataset_name": DATASET_NAME,
        }

    # =========================================================================
    # VIZ SAMPLE (full queries, no subsampling)
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """
        Return full (non-subsampled) queries for visualization.
        Includes raw image and labels for overlay rendering.
        """
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        time_idx = self._get_time_idx(tile_idx)
        C = image.shape[0]

        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
        tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.all_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=time_idx,
        )
        attention_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        first_spectral_idx = self.all_spectral_indices[0]
        tasks = {}
        labels = {}

        if "esa_worldcover" in self.enabled_tasks:
            esa_label = self._load_esa_worldcover(tile_idx)
            labels["esa_worldcover"] = esa_label
            queries = self.token_builder.build_queries(
                label=esa_label, resolution=self.RESOLUTION,
                first_spectral_idx=first_spectral_idx,
                resolution_idx=self.resolution_idx,
                time_idx=time_idx,
            )
            tasks["esa_worldcover"] = {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }

        if "dynamic_world" in self.enabled_tasks:
            dw_label = self._load_dynamic_world(tile_idx)
            labels["dynamic_world"] = dw_label
            queries = self.token_builder.build_queries(
                label=dw_label, resolution=self.RESOLUTION,
                first_spectral_idx=first_spectral_idx,
                resolution_idx=self.resolution_idx,
                time_idx=time_idx,
            )
            tasks["dynamic_world"] = {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }

        if "reconstruction" in self.enabled_tasks:
            recon_queries = tokens.clone()
            recon_queries[:, 4] = recon_queries[:, 0].clone()
            tasks["reconstruction"] = {
                "queries": recon_queries,
                "queries_mask": torch.zeros(recon_queries.shape[0], dtype=torch.bool),
            }

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (C, H, W),
                },
            },
            "tasks": tasks,
            "target_resolution": self.RESOLUTION,
            "dataset_name": DATASET_NAME,
            # Viz extras
            "image": image,
            "labels": labels,
            "image_shape": (C, H, W),
            "n_real": tokens.shape[0],
        }


# =============================================================================
# BANDS INFO HELPER (for Lookup_encoding initialization)
# =============================================================================

def create_mmearth_bands_info() -> dict:
    """
    Create bands_info dict for MMEarth, compatible with Lookup_encoding.

    Covers: Sentinel-2 (12 bands), Sentinel-1 (VV, VH),
            ASTER (elevation), Canopy Height.
    """
    from .lookup_encoding import ABSTRACT_CHANNELS

    bands_info = {
        "bands_mmearth_s2": {
            "B01": {"bandwidth": 20,  "central_wavelength": 443, "idx": 0},
            "B02": {"bandwidth": 65,  "central_wavelength": 490, "idx": 1},
            "B03": {"bandwidth": 35,  "central_wavelength": 560, "idx": 2},
            "B04": {"bandwidth": 30,  "central_wavelength": 665, "idx": 3},
            "B05": {"bandwidth": 15,  "central_wavelength": 705, "idx": 4},
            "B06": {"bandwidth": 15,  "central_wavelength": 740, "idx": 5},
            "B07": {"bandwidth": 20,  "central_wavelength": 783, "idx": 6},
            "B08": {"bandwidth": 115, "central_wavelength": 842, "idx": 7},
            "B8A": {"bandwidth": 20,  "central_wavelength": 865, "idx": 8},
            "B09": {"bandwidth": 20,  "central_wavelength": 945, "idx": 9},
            "B11": {"bandwidth": 90,  "central_wavelength": 1610, "idx": 10},
            "B12": {"bandwidth": 180, "central_wavelength": 2190, "idx": 11},
        },
        "bands_mmearth_s1": {
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
        "bands_mmearth_aster": {
            "ELEVATION": {
                "bandwidth": ABSTRACT_CHANNELS["ELEVATION"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["ELEVATION"]["central_wavelength"],
                "idx": 0,
            },
        },
        "bands_mmearth_canopy": {
            "CANOPY_HEIGHT": {
                "bandwidth": -13,
                "central_wavelength": -13,
                "idx": 0,
            },
        },
    }
    return bands_info