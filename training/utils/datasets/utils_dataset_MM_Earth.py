"""
MMEarth MAE Dataset — Multi-modal MAE pre-training
====================================================

Loads MMEarth HDF5 directly (no torchgeo dependency).

Modalities used (continuous, available at inference):
    - sentinel2:          12 bands @ 10m (B10 cirrus excluded)
    - sentinel1_asc:      2 bands (VV, VH) @ 10m
    - sentinel1_desc:     2 bands (VV, VH) @ 10m
    - aster:              1 band (elevation) @ 10m (resampled)
    - canopy_height_eth:  1 band (height) @ 10m (resampled)

Modes:
    - reconstruction: queries = image tokens, col 4 = reflectance (default)
    - segmentation:   queries = 1 per pixel, col 4 = class label (future)

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7
"""

import os
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .token_builder import TokenBuilder
from .block_masking import (
    generate_spatial_block_mask,
    expand_mask_to_tokens,
    apply_mask_to_tokens,
    build_mae_queries,
)


class MMEarthMAEDataset(Dataset):

    # ── Constants ───────────────────────────────────────────
    RESOLUTION = 10.0
    IMAGE_SIZE = 128
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1

    # ── Modality → YAML key mapping ────────────────────────
    MODALITY_BAND_CONFIGS = {
        "sentinel2":          "bands_mmearth_s2",
        "sentinel1_asc":      "bands_mmearth_s1",
        "sentinel1_desc":     "bands_mmearth_s1",
        "aster":              "bands_mmearth_aster",
        "canopy_height_eth":  "bands_mmearth_canopy",
    }

    # ── Subset → filename mapping ──────────────────────────
    SUBSET_FILENAMES = {
        "MMEarth":     "data_1M_v001",
        "MMEarth64":   "data_1M_v001_64",
        "MMEarth100k": "data_100k_v001",
    }

    # ── Band indices within HDF5 arrays ──────────────────────
    S2_HDF5_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # 12 bands (skip B10)
    S1_ASC_INDICES  = [0, 1]
    S1_DESC_INDICES = [4, 5]

    def __init__(
        self,
        root_path: str = "./data/MM-Earth",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        mask_ratio: float = 0.75,
        block_size: int = 8,
        max_queries: int = 200_000,
        subset: str = "MMEarth",
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model
        self.dataset_config = dataset_config
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.max_queries = max_queries
        self.subset = subset

        # Token builder
        self.token_builder = TokenBuilder(look_up)

        # Mode: reconstruction or segmentation
        self.reconstruction = config_model["trainer"].get("mode", "reconstruction") == "reconstruction"

        if self.reconstruction:
            print(f"[MMEarth-MAE] Mode: RECONSTRUCTION (queries = image tokens, col 4 = reflectance)")
        else:
            print(f"[MMEarth-MAE] Mode: SEGMENTATION (queries = pixels, col 4 = class label)")

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Which modalities to use
        self.modalities = list(self.MODALITY_BAND_CONFIGS.keys())

        # Parse band info from YAML and build spectral indices
        self._setup_all_band_indices()

        # Load HDF5 dataset
        self._load_h5_dataset()
        
        if len(self.tile_indices) > 100_000:
            self.tile_indices = self.tile_indices[:100_000]

        # Load normalization stats
        self._load_band_stats()

        # Compute sizes
        self._compute_target_sizes()

        print(f"[MMEarth-MAE] Loaded {len(self.tile_indices)} samples for '{mode}'")
        print(f"[MMEarth-MAE] Modalities: {self.modalities}")
        print(f"[MMEarth-MAE] Total bands: {self.total_bands} → "
              f"{self.total_tokens} tokens/sample")
        print(f"[MMEarth-MAE] Mask ratio: {self.mask_ratio} → "
              f"~{self.target_visible} visible tokens")
        print(f"[MMEarth-MAE] Max queries: {self.max_queries}")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _setup_all_band_indices(self):
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
        print(f"[MMEarth-MAE] Merged spectral indices: {self.all_spectral_indices.shape[0]} bands")

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
                    print(f"[MMEarth-MAE] Registered abstract channel "
                          f"{band['name']} {key} → spectral idx {new_idx}")
                else:
                    raise KeyError(
                        f"[MMEarth-MAE] Band {band['name']} key={key} not in "
                        f"lookup table for modality '{modality_name}'.\n"
                        f"Available keys: {list(self.look_up.table_wave.keys())}"
                    )
            indices.append(self.look_up.table_wave[key])

        tag = " (abstract)" if all_bands and all_bands[0]["bandwidth"] < 0 else ""
        print(f"[MMEarth-MAE] {modality_name}: {len(all_bands)} bands{tag}")

        return all_bands, torch.tensor(indices, dtype=torch.long)

    def _load_h5_dataset(self):
        filename = self.SUBSET_FILENAMES.get(self.subset, self.subset)
        self.h5_path = os.path.join(self.root_path, f"{filename}.h5")
        splits_path = os.path.join(self.root_path, f"{filename}_splits.json")

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}\n"
                f"Download MMEarth from https://vishalned.github.io/mmearth/"
            )

        split_map = {"train": "train", "validation": "val", "test": "test"}
        split_key = split_map.get(self.split, self.split)

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self.tile_indices = splits[split_key]
        self.h5 = None

    def _load_band_stats(self):
        filename = self.SUBSET_FILENAMES.get(self.subset, self.subset)
        stats_path = os.path.join(self.root_path, f"{filename}_band_stats.json")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.band_stats = json.load(f)
        else:
            print(f"[MMEarth-MAE] Warning: band stats not found at {stats_path}")
            self.band_stats = None

    def _get_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def _compute_target_sizes(self):
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE
        self.total_bands = sum(self.modality_num_bands.values())
        self.total_tokens = self.total_bands * H * W
        self.target_visible = int((1.0 - self.mask_ratio) * self.total_tokens)

    # =========================================================================
    # LOADING
    # =========================================================================

    def _load_and_merge(self, tile_idx: int) -> torch.Tensor:
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
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, index: int) -> dict:
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        spectral_indices = self.all_spectral_indices
        C = image.shape[0]
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask = torch.zeros(tokens.shape[0])

        if self.reconstruction:
            queries = tokens.clone()
            queries[:, 4] = queries[:, 0].clone()

            # Subsample queries if needed
            N = queries.shape[0]
            n_queries = min(N, self.max_queries)
            if N > n_queries:
                perm = torch.randperm(N)[:n_queries]
                queries = queries[perm]
        else:
            first_spectral_idx = spectral_indices[0]
            queries = self.token_builder.build_queries(
                label=dummy_label,
                resolution=self.RESOLUTION,
                first_spectral_idx=first_spectral_idx,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
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
        }

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """
        Viz sample — mode-aware.
        Reconstruction: all tokens as queries, col 4 = reflectance.
        Segmentation: all pixels as queries (future).
        """
        tile_idx = self.tile_indices[index]
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        image = self._load_and_merge(tile_idx)
        spectral_indices = self.all_spectral_indices
        C = image.shape[0]
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        if self.reconstruction:
            tokens[:, 4] = tokens[:, 0].clone()

            N_real = tokens.shape[0]
            mask = torch.zeros(tokens.shape[0], dtype=torch.bool)
            queries = tokens.clone()
            queries_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

            return {
                "groups": {
                    self.RESOLUTION: {
                        "tokens": tokens,
                        "mask": mask,
                        "shape": (C, H, W),
                    },
                },
                "queries": queries,
                "queries_mask": queries_mask,
                "target_resolution": self.RESOLUTION,
                "image": image,
                "image_shape": (C, H, W),
                "n_real": N_real,
            }
        else:
            first_spectral_idx = spectral_indices[0]
            queries = self.token_builder.build_queries(
                label=dummy_label,
                resolution=self.RESOLUTION,
                first_spectral_idx=first_spectral_idx,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
            attention_mask = torch.zeros(tokens.shape[0])

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
                "image_shape": (C, H, W),
                "n_real": tokens.shape[0],  
            }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _empty_sample(self) -> dict:
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE
        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": torch.zeros(self.total_tokens, 8),
                    "mask": torch.ones(self.total_tokens),
                    "shape": (self.total_bands, H, W),
                },
            },
            "queries": torch.zeros(1, 8),
            "queries_mask": torch.ones(1, dtype=torch.bool),
        }