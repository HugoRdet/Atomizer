"""
Cashew Segmentation Dataset for Atomiser
==========================================

7-class semantic segmentation of cashew plantations on Sentinel-2 imagery
(geo-bench m-cashew-plant).

Source files (geo-bench-1.0/segmentation_v1.0/m-cashew-plant/):
    band_stats.json         — per-band mean/std/percentiles
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
    {sample_name}.hdf5      — 12 bands + label, [256, 256] each

HDF5 contents (date-suffixed keys):
    "01 - Coastal aerosol_YYYY-MM-DD"   (60m)
    "02 - Blue_YYYY-MM-DD"               (10m)
    "03 - Green_YYYY-MM-DD"              (10m)
    "04 - Red_YYYY-MM-DD"                (10m)
    "05 - Vegetation Red Edge_YYYY-MM-DD"  (20m)
    "06 - Vegetation Red Edge_YYYY-MM-DD"  (20m)
    "07 - Vegetation Red Edge_YYYY-MM-DD"  (20m)
    "08 - NIR_YYYY-MM-DD"                (10m)
    "08A - Vegetation Red Edge_YYYY-MM-DD" (20m)
    "09 - Water vapour_YYYY-MM-DD"       (60m)
    "11 - SWIR_YYYY-MM-DD"               (20m)
    "12 - SWIR_YYYY-MM-DD"               (20m)
    "Cloud Probability_YYYY-MM-DD"        (dropped)
    "label"                               [256, 256] int64 in [0, 6]

Output format:
    {
        "groups": {
            10.0: {                          # S2 placeholder resolution
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (12, H, W),
            },
        },
        "queries":           [M, 8],
        "queries_mask":      [M],
        "label":             [H, W] long (0..6),
        "task":              "segmentation",
        "target_resolution": 10.0,
        "image":             [12, H, W],
    }

Augmentations (training only): D4 group.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .token_grouping import *
from .token_builder import TokenBuilder


class CashewDataset(Dataset):
    """Cashew (geo-bench m-cashew-plant) 7-class segmentation dataset for Atomiser."""

    S2_RESOLUTION = 10.0
    NUM_BANDS = 12
    NUM_CLASSES = 6
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE = 256
    TASK_NAME = "segmentation"

    # Label scheme (from cashew README):
    #   0 = unlabeled / no-data       → IGNORE_INDEX
    #   1 = Well-managed plantation   → 0
    #   2 = Poorly-managed plantation → 1
    #   3 = Non-plantation            → 2
    #   4 = Residential               → 3
    #   5 = Background                → 4
    #   6 = Uncertain                 → 5
    # Stored labels are 1-indexed; we remap to 0-indexed for cross-entropy
    # and treat 0 (no-data) as IGNORE.

    # Band key prefixes inside HDF5 (date-suffixed at load time).
    # Order: 10m bands first (B02 B03 B04 B08), then 20m (B05 B06 B07 B08A B11 B12),
    # then 60m (B01 B09). Drops Cirrus (B10) and Cloud Probability (geo-bench convention).
    BAND_PREFIXES = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "08 - NIR",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
        "01 - Coastal aerosol",
        "09 - Water vapour",
    ]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path     = root_path
        self.split         = mode
        self.look_up       = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens                  = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction  = config_model["trainer"]["max_tokens_reconstruction"]

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            self.band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Normalization tensors ───────────────────────────
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in self.band_stats:
                raise KeyError(
                    f"[Cashew] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[prefix]["mean"])
            stds.append(self.band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        self.bands_info = dataset_config["bands_cashew"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.S2_RESOLUTION)

        print(f"[Cashew] task={self.TASK_NAME}, split={mode} → "
              f"{len(self.sample_names)} samples")
        print(f"[Cashew] bands ({self.NUM_BANDS}): "
              f"{[p.split(' - ')[0] for p in self.BAND_PREFIXES]}")
        print(f"[Cashew] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"[Cashew] resolution idx: {self.resolution_idx} "
              f"(GSD={self.S2_RESOLUTION} m/px)")
        print(f"[Cashew] D4 augment: {'ON' if mode == 'train' else 'OFF'}")
        print(f"[Cashew] num_classes: {self.NUM_CLASSES}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[Cashew] No key with prefix '{prefix}' in {path}"
                    )
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            label = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))   # [12, 256, 256]
        label = torch.from_numpy(label)                      # [256, 256]

        # ── NaN cleanup, normalize ──────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        # ── Label remap: 1-indexed → 0-indexed, 0 → IGNORE ──
        # README: classes 1..6 are real, 0 is no-data/unlabeled.
        # CE expects 0-indexed targets, so shift down by 1.
        ignore_mask = (label == 0) | (label > self.NUM_CLASSES)
        label = label - 1
        label = torch.where(ignore_mask, torch.full_like(label, self.IGNORE_INDEX), label)

        # ── D4 augmentation (training only) ─────────────────
        if self.split == "train":
            image, label = self._d4_augment(image, label)

        C, H, W = image.shape

        # ── Build tokens ────────────────────────────────────
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label.long(),
            resolution=self.S2_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Build queries: one per pixel ─────────────────────
        first_spectral_idx = self.spectral_indices[0]
        seg_queries = self.token_builder.build_queries(
            label=label.long(),
            resolution=self.S2_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Subsample image tokens if too many ──────────────
        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        # ── Subsample queries: training only ────────────────
        if self.split == "train":
            queries = self.token_builder.subsample_queries(
                seg_queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )
        else:
            queries = seg_queries

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(queries.shape[0])

        return {
            "groups": {
                self.S2_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label.long(),
            "task":              self.TASK_NAME,
            "target_resolution": self.S2_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        name = self.sample_names[index]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            label = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))
        label = torch.from_numpy(label)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        # ── Label remap: 1-indexed → 0-indexed, 0 → IGNORE ──
        ignore_mask = (label == 0) | (label > self.NUM_CLASSES)
        label = label - 1
        label = torch.where(ignore_mask, torch.full_like(label, self.IGNORE_INDEX), label)

        C, H, W = image.shape

        image_tokens = self.token_builder.build_tokens(
            image=image, label=label.long(),
            resolution=self.S2_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label.long(),
            resolution=self.S2_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(queries.shape[0])

        return {
            "groups": {
                self.S2_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label.long(),
            "task":              self.TASK_NAME,
            "target_resolution": self.S2_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw    = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[Cashew] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:25s} → "
                  f"bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[Cashew] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)