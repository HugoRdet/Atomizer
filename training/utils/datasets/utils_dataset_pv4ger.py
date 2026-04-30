"""
PV4GER Segmentation Dataset for Atomiser
==========================================

Binary segmentation of solar PV panels in aerial RGB imagery (geo-bench m-pv4ger-seg).

Source files (geo-bench-1.0/segmentation_v1.0/m-pv4ger-seg/):
    band_stats.json         — per-band mean/std/percentiles
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
    {sample_name}.hdf5      — 3 bands + label, [320, 320] uint8 each

HDF5 contents (clean keys, no date suffix):
    "Red":   [320, 320] uint8
    "Green": [320, 320] uint8
    "Blue":  [320, 320] uint8
    "label": [320, 320] uint8 in {0, 1}

Output format (compatible with Atomiser segmentation training):
    {
        "groups": {
            0.1: {                              # placeholder resolution (aerial)
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (3, H, W),
            },
        },
        "queries":           [M, 8],
        "queries_mask":      [M],
        "label":             [H, W] long (0 or 1),
        "task":              "segmentation",
        "target_resolution": 0.1,
        "image":             [3, H, W],
    }

Augmentations (training only):
    D4 group: 4 rotations × 2 flips = 8 transforms.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .token_grouping import *
from .token_builder import TokenBuilder


class PV4GERDataset(Dataset):
    """PV4GER (geo-bench m-pv4ger-seg) binary segmentation dataset for Atomiser."""

    AERIAL_RESOLUTION = 0.1   # meters per pixel — placeholder for aerial imagery
    NUM_BANDS = 3
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE = 320           # native size, no cropping needed
    TASK_NAME = "segmentation"

    # Band keys are clean (no date suffix, no "02 - " prefix).
    # Order matches HLS BurnScars / Sen1Floods11: [Blue, Green, Red].
    BAND_KEYS = ["Blue", "Green", "Red"]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/segmentation_v1.0/m-pv4ger-seg",
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

        # ── Normalization tensors (bands only, ignore "label" entry) ──
        means, stds = [], []
        for key in self.BAND_KEYS:
            if key not in self.band_stats:
                raise KeyError(
                    f"[PV4GER] Band '{key}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[key]["mean"])
            stds.append(self.band_stats[key]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        # Reads bands_pv4ger from the YAML dataset config.
        self.bands_info = dataset_config["bands_pv4ger"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.AERIAL_RESOLUTION)

        print(f"[PV4GER] task={self.TASK_NAME}, split={mode} → "
              f"{len(self.sample_names)} samples")
        print(f"[PV4GER] bands ({self.NUM_BANDS}): {self.BAND_KEYS}")
        print(f"[PV4GER] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE} (no cropping)")
        print(f"[PV4GER] resolution idx: {self.resolution_idx} "
              f"(GSD={self.AERIAL_RESOLUTION} m/px)")
        print(f"[PV4GER] D4 augment: {'ON' if mode == 'train' else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 group: random flip + 90° rotation, applied identically."""
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

        # ── Load 3 bands + label ────────────────────────────
        bands = []
        with h5py.File(path, "r") as f:
            for key in self.BAND_KEYS:
                if key not in f:
                    raise KeyError(
                        f"[PV4GER] Band '{key}' not in {path}. "
                        f"Keys: {list(f.keys())}"
                    )
                bands.append(np.asarray(f[key], dtype=np.float32))
            label = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))   # [3, 320, 320]
        label = torch.from_numpy(label)                      # [320, 320]

        # ── NaN cleanup, normalize ──────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        # ── D4 augmentation (training only) ─────────────────
        if self.split == "train":
            image, label = self._d4_augment(image, label)

        C, H, W = image.shape

        # ── Build tokens: per-pixel-per-band ────────────────
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label.long(),
            resolution=self.AERIAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Build queries: one per pixel ─────────────────────
        first_spectral_idx = self.spectral_indices[0]
        seg_queries = self.token_builder.build_queries(
            label=label.long(),
            resolution=self.AERIAL_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Subsample image tokens if too many ──────────────
        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        # ── Subsample queries: training only (memory) ───────
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
                self.AERIAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label.long(),
            "task":              self.TASK_NAME,
            "target_resolution": self.AERIAL_RESOLUTION,
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
            for key in self.BAND_KEYS:
                bands.append(np.asarray(f[key], dtype=np.float32))
            label = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))
        label = torch.from_numpy(label)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        C, H, W = image.shape

        image_tokens = self.token_builder.build_tokens(
            image=image, label=label.long(),
            resolution=self.AERIAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label.long(),
            resolution=self.AERIAL_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(queries.shape[0])

        return {
            "groups": {
                self.AERIAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label.long(),
            "task":              self.TASK_NAME,
            "target_resolution": self.AERIAL_RESOLUTION,
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

        print(f"[PV4GER] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:6s} → "
                  f"bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[PV4GER] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)