"""
ForestNet Dataset for Atomiser (Classification, multi-task ready)
==================================================================

Single-temporal classification dataset in the grouped-token format.

Emits "task": "classification" so the collate function can stack labels
correctly and the trainer can dispatch to the right loss/metrics.

Source files (geo-bench-1.0/classification_v1.0/m-forestnet/):
    band_stats.json         — per-band mean/std
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
    label_map.json          — {class_idx_str: [sample_name, ...]}
    {sample_name}.hdf5      — 6 bands, [332, 332] uint8 each

HDF5 keys are date-suffixed (e.g. "02 - Blue_2014-01-01"). We match keys
by prefix at load time so the date suffix is handled transparently.

Output format:
    {
        "groups": {
            15.0: {                          # Landsat resolution
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (6, H, W),
            },
        },
        "queries":           [1, 8],          # dummy — unused for classification
        "queries_mask":      [1],
        "label":             0-d long tensor (class index in [0, 12)),
        "task":              "classification",
        "target_resolution": 15.0,
        "image":             [6, H, W],
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


class ForestNetDataset(Dataset):
    """ForestNet (geo-bench m-forestnet) classification dataset for Atomiser."""

    LANDSAT_RESOLUTION = 15.0
    NUM_BANDS = 6
    NUM_CLASSES = 12
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE_NATIVE = 332
    TASK_NAME = "classification"

    BAND_PREFIXES = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - NIR",
        "06 - SWIR1",
        "07 - SWIR2",
    ]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-forestnet",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        crop_size: int = 320,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        assert crop_size <= self.PATCH_SIZE_NATIVE

        self.root_path     = root_path
        self.split         = mode
        self.crop_size     = crop_size
        self.look_up       = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens = config_model["trainer"]["max_tokens"]

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "label_map.json")) as f:
            label_map = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            self.band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── sample_name → class_idx lookup ──────────────────
        self.name_to_label = {}
        for cls_str, names in label_map.items():
            cls_idx = int(cls_str)
            for n in names:
                self.name_to_label[n] = cls_idx

        missing = [n for n in self.sample_names if n not in self.name_to_label]
        if missing:
            raise RuntimeError(
                f"[ForestNet] {len(missing)} samples missing labels. "
                f"First missing: {missing[0]}"
            )

        # ── Normalization tensors ───────────────────────────
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in self.band_stats:
                raise KeyError(
                    f"[ForestNet] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[prefix]["mean"])
            stds.append(self.band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        self.bands_info = dataset_config["bands_forestnet"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.LANDSAT_RESOLUTION)

        print(f"[ForestNet] task={self.TASK_NAME}, split={mode} → "
              f"{len(self.sample_names)} samples")
        print(f"[ForestNet] bands ({self.NUM_BANDS}): "
              f"{[p.split(' - ')[1] for p in self.BAND_PREFIXES]}")
        print(f"[ForestNet] center crop: {crop_size}×{crop_size}")
        print(f"[ForestNet] resolution idx: {self.resolution_idx}")
        print(f"[ForestNet] D4 augment: {'ON' if mode == 'train' else 'OFF'}")

        cls_counts = np.zeros(self.NUM_CLASSES, dtype=int)
        for n in self.sample_names:
            cls_counts[self.name_to_label[n]] += 1
        print(f"[ForestNet] class counts: {cls_counts.tolist()}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor):
        """D4 group on [C, H, W] only — classification has no spatial label."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    @staticmethod
    def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = image.shape
        if H == size and W == size:
            return image
        top  = (H - size) // 2
        left = (W - size) // 2
        return image[:, top:top + size, left:left + size]

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        cls_idx = self.name_to_label[name]

        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[ForestNet] No key with prefix '{prefix}' in {path}"
                    )
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
        image = torch.from_numpy(np.stack(bands, axis=0))

        image = self._center_crop(image, self.crop_size)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        if self.split == "train":
            image = self._d4_augment(image)

        C, H, W = image.shape

        # Build tokens — classification doesn't use per-pixel labels,
        # so dummy IGNORE_INDEX label is fine.
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.LANDSAT_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        # Dummy queries — classification path doesn't use them, but the
        # batch format expects the field. Single zero-vector query.
        dummy_query      = torch.zeros(1, 8, dtype=image_tokens.dtype)
        dummy_query_mask = torch.zeros(1, dtype=torch.float32)

        attention_mask = torch.zeros(image_tokens.shape[0])

        return {
            "groups": {
                self.LANDSAT_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           dummy_query,
            "queries_mask":      dummy_query_mask,
            "label":             torch.tensor(cls_idx, dtype=torch.long),
            "task":              self.TASK_NAME,
            "target_resolution": self.LANDSAT_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        name = self.sample_names[index]
        cls_idx = self.name_to_label[name]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))

        image = torch.from_numpy(np.stack(bands, axis=0))
        image = self._center_crop(image, self.crop_size)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.LANDSAT_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask   = torch.zeros(image_tokens.shape[0])
        dummy_query      = torch.zeros(1, 8, dtype=image_tokens.dtype)
        dummy_query_mask = torch.zeros(1, dtype=torch.float32)

        return {
            "groups": {
                self.LANDSAT_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           dummy_query,
            "queries_mask":      dummy_query_mask,
            "label":             torch.tensor(cls_idx, dtype=torch.long),
            "task":              self.TASK_NAME,
            "target_resolution": self.LANDSAT_RESOLUTION,
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

        print(f"[ForestNet] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:8s} → "
                  f"bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[ForestNet] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)