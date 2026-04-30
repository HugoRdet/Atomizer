"""
Cashew Baseline Dataset
=========================

Plain tensor dataset for non-Atomiser baselines (UNet, ViT, ResNet+UPerNet)
on geo-bench m-cashew-plant (7-class semantic segmentation, single-frame S2).

Output format (compatible with BaselineTrainer):
    {
        "image":  {"s2": [12, H, W]},
        "target": [H, W] long (0..6),
        "metadata": {...},
    }

Splits: from default_partition.json (train/valid/test → 1350/400/50).
Native size: 256×256 (no cropping needed; divides cleanly by 16 for ViT).
Bands: 12 S2 (B02 B03 B04 B08 B05 B06 B07 B08A B11 B12 B01 B09).
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CashewBaselineDataset(Dataset):
    """Cashew (m-cashew-plant) dataset for baseline segmentation models."""

    NUM_CHANNELS = 12
    NUM_CLASSES = 6
    IGNORE_INDEX = 255
    PATCH_SIZE = 256

    # Label scheme: 1..6 are real classes, 0 is no-data/unlabeled.
    # We remap to 0..5 for cross-entropy and use IGNORE for no-data.

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
        mode: str = "train",
        crop_size: int = None,        # None = full 256×256
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split     = mode
        self.crop_size = crop_size
        self.augment   = augment and (mode == "train")

        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in band_stats:
                raise KeyError(
                    f"[Cashew-BL] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[prefix]["mean"])
            stds.append(band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        print(f"[Cashew-BL] split={mode}, samples={len(self.sample_names)}")
        print(f"[Cashew-BL] channels: {self.NUM_CHANNELS} S2 bands")
        print(f"[Cashew-BL] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        if self.crop_size is not None:
            crop_kind = "random" if mode == "train" else "center"
            print(f"[Cashew-BL] {crop_kind} crop: {self.crop_size}×{self.crop_size}")
        else:
            print(f"[Cashew-BL] no crop (full image)")
        print(f"[Cashew-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[Cashew-BL] num_classes: {self.NUM_CLASSES}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image, label):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _random_crop(image, label, size):
        C, H, W = image.shape
        assert H >= size and W >= size
        top = torch.randint(0, H - size + 1, (1,)).item()
        left = torch.randint(0, W - size + 1, (1,)).item()
        return image[:, top:top + size, left:left + size], label[top:top + size, left:left + size]

    @staticmethod
    def _center_crop(image, label, size):
        C, H, W = image.shape
        assert H >= size and W >= size
        top  = (H - size) // 2
        left = (W - size) // 2
        return image[:, top:top + size, left:left + size], label[top:top + size, left:left + size]

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
                        f"[Cashew-BL] No key with prefix '{prefix}' in {path}"
                    )
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            target = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))
        target = torch.from_numpy(target).long()

        # ── NaN cleanup ─────────────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Label remap: 1-indexed → 0-indexed, 0 → IGNORE ──
        # README: classes 1..6 are real, 0 is no-data/unlabeled.
        ignore_mask = (target == 0) | (target > self.NUM_CLASSES)
        target = target - 1
        target = torch.where(ignore_mask, torch.full_like(target, self.IGNORE_INDEX), target)

        # ── Normalize ───────────────────────────────────────
        image = (image - self.norm_mean) / self.norm_std

        # ── D4 augmentation (training only) ─────────────────
        if self.augment:
            image, target = self._d4_augment(image, target)

        # ── Crop if requested ───────────────────────────────
        if self.crop_size is not None:
            if self.split == "train":
                image, target = self._random_crop(image, target, self.crop_size)
            else:
                image, target = self._center_crop(image, target, self.crop_size)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image":  {"s2": image},
            "target": target,
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "sample_name": name,
            },
        }