"""
PV4GER Baseline Dataset
=========================

Plain tensor dataset for non-Atomiser baselines (UNet, ViT, ResNet+UPerNet)
on geo-bench m-pv4ger-seg (binary PV panel segmentation, RGB aerial).

Output format (compatible with BaselineTrainer):
    {
        "image":  {"rgb": [3, H, W]},
        "target": [H, W] long (0 or 1),
        "metadata": {...},
    }

Splits: from default_partition.json (train/valid/test → 3000/403/403).
Native size: 320×320 (no cropping needed).
Bands: 3 RGB uint8 → float32 → per-band z-score from band_stats.json.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PV4GERBaselineDataset(Dataset):
    """PV4GER (m-pv4ger-seg) dataset for baseline segmentation models."""

    NUM_CHANNELS = 3
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    PATCH_SIZE = 320
    BAND_KEYS = ["Blue", "Green", "Red"]   # match HLS BurnScars / Sen1Floods11 ordering

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/segmentation_v1.0/m-pv4ger-seg",
        mode: str = "train",
        crop_size: int = None,        # None = full 320×320 (default; native size)
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split     = mode
        self.crop_size = crop_size
        self.augment   = augment and (mode == "train")

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Build normalization tensors (bands only) ────────
        means, stds = [], []
        for key in self.BAND_KEYS:
            if key not in band_stats:
                raise KeyError(
                    f"[PV4GER-BL] Band '{key}' not in band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[key]["mean"])
            stds.append(band_stats[key]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        print(f"[PV4GER-BL] split={mode}, samples={len(self.sample_names)}")
        print(f"[PV4GER-BL] channels: {self.NUM_CHANNELS} ({', '.join(self.BAND_KEYS)})")
        print(f"[PV4GER-BL] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        if self.crop_size is not None:
            crop_kind = "random" if mode == "train" else "center"
            print(f"[PV4GER-BL] {crop_kind} crop: {self.crop_size}×{self.crop_size}")
        else:
            print(f"[PV4GER-BL] no crop (full image)")
        print(f"[PV4GER-BL] D4 augment: {'ON' if self.augment else 'OFF'}")

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
            for key in self.BAND_KEYS:
                bands.append(np.asarray(f[key], dtype=np.float32))
            target = np.asarray(f["label"], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))
        target = torch.from_numpy(target).long()

        # ── NaN cleanup ─────────────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Defensive label remap (anything outside {0, 1} → IGNORE) ──
        valid = (target >= 0) & (target < self.NUM_CLASSES)
        target = torch.where(valid, target, torch.full_like(target, self.IGNORE_INDEX))

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
            "image":  {"rgb": image},
            "target": target,
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "sample_name": name,
            },
        }