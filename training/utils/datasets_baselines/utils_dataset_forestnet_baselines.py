"""
ForestNet Baseline Dataset (geo-bench format)
==============================================

Plain tensor dataset for non-Atomiser baselines on m-forestnet classification.

Source files (geo-bench-1.0/classification_v1.0/m-forestnet/):
    band_stats.json           — per-band mean/std/min/max/percentiles
    default_partition.json    — {"train": [...], "valid": [...], "test": [...]}
    label_map.json            — {class_idx_str: [sample_name, ...]}
    {sample_name}.hdf5        — 6 bands per file, [332, 332] uint8 each

Inside each HDF5 file, the 6 bands are stored under date-suffixed keys:
    "02 - Blue_YYYY-MM-DD"
    "03 - Green_YYYY-MM-DD"
    "04 - Red_YYYY-MM-DD"
    "05 - NIR_YYYY-MM-DD"
    "06 - SWIR1_YYYY-MM-DD"
    "07 - SWIR2_YYYY-MM-DD"

We match keys by prefix at load time so the date-varying suffix is handled
transparently.

Output format (compatible with ClassificationBaselineTrainer):
    {
        "image":  {"landsat": [6, H, W]},   # H=W=320 by default
        "target": int (class index in [0, 12)),
        "metadata": {"sample_name": str, "n_bands": 6},
    }

Pre-processing:
    - Center crop 332 → 320 (divisible by 16, retains 92% area)
    - Per-band z-score using band_stats.json mean/std
    - D4 augmentation (train only)

Bands ordered: [Blue, Green, Red, NIR, SWIR1, SWIR2] (matches HLS BurnScars).
"""

import json
import os
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ForestNetBaselineDataset(Dataset):
    """
    Geo-bench m-forestnet dataset for baseline classification models.

    Args:
        root_path:  path containing band_stats.json, default_partition.json,
                    label_map.json, and {sample_name}.hdf5 files.
        mode:       "train", "validation", or "test".
        crop_size:  center-crop spatial size (default 320, native is 332).
        augment:    D4 augmentation — train only.
    """

    NUM_CLASSES = 12
    NUM_CHANNELS = 6
    PATCH_SIZE_NATIVE = 332

    # Band key prefixes inside HDF5 (matched ignoring the trailing _YYYY-MM-DD).
    # Ordered to match HLS BurnScars: [Blue, Green, Red, NIR, SWIR1, SWIR2].
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
        "validation": "valid",   # geo-bench convention
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-forestnet",
        mode: str = "train",
        crop_size: int = 320,
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        assert crop_size <= self.PATCH_SIZE_NATIVE, (
            f"crop_size {crop_size} > native size {self.PATCH_SIZE_NATIVE}"
        )

        self.root_path = root_path
        self.split = mode
        self.crop_size = crop_size
        self.augment = augment and (mode == "train")

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "label_map.json")) as f:
            label_map = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        # ── Resolve sample list for this split ──────────────
        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Build sample_name → class_idx lookup ────────────
        # label_map is {class_idx_str: [name1, name2, ...]}
        self.name_to_label = {}
        for cls_str, names in label_map.items():
            cls_idx = int(cls_str)
            for n in names:
                self.name_to_label[n] = cls_idx

        # Verify all samples in the split have a label
        missing = [n for n in self.sample_names if n not in self.name_to_label]
        if missing:
            raise RuntimeError(
                f"[ForestNet-BL] {len(missing)} samples in split '{mode}' "
                f"have no label in label_map.json. First missing: {missing[0]}"
            )

        # ── Verify HDF5 files exist ─────────────────────────
        # (Cheap check on a few; full check would be slow with 8k files.)
        for n in self.sample_names[:3]:
            path = os.path.join(root_path, f"{n}.hdf5")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[ForestNet-BL] Sample HDF5 not found: {path}"
                )

        # ── Build normalization tensors from band_stats.json ──
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in band_stats:
                raise KeyError(
                    f"[ForestNet-BL] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[prefix]["mean"])
            stds.append(band_stats[prefix]["std"])

        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        print(f"[ForestNet-BL] split={mode} → {len(self.sample_names)} samples")
        print(f"[ForestNet-BL] bands ({self.NUM_CHANNELS}): "
              f"{[p.split(' - ')[1] for p in self.BAND_PREFIXES]}")
        print(f"[ForestNet-BL] center crop: {crop_size}×{crop_size} "
              f"(from native {self.PATCH_SIZE_NATIVE}×{self.PATCH_SIZE_NATIVE})")
        print(f"[ForestNet-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[ForestNet-BL] norm mean: {means}")
        print(f"[ForestNet-BL] norm std:  {stds}")

        # Class distribution for this split
        cls_counts = np.zeros(self.NUM_CLASSES, dtype=int)
        for n in self.sample_names:
            cls_counts[self.name_to_label[n]] += 1
        print(f"[ForestNet-BL] class counts: {cls_counts.tolist()}")

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        label = self.name_to_label[name]

        # ── Load 6 bands from HDF5 ──────────────────────────
        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                # Find the key starting with this prefix (date suffix varies).
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[ForestNet-BL] No key with prefix '{prefix}' in "
                        f"{path}. Keys: {keys}"
                    )
                # If multiple match (shouldn't happen), take the first.
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))

        # Stack: [6, H, W] in float32
        image = torch.from_numpy(np.stack(bands, axis=0))

        # ── Center crop 332 → 320 (or whatever crop_size) ───
        image = self._center_crop(image, self.crop_size)

        # ── NaN cleanup (defensive) ─────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────
        image = (image - self.norm_mean) / self.norm_std

        # ── D4 augmentation (training only) ─────────────────
        if self.augment:
            image = self._d4_augment(image)

        return {
            "image": {"landsat": image},   # [6, H, W]
            "target": int(label),           # scalar class index
            "metadata": {
                "sample_name": name,
                "n_bands": self.NUM_CHANNELS,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        """Center crop a [C, H, W] tensor to size×size."""
        C, H, W = image.shape
        if H == size and W == size:
            return image
        top  = (H - size) // 2
        left = (W - size) // 2
        return image[:, top:top + size, left:left + size]

    @staticmethod
    def _d4_augment(image: torch.Tensor) -> torch.Tensor:
        """D4 group augmentation on [C, H, W] image (no label to align)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image