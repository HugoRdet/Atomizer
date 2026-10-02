"""
EuroSAT Baseline Dataset
==========================

Plain tensor dataset for non-Atomiser baselines (ResNet, ViT, etc.) on
geo-bench m-eurosat (10-class land-cover classification, single-frame S2,
13 optical bands including B10 cirrus).

Output format (compatible with a classification BaselineTrainer):
    {
        "image":  {"s2": [13, H, W]},
        "target": 0-d long tensor (class index in [0, 10)),
        "metadata": {...},
    }

Splits: from default_partition.json (train/valid/test -> 2000/1000/1000).
Native size: 64x64 (no cropping needed by default).
Bands: 13 S2 (01-CoastalAerosol, 02-Blue, 03-Green, 04-Red, 05/06/07-
RedEdge, 08-NIR, 08A-RedEdge, 09-WaterVapour, 10-Cirrus, 11-SWIR, 12-SWIR).

Label extraction: EuroSAT's HDF5 files have NO 'label' dataset key and NO
separate label_map.json (unlike ForestNet/Cashew) — the class label is a
plain Python int embedded in the HDF5's top-level 'pickle' attribute,
itself a pickled dict. See _PermissiveUnpickler below for why a custom
unpickler is needed (the pickled band-metadata entries reference
geobench.dataset.Sentinel2, which this codebase doesn't depend on).
"""

import ast
import io
import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class _PermissiveUnpickler(pickle.Unpickler):
    """
    Substitutes a harmless stub class for anything it can't import (e.g.
    geobench.dataset.Sentinel2), so the pickle can still be fully
    reconstructed. Only the plain-int 'label' field is ever read out of
    the result.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {})


class EurosatBaselineDataset(Dataset):
    """EuroSAT (m-eurosat) dataset for baseline classification models."""

    NUM_CHANNELS = 13
    NUM_CLASSES = 10
    PATCH_SIZE = 64

    BAND_PREFIXES = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
        "11 - SWIR",
        "12 - SWIR",
    ]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-eurosat",
        mode: str = "train",
        crop_size: int = None,        # None = full 64x64
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
                    f"[EuroSAT-BL] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[prefix]["mean"])
            stds.append(band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        print(f"[EuroSAT-BL] split={mode}, samples={len(self.sample_names)}")
        print(f"[EuroSAT-BL] channels: {self.NUM_CHANNELS} S2 bands (incl. B10 cirrus)")
        print(f"[EuroSAT-BL] patch size: {self.PATCH_SIZE}x{self.PATCH_SIZE}")
        if self.crop_size is not None:
            crop_kind = "random" if mode == "train" else "center"
            print(f"[EuroSAT-BL] {crop_kind} crop: {self.crop_size}x{self.crop_size}")
        else:
            print(f"[EuroSAT-BL] no crop (full image)")
        print(f"[EuroSAT-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[EuroSAT-BL] num_classes: {self.NUM_CLASSES}")

    # ─────────────────────────────────────────────────────────────────────
    # LABEL EXTRACTION (pickle-embedded, see module docstring)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_label(h5_file: h5py.File) -> int:
        raw = h5_file.attrs["pickle"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        raw_bytes = ast.literal_eval(raw)
        meta = _PermissiveUnpickler(io.BytesIO(raw_bytes)).load()
        return int(meta["label"])

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION (image-only — classification has no spatial label)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    @staticmethod
    def _random_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = image.shape
        assert H >= size and W >= size
        top  = torch.randint(0, H - size + 1, (1,)).item()
        left = torch.randint(0, W - size + 1, (1,)).item()
        return image[:, top:top + size, left:left + size]

    @staticmethod
    def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = image.shape
        assert H >= size and W >= size
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
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[EuroSAT-BL] No key with prefix '{prefix}' in {path}"
                    )
                # Raw digital numbers are int16 -- cast to float32 before
                # any arithmetic.
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            cls_idx = self._extract_label(f)

        image = torch.from_numpy(np.stack(bands, axis=0))

        # ── NaN cleanup ─────────────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────
        image = (image - self.norm_mean) / self.norm_std

        # ── D4 augmentation (training only) ─────────────────
        if self.augment:
            image = self._d4_augment(image)

        # ── Crop if requested ───────────────────────────────
        if self.crop_size is not None:
            if self.split == "train":
                image = self._random_crop(image, self.crop_size)
            else:
                image = self._center_crop(image, self.crop_size)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image":  {"s2": image},
            "target": torch.tensor(cls_idx, dtype=torch.long),
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "sample_name": name,
            },
        }
