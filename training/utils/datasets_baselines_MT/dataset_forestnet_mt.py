"""
ForestNet (geo-bench m-forestnet) dataset for multi-task baseline training.

Landsat-8 (Blue, Green, Red, NIR, SWIR1, SWIR2), single-frame, 332x332 native,
12-class classification.

Spectral canonicalization: 6 Landsat-8 bands -> 13 canonical S2 bands via
linear interpolation in wavelength (build_interpolation_matrix). Mapping:
  B01 (443 nm)  -> zero (out of source range, < Blue)
  B02 (490)     -> 0.90*Blue(482) + 0.10*Green(561)
  B03 (560)     -> ~Green
  B04 (665)     -> ~Red
  B05/B06/B07   -> mixtures of Red(655) and NIR(865) (Landsat doesn't sample
                   these red-edge wavelengths; the mapping is structural,
                   not physically meaningful)
  B08 (842)     -> 0.11*Red + 0.89*NIR
  B8A (865)     -> identity from NIR
  B09 (945)     -> 0.89*NIR + 0.11*SWIR1
  B10 (1375)    -> 0.32*NIR + 0.68*SWIR1
  B11 (1610)    -> ~SWIR1
  B12 (2190)    -> ~SWIR2

Spatial: center-crop 332 -> 320 (matches the single-task baseline; 320 is
divisible by common patch sizes), then pad 320 -> 512 to fit the canonical
multi-task layout.

Output (matches every other multi-task dataset):
    {
        "image": {"input": [15, 512, 512]},   # 13 canonical S2 + 2 zero SAR
        "target": int,                         # class index in [0, 12)
        "valid_mask": [512, 512],              # uint8, 1 where real (top-left 320x320)
        "original_size": [2],                  # long, (320, 320)
        "metadata": {...},
    }

Splits (geo-bench convention):
    train -> 'train'
    val   -> 'valid'
    test  -> 'test'

Reference: https://github.com/ServiceNow/geo-bench
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .multitask_utils import (
    CANONICAL_SIZE,
    apply_interpolation_matrix,
    build_canonical_image,
    build_interpolation_matrix,
    pad_to_canonical,
)


class ForestNetMTDataset(Dataset):
    """ForestNet (geo-bench m-forestnet) dataset for multi-task baselines."""

    NUM_CLASSES = 12
    NUM_NATIVE_BANDS = 6
    NATIVE_SIZE = 332

    # Center-cropped to this size before spatial padding to CANONICAL_SIZE.
    # 320 matches the single-task baseline default.
    CROP_SIZE = 320

    # Landsat-8 OLI bands in the order ForestNet stores them. Wavelengths
    # are the OLI band centers (nm), used to build the interpolation matrix.
    BAND_PREFIXES = [
        "02 - Blue",     # ~482 nm
        "03 - Green",    # ~561 nm
        "04 - Red",      # ~655 nm
        "05 - NIR",      # ~865 nm
        "06 - SWIR1",    # ~1609 nm
        "07 - SWIR2",    # ~2201 nm
    ]
    LANDSAT_WAVELENGTHS_NM = [482, 561, 655, 865, 1609, 2201]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-forestnet",
        mode: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split = mode
        self.augment = augment and (mode == "train")

        # ── Load metadata ────────────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "label_map.json")) as f:
            label_map = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Build sample_name -> class_idx lookup ─────────
        self.name_to_label = {}
        for cls_str, names in label_map.items():
            cls_idx = int(cls_str)
            for n in names:
                self.name_to_label[n] = cls_idx

        missing = [n for n in self.sample_names if n not in self.name_to_label]
        if missing:
            raise RuntimeError(
                f"[ForestNet-MT] {len(missing)} samples in split '{mode}' "
                f"have no label. First missing: {missing[0]}"
            )

        # ── Verify a few HDF5 files exist (cheap check) ──
        for n in self.sample_names[:3]:
            path = os.path.join(root_path, f"{n}.hdf5")
            if not os.path.exists(path):
                raise FileNotFoundError(f"[ForestNet-MT] Missing HDF5: {path}")

        # ── Native normalization stats (per-band) ──────────
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in band_stats:
                raise KeyError(
                    f"[ForestNet-MT] Band '{prefix}' missing from band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[prefix]["mean"])
            stds.append(band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Build the [13, 6] interpolation matrix ──────
        self.interp_matrix = build_interpolation_matrix(self.LANDSAT_WAVELENGTHS_NM)

        # ── Summary ──────────────────────────────────────
        from collections import Counter
        label_counts = Counter(self.name_to_label[n] for n in self.sample_names)
        print(f"[ForestNet-MT] split={mode} -> {len(self.sample_names)} samples")
        print(f"[ForestNet-MT] {self.NUM_NATIVE_BANDS} Landsat bands -> "
              f"canonical 13 S2 (interpolation), SAR zero-filled")
        print(f"[ForestNet-MT] {self.NATIVE_SIZE}x{self.NATIVE_SIZE} -> "
              f"center-cropped to {self.CROP_SIZE} -> padded to {CANONICAL_SIZE}")
        print(f"[ForestNet-MT] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[ForestNet-MT] class distribution: {dict(sorted(label_counts.items()))}")

    # ─────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        """Center crop a [C, H, W] tensor to size x size."""
        C, H, W = image.shape
        if H == size and W == size:
            return image
        top  = (H - size) // 2
        left = (W - size) // 2
        return image[:, top:top + size, left:left + size]

    @staticmethod
    def _d4_augment(image: torch.Tensor) -> torch.Tensor:
        """D4 group on [C, H, W]. No label to align (scalar target)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    # ─────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        label = self.name_to_label[name]

        # ── Load 6 Landsat bands ──────────────────────
        # Geo-bench keys are '<prefix>_YYYY-MM-DD'; match by prefix.
        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[ForestNet-MT] No key with prefix '{prefix}' in "
                        f"{path}. Keys: {keys}"
                    )
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))

        image = torch.from_numpy(np.stack(bands, axis=0))    # [6, 332, 332]

        # ── Center crop 332 -> 320 ─────────────────────
        image = self._center_crop(image, self.CROP_SIZE)     # [6, 320, 320]

        # ── Defensive clean ────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Native per-band z-score ────────────────────
        image = (image - self.norm_mean) / self.norm_std
        image = torch.clamp(image, -10, 10)

        # ── D4 augmentation (channels stay aligned) ────
        if self.augment:
            image = self._d4_augment(image)

        # ── Spectral canonicalization: 6 -> 13 ─────────
        optical_canonical = apply_interpolation_matrix(image, self.interp_matrix)
        # [13, 320, 320]

        # ── Concat with zero SAR -> [15, 320, 320] ─────
        canonical = build_canonical_image(optical_canonical, sar=None)

        # ── Spatial padding 320 -> 512 ─────────────────
        canonical, _, valid_mask, original_size = pad_to_canonical(
            canonical, target=None, size=CANONICAL_SIZE,
        )

        return {
            "image": {"input": canonical},          # [15, 512, 512]
            "target": int(label),                    # scalar
            "valid_mask": valid_mask,                # [512, 512]
            "original_size": original_size,          # [2] -> (320, 320)
            "metadata": {
                "sample_name": name,
                "n_native_bands": self.NUM_NATIVE_BANDS,
                "native_band_names": [
                    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
                ],
            },
        }