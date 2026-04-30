"""
Sen1Floods11 dataset for multi-task baseline training.

Differences from the single-task Sen1Floods11BaselineDataset:
    - Drops the random crop step. Multi-task uses full 512x512.
      Sen1Floods11 is already 512x512 natively, so spatial padding is a no-op.
    - Canonicalizes the input to the 15-channel layout. The 13 S2 bands are
      assumed to be in canonical S2 order [B01, B02, ..., B12] in the .tif
      files (Sen1Floods11 standard distribution), so the interpolation
      matrix is the 13x13 identity. The 2 S1 channels are placed at the
      canonical VV (slot 13) and VH (slot 14) positions.
    - Returns "valid_mask" (always all ones for Sen1Floods11) and
      "original_size" alongside the image and target.
    - Uses unified image key "input" instead of modality-specific "s2s1".
    - Native per-band normalization (separate stats for S2 and S1) is
      preserved from the single-task version. The interpolation matrix
      is applied on already-normalized values.

Output format:
    {
        "image": {"input": [15, 512, 512]},   # float32, normalized
        "target": [512, 512],                  # long; {0, 1, 255}
        "valid_mask": [512, 512],              # uint8; 1 = real pixel
        "original_size": [2],                  # long; (512, 512)
        "metadata": {...},
    }

Splits, normalization stats, and invalid-sample filtering match the
single-task version exactly.
"""

import csv
import os

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .multitask_utils import (
    CANONICAL_SIZE,
    IGNORE_INDEX,
    S2_BAND_NAMES,
    S2_CANONICAL_WAVELENGTHS_NM,
    apply_interpolation_matrix,
    build_canonical_image,
    build_interpolation_matrix,
    pad_to_canonical,
)


class Sen1Floods11MTDataset(Dataset):
    """Sen1Floods11 dataset for multi-task baselines."""

    OPTICAL_RESOLUTION = 10.0
    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_NATIVE_CHANNELS = NUM_S2_BANDS + NUM_S1_BANDS  # 15
    NUM_CLASSES = 2

    SPLIT_MAPPING = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    # S2 bands are stored in canonical order in Sen1Floods11 .tif files.
    S2_BANDS = S2_BAND_NAMES                            # all 13, canonical order
    S2_WAVELENGTHS_NM = S2_CANONICAL_WAVELENGTHS_NM     # [443, 490, ..., 2190]

    def __init__(
        self,
        root_path: str = "./data/SENFLOOD",
        mode: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        self.root_path = root_path
        self.split = mode
        self.augment = augment and (mode == "train")

        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.SPLIT_MAPPING[mode]}_data.csv",
        )

        # File lists + filter invalid samples (preserves single-task behavior)
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        # Normalization (loads existing stats, or computes them on train split)
        self.norm_stats = self._load_or_compute_normalization()

        # Spectral interpolation matrix (precomputed once).
        # For Sen1Floods11 with all 13 S2 bands at canonical wavelengths,
        # this is the 13x13 identity — the matmul is effectively a no-op
        # but keeps the canonicalization pipeline uniform across datasets.
        self.interp_matrix = build_interpolation_matrix(self.S2_WAVELENGTHS_NM)

        print(f"[Sen1Floods11-MT] split={mode}, samples={len(self.s1_image_list)}")
        print(f"[Sen1Floods11-MT] native: 13 S2 + 2 S1 -> canonical 15ch (identity)")
        print(f"[Sen1Floods11-MT] D4 augment: {'ON' if self.augment else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_chw(arr, expected_channels: int):
        """
        Ensure a multi-band array is in [C, H, W] order, regardless of
        whether tifffile returned [C, H, W] or [H, W, C].

        The convention varies by file: Sen1Floods11 .tif files are stored
        channels-first ([C, H, W]); other GeoTIFFs (e.g. HLS for BurnScars)
        are channels-last ([H, W, C]). We detect by matching against the
        expected band count.
        """
        if arr.ndim != 3:
            return arr
        if arr.shape[0] == expected_channels:
            return arr                                # already [C, H, W]
        if arr.shape[-1] == expected_channels:
            return arr.transpose(2, 0, 1)             # [H, W, C] -> [C, H, W]
        raise RuntimeError(
            f"Unexpected TIFF shape {arr.shape}, expected {expected_channels} channels"
        )

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
        # ── Load (tifffile + layout-robust channels-first) ──
        image_s2 = tiff.imread(self.s2_image_list[index]).astype(np.float32)
        image_s2 = self._ensure_chw(image_s2, self.NUM_S2_BANDS)        # [13, H, W]
        image_s1 = tiff.imread(self.s1_image_list[index]).astype(np.float32)
        image_s1 = self._ensure_chw(image_s1, self.NUM_S1_BANDS)        # [2, H, W]
        label = tiff.imread(self.label_list[index]).astype(np.int64)
        # label is [H, W] for single-band TIFF

        # ── Clean ───────────────────────────────────────────
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        target = torch.from_numpy(label).long()

        # ── Native normalization (separate per-band z-score for S2 / S1) ──
        image_s2, image_s1 = self._normalize_native(image_s2, image_s1)

        # ── D4 augmentation (train only) ─────────────────────
        # Concat S2 + S1 along channel so the same flip/rotation is applied
        # to both modalities and the label, then split back.
        if self.augment:
            full = torch.cat([image_s2, image_s1], dim=0)   # [15, H, W]
            full, target = self._d4_augment(full, target)
            image_s2 = full[:self.NUM_S2_BANDS]
            image_s1 = full[self.NUM_S2_BANDS:]

        # ── Spectral canonicalization ────────────────────────
        # Optical: identity matmul (13 -> 13, same wavelengths in same order).
        optical_canonical = apply_interpolation_matrix(image_s2, self.interp_matrix)
        # Combine with SAR (S1 already in [VV, VH] order) -> [15, H, W].
        canonical = build_canonical_image(optical_canonical, sar=image_s1)

        # ── Spatial padding (no-op for Sen1Floods11 at 512x512) ──
        canonical, target, valid_mask, original_size = pad_to_canonical(
            canonical, target, size=CANONICAL_SIZE,
        )

        return {
            "image": {"input": canonical},          # [15, 512, 512]
            "target": target,                       # [512, 512]
            "valid_mask": valid_mask,               # [512, 512]
            "original_size": original_size,         # [2]
            "metadata": {
                "filename": os.path.basename(self.s1_image_list[index]),
                "resolution": self.OPTICAL_RESOLUTION,
                "native_bands_optical": self.S2_BANDS,
                "native_bands_sar": ["VV", "VH"],
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 group: random horizontal flip + 90 degree rotation."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # FILE LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_file_lists(self):
        s1_images, s2_images, labels = [], [], []
        print(f"[Sen1Floods11-MT] Loading split file: {self.split_file}")

        with open(self.split_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                s1_filename = row[0].replace("S1Hand/", "")
                label_filename = row[1].replace("LabelHand/", "")
                s2_filename = s1_filename.replace("_S1Hand", "_S2Hand")

                s1_images.append(os.path.join(self.data_root, "S1Hand", s1_filename))
                s2_images.append(os.path.join(self.data_root, "S2Hand", s2_filename))
                labels.append(os.path.join(self.data_root, "LabelHand", label_filename))

        return s1_images, s2_images, labels

    def _filter_invalid_samples(self):
        """Skip samples with <100 valid label pixels (matches single-task version)."""
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0

        print(f"[Sen1Floods11-MT] Filtering invalid samples...")
        for i in tqdm(range(len(self.label_list)), desc="Checking labels"):
            try:
                lbl = tiff.imread(self.label_list[i])           # [H, W]
                lbl[lbl == -1] = IGNORE_INDEX
                if (lbl != IGNORE_INDEX).sum() > 100:
                    valid_s1.append(self.s1_image_list[i])
                    valid_s2.append(self.s2_image_list[i])
                    valid_labels.append(self.label_list[i])
                else:
                    skipped += 1
            except Exception as e:
                print(f"[Warning] Could not read {self.label_list[i]}: {e}")
                skipped += 1

        print(f"[Sen1Floods11-MT] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list = valid_labels

    # ─────────────────────────────────────────────────────────────────────
    # NORMALIZATION (compatible with single-task stats file)
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[Sen1Floods11-MT] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[Sen1Floods11-MT] WARNING: No normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(self.NUM_S2_BANDS),
                "s2_std":  torch.ones(self.NUM_S2_BANDS),
                "s1_mean": torch.zeros(self.NUM_S1_BANDS),
                "s1_std":  torch.ones(self.NUM_S1_BANDS),
            }

        print(
            f"[Sen1Floods11-MT] Computing normalization from "
            f"{len(self.s1_image_list)} samples..."
        )
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[Sen1Floods11-MT] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_sq  = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_n   = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s1_sum = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_sq  = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_n   = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)

        for idx in tqdm(range(len(self.s2_image_list)), desc="Computing normalization"):
            try:
                s2 = tiff.imread(self.s2_image_list[idx]).astype(np.float64)
                s2 = self._ensure_chw(s2, self.NUM_S2_BANDS)            # [13, H, W]
                s2 = np.nan_to_num(s2)
                for c in range(self.NUM_S2_BANDS):
                    valid = s2[c].flatten()
                    valid = valid[valid > 0]
                    if len(valid):
                        s2_sum[c] += valid.sum()
                        s2_sq[c]  += (valid ** 2).sum()
                        s2_n[c]   += len(valid)
            except Exception:
                continue

            try:
                s1 = tiff.imread(self.s1_image_list[idx]).astype(np.float64)
                s1 = self._ensure_chw(s1, self.NUM_S1_BANDS)            # [2, H, W]
                s1 = np.nan_to_num(s1)
                for c in range(self.NUM_S1_BANDS):
                    valid = s1[c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        s1_sum[c] += valid.sum()
                        s1_sq[c]  += (valid ** 2).sum()
                        s1_n[c]   += len(valid)
            except Exception:
                continue

        s2_mean = (s2_sum / s2_n.clamp(min=1)).float()
        s2_std  = ((s2_sq / s2_n.clamp(min=1) - s2_mean.double() ** 2).sqrt()).float()
        s1_mean = (s1_sum / s1_n.clamp(min=1)).float()
        s1_std  = ((s1_sq / s1_n.clamp(min=1) - s1_mean.double() ** 2).sqrt()).float()

        return {
            "s2_mean": s2_mean, "s2_std": s2_std,
            "s1_mean": s1_mean, "s1_std": s1_std,
        }

    def _print_norm_stats(self, stats):
        print(f"[Sen1Floods11-MT] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[Sen1Floods11-MT] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[Sen1Floods11-MT] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[Sen1Floods11-MT] S1 std:  {stats['s1_std'].numpy()}")

    def _normalize_native(self, s2, s1):
        """Per-band z-score using precomputed train stats."""
        s2_mean = self.norm_stats["s2_mean"].view(self.NUM_S2_BANDS, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(self.NUM_S2_BANDS, 1, 1).clamp(min=1e-6)
        s1_mean = self.norm_stats["s1_mean"].view(self.NUM_S1_BANDS, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(self.NUM_S1_BANDS, 1, 1).clamp(min=1e-6)
        return (s2 - s2_mean) / s2_std, (s1 - s1_mean) / s1_std