"""
HLS BurnScars dataset for multi-task baseline training.

Differences from the single-task BurnScarsBaselineDataset:
    - Drops the random/center crop step. Multi-task uses full image,
      then pad to the canonical 512x512 (no-op for BurnScars — already 512).
    - Canonicalizes 6 HLS bands to the 15-channel layout
      (13 S2 optical + 2 zero SAR) via linear interpolation onto the
      S2 wavelength grid.
    - Returns "valid_mask" (always all ones for BurnScars) and
      "original_size" alongside the image and target.
    - Uses unified image key "input" instead of modality-specific "hls".
    - Native normalization (per HLS band) is preserved from the
      single-task version. Interpolation runs on already-normalized
      values.

Output format:
    {
        "image": {"input": [15, 512, 512]},   # float32, normalized
        "target": [512, 512],                  # long; {0, 1, 255}
        "valid_mask": [512, 512],              # uint8; 1 = real pixel
        "original_size": [2],                  # long; (H_orig, W_orig)
        "metadata": {...},
    }

Splits and normalization match the single-task version exactly:
    - train: 90% of training/ (stratified, random_state=23)
    - val:   10% of training/ (stratified, random_state=23)
    - test:  all of validation/
    - normalization stats loaded from training/normalization_stats.pt
      if present, else computed on train and saved.

Reference: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
"""

import os
from glob import glob

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from .multitask_utils import (
    CANONICAL_SIZE,
    IGNORE_INDEX,
    S2_WAVELENGTHS_NM,
    apply_interpolation_matrix,
    build_canonical_image,
    build_interpolation_matrix,
    pad_to_canonical,
)


class BurnScarsMTDataset(Dataset):
    """HLS BurnScars dataset for multi-task baselines."""

    # HLS bands present in the merged.tif files. Names match S2.
    HLS_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    HLS_WAVELENGTHS_NM = [S2_WAVELENGTHS_NM[b] for b in HLS_BANDS]
    NUM_NATIVE_CHANNELS = 6
    NUM_CLASSES = 2
    INVALID_VALUE = 9999  # HLS no-data sentinel

    SPLIT_RANDOM_STATE = 23
    VAL_FRACTION = 0.1

    def __init__(
        self,
        root_path: str = "./data/hls_burn_scars",
        mode: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        assert mode in ("train", "validation", "test"), f"Unknown split: {mode}"
        self.root_path = root_path
        self.split = mode
        self.augment = augment and (mode == "train")

        # PANGAEA's split_mapping: train and val from training/, test from validation/
        self.split_mapping = {
            "train": "training",
            "validation": "training",
            "test": "validation",
        }

        # ── File lists ──────────────────────────────────────
        self.image_list, self.target_list = self._load_file_lists()

        # ── 90/10 stratified split for train/val ────────────
        if mode in ("train", "validation"):
            split_indices = self._get_train_val_split(self.image_list)
            indices = split_indices[mode]
            self.image_list = [self.image_list[i] for i in indices]
            self.target_list = [self.target_list[i] for i in indices]

        # ── Native normalization stats ──────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        # ── Spectral interpolation matrix (precomputed once) ─
        # [13, 6]: maps the 6 HLS bands to the canonical 13 S2 positions.
        # In-source-range S2 wavelengths get interpolated values; the rest
        # (e.g. B01 at 443 nm) stay zero.
        self.interp_matrix = build_interpolation_matrix(self.HLS_WAVELENGTHS_NM)

        print(f"[BurnScars-MT] split={mode}, samples={len(self.image_list)}")
        print(f"[BurnScars-MT] native bands: {self.HLS_BANDS} -> canonical 15ch")
        print(f"[BurnScars-MT] D4 augment: {'ON' if self.augment else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # SPLIT
    # ─────────────────────────────────────────────────────────────────────

    def _get_train_val_split(self, all_files):
        """Stratified 90/10 split of training/ folder (matches PANGAEA)."""
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=self.VAL_FRACTION,
            random_state=self.SPLIT_RANDOM_STATE,
        )
        return {"train": train_idxs, "validation": val_idxs}

    # ─────────────────────────────────────────────────────────────────────
    # FILE LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_file_lists(self):
        folder = os.path.join(self.root_path, self.split_mapping[self.split])
        images = sorted(glob(os.path.join(folder, "*merged.tif")))
        targets = sorted(glob(os.path.join(folder, "*mask.tif")))

        if not images:
            raise FileNotFoundError(
                f"[BurnScars-MT] No '*merged.tif' in {folder}. "
                f"Check that the dataset is at {self.root_path}."
            )
        if len(images) != len(targets):
            raise RuntimeError(
                f"[BurnScars-MT] Mismatch: {len(images)} images vs "
                f"{len(targets)} targets in {folder}"
            )
        return images, targets

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # ── Load image: tiff [H, W, C] → [C, H, W] ──────────
        image = tiff.imread(self.image_list[index]).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # ── Load target ─────────────────────────────────────
        target = tiff.imread(self.target_list[index]).astype(np.int64)
        target = torch.from_numpy(target).long()

        # Remap label values: anything not in [0, NUM_CLASSES) → IGNORE_INDEX
        valid_classes = (target >= 0) & (target < self.NUM_CLASSES)
        target = torch.where(
            valid_classes, target,
            torch.full_like(target, IGNORE_INDEX),
        )

        # ── Clean invalid + NaN ─────────────────────────────
        image[image == self.INVALID_VALUE] = 0
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Native normalization (per HLS band) ─────────────
        image = self._normalize_native(image)

        # ── D4 augmentation (train only) ────────────────────
        # Applied in native-band space — same effect as after canonicalization
        # since D4 is purely spatial, but operates on 6 channels not 15.
        if self.augment:
            image, target = self._d4_augment(image, target)

        # ── Spectral canonicalization: 6 HLS → 13 S2 → 15ch ─
        optical_canonical = apply_interpolation_matrix(image, self.interp_matrix)
        canonical = build_canonical_image(optical_canonical, sar=None)  # [15, H, W]

        # ── Spatial padding (no-op for BurnScars at 512x512) ─
        canonical, target, valid_mask, original_size = pad_to_canonical(
            canonical, target, size=CANONICAL_SIZE,
        )

        return {
            "image": {"input": canonical},          # [15, 512, 512]
            "target": target,                       # [512, 512]
            "valid_mask": valid_mask,               # [512, 512]
            "original_size": original_size,         # [2]
            "metadata": {
                "filename": os.path.basename(self.image_list[index]),
                "native_bands": self.HLS_BANDS,
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
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[BurnScars-MT] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            mean, std = self._extract_mean_std(stats)
            self._print_norm_stats(mean, std)
            return {"mean": mean, "std": std}

        if self.split != "train":
            print(f"[BurnScars-MT] WARNING: No normalization file at {norm_file}")
            return {
                "mean": torch.zeros(self.NUM_NATIVE_CHANNELS),
                "std":  torch.ones(self.NUM_NATIVE_CHANNELS),
            }

        print(
            f"[BurnScars-MT] Computing normalization from "
            f"{len(self.image_list)} train samples..."
        )
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[BurnScars-MT] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats["mean"], stats["std"])
        return stats

    def _extract_mean_std(self, stats: dict):
        """Try a few key naming conventions for mean/std in the saved stats dict."""
        for mkey, skey in [
            ("mean", "std"),
            ("hls_mean", "hls_std"),
            ("optical_mean", "optical_std"),
        ]:
            if mkey in stats and skey in stats:
                return stats[mkey], stats[skey]

        raise KeyError(
            f"[BurnScars-MT] Could not find mean/std in normalization file. "
            f"Available keys: {list(stats.keys())}. "
            f"Expected one of: 'mean'/'std', 'hls_mean'/'hls_std', "
            f"'optical_mean'/'optical_std'."
        )

    def _compute_normalization_stats(self):
        s   = torch.zeros(self.NUM_NATIVE_CHANNELS, dtype=torch.float64)
        sq  = torch.zeros(self.NUM_NATIVE_CHANNELS, dtype=torch.float64)
        n   = torch.zeros(self.NUM_NATIVE_CHANNELS, dtype=torch.float64)

        for path in tqdm(self.image_list, desc="Computing normalization"):
            try:
                img = tiff.imread(path).astype(np.float64)  # [H, W, C]
                mask = (img != self.INVALID_VALUE) & np.isfinite(img)
                for c in range(self.NUM_NATIVE_CHANNELS):
                    if c >= img.shape[-1]:
                        continue
                    valid = img[..., c][mask[..., c]]
                    if len(valid):
                        s[c]  += valid.sum()
                        sq[c] += (valid ** 2).sum()
                        n[c]  += len(valid)
            except Exception as e:
                print(f"[Warning] Could not read {path}: {e}")
                continue

        mean = (s / n.clamp(min=1)).float()
        std  = ((sq / n.clamp(min=1) - mean.double() ** 2).sqrt()).float()
        return {"mean": mean, "std": std}

    def _print_norm_stats(self, mean, std):
        print(f"[BurnScars-MT] Mean: {mean.numpy()}")
        print(f"[BurnScars-MT] Std:  {std.numpy()}")

    def _normalize_native(self, image):
        """Per-band z-score in native HLS space (6 channels)."""
        mean = self.norm_stats["mean"].view(self.NUM_NATIVE_CHANNELS, 1, 1)
        std  = self.norm_stats["std"].view(self.NUM_NATIVE_CHANNELS, 1, 1).clamp(min=1e-6)
        return (image - mean) / std