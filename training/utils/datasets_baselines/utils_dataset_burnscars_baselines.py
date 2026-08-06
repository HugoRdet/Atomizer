"""
HLS BurnScars Baseline Dataset
================================

Plain tensor dataset for non-Atomiser baselines (UNet, ViT, ResNet + UPerNet).

Returns simple [C, H, W] image tensors and [H, W] labels — no token building.

Output format (compatible with BaselineTrainer):
    {
        "image":  {"hls": [6, H, W]},     # 6 HLS bands
        "target": [H, W],                  # binary burn scar label
        "metadata": {...},
    }

Splits (matches PANGAEA's HLSBurnScars exactly):
    - train: 90% of training/ folder (stratified, random_state=23)
    - val:   10% of training/ folder (stratified, random_state=23)
    - test:  all of validation/ folder

Normalization:
    Loads training/normalization_stats.pt if available, else computes on
    train split and saves it. Same convention as the Sen1Floods11 baseline.

Invalid values:
    Pixel value 9999 (HLS no-data sentinel) → 0 (matches PANGAEA).

Augmentations (train only):
    D4 group: random horizontal flip + 90° rotation = 8 transforms.

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


class BurnScarsBaselineDataset(Dataset):
    """
    HLS BurnScars dataset for baseline segmentation models.

    Args:
        root_path:   Path to dataset root containing training/ and validation/
        mode:        "train", "validation", or "test"
        crop_size:   Crop size (random for train, center for val/test).
                     None = no crop (use full image).
        augment:     D4 augmentation — train only.
    """

    HLS_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
    NUM_CHANNELS = 6
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    INVALID_VALUE = 9999

    # PANGAEA splits training/ into 90% train + 10% val with this exact seed
    SPLIT_RANDOM_STATE = 23
    VAL_FRACTION = 0.1

    def __init__(
        self,
        root_path: str = "./data/hls_burn_scars",
        mode: str = "train",
        crop_size: int = 256,
        augment: bool = True,
    ):
        super().__init__()
        assert mode in ("train", "validation", "test"), f"Unknown split: {mode}"

        self.root_path = root_path
        self.split = mode
        self.crop_size = crop_size
        self.augment = augment and (mode == "train")

        # PANGAEA's split_mapping: train and val both come from training/, test from validation/
        self.split_mapping = {
            "train": "training",
            "validation": "training",
            "test": "validation",
        }

        # ── Load file lists ─────────────────────────────────
        self.image_list, self.target_list = self._load_file_lists()

        # ── Apply 90/10 stratified split for train/val ──────
        if mode in ("train", "validation"):
            split_indices = self._get_train_val_split(self.image_list)
            indices = split_indices[mode]
            self.image_list = [self.image_list[i] for i in indices]
            self.target_list = [self.target_list[i] for i in indices]

        # ── Normalization ───────────────────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[BurnScars-BL] split={mode}, samples={len(self.image_list)}")
        print(f"[BurnScars-BL] channels: {self.NUM_CHANNELS} ({', '.join(self.HLS_BANDS)})")
        if self.crop_size is not None:
            crop_kind = "random" if mode == "train" else "center"
            print(f"[BurnScars-BL] {crop_kind} crop: {self.crop_size}×{self.crop_size}")
        else:
            print(f"[BurnScars-BL] full image (no crop)")
        print(f"[BurnScars-BL] D4 augment: {'ON' if self.augment else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # SPLIT (matches PANGAEA exactly)
    # ─────────────────────────────────────────────────────────────────────

    def _get_train_val_split(self, all_files):
        """
        Stratified 90/10 split of training/ folder.

        Identical to PANGAEA's HLSBurnScars.get_train_val_split:
            train_test_split(test_size=0.1, random_state=23)
        """
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

        if len(images) == 0:
            raise FileNotFoundError(
                f"[BurnScars-BL] No '*merged.tif' found in {folder}. "
                f"Check that the dataset is at {self.root_path}."
            )
        if len(images) != len(targets):
            raise RuntimeError(
                f"[BurnScars-BL] Mismatch: {len(images)} images vs "
                f"{len(targets)} targets in {folder}"
            )

        return images, targets

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # ── Load image: tiff is [H, W, C] → permute to [C, H, W] ──
        image = tiff.imread(self.image_list[index]).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # ── Load target ─────────────────────────────────────
        target = tiff.imread(self.target_list[index]).astype(np.int64)
        target = torch.from_numpy(target).long()

        # Remap label values:
        #   HLS BurnScars masks contain {0: no-burn, 1: burn, -1: no-data}.
        #   torchmetrics requires labels in [0, num_classes) ∪ {ignore_index},
        #   so map any value not in {0, 1} to IGNORE_INDEX (255).
        valid_classes = (target >= 0) & (target < self.NUM_CLASSES)
        target = torch.where(valid_classes, target,
                             torch.full_like(target, self.IGNORE_INDEX))

        # ── Mask invalid values (9999) → 0 ──────────────────
        image[image == self.INVALID_VALUE] = 0

        # ── NaN cleanup ─────────────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────
        image = self.normalize_image(image)

        # ── D4 augmentation (training only) ─────────────────
        if self.augment:
            image, target = self._d4_augment(image, target)

        # ── Crop: random (train) or center (val/test) ───────
        if self.crop_size is not None:
            if self.split == "train":
                image, target = self._random_crop(image, target, self.crop_size)
            else:
                image, target = self._center_crop(image, target, self.crop_size)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image": {"hls": image},        # [6, H, W]
            "target": target,                # [H, W]
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "bands": self.HLS_BANDS,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 group: random flip + 90° rotation."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _random_crop(image: torch.Tensor, label: torch.Tensor, size: int):
        """Random spatial crop to size×size."""
        C, H, W = image.shape
        assert H >= size and W >= size, (
            f"Crop size {size} exceeds image size ({H}×{W})"
        )
        top = torch.randint(0, H - size + 1, (1,)).item()
        left = torch.randint(0, W - size + 1, (1,)).item()
        image = image[:, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return image, label

    @staticmethod
    def _center_crop(image: torch.Tensor, label: torch.Tensor, size: int):
        """Deterministic center crop to size×size."""
        C, H, W = image.shape
        assert H >= size and W >= size, (
            f"Crop size {size} exceeds image size ({H}×{W})"
        )
        top = (H - size) // 2
        left = (W - size) // 2
        image = image[:, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[BurnScars-BL] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)

            # Try to find mean/std under any reasonable key naming
            mean, std = self._extract_mean_std(stats)
            self._print_norm_stats(mean, std)
            return {"mean": mean, "std": std}

        if self.split != "train":
            print(f"[BurnScars-BL] WARNING: No normalization file at {norm_file}")
            return {
                "mean": torch.zeros(self.NUM_CHANNELS),
                "std":  torch.ones(self.NUM_CHANNELS),
            }

        print(f"[BurnScars-BL] Computing normalization from "
              f"{len(self.image_list)} train samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[BurnScars-BL] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats["mean"], stats["std"])
        return stats

    def _extract_mean_std(self, stats: dict):
        """
        Try a few key naming conventions for mean/std in the saved stats dict.

        Supported:
            {"mean", "std"}
            {"hls_mean", "hls_std"}
            {"optical_mean", "optical_std"}
        """
        for mkey, skey in [
            ("mean", "std"),
            ("hls_mean", "hls_std"),
            ("optical_mean", "optical_std"),
        ]:
            if mkey in stats and skey in stats:
                return stats[mkey], stats[skey]

        raise KeyError(
            f"[BurnScars-BL] Could not find mean/std in normalization file. "
            f"Available keys: {list(stats.keys())}. "
            f"Expected one of: 'mean'/'std', 'hls_mean'/'hls_std', "
            f"'optical_mean'/'optical_std'."
        )

    def _compute_normalization_stats(self):
        s   = torch.zeros(self.NUM_CHANNELS, dtype=torch.float64)
        sq  = torch.zeros(self.NUM_CHANNELS, dtype=torch.float64)
        n   = torch.zeros(self.NUM_CHANNELS, dtype=torch.float64)

        for path in tqdm(self.image_list, desc="Computing normalization"):
            try:
                img = tiff.imread(path).astype(np.float64)  # [H, W, C]
                # Drop invalid + NaN
                mask = (img != self.INVALID_VALUE) & np.isfinite(img)
                for c in range(self.NUM_CHANNELS):
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
        print(f"[BurnScars-BL] Mean: {mean.numpy()}")
        print(f"[BurnScars-BL] Std:  {std.numpy()}")

    def normalize_image(self, image):
        """Per-band z-score using precomputed train stats."""
        mean = self.norm_stats["mean"].view(self.NUM_CHANNELS, 1, 1)
        std  = self.norm_stats["std"].view(self.NUM_CHANNELS, 1, 1).clamp(min=1e-6)
        return (image - mean) / std
