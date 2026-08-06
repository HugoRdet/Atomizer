"""
Sen1Floods11 Baseline Dataset
==============================

Plain tensor dataset for non-Atomiser baselines (ResNet, UNet, ViT + UPerNet).

Returns simple [C, H, W] image tensors and [H, W] labels — no token building,
no lookup tables, no grouped format.

Output format (compatible with BaselineTrainer):
    {
        "image":  {"s2s1": [C, H, W]},   # 15 channels = 13 S2 + 2 S1
        "target": [H, W],                 # binary flood label
        "metadata": {...},
    }

Shared with the Atomiser dataset:
    - Same train/val/test splits (split CSVs from Sen1Floods11)
    - Same normalization (per-band z-score, stats from normalization_stats.pt)
    - Same NaN cleanup and label remapping
    - Same invalid-sample filtering (<100 valid label pixels skipped)
    - Same D4 augmentation (train only)

Specific to baselines:
    - No token grouping
    - Optional random crop to 256×256 during training (full 512 at val/test)
    - Channels are merged in fixed order: 13 S2 bands first, then 2 S1 bands

Band-dropout augmentation (train only):
    Zeroes whole modalities or random individual bands during training, so
    baselines get SOME training-time exposure to missing-band inputs —
    matching the semantics of the test-time modality-drop ablation
    (script_test_senflood_baseline_modality_drop.py's ChannelDropWrapper,
    which zeros already-normalized channels). Applied AFTER normalization
    for the same reason: zeroing raw pixel values before normalization
    would leave a nonzero value post-z-score ((0-mean)/std != 0), which
    is a different signal than what the model actually sees at ablation
    eval time (a literal zero). Sampled per-sample, independently of the
    fixed eval ablations (whole-S1-drop, whole-S2-drop, or a random
    per-band subset) — NOT limited to replaying the exact eval-time
    combinations, since always training on only those exact combinations
    would be a soft form of eval-set leakage into training. See
    Atomiser vs. baseline fairness note in the training script: this
    augmentation is baseline-only, compensating for these architectures'
    lack of a native way to represent "this band is absent" (unlike
    Atomiser's padding tokens) — not neutral, and should be reported as
    such in any writeup.
"""

import csv
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Sen1Floods11BaselineDataset(Dataset):
    """
    Sen1Floods11 dataset for baseline segmentation models.

    Args:
        root_path:    Path to dataset root (containing data/, splits/, etc.)
        mode:         "train", "validation", or "test"
        crop_size:    Random crop size for training (None = no crop, full 512).
                      Validation/test always use full 512.
        augment:      D4 augmentation (rotations + flip) — train only.
        band_dropout: Whole-modality/per-band zeroing augmentation — train
                      only (see module docstring). Independent of `augment`
                      so it can be toggled separately if needed.
        p_dropout_applied: Probability that band dropout happens at all for
                      a given training sample. The rest of the time, the
                      sample is unmodified (all bands present) — keeps the
                      "no dropout" regime well-represented so standard
                      full-band performance doesn't degrade.
        p_whole_modality: Given that dropout is applied, probability it's a
                      whole-modality drop (all S1 or all S2, mirroring the
                      "S2 only"/"S1 only" eval ablations) rather than a
                      random per-band subset.
        p_band_drop:  Given a per-band (not whole-modality) drop, the
                      independent Bernoulli probability each of the 15
                      bands is individually zeroed.

    Returns dict per sample:
        {
            "image":  {"s2s1": [15, H, W]},
            "target": [H, W] (long, IGNORE_INDEX=255 for invalid pixels),
            "metadata": {"H": int, "W": int, "n_bands": 15},
        }
    """

    OPTICAL_RESOLUTION = 10.0
    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_CHANNELS = NUM_S2_BANDS + NUM_S1_BANDS  # 15
    NUM_CLASSES = 2
    IGNORE_INDEX = 255

    SPLIT_MAPPING = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    def __init__(
        self,
        root_path: str = "./data/SENFLOOD",
        mode: str = "train",
        crop_size: int = 256,
        augment: bool = True,
        band_dropout: bool = True,
        p_dropout_applied: float = 0.5,
        p_whole_modality: float = 0.5,
        p_band_drop: float = 0.15,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split = mode
        self.crop_size = crop_size if mode == "train" else None
        self.augment = augment and (mode == "train")
        self.band_dropout = band_dropout and (mode == "train")
        self.p_dropout_applied = p_dropout_applied
        self.p_whole_modality = p_whole_modality
        self.p_band_drop = p_band_drop

        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.SPLIT_MAPPING[mode]}_data.csv",
        )

        # File lists + filter invalid samples
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        # Normalization (loads existing stats, or computes them on train split)
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[Sen1Floods11-BL] split={mode}, samples={len(self.s1_image_list)}")
        print(f"[Sen1Floods11-BL] channels: {self.NUM_CHANNELS} "
              f"({self.NUM_S2_BANDS} S2 + {self.NUM_S1_BANDS} S1)")
        if self.crop_size is not None:
            print(f"[Sen1Floods11-BL] random crop: {self.crop_size}×{self.crop_size}")
        else:
            print(f"[Sen1Floods11-BL] full image (512×512)")
        print(f"[Sen1Floods11-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        if self.band_dropout:
            print(f"[Sen1Floods11-BL] Band dropout: ON "
                  f"(p_applied={self.p_dropout_applied}, "
                  f"p_whole_modality={self.p_whole_modality}, "
                  f"p_band_drop={self.p_band_drop})")
        else:
            print(f"[Sen1Floods11-BL] Band dropout: OFF")

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
        # ── Load ────────────────────────────────────────────
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)

        # ── Clean ───────────────────────────────────────────
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label = torch.from_numpy(label)

        # ── Normalize ───────────────────────────────────────
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)

        # ── Merge channels: [13 S2 + 2 S1] = [15, H, W] ─────
        image = torch.cat([image_s2, image_s1], dim=0)

        # ── Band-dropout augmentation (training only, AFTER normalize) ──
        if self.band_dropout:
            image = self._band_dropout_augment(
                image, self.p_dropout_applied, self.p_whole_modality,
                self.p_band_drop, self.NUM_S2_BANDS, self.NUM_S1_BANDS,
            )

        # ── D4 augmentation (training only) ─────────────────
        if self.augment:
            image, label = self._d4_augment(image, label)

        # ── Random crop (training only) ─────────────────────
        if self.crop_size is not None:
            image, label = self._random_crop(image, label, self.crop_size)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image": {"s2s1": image},      # [15, H, W]
            "target": label,                # [H, W]
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "resolution": self.OPTICAL_RESOLUTION,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _band_dropout_augment(
        image: torch.Tensor,
        p_dropout_applied: float,
        p_whole_modality: float,
        p_band_drop: float,
        num_s2_bands: int,
        num_s1_bands: int,
    ) -> torch.Tensor:
        """
        Zero out whole modalities or random individual bands, applied to
        the already-normalized, already-merged [15, H, W] tensor.

        With probability (1 - p_dropout_applied): no-op, sample keeps all
        bands (keeps the full-band regime well-represented in training).

        Otherwise, with probability p_whole_modality: zero either all S2
        or all S1 bands (mirrors the "S2 only" / "S1 only" eval
        ablations). With probability (1 - p_whole_modality): zero each of
        the 15 bands independently with probability p_band_drop (covers
        the RGB-only / no-SWIR / no-red-edge style subset ablations
        without hardcoding to those exact combinations).
        """
        if torch.rand(1).item() >= p_dropout_applied:
            return image

        image = image.clone()

        if torch.rand(1).item() < p_whole_modality:
            if torch.rand(1).item() < 0.5:
                image[num_s2_bands:] = 0.0                    # drop S1
            else:
                image[:num_s2_bands] = 0.0                    # drop S2
        else:
            total_bands = num_s2_bands + num_s1_bands
            band_mask = torch.rand(total_bands) < p_band_drop
            image[band_mask] = 0.0

        return image

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """
        D4 group: random horizontal flip + 90° rotation (k ∈ {0,1,2,3}).
        Applied identically to image [C, H, W] and label [H, W].
        """
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

    # ─────────────────────────────────────────────────────────────────────
    # FILE LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_file_lists(self):
        s1_images, s2_images, labels = [], [], []
        print(f"[Sen1Floods11-BL] Loading split file: {self.split_file}")

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
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0

        print(f"[Sen1Floods11-BL] Filtering invalid samples...")
        for i in tqdm(range(len(self.label_list)), desc="Checking labels"):
            try:
                with rasterio.open(self.label_list[i]) as src:
                    lbl = src.read(1)
                lbl[lbl == -1] = 255
                if (lbl != 255).sum() > 100:
                    valid_s1.append(self.s1_image_list[i])
                    valid_s2.append(self.s2_image_list[i])
                    valid_labels.append(self.label_list[i])
                else:
                    skipped += 1
            except Exception as e:
                print(f"[Warning] Could not read {self.label_list[i]}: {e}")
                skipped += 1

        print(f"[Sen1Floods11-BL] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list = valid_labels

    # ─────────────────────────────────────────────────────────────────────
    # NORMALIZATION (shared file with Atomiser dataset)
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[Sen1Floods11-BL] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[Sen1Floods11-BL] WARNING: No normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(self.NUM_S2_BANDS),
                "s2_std":  torch.ones(self.NUM_S2_BANDS),
                "s1_mean": torch.zeros(self.NUM_S1_BANDS),
                "s1_std":  torch.ones(self.NUM_S1_BANDS),
            }

        print(f"[Sen1Floods11-BL] Computing normalization from "
              f"{len(self.s1_image_list)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[Sen1Floods11-BL] Saved normalization stats to {norm_file}")
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
                with rasterio.open(self.s2_image_list[idx]) as src:
                    s2 = src.read().astype(np.float64)
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
                with rasterio.open(self.s1_image_list[idx]) as src:
                    s1 = src.read().astype(np.float64)
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
        print(f"[Sen1Floods11-BL] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[Sen1Floods11-BL] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[Sen1Floods11-BL] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[Sen1Floods11-BL] S1 std:  {stats['s1_std'].numpy()}")

    def normalize_image(self, s2, s1):
        """Per-band z-score using precomputed train stats."""
        s2_mean = self.norm_stats["s2_mean"].view(self.NUM_S2_BANDS, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(self.NUM_S2_BANDS, 1, 1).clamp(min=1e-6)
        s1_mean = self.norm_stats["s1_mean"].view(self.NUM_S1_BANDS, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(self.NUM_S1_BANDS, 1, 1).clamp(min=1e-6)
        return (s2 - s2_mean) / s2_std, (s1 - s1_mean) / s1_std
