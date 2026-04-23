"""
BigEarthNet-S2 Dataset for Atomizer-IO Pretraining
====================================================

Multi-label classification on 10 Sentinel-2 bands (10m+20m, no 60m),
matching the reBEN benchmark paper exactly.

Normalization: per-band percentile min-max (p2/p98) → [0, 1],
matching C2Seg's per-band min-max convention for smooth transfer.

Run compute_ben_stats.py first to generate the stats file.

Augmentations (training only):
    - D4: random rotations (0/90/180/270°) + horizontal flip
    - Band dropout: randomly drop 0-3 bands per sample, teaching the
      encoder to work with variable band counts (cross-sensor prep)
    - Random crop: crop to crop_size×crop_size from 120×120

Token format (matches C2Seg exactly):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7

Output batch format:
    {
        "groups": {10.0: {"tokens": [N, 8], "mask": [N], "shape": (C, H, H)}},
        "label": [19] multi-hot,
        "target_resolution": 10.0,
        "dataset_name": "BigEarthNet",
    }

No queries — classification uses attention pooling on latents directly.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from configilm.extra.DataSets import BENv2_DataSet
from typing import Literal, Optional

from .token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

DATASET_NAME = "BigEarthNet"
NUM_CLASSES = 19
STATIC_TIME_IDX = -1

# Channel mapping for configilm's 14-channel config:
# [S1_VV, S1_VH, B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
#  idx 0    1      2    3    4    5    6    7    8    9   10   11   12   13

# 10 S2 bands (matching reBEN paper: no 60m bands B01/B09)
S2_CHANNELS = {
    # ch_idx: (band_name, wavelength_nm, bandwidth_nm)
    2:  ("B02",  490,   65),
    3:  ("B03",  560,   35),
    4:  ("B04",  665,   30),
    5:  ("B08",  842,  115),
    6:  ("B05",  705,   15),
    7:  ("B06",  740,   15),
    8:  ("B07",  783,   20),
    9:  ("B8A",  865,   20),
    10: ("B11", 1610,   90),
    11: ("B12", 2190,  180),
}

# 12-band variant (includes 60m bands) — for ablation
S2_CHANNELS_12 = {
    **S2_CHANNELS,
    12: ("B01",  443,   20),
    13: ("B09",  945,   20),
}

GSD = 10.0  # All bands resampled to 10m
IMG_SIZE = 120  # 120×120 pixels

# Fallback normalization (divide by 10000) if stats file not found
FALLBACK_NORM_SCALE = 10000.0


# ═══════════════════════════════════════════════════════════════════════
# BAND REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

def register_ben_bands(look_up, include_60m: bool = False) -> int:
    """
    Pre-register all BigEarthNet S2 bands into the lookup table.

    Must be called in the main process BEFORE DataLoader workers spawn
    to avoid DDP deadlocks from lazy registration in forked workers.
    """
    channels = S2_CHANNELS_12 if include_60m else S2_CHANNELS
    n_new = 0
    for ch_idx, (band_name, wl, bw) in channels.items():
        wave_key = (bw, wl)
        if wave_key not in look_up.table_wave:
            look_up.table_wave[wave_key] = len(look_up.table_wave)
            n_new += 1
    n_bands = len(channels)
    print(f"[BigEarthNet] Pre-registered {n_new} new bands ({n_bands} total S2 bands) "
          f"into lookup table (total: {len(look_up.table_wave)})")
    return n_new


def create_ben_bands_info(include_60m: bool = False) -> dict:
    """Create bands_info dict for BigEarthNet."""
    channels = S2_CHANNELS_12 if include_60m else S2_CHANNELS
    bands = {}
    for ch_idx, (band_name, wl, bw) in channels.items():
        bands[band_name] = {
            "central_wavelength": wl,
            "bandwidth": bw,
            "idx": ch_idx,
        }
    return {"bands_ben_s2": bands}


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

class BENNormalization:
    """
    Per-band percentile min-max normalization.

    Loads p2/p98 stats from JSON (computed by compute_ben_stats.py),
    normalizes: (value - p2) / (p98 - p2) → approximately [0, 1].
    Clamps to [-0.5, 1.5] for safety.

    Matches the per-band min-max normalization used in C2Seg
    for smooth pretraining → fine-tuning transfer.
    """

    def __init__(self, stats_path: Optional[str] = None, channels: dict = None):
        self.channels = channels or S2_CHANNELS
        self.band_names = [self.channels[ch][0]
                           for ch in sorted(self.channels.keys())]

        if stats_path is not None:
            self._load_stats(stats_path)
        else:
            self.use_fallback = True
            print(f"[BigEarthNet] No stats file — using fallback /10000 normalization")

    def _load_stats(self, stats_path: str):
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)

            band_p2 = stats["band_p2"]
            band_p98 = stats["band_p98"]
            stats_names = stats["band_names"]

            # Map stats to our channel order
            stats_lookup = {name: (p2, p98) for name, p2, p98
                           in zip(stats_names, band_p2, band_p98)}

            p2_list = []
            p98_list = []
            for name in self.band_names:
                if name in stats_lookup:
                    p2, p98 = stats_lookup[name]
                    p2_list.append(p2)
                    p98_list.append(p98)
                else:
                    p2_list.append(0.0)
                    p98_list.append(10000.0)
                    print(f"[BigEarthNet] WARNING: no stats for {name}, using [0, 10000]")

            self.band_min = torch.tensor(p2_list, dtype=torch.float32)
            self.band_range = torch.tensor(
                [max(p98 - p2, 1.0) for p2, p98 in zip(p2_list, p98_list)],
                dtype=torch.float32,
            )
            self.use_fallback = False

            print(f"[BigEarthNet] Loaded percentile normalization from {stats_path}")
            print(f"  Bands: {self.band_names}")
            print(f"  p2:    {[f'{v:.1f}' for v in p2_list]}")
            print(f"  p98:   {[f'{v:.1f}' for v in p98_list]}")

        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            self.use_fallback = True
            print(f"[BigEarthNet] Could not load stats from {stats_path}: {e}")
            print(f"[BigEarthNet] Using fallback /10000 normalization")

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image [C, H, W] per-band.

        Returns [C, H, W] normalized to approximately [0, 1], clamped to [-0.5, 1.5].
        """
        if self.use_fallback:
            image = image / FALLBACK_NORM_SCALE
        else:
            band_min = self.band_min[:, None, None]
            band_range = self.band_range[:, None, None]
            image = (image - band_min) / band_range

        return torch.clamp(image, min=-0.5, max=1.5)


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

class BENAugmentation:
    """
    Training augmentations for BigEarthNet.

    All augmentations operate on [C, H, W] image tensors.

    D4:           Random rotation (0/90/180/270°) + horizontal flip.
    Band dropout: Randomly zero out 0 to max_drop bands per sample.
                  Teaches the encoder to handle variable band counts,
                  directly preparing it for cross-sensor inference.
    Random crop:  Crop to crop_size×crop_size from a random position.
                  Provides spatial variation and reduces token count.
    """

    def __init__(
        self,
        d4: bool = True,
        band_dropout: bool = True,
        max_band_drop: int = 3,
        random_crop: bool = True,
        crop_size: int = 96,
        img_size: int = 120,
    ):
        self.d4 = d4
        self.band_dropout = band_dropout
        self.max_band_drop = max_band_drop
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.img_size = img_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to image [C, H, W].

        Returns augmented image (possibly different H, W if cropped).
        """
        # ── D4: rotation + flip ──
        if self.d4:
            k = random.randint(0, 3)
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))
            if random.random() > 0.5:
                image = torch.flip(image, dims=(-1,))

        # ── Random crop ──
        if self.random_crop and self.crop_size < image.shape[-1]:
            _, H, W = image.shape
            max_r = H - self.crop_size
            max_c = W - self.crop_size
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, max_c)
            image = image[:, r0:r0 + self.crop_size, c0:c0 + self.crop_size]

        # ── Band dropout ──
        if self.band_dropout and self.max_band_drop > 0:
            C = image.shape[0]
            n_drop = random.randint(0, min(self.max_band_drop, C - 1))
            if n_drop > 0:
                drop_idx = random.sample(range(C), n_drop)
                image[drop_idx] = 0.0

        return image


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class BigEarthNetAtomizer(Dataset):
    """
    BigEarthNet-S2 dataset with atomic tokenization for Atomizer-IO pretraining.

    Wraps configilm's BENv2DataSet and converts each sample into the
    grouped token format expected by the Atomizer encoder.

    10 S2 bands at 120×120 (resampled to 10m by configilm), matching reBEN paper.
    Single resolution group: {10.0: ...}.

    Args:
        data_dirs: dict with keys "images_lmdb", "metadata_parquet",
                   "metadata_snow_cloud_parquet"
        split: "train", "val", or "test"
        look_up: Lookup_encoding instance (must have BEN bands pre-registered)
        stats_path: Path to ben_norm_stats.json (from compute_ben_stats.py).
                    If None, uses fallback /10000 normalization.
        include_60m: If True, include B01/B09 (12 bands). Default: False (10 bands).
        augment: If True, apply augmentations during training. Default: True.
        crop_size: Random crop size for training. Set to e.g. 96 to enable.
                   Default: None (disabled, use full 120×120).
        max_band_drop: Max bands to drop per sample during training.
                       Default: 0 (disabled). Set to 3 for C2Seg pretraining.
    """

    TASK_NAME = "ben_classification"

    def __init__(
        self,
        data_dirs: dict,
        split: Literal["train", "val", "test"],
        look_up,
        stats_path: Optional[str] = None,
        include_60m: bool = False,
        augment: bool = True,
        crop_size: Optional[int] = None,
        max_band_drop: int = 0,
    ):
        self.split = split
        self.look_up = look_up
        self.include_60m = include_60m
        self.channels = S2_CHANNELS_12 if include_60m else S2_CHANNELS
        self.is_train = (split == "train")

        # ── configilm handles LMDB, splits, snow/cloud filtering ──
        configilm_split = {"val": "validation"}.get(split, split)
        self.ben = BENv2_DataSet.BENv2DataSet(
            data_dirs=data_dirs,
            split=configilm_split,
            img_size=(14, IMG_SIZE, IMG_SIZE),
            include_snowy=False,
            include_cloudy=False,
        )

        # ── Normalization ──
        self.normalizer = BENNormalization(
            stats_path=stats_path,
            channels=self.channels,
        )

        # ── Augmentations (training only) ──
        self.augment = None
        if augment and self.is_train:
            use_crop = crop_size is not None and crop_size < IMG_SIZE
            use_band_drop = max_band_drop > 0
            self.augment = BENAugmentation(
                d4=True,
                band_dropout=use_band_drop,
                max_band_drop=max_band_drop,
                random_crop=use_crop,
                crop_size=crop_size if use_crop else IMG_SIZE,
                img_size=IMG_SIZE,
            )
            aug_parts = ["D4"]
            if use_crop:
                aug_parts.append(f"crop={crop_size}")
            if use_band_drop:
                aug_parts.append(f"band_drop≤{max_band_drop}")
            print(f"[BigEarthNet] Augmentations: {', '.join(aug_parts)}")

        # ── Token builder (reference grid system) ──
        self.token_builder = TokenBuilder(look_up)

        # ── Resolution index (shared across all tokens) ──
        self.resolution_idx = look_up.get_resolution_idx(GSD)

        # ── Spectral indices for each S2 band ──
        self.spectral_indices = self._build_spectral_indices()

        n_bands = len(self.channels)
        n_tokens_full = n_bands * IMG_SIZE * IMG_SIZE
        crop_str = f" (train crops to {crop_size}×{crop_size})" if (augment and self.is_train and crop_size is not None and crop_size < IMG_SIZE) else ""
        print(f"[BigEarthNet] {split}: {len(self)} samples, "
              f"{n_bands} S2 bands @ {GSD}m, "
              f"{n_tokens_full:,} tokens/sample{crop_str}, "
              f"res_idx={self.resolution_idx}")

    def _build_spectral_indices(self) -> torch.Tensor:
        """Build spectral index tensor matching token builder's convention."""
        indices = []
        for ch_idx in sorted(self.channels.keys()):
            _, wl, bw = self.channels[ch_idx]
            wave_key = (bw, wl)
            if wave_key not in self.look_up.table_wave:
                raise RuntimeError(
                    f"Band ({bw}nm, {wl}nm) not in lookup table. "
                    f"Call register_ben_bands(look_up) before creating dataset."
                )
            indices.append(self.look_up.table_wave[wave_key])
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.ben)

    def __getitem__(self, idx: int) -> dict:
        img, label = self.ben[idx]  # img: [14, 120, 120], label: [19]

        # ── Extract S2 bands only (skip S1 at ch 0-1) ──
        s2_channels = sorted(self.channels.keys())
        image = img[s2_channels]  # [10, 120, 120] or [12, 120, 120]

        # ── Per-band percentile normalization ──
        image = self.normalizer.normalize(image)

        # ── Augmentations (training only) ──
        if self.augment is not None:
            image = self.augment(image)

        # ── Build tokens via TokenBuilder ──
        C, H, W = image.shape
        dummy_label = torch.full((H, W), 255, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=GSD,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=STATIC_TIME_IDX,
        )

        token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        return {
            "groups": {
                GSD: {
                    "tokens": tokens,
                    "mask": token_mask,
                    "shape": (C, H, W),
                },
            },
            "label": label,  # [19] multi-hot float
            "target_resolution": GSD,
            "dataset_name": DATASET_NAME,
        }


# ═══════════════════════════════════════════════════════════════════════
# COLLATE
# ═══════════════════════════════════════════════════════════════════════

def collate_ben(batch: list) -> dict:
    """
    Collate BigEarthNet samples.

    With random crop, token counts may vary across samples.
    Pads to max length in batch (or stacks directly if all same size).
    """
    B = len(batch)
    res = GSD

    tokens_list = [s["groups"][res]["tokens"] for s in batch]
    masks_list = [s["groups"][res]["mask"] for s in batch]

    # Check if all same length (no crop or all same crop)
    lengths = [t.shape[0] for t in tokens_list]
    all_same = all(l == lengths[0] for l in lengths)

    if all_same:
        tokens = torch.stack(tokens_list, dim=0)
        masks = torch.stack(masks_list, dim=0)
    else:
        # Pad to max length
        max_len = max(lengths)
        tokens = torch.zeros(B, max_len, 8)
        masks = torch.ones(B, max_len, dtype=torch.bool)  # True = padded
        for i, (t, m) in enumerate(zip(tokens_list, masks_list)):
            n = t.shape[0]
            tokens[i, :n] = t
            masks[i, :n] = m  # False = valid

    # Shape from first sample (used for grid config)
    shape = batch[0]["groups"][res]["shape"]

    groups = {
        res: {
            "tokens": tokens,
            "mask": masks,
            "shape": shape,
        }
    }

    labels = torch.stack([s["label"] for s in batch], dim=0)

    return {
        "groups": groups,
        "label": labels,
        "target_resolution": res,
        "dataset_name": DATASET_NAME,
    }