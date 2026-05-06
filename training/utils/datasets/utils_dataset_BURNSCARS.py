"""
HLS BurnScars Atomizer Dataset
================================

Atomizer-format dataset for HLS BurnScars binary segmentation.

Built on the converging structure of the original HLSBurnScarsDataset:
    - rasterio loading ([C, H, W] directly — no permute)
    - No D4 augmentation
    - Norm stats: per-channel z-score, computed on (>0 & !=9999) pixels
    - 9999 sentinel → 0 before normalization
    - clamp(-10, 10) post-normalization
    - Random shuffle + truncate for query subsampling (no prioritize_valid)

Only change vs the original: use TokenBuilder for token + query construction
(matches the convention used by Sen1Floods11Dataset and PASTIS-HD), instead
of the older custom _build_tokens / _get_position_coordinates path.

Output format (canonical Atomizer single-task seg sample, matches
Sen1Floods11Dataset):
    {
        "groups": {
            30.0: {
                "tokens": [N, 8],     # one token per pixel × band
                "mask":   [N],         # bool, True = ignore
                "shape":  (6, H, W),
            },
        },
        "queries":           [M, 8],   # one query per pixel
        "queries_mask":      [M],
        "label":             [H, W],
        "target_resolution": 30.0,
        "image":             [6, H, W],
    }

Token format (8 cols):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]

Bands metadata:
    Pulled from `dataset_config["bands_hls_info"]` — must contain entries
    for B02, B03, B04, B8A, B11, B12 (HLS surface reflectance bands at
    Sentinel-2-harmonized wavelengths).
"""

import os
import glob

import numpy as np
import rasterio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from .token_grouping import *
from .token_builder import TokenBuilder


class BurnScarsDataset(Dataset):
    """
    HLS BurnScars binary segmentation, Atomizer format.

    Drop-in replacement for the older HLSBurnScarsDataset, with
    TokenBuilder used for token construction (matching Sen1Floods11
    convention).
    """

    NUM_BANDS    = 6
    NUM_CLASSES  = 2
    IGNORE_INDEX = 255
    RESOLUTION   = 30.0       # meters per pixel (HLS common grid)
    IMG_SIZE     = 512
    TIME_IDX_NA  = -1

    # PANGAEA convention: training/ folder split 90/10 with this seed.
    SPLIT_RANDOM_STATE = 23
    VAL_FRACTION       = 0.1

    SPLIT_DIR_MAPPING = {
        "train":      "training",
        "validation": "training",
        "test":       "validation",
    }

    def __init__(
        self,
        root_path: str = "./data/hls_burn_scars",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()
        assert mode in self.SPLIT_DIR_MAPPING, f"Unknown split: {mode}"

        self.root_path     = root_path
        self.split         = mode
        self.look_up       = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config

        # TokenBuilder — same instance type used by Sen1Floods11.
        # This is the only structural change vs the old converging
        # HLSBurnScarsDataset: token construction goes through TokenBuilder
        # rather than custom _build_tokens.
        self.token_builder = TokenBuilder(look_up)

        # Config parameters
        self.nb_tokens                  = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction  = config_model["trainer"]["max_tokens_reconstruction"]

        # ── Load file lists + apply 90/10 train/val split ──
        self.scene_list, self.mask_list = self._load_file_lists()

        # ── Band metadata ────────────────────────────────────
        self.bands_info = dataset_config["bands_hls_info"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        # ── Resolution index ────────────────────────────────
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # ── Normalization ───────────────────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[BurnScars] split={mode}, samples={len(self.scene_list)}")
        print(f"[BurnScars] bands ({self.NUM_BANDS}): {self.band_names}")
        print(f"[BurnScars] resolution idx: {self.resolution_idx} "
              f"(GSD={self.RESOLUTION} m/px)")
        print(f"[BurnScars] No augmentation (matches converging baseline).")

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_file_lists(self):
        """
        Discover scene/mask pairs from disk, then apply 90/10 stratified
        split for train/validation (matching the old converging behavior
        and PANGAEA convention).
        """
        split_dir = os.path.join(
            self.root_path, self.SPLIT_DIR_MAPPING[self.split]
        )
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Contents of {self.root_path}: {os.listdir(self.root_path)}"
            )

        all_scenes = sorted(glob.glob(os.path.join(split_dir, "*_merged.tif")))
        all_masks  = sorted(glob.glob(os.path.join(split_dir, "*.mask.tif")))

        # Pair scenes with their masks by name match
        scenes, masks = [], []
        all_masks_set = set(all_masks)
        for scene_path in all_scenes:
            mask_path = scene_path.replace("_merged.tif", ".mask.tif")
            if mask_path in all_masks_set:
                scenes.append(scene_path)
                masks.append(mask_path)

        if len(scenes) == 0:
            raise RuntimeError(
                f"No valid scene/mask pairs found in {split_dir}. "
                f"Expected '*_merged.tif' and '*.mask.tif' file pairs."
            )

        # 90/10 split for train/val. Test uses the entire validation/ folder.
        if self.split in ("train", "validation"):
            train_idxs, val_idxs = train_test_split(
                np.arange(len(scenes)),
                test_size=self.VAL_FRACTION,
                random_state=self.SPLIT_RANDOM_STATE,
            )
            indices = train_idxs if self.split == "train" else val_idxs
            scenes = [scenes[i] for i in indices]
            masks  = [masks[i]  for i in indices]

        print(f"[BurnScars] Found {len(scenes)} scenes in "
              f"{split_dir} (split={self.split})")
        return scenes, masks

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        # ── Load: rasterio returns [C, H, W] directly for multiband TIFs ──
        # No permute needed — same as the converging old dataset.
        with rasterio.open(self.scene_list[index]) as src:
            image = src.read().astype(np.float32)              # [6, 512, 512]
        with rasterio.open(self.mask_list[index]) as src:
            label = src.read(1).astype(np.int64)                # [512, 512]

        # ── Clean ───────────────────────────────────────────
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[image == 9999] = 0.0                              # HLS no-data
        label[label == -1]   = self.IGNORE_INDEX

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # ── Normalize + clamp (matches old converging behavior) ──
        image = self.normalize_image(image)
        image = torch.clamp(image, -10, 10)

        # ── Build tokens via TokenBuilder ───────────────────
        # Same code path as Sen1Floods11Dataset._build_tokens. Produces
        # canonical [N, 8] tokens with positions, spectral indices,
        # labels, resolution_idx, and TIME_IDX_NA in the right columns.
        resolution = self.RESOLUTION
        image_tokens, queries = self._build_tokens(image, label, resolution)

        # ── Query subsampling: random shuffle + truncate ────
        # Matches old converging behavior — no prioritize_valid flag,
        # which previously was suspected of skewing the class balance.
        queries_mask = torch.zeros(queries.shape[0])
        queries, queries_mask = self._shuffle_arrays([queries, queries_mask])
        nb_q = self.max_tokens_reconstruction
        queries      = queries[:nb_q]
        queries_mask = queries_mask[:nb_q]

        # ── Image-token attention mask (no padding here) ────
        attention_mask = torch.zeros(image_tokens.shape[0])

        # ── Return ──────────────────────────────────────────
        return {
            "groups": {
                resolution: {
                    "tokens": image_tokens,                     # [C*H*W, 8]
                    "mask":   attention_mask,                    # [C*H*W]
                    "shape":  tuple(image.shape),                # (6, H, W)
                },
            },
            "queries":           queries,                        # [M, 8]
            "queries_mask":      queries_mask,                    # [M]
            "label":             label,                           # [H, W]
            "target_resolution": resolution,
            "image":             image,                           # [6, H, W]
        }

    # =========================================================================
    # TOKEN BUILDING (via TokenBuilder)
    # =========================================================================

    def _build_tokens(self, image, label, resolution):
        """
        Produce [N, 8] image tokens and [M, 8] queries via TokenBuilder.
        Mirrors Sen1Floods11Dataset._build_tokens exactly.
        """
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=resolution,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=resolution,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        return image_tokens, queries

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        """
        Load or compute per-band z-score stats. Same convention as old
        HLSBurnScarsDataset: stats keyed by 'mean' / 'std', valid pixels
        defined as (channel > 0) & (channel != 9999).
        """
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[BurnScars] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[BurnScars] WARNING: No normalization file at {norm_file}. "
                  f"Using zero-mean / unit-std on val/test.")
            return {
                "mean": torch.zeros(self.NUM_BANDS),
                "std":  torch.ones(self.NUM_BANDS),
            }

        print(f"[BurnScars] Computing normalization from "
              f"{len(self.scene_list)} train samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[BurnScars] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        ch_sum    = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_sum_sq = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_count  = torch.zeros(self.NUM_BANDS, dtype=torch.float64)

        for scene_path in tqdm(self.scene_list, desc="Computing normalization"):
            try:
                with rasterio.open(scene_path) as src:
                    data = src.read().astype(np.float64)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                for c in range(self.NUM_BANDS):
                    channel = data[c].flatten()
                    valid = channel[(channel > 0) & (channel != 9999)]
                    if len(valid) > 0:
                        ch_sum[c]    += valid.sum()
                        ch_sum_sq[c] += (valid ** 2).sum()
                        ch_count[c]  += len(valid)
            except Exception as e:
                print(f"[Warning] Could not read {scene_path}: {e}")
                continue

        mean = ch_sum / ch_count.clamp(min=1)
        var  = (ch_sum_sq / ch_count.clamp(min=1)) - (mean ** 2)
        std  = torch.sqrt(var.clamp(min=1e-8))
        return {"mean": mean.float(), "std": std.float()}

    def _print_norm_stats(self, stats):
        print(f"[BurnScars] Normalization stats:")
        for i, name in enumerate(self.band_names):
            print(f"  {name}: mean={stats['mean'][i]:.4f}, "
                  f"std={stats['std'][i]:.4f}")

    def normalize_image(self, image):
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std  = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _parse_bands_info(self):
        """
        Parse the bands_hls_info YAML entry. Same structure as
        Sen1Floods11._parse_bands_info — bands sorted by `idx`, with
        `bandwidth` and `central_wavelength` looked up against the
        shared spectral lookup table.
        """
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": data["bandwidth"],
                    "central_wavelength": data["central_wavelength"],
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw    = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[BurnScars] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:4s} → "
                  f"bw={b['bandwidth']:4d}nm, wl={b['central_wavelength']:4d}nm")
        return bw, wl, names

    def _build_spectral_indices(self):
        """
        Look up each band's (bandwidth, central_wavelength) in the shared
        spectral lookup table. Same logic as Sen1Floods11 — guarantees that
        a B02 token from BurnScars and a B02 token from Sen1Floods11 share
        the same spectral_idx, which is what allows the shared encoder to
        reason across tasks coherently.
        """
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[BurnScars] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _shuffle_arrays(arrays: list):
        """Shuffle a list of tensors with the same first-dim permutation."""
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]