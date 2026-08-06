"""
HLS BurnScars Atomizer Dataset — SKIP variant
================================================

Ported from HLSBurnScarsDataset to match the Sen1Floods11SkipDataset
conventions: band selection (keep/drop), D4 augmentation, prioritized
query subsampling, and the per-query `query_token_idx` gather index used
by the decoder's skip-connection path.

Single-sensor dataset (HLS, 6 bands, 30m) — no fusion, so this is
structurally simpler than Sen1Floods11: one image tensor, one set of
normalization stats.

Token format (8 cols):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]

Bands metadata pulled from `dataset_config["bands_hls_info"]`.

CHANGES vs the old converging HLSBurnScarsDataset (flagged explicitly,
please confirm these are intended before training):
    1. D4 augmentation is now ON for train split (old version had none).
    2. Query subsampling now uses `prioritize_valid=True` (old version
       used plain random shuffle + truncate, specifically to avoid class
       skew — if that was a deliberate choice for BurnScars, set
       `prioritize_valid=False` in subsample_queries below).
    3. Validation/test queries are now the full pixel grid in raster
       order (old version applied the same shuffle+truncate at eval time
       too, which is unusual — full-grid eval matches Sen1Floods11 and
       is almost certainly what you want for mIoU reporting).
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
    HLS BurnScars binary segmentation, Atomizer format, SKIP variant.
    """

    NUM_CLASSES  = 2
    IGNORE_INDEX = 255
    RESOLUTION   = 30.0       # meters per pixel (HLS common grid)
    IMG_SIZE     = 512
    TIME_IDX_NA  = -1

    ALL_BAND_NAMES = ["B02", "B03", "B04", "B8A", "B11", "B12"]

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

        self.root_path      = root_path
        self.split          = mode
        self.look_up        = look_up
        self.config_model   = config_model
        self.dataset_config = dataset_config

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction = config_model["trainer"].get("mode", "segmentation") == "reconstruction"

        # ── Band selection ──────────────────────────────────────────────────
        bands_cfg = config_model["trainer"].get("bands", {}) or {}
        keep_names = bands_cfg.get("keep", None)
        drop_names = bands_cfg.get("drop", None)

        if keep_names is None:
            sc = config_model["trainer"].get("single_channel", -1)
            if isinstance(sc, list):
                keep_names = [self.ALL_BAND_NAMES[i] for i in sorted(sc)]
            elif isinstance(sc, int) and sc >= 0:
                keep_names = [self.ALL_BAND_NAMES[sc]]

        self.selected_channels = self._resolve_band_names(keep_names)
        self.drop_band_names   = set(drop_names) if drop_names else set()

        # ── Load file lists + apply 90/10 train/val split ──
        self.scene_list, self.mask_list = self._load_file_lists()

        # ── Band metadata ────────────────────────────────────
        self.bands_info = dataset_config["bands_hls_info"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        if self.selected_channels is not None:
            self.bandwidths       = self.bandwidths[self.selected_channels]
            self.wavelengths      = self.wavelengths[self.selected_channels]
            self.band_names       = [self.band_names[i] for i in self.selected_channels]
            self.spectral_indices = self.spectral_indices[self.selected_channels]

        self.dropped_spectral_indices = self._resolve_drop_indices()
        self._print_band_selection()

        # ── Resolution index ────────────────────────────────
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # ── Normalization ───────────────────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        if self.reconstruction:
            print(f"[BurnScarsSkip] Mode: RECONSTRUCTION")
        else:
            print(f"[BurnScarsSkip] Mode: SEGMENTATION")
        print(f"[BurnScarsSkip] split={mode}, samples={len(self.scene_list)}")
        print(f"[BurnScarsSkip] Loaded {len(self.bandwidths)} bands")
        print(f"[BurnScarsSkip] resolution idx: {self.resolution_idx} "
              f"(GSD={self.RESOLUTION} m/px)")
        print(f"[BurnScarsSkip] D4 augmentations: {'ON' if self.split == 'train' else 'OFF'}")

    # =========================================================================
    # BAND SELECTION HELPERS
    # =========================================================================

    def _resolve_band_names(self, names):
        if names is None:
            return None
        invalid = set(names) - set(self.ALL_BAND_NAMES)
        if invalid:
            raise ValueError(
                f"Unknown band names: {invalid}. Valid names: {self.ALL_BAND_NAMES}"
            )
        return [self.ALL_BAND_NAMES.index(n) for n in names]

    def _resolve_drop_indices(self):
        if not self.drop_band_names:
            return set()

        kept = set(self.band_names)
        unknown = self.drop_band_names - set(self.ALL_BAND_NAMES)
        if unknown:
            raise ValueError(f"bands.drop contains unknown names: {unknown}")
        not_kept = self.drop_band_names - kept
        if not_kept:
            raise ValueError(
                f"bands.drop {not_kept} are not in bands.keep {kept}. "
                f"You can only drop bands that were kept."
            )

        dropped = set()
        for name in self.drop_band_names:
            data = self.bands_info[name]
            key = (int(data["bandwidth"]), int(data["central_wavelength"]))
            if key in self.look_up.table_wave:
                dropped.add(self.look_up.table_wave[key])
            else:
                raise KeyError(f"Band '{name}' key={key} not found in lookup table.")
        return dropped

    def _print_band_selection(self):
        if self.selected_channels is None:
            kept_str = "ALL"
        else:
            kept_str = str([self.ALL_BAND_NAMES[i] for i in self.selected_channels])
        drop_str = str(sorted(self.drop_band_names)) if self.drop_band_names else "none"
        print(f"[BurnScarsSkip] Bands kept    : {kept_str}")
        print(f"[BurnScarsSkip] Bands dropped : {drop_str} (padding tokens, grid unchanged)")

    @property
    def NUM_BANDS(self):
        return len(self.band_names) if hasattr(self, "band_names") else len(self.ALL_BAND_NAMES)

    # =========================================================================
    # D4 AUGMENTATION
    # =========================================================================

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    # =========================================================================
    # CHANNEL SELECTION
    # =========================================================================

    def _select_channels(self, image):
        if self.selected_channels is None:
            return image
        return image[self.selected_channels]

    def _apply_drop_mask(self, tokens: torch.Tensor, mask: torch.Tensor):
        if not self.dropped_spectral_indices:
            return tokens, mask

        tokens = tokens.clone()
        mask   = mask.clone().float()

        spec_idx = tokens[:, 3]
        drop = torch.zeros(tokens.shape[0], dtype=torch.bool)
        for sid in self.dropped_spectral_indices:
            drop |= (spec_idx == sid)

        tokens[drop, 0] = 0.0
        mask[drop]      = 1.0

        return tokens, mask

    # =========================================================================
    # >>> SKIP: per-query gather index into own band-tokens
    # =========================================================================

    def _build_full_pixel_index(self, C, H, W):
        """
        Closed-form gather index for ALL pixels, in pixel order p = h*W + w.
        TokenBuilder flattens channel-major: pixel p's band-tokens live at
        rows {p + c*H*W : c in 0..C-1}. Returns [H*W, C] long.
        """
        HW = H * W
        p = torch.arange(HW)
        c = torch.arange(C)
        return p.unsqueeze(1) + c.unsqueeze(0) * HW

    def _build_query_token_index(self, C, H, W, kept_indices=None):
        """
        Vectorized per-query gather index into own band-tokens.
        See Sen1Floods11SkipDataset for full docstring — identical logic,
        single-modality so no fusion offset needed.
        """
        full = self._build_full_pixel_index(C, H, W)
        if kept_indices is None:
            idx = full
        else:
            idx = full[kept_indices]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_file_lists(self):
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

        if self.split in ("train", "validation"):
            train_idxs, val_idxs = train_test_split(
                np.arange(len(scenes)),
                test_size=self.VAL_FRACTION,
                random_state=self.SPLIT_RANDOM_STATE,
            )
            indices = train_idxs if self.split == "train" else val_idxs
            scenes = [scenes[i] for i in indices]
            masks  = [masks[i]  for i in indices]

        print(f"[BurnScarsSkip] Found {len(scenes)} scenes in "
              f"{split_dir} (split={self.split})")
        return scenes, masks

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        with rasterio.open(self.scene_list[index]) as src:
            image = src.read().astype(np.float32)              # [6, 512, 512]
        with rasterio.open(self.mask_list[index]) as src:
            label = src.read(1).astype(np.int64)                # [512, 512]

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[image == 9999] = 0.0                              # HLS no-data
        label[label == -1]   = self.IGNORE_INDEX

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        image = self.normalize_image(image)
        image = torch.clamp(image, -10, 10)

        image = self._select_channels(image)                    # [C', H, W]

        if self.split == "train":
            image, label = self._d4_augment(image, label)

        resolution = self.RESOLUTION
        image_tokens, seg_queries = self._build_tokens(image, label, resolution)

        attention_mask = torch.zeros(image_tokens.shape[0])
        image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)

        if self.reconstruction:
            queries = image_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()
            perm    = torch.randperm(queries.shape[0])[:self.max_tokens_reconstruction]
            queries = queries[perm]
            kept_indices = None
        else:
            if self.split == "train":
                # CHANGE vs old version: prioritize_valid=True (was plain
                # random shuffle+truncate). See module docstring note #2.
                queries, kept_indices = self.token_builder.subsample_queries(
                    seg_queries,
                    max_queries=self.max_tokens_reconstruction,
                    ignore_index=self.IGNORE_INDEX,
                    prioritize_valid=True,
                    return_indices=True,
                )
            else:
                # CHANGE vs old version: full pixel grid at eval, not
                # shuffled+truncated. See module docstring note #3.
                queries = seg_queries
                kept_indices = None

        queries_mask = torch.zeros(queries.shape[0])

        C_img, H_img, W_img = image.shape
        query_token_idx, query_token_valid = self._build_query_token_index(
            C_img, H_img, W_img, kept_indices=kept_indices
        )

        result = {
            "groups": {
                resolution: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "target_resolution": resolution,
            "image":             image,
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }

        if not self.reconstruction:
            result["label"] = label

        return result

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_tokens(self, image, label, resolution):
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
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        with rasterio.open(self.scene_list[index]) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(self.mask_list[index]) as src:
            label = src.read(1).astype(np.int64)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[image == 9999] = 0.0
        label[label == -1]   = self.IGNORE_INDEX

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image = self.normalize_image(image)
        image = torch.clamp(image, -10, 10)
        image = self._select_channels(image)
        C, H, W = image.shape

        if self.reconstruction:
            dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
            tokens = self.token_builder.build_tokens(
                image=image, label=dummy_label,
                resolution=self.RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )
            tokens[:, 4] = tokens[:, 0].clone()
            attention_mask = torch.zeros(tokens.shape[0])
            tokens, attention_mask = self._apply_drop_mask(tokens, attention_mask)
            queries      = tokens.clone()
            queries_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

            query_token_idx, query_token_valid = self._build_query_token_index(
                C, H, W, kept_indices=None
            )

            return {
                "groups": {self.RESOLUTION: {
                    "tokens": tokens, "mask": attention_mask, "shape": (C, H, W),
                }},
                "queries": queries, "queries_mask": queries_mask,
                "target_resolution": self.RESOLUTION,
                "image": image, "image_shape": (C, H, W),
                "n_real": (attention_mask == 0).sum().item(),
                "query_token_idx": query_token_idx,
                "query_token_valid": query_token_valid,
            }
        else:
            image_tokens, queries = self._build_tokens(image, label, self.RESOLUTION)
            attention_mask = torch.zeros(image_tokens.shape[0])
            image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

            query_token_idx, query_token_valid = self._build_query_token_index(
                C, H, W, kept_indices=None
            )

            return {
                "groups": {self.RESOLUTION: {
                    "tokens": image_tokens, "mask": attention_mask, "shape": (C, H, W),
                }},
                "queries": queries, "queries_mask": queries_mask,
                "label": label,
                "target_resolution": self.RESOLUTION,
                "image": image,
                "query_token_idx": query_token_idx,
                "query_token_valid": query_token_valid,
            }

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[BurnScarsSkip] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[BurnScarsSkip] WARNING: No normalization file at {norm_file}. "
                  f"Using zero-mean / unit-std on val/test.")
            n = len(self.ALL_BAND_NAMES)
            return {"mean": torch.zeros(n), "std": torch.ones(n)}

        print(f"[BurnScarsSkip] Computing normalization from "
              f"{len(self.scene_list)} train samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[BurnScarsSkip] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        # NOTE: computed over ALL bands (pre band-selection), same as the
        # old version and as Sen1Floods11 — normalize_image always indexes
        # by absolute band position, selection happens after normalization.
        n = len(self.ALL_BAND_NAMES)
        ch_sum    = torch.zeros(n, dtype=torch.float64)
        ch_sum_sq = torch.zeros(n, dtype=torch.float64)
        ch_count  = torch.zeros(n, dtype=torch.float64)

        for scene_path in tqdm(self.scene_list, desc="Computing normalization"):
            try:
                with rasterio.open(scene_path) as src:
                    data = src.read().astype(np.float64)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                for c in range(n):
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
        print(f"[BurnScarsSkip] Normalization stats:")
        for i, name in enumerate(self.ALL_BAND_NAMES):
            print(f"  {name}: mean={stats['mean'][i]:.4f}, "
                  f"std={stats['std'][i]:.4f}")

    def normalize_image(self, image):
        """Normalize BEFORE channel selection — stats indexed by absolute
        band position, matching self.ALL_BAND_NAMES order."""
        n = len(self.ALL_BAND_NAMES)
        mean = self.norm_stats["mean"].view(n, 1, 1)
        std  = self.norm_stats["std"].view(n, 1, 1)
        return (image - mean) / std

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _parse_bands_info(self):
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

        print(f"[BurnScarsSkip] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:4s} → "
                  f"bw={b['bandwidth']:4d}nm, wl={b['central_wavelength']:4d}nm")
        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[BurnScarsSkip] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _shuffle_arrays(arrays: list):
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]
