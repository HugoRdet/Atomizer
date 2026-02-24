"""
MMEarth MAE Dataset — Multi-modal MAE pre-training
====================================================

Wraps torchgeo's MMEarth dataset for MAE pre-training with Atomizer.

Modalities used (continuous, available at inference):
    - sentinel2:          12 bands @ 10m (B10 cirrus excluded)
    - sentinel1_asc:      2 bands (VV, VH) @ 10m
    - sentinel1_desc:     2 bands (VV, VH) @ 10m
    - aster:              1 band (elevation) @ 10m (resampled)
    - canopy_height_eth:  1 band (height) @ 10m (resampled)

Excluded (categorical / not available at inference):
    - esa_worldcover, dynamic_world, biome, eco_region, era5

Masking:
    Per-modality spatial block masking. Each modality gets an independent
    random block mask, forcing cross-modal reconstruction learning.

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7

Directory structure:
    ./data/MM-Earth/
    ├── data_1M_v001.h5
    ├── data_1M_v001_band_stats.json
    ├── data_1M_v001_splits.json
    └── data_1M_v001_tile_info.json

Returns:
{
    "groups": {
        10.0: {
            "tokens": [N_vis, 8],
            "mask":   [N_vis],
            "shape":  (128, 128),
        },
    },
    "queries":      [M, 8],
    "queries_mask": [M],
    "ground_truth": [M],
}
"""

import os
import json

import torch
from torch.utils.data import Dataset

from .token_builder import TokenBuilder
from .block_masking import (
    generate_spatial_block_mask,
    expand_mask_to_tokens,
    apply_mask_to_tokens,
    build_mae_queries,
)

try:
    from torchgeo.datasets import MMEarth
    HAS_TORCHGEO = True
except ImportError:
    HAS_TORCHGEO = False
    print("[Warning] torchgeo not installed. Install with: pip install torchgeo")


class MMEarthMAEDataset(Dataset):
    """
    MMEarth dataset for MAE pre-training.

    Uses torchgeo MMEarth as backend. Applies per-modality spatial block
    masking and returns visible tokens + MAE reconstruction queries.

    Follows the same pattern as Sen1Floods11Dataset and PastisHDDataset:
        - Band info loaded from YAML config (dataset_config)
        - Spectral indices looked up via lookup table
        - Normalization handled by torchgeo (z-score)
        - TokenBuilder used for token construction
    """

    # ── Constants ───────────────────────────────────────────
    RESOLUTION = 10.0    # m/px — all modalities on same grid
    IMAGE_SIZE = 128     # px
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1     # No temporal info → zeroed by encoder

    # ── Modality → YAML key mapping ────────────────────────
    MODALITY_BAND_CONFIGS = {
        "sentinel2":          "bands_mmearth_s2",
        "sentinel1_asc":      "bands_mmearth_s1",
        "sentinel1_desc":     "bands_mmearth_s1",    # same bands as asc
        "aster":              "bands_mmearth_aster",
        "canopy_height_eth":  "bands_mmearth_canopy",
    }

    # ── torchgeo band selection (skip B10, HV, HH, slope, std) ──
    MODALITY_BANDS = {
        "sentinel2":          ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B11", "B12"],
        "sentinel1_asc":      ["VV", "VH"],
        "sentinel1_desc":     ["VV", "VH"],
        "aster":              ["elevation"],
        "canopy_height_eth":  ["canopy_height"],
    }

    def __init__(
        self,
        root_path: str = "./data/MM-Earth",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        mask_ratio: float = 0.75,
        block_size: int = 8,
        max_queries: int = 200_000,
        subset: str = "MMEarth",
    ):
        """
        Same interface as Sen1Floods11Dataset + MAE-specific params.

        Args:
            root_path:      Path to MM-Earth data directory
            mode:           "train" / "validation" / "test"
            dataset_config: YAML config dict (must contain bands_mmearth_* keys)
            config_model:   Model config dict
            look_up:        Lookup_encoding instance
            mask_ratio:     Fraction of pixels to mask per modality
            block_size:     Low-res noise grid size for block masking
            max_queries:    Max MAE reconstruction queries per sample
            subset:         "MMEarth", "MMEarth64", or "MMEarth100k"
        """
        super().__init__()

        if not HAS_TORCHGEO:
            raise ImportError("torchgeo is required for MMEarth. pip install torchgeo")

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model
        self.dataset_config = dataset_config
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.max_queries = max_queries
        self.subset = subset

        # Token builder (same as Sen1Floods11)
        self.token_builder = TokenBuilder(look_up)

        # Resolution index: same for all bands (everything on 10m grid)
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Which modalities to use
        self.modalities = list(self.MODALITY_BAND_CONFIGS.keys())

        # Parse band info from YAML and build spectral indices
        self._setup_all_band_indices()

        # Load torchgeo dataset
        self._load_torchgeo_dataset()

        # Compute target visible token count (fixed output size)
        self._compute_target_sizes()

        print(f"[MMEarth-MAE] Loaded {len(self.dataset)} samples for '{mode}'")
        print(f"[MMEarth-MAE] Modalities: {self.modalities}")
        print(f"[MMEarth-MAE] Total bands: {self.total_bands} → "
              f"{self.total_tokens} tokens/sample")
        print(f"[MMEarth-MAE] Mask ratio: {self.mask_ratio} → "
              f"~{self.target_visible} visible tokens")
        print(f"[MMEarth-MAE] Max queries: {self.max_queries}")

    # =========================================================================
    # INITIALIZATION (follows Sen1Floods11 pattern)
    # =========================================================================

    def _setup_all_band_indices(self):
        """
        Parse band info from YAML and build spectral indices.
        Same pattern as Sen1Floods11._parse_bands_info + _build_spectral_indices.
        """
        self.modality_band_info = {}      # {modality: [sorted band dicts]}
        self.modality_spectral_idx = {}   # {modality: tensor of spectral indices}
        self.modality_num_bands = {}      # {modality: int}

        for modality in self.modalities:
            yaml_key = self.MODALITY_BAND_CONFIGS[modality]
            bands_info = self.dataset_config[yaml_key]

            # Parse (same as Sen1Floods11._parse_bands_info)
            parsed, spectral_indices = self._parse_and_build_indices(
                bands_info, modality
            )

            self.modality_band_info[modality] = parsed
            self.modality_spectral_idx[modality] = spectral_indices
            self.modality_num_bands[modality] = len(parsed)

    def _parse_and_build_indices(self, bands_info: dict, modality_name: str):
        """
        Parse YAML band info dict and build spectral indices.
        Same logic as Sen1Floods11._parse_bands_info + _build_spectral_indices.

        Args:
            bands_info: YAML dict {band_name: {bandwidth, central_wavelength, idx, ...}}
            modality_name: For error messages

        Returns:
            parsed: Sorted list of band dicts
            spectral_indices: torch.Tensor of lookup table indices
        """
        # Parse all bands
        all_bands = []
        for name, data in bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        # Build spectral indices (same as Sen1Floods11._build_spectral_indices)
        indices = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                # Auto-register abstract channels (negative wavelengths)
                if band["bandwidth"] < 0:
                    new_idx = len(self.look_up.table_wave)
                    self.look_up.table_wave[key] = new_idx
                    print(f"[MMEarth-MAE] Registered abstract channel "
                          f"{band['name']} {key} → spectral idx {new_idx}")
                else:
                    raise KeyError(
                        f"[MMEarth-MAE] Band {band['name']} key={key} not in "
                        f"lookup table for modality '{modality_name}'.\n"
                        f"Available keys: {list(self.look_up.table_wave.keys())}"
                    )
            indices.append(self.look_up.table_wave[key])

        tag = " (abstract)" if all_bands and all_bands[0]["bandwidth"] < 0 else ""
        print(f"[MMEarth-MAE] {modality_name}: {len(all_bands)} bands{tag}")

        return all_bands, torch.tensor(indices, dtype=torch.long)

    def _load_torchgeo_dataset(self):
        """Load MMEarth via torchgeo with band selection."""
        split_map = {"train": "train", "validation": "val", "test": "test"}
        tg_split = split_map.get(self.split, self.split)

        self.dataset = MMEarth(
            root=self.root_path,
            subset=self.subset,
            modalities=self.modalities,
            modality_bands=self.MODALITY_BANDS,
            split=tg_split,
            normalization_mode="z-score",
        )

    def _compute_target_sizes(self):
        """Compute fixed output sizes for padding."""
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE

        self.total_bands = sum(self.modality_num_bands.values())
        self.total_tokens = self.total_bands * H * W

        # Fixed visible count: (1 - mask_ratio) × total
        self.target_visible = int((1.0 - self.mask_ratio) * self.total_tokens)

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        # ── Load from torchgeo (already z-score normalized) ─
        sample = self.dataset[index]

        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        all_visible = []
        all_vis_masks = []
        all_masked = []

        # ── Process each modality independently ─────────────
        for modality in self.modalities:
            if modality not in sample:
                continue

            data = sample[modality]
            if data is None or data.numel() == 0:
                continue

            data = data.float()
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = torch.clamp(data, -10, 10)

            C_m = data.shape[0]

            # Resample to 128×128 if needed (ASTER/canopy may differ)
            if data.shape[1] != H or data.shape[2] != W:
                data = torch.nn.functional.interpolate(
                    data.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False
                ).squeeze(0)

            # Get spectral indices for this modality
            spectral_indices = self.modality_spectral_idx[modality]

            # Build tokens (same as Sen1Floods11._build_tokens)
            tokens = self.token_builder.build_tokens(
                image=data,
                label=dummy_label,
                resolution=self.RESOLUTION,
                spectral_indices=spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )  # [C_m * H * W, 8]

            # Independent spatial block mask for this modality
            spatial_mask = generate_spatial_block_mask(
                H, W,
                mask_ratio=self.mask_ratio,
                block_size=self.block_size,
            )
            token_mask = expand_mask_to_tokens(spatial_mask, num_bands=C_m)

            # Split visible / masked (no padding yet — concat first)
            visible, vis_pad_mask, masked = apply_mask_to_tokens(
                tokens, token_mask, target_visible=None
            )

            all_visible.append(visible)
            all_vis_masks.append(vis_pad_mask)
            all_masked.append(masked)

        # ── Handle empty sample ─────────────────────────────
        if len(all_visible) == 0:
            return self._empty_sample()

        # ── Concatenate all modalities ──────────────────────
        visible_tokens = torch.cat(all_visible, dim=0)
        visible_mask = torch.cat(all_vis_masks, dim=0)
        masked_tokens = torch.cat(all_masked, dim=0)

        # ── Pad/trim to fixed size ──────────────────────────
        n_vis = visible_tokens.shape[0]
        if n_vis < self.target_visible:
            pad_n = self.target_visible - n_vis
            visible_tokens = torch.cat([
                visible_tokens,
                torch.zeros(pad_n, 8, dtype=visible_tokens.dtype),
            ], dim=0)
            visible_mask = torch.cat([
                visible_mask,
                torch.ones(pad_n, dtype=torch.bool),
            ], dim=0)
        elif n_vis > self.target_visible:
            perm = torch.randperm(n_vis)[:self.target_visible]
            visible_tokens = visible_tokens[perm]
            visible_mask = torch.zeros(self.target_visible, dtype=torch.bool)

        # ── Build MAE queries ───────────────────────────────
        queries, ground_truth, queries_mask = build_mae_queries(
            masked_tokens, max_queries=self.max_queries
        )

        # ── Return ──────────────────────────────────────────
        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": visible_tokens,   # [target_visible, 8]
                    "mask": visible_mask,        # [target_visible] bool
                    "shape": (H, W),
                },
            },
            "queries": queries,                 # [M, 8]
            "queries_mask": queries_mask,        # [M] bool
            "ground_truth": ground_truth,        # [M] float
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _empty_sample(self) -> dict:
        """Minimal valid sample when all modalities are missing."""
        H, W = self.IMAGE_SIZE, self.IMAGE_SIZE
        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": torch.zeros(self.target_visible, 8),
                    "mask": torch.ones(self.target_visible, dtype=torch.bool),
                    "shape": (H, W),
                },
            },
            "queries": torch.zeros(1, 8),
            "queries_mask": torch.ones(1, dtype=torch.bool),
            "ground_truth": torch.zeros(1),
        }