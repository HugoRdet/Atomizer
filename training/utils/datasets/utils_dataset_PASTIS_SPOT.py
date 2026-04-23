"""
PASTIS SPOT Reconstruction Dataset
====================================

Reconstruction-only dataset using SPOT6 RGB imagery from PASTIS-HD.
Single frame, 3 bands (R, G, B) at 1m resolution, ~1280×1280 pixels.

This is a diagnostic dataset to validate reconstruction in a regime
similar to FLAIR (VHR, few bands, large spatial extent).

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

Returns:
{
    "groups": {
        1.0: {
            "tokens": [N, 8],       # 3 * H * W tokens
            "mask":   [N],
            "shape":  (3, H, W),
        },
    },
    "queries":           [M, 8],    # col 4 = reflectance
    "queries_mask":      [M],
    "target_resolution": 1.0,
}
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .token_builder import TokenBuilder

try:
    import pandas as pd
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


class PastisSpotReconDataset(Dataset):
    """
    PASTIS SPOT6 Reconstruction Dataset.

    Single resolution group at 1m, 3 RGB bands, no temporal dimension.
    Queries are image tokens with col 4 = reflectance value.
    """

    RESOLUTION = 10.0         # m/px
    NUM_BANDS = 3            # R, G, B
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1

    SPOT_BANDS_INFO = {
        "SPOT_R": {"central_wavelength": 660, "bandwidth": 120, "idx": 0},
        "SPOT_G": {"central_wavelength": 560, "bandwidth": 120, "idx": 1},
        "SPOT_B": {"central_wavelength": 490, "bandwidth": 140, "idx": 2},
    }

    def __init__(
        self,
        root_path: str = "./data/PASTIS-HD",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model

        self.token_builder = TokenBuilder(look_up)

        self.max_queries = config_model["trainer"].get("max_tokens_reconstruction", 200_000)

        self.split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

        # SPOT directory
        self.spot_dir = os.path.join(
            root_path, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019"
        )
        if not os.path.isdir(self.spot_dir):
            raise FileNotFoundError(
                f"SPOT directory not found: {self.spot_dir}\n"
                f"This dataset requires PASTIS-HD SPOT6 imagery."
            )
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for SPOT imagery.")

        # Load metadata → filter split → filter SPOT availability
        self._load_metadata()

        # Band indices
        self._setup_band_indices()

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[PASTIS-SPOT] Loaded {len(self.patch_ids)} patches for split '{self.split}'")
        print(f"[PASTIS-SPOT] {self.NUM_BANDS} bands (RGB) @ {self.RESOLUTION}m")
        print(f"[PASTIS-SPOT] Mode: RECONSTRUCTION (DEBUG: single band, 128x128 crop)")

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required. pip install geopandas")

        metadata_path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.geojson not found at {metadata_path}")

        print(f"[PASTIS-SPOT] Loading metadata from {metadata_path}")
        metadata = gpd.read_file(metadata_path)

        fold_mapping = {
            "train": [1, 2, 3],
            "val": [4],
            "test": [5],
        }
        mapped_split = self.split_mapping.get(self.split, self.split)
        folds = fold_mapping.get(mapped_split)
        if folds is None:
            raise ValueError(f"Invalid split '{self.split}'")

        metadata = pd.concat([metadata[metadata["Fold"] == f] for f in folds])
        metadata = metadata.reset_index(drop=True)

        # Filter to patches that have SPOT files
        valid_ids = []
        valid_rows = []
        for i in range(len(metadata)):
            pid = metadata.iloc[i]["ID_PATCH"]
            spot_path = os.path.join(self.spot_dir, f"SPOT6_RVB_1M00_2019_{pid}.tif")
            if os.path.exists(spot_path):
                valid_ids.append(pid)
                valid_rows.append(i)

        self.metadata = metadata.iloc[valid_rows].reset_index(drop=True)
        self.patch_ids = valid_ids

        total_in_split = len(metadata)
        print(f"[PASTIS-SPOT] {len(self.patch_ids)}/{total_in_split} patches have SPOT imagery")

    def _setup_band_indices(self):
        self.spectral_indices = []
        for band_name in sorted(self.SPOT_BANDS_INFO.keys(), key=lambda b: self.SPOT_BANDS_INFO[b]["idx"]):
            info = self.SPOT_BANDS_INFO[band_name]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                new_idx = len(self.look_up.table_wave)
                self.look_up.table_wave[key] = new_idx
                print(f"[PASTIS-SPOT] Registered SPOT band {band_name} {key} → idx {new_idx}")
            self.spectral_indices.append(self.look_up.table_wave[key])
        self.spectral_indices = torch.tensor(self.spectral_indices, dtype=torch.long)

    # =========================================================================
    # LOADING
    # =========================================================================

    def _load_spot(self, patch_id: int) -> torch.Tensor:
        spot_path = os.path.join(self.spot_dir, f"SPOT6_RVB_1M00_2019_{patch_id}.tif")
        with rasterio.open(spot_path) as src:
            data = src.read().astype(np.float32)
        return torch.from_numpy(data)  # [3, H, W]

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> dict:
        patch_id = self.patch_ids[index]

        # ── Load + clean ────────────────────────────────────
        image = self._load_spot(patch_id)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = self._normalize(image)
        image = torch.clamp(image, -10, 10)

        # DEBUG: center-crop to 128×128
        _, _H, _W = image.shape
        _cy, _cx = _H // 2, _W // 2
        image = image[:, _cy - 64:_cy + 64, _cx - 64:_cx + 64]

        # DEBUG: single band only → 16K tokens
        image = image[0:1]

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        # ── Build tokens ────────────────────────────────────
        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.spectral_indices[0:1],
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask = torch.zeros(tokens.shape[0])

        # ── Queries = tokens with col 4 = reflectance ──────
        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()

        # Subsample queries
        N = queries.shape[0]
        n_queries = min(N, self.max_queries)
        perm = torch.randperm(N)[:n_queries]
        queries = queries[perm]
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (1, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
        }

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """
        Viz sample for reconstruction — all tokens as queries, no subsampling.
        """
        patch_id = self.patch_ids[index]

        image = self._load_spot(patch_id)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = self._normalize(image)
        image = torch.clamp(image, -10, 10)

        # DEBUG: center-crop to 128×128
        _, _H, _W = image.shape
        _cy, _cx = _H // 2, _W // 2
        image = image[:, _cy - 64:_cy + 64, _cx - 64:_cx + 64]

        # DEBUG: single band only → 16K tokens
        image = image[0:1]

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.spectral_indices[0:1],
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        tokens[:, 4] = tokens[:, 0].clone()

        queries = tokens.clone()
        queries_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)
        attention_mask = torch.zeros(tokens.shape[0])

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": (1, H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "image": image,
            "image_shape": (1, H, W),
            "n_real": tokens.shape[0],
        }

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "spot_normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[PASTIS-SPOT] Loading normalization from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            print(f"[PASTIS-SPOT] mean: {stats['mean'].numpy()}")
            print(f"[PASTIS-SPOT] std:  {stats['std'].numpy()}")
            return stats

        if self.split != "train":
            print(f"[PASTIS-SPOT] WARNING: No normalization file at {norm_file}")
            return {"mean": torch.zeros(self.NUM_BANDS), "std": torch.ones(self.NUM_BANDS)}

        print(f"[PASTIS-SPOT] Computing normalization from {len(self.patch_ids)} patches...")
        stats = self._compute_normalization()
        torch.save(stats, norm_file)
        print(f"[PASTIS-SPOT] Saved to {norm_file}")
        print(f"[PASTIS-SPOT] mean: {stats['mean'].numpy()}")
        print(f"[PASTIS-SPOT] std:  {stats['std'].numpy()}")
        return stats

    def _compute_normalization(self):
        running_sum = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        running_sq = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        running_n = torch.zeros(self.NUM_BANDS, dtype=torch.float64)

        for pid in tqdm(self.patch_ids, desc="SPOT normalization"):
            try:
                img = self._load_spot(pid).double()
                img = torch.nan_to_num(img)
                for c in range(self.NUM_BANDS):
                    valid = img[c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        running_sum[c] += valid.sum()
                        running_sq[c] += (valid ** 2).sum()
                        running_n[c] += len(valid)
            except Exception as e:
                print(f"[PASTIS-SPOT] Warning: skipping {pid}: {e}")
                continue

        mean = (running_sum / running_n.clamp(min=1)).float()
        std = ((running_sq / running_n.clamp(min=1) - mean.double() ** 2).sqrt()).float()

        return {"mean": mean, "std": std}

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std.clamp(min=1e-6)