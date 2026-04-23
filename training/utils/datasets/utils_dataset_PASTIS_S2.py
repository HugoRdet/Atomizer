"""
PASTIS S2 Reconstruction Dataset
==================================

Reconstruction-only dataset using Sentinel-2 imagery from PASTIS-HD.
Single random timestep, 10 bands at 10m resolution, ~128×128 pixels.

Diagnostic dataset to test reconstruction on 10m imagery.

Single-band diagnostic mode:
  Set SINGLE_BAND_IDX class variable to 0-9 to encode+reconstruct
  only that band. -1 = all bands (default).
  Both encoder and decoder see only the selected band.
  
  Band indices:
    0=B02(Blue) 1=B03(Green) 2=B04(Red) 3=B05(RE1)
    4=B06(RE2) 5=B07(RE3) 6=B08(NIR) 7=B8A(NIR-n)
    8=B11(SWIR1) 9=B12(SWIR2)

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

Returns:
{
    "groups": {
        10.0: {
            "tokens": [N, 8],
            "mask":   [N],
            "shape":  (C, H, W),   # C=1 in single-band, C=10 otherwise
        },
    },
    "queries":           [M, 8],
    "queries_mask":      [M],
    "target_resolution": 10.0,
}
"""

import os
import json
import random

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
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


class PastisS2ReconDataset(Dataset):
    """
    PASTIS Sentinel-2 Reconstruction Dataset.

    Single resolution group at 10m, 10 optical bands, single random timestep.
    Queries are image tokens with col 4 = reflectance value.

    Single-band diagnostic mode:
        Set SINGLE_BAND_IDX = 2 (Red) to encode + reconstruct only B04.
        Both encoder and decoder see only that band (128×128×1 = 16K tokens).
        Isolates: can the architecture handle 10m spatial reconstruction?
    """

    # ── Single-band diagnostic mode ────────────────────────
    # Set to band index (0-9) to encode+reconstruct only that band.
    # -1 = all bands (default). No config change needed.
    #
    # Band indices:
    #   0=B02(Blue) 1=B03(Green) 2=B04(Red) 3=B05(RE1)
    #   4=B06(RE2) 5=B07(RE3) 6=B08(NIR) 7=B8A(NIR-n)
    #   8=B11(SWIR1) 9=B12(SWIR2)
    SINGLE_BAND_IDX = -1

    RESOLUTION = 10.0
    NUM_BANDS = 10
    IGNORE_INDEX = 255
    TIME_IDX_NA = 1

    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490, "bandwidth": 65, "idx": 0},
        "B03": {"central_wavelength": 560, "bandwidth": 35, "idx": 1},
        "B04": {"central_wavelength": 665, "bandwidth": 30, "idx": 2},
        "B05": {"central_wavelength": 705, "bandwidth": 15, "idx": 3},
        "B06": {"central_wavelength": 740, "bandwidth": 15, "idx": 4},
        "B07": {"central_wavelength": 783, "bandwidth": 20, "idx": 5},
        "B08": {"central_wavelength": 842, "bandwidth": 115, "idx": 6},
        "B8A": {"central_wavelength": 865, "bandwidth": 20, "idx": 7},
        "B11": {"central_wavelength": 1610, "bandwidth": 90, "idx": 8},
        "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
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

        # Load metadata
        self._load_metadata()

        # Band indices
        self._setup_band_indices()

        # Single-band mode: resolve to spectral index
        self.recon_spectral_idx = None
        if self.SINGLE_BAND_IDX >= 0:
            self._setup_single_band()

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[PASTIS-S2] Loaded {len(self.patch_ids)} patches for split '{self.split}'")
        if self.SINGLE_BAND_IDX >= 0:
            band_name = self._idx_to_band_name(self.SINGLE_BAND_IDX)
            print(f"[PASTIS-S2] *** SINGLE-BAND MODE: {band_name} (idx={self.SINGLE_BAND_IDX}) ***")
            print(f"[PASTIS-S2]     spectral_idx = {self.recon_spectral_idx}")
            print(f"[PASTIS-S2]     Encoder + decoder: SINGLE BAND ONLY")
        else:
            print(f"[PASTIS-S2] {self.NUM_BANDS} bands @ {self.RESOLUTION}m")
            print(f"[PASTIS-S2] Mode: RECONSTRUCTION (all bands, single random timestep)")

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required. pip install geopandas")

        metadata_path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.geojson not found at {metadata_path}")

        print(f"[PASTIS-S2] Loading metadata from {metadata_path}")
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

        # Filter to patches that have S2 files
        valid_ids = []
        valid_rows = []
        for i in range(len(metadata)):
            pid = metadata.iloc[i]["ID_PATCH"]
            s2_path = os.path.join(self.root_path, "DATA_S2", f"S2_{pid}.npy")
            if os.path.exists(s2_path):
                valid_ids.append(pid)
                valid_rows.append(i)

        self.metadata = metadata.iloc[valid_rows].reset_index(drop=True)
        self.patch_ids = valid_ids

        total_in_split = len(metadata)
        print(f"[PASTIS-S2] {len(self.patch_ids)}/{total_in_split} patches have S2 data")

    def _setup_band_indices(self):
        self.spectral_indices = []
        for band_name in sorted(self.S2_BANDS_INFO.keys(), key=lambda b: self.S2_BANDS_INFO[b]["idx"]):
            info = self.S2_BANDS_INFO[band_name]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"S2 band {band_name} key={key} not in lookup table.\n"
                    f"Available keys: {list(self.look_up.table_wave.keys())}"
                )
            self.spectral_indices.append(self.look_up.table_wave[key])
        self.spectral_indices = torch.tensor(self.spectral_indices, dtype=torch.long)

    def _setup_single_band(self):
        """Resolve SINGLE_BAND_IDX to its lookup table spectral index."""
        band_name = self._idx_to_band_name(self.SINGLE_BAND_IDX)
        if band_name is None:
            raise ValueError(
                f"SINGLE_BAND_IDX={self.SINGLE_BAND_IDX} out of range. "
                f"Valid: 0-{self.NUM_BANDS - 1}"
            )
        info = self.S2_BANDS_INFO[band_name]
        key = (info["bandwidth"], info["central_wavelength"])
        self.recon_spectral_idx = self.look_up.table_wave[key]

    def _idx_to_band_name(self, idx: int):
        """Convert band array index (0-9) to band name (B02, B03, ...)."""
        for name, info in self.S2_BANDS_INFO.items():
            if info["idx"] == idx:
                return name
        return None

    # =========================================================================
    # LOADING
    # =========================================================================
    

    def _load_s2(self, patch_id: int) -> torch.Tensor:
        s2_path = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
        s2_data = np.load(s2_path).astype(np.float32)
        s2_data = torch.from_numpy(s2_data)
        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)
        return s2_data[0]  # DEBUG: always first timestep

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> dict:
        patch_id = self.patch_ids[index]

        image = self._load_s2(patch_id)
        image = self._normalize(image)

        
        image = torch.clamp(image, -10, 10)
        

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        # Single-band mode: slice image to one channel before tokenization
        if self.recon_spectral_idx is not None:
            b = self.SINGLE_BAND_IDX
            image_for_tokens = image[b:b+1]                     # [1, H, W]
            spectral_indices = self.spectral_indices[b:b+1]     # [1]
            shape = (1, H, W)
        else:
            image_for_tokens = image
            spectral_indices = self.spectral_indices
            shape = (C, H, W)

        tokens = self.token_builder.build_tokens(
            image=image_for_tokens,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask = torch.zeros(tokens.shape[0])

        # Queries = tokens with col 4 = reflectance
        queries = tokens.clone()
        queries[:, 4] = queries[:, 0].clone()

        # Subsample
        N = queries.shape[0]
        n_queries = min(N, self.max_queries)
        if N > n_queries:
            perm = torch.randperm(N)[:n_queries]
            queries = queries[perm]
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)


        

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": shape,
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
        patch_id = self.patch_ids[index]

        image = self._load_s2(patch_id)
        image = self._normalize(image)
        image = torch.clamp(image, -10, 10)

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        # Single-band mode: slice image to one channel
        if self.recon_spectral_idx is not None:
            b = self.SINGLE_BAND_IDX
            image_for_tokens = image[b:b+1]
            spectral_indices = self.spectral_indices[b:b+1]
            shape = (1, H, W)
            viz_image = image[b:b+1]                            # single-band for viz
        else:
            image_for_tokens = image
            spectral_indices = self.spectral_indices
            shape = (C, H, W)
            viz_image = image

        tokens = self.token_builder.build_tokens(
            image=image_for_tokens,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        tokens[:, 4] = tokens[:, 0].clone()

        queries = tokens.clone()
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        attention_mask = torch.zeros(tokens.shape[0])

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": tokens,
                    "mask": attention_mask,
                    "shape": shape,
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": self.RESOLUTION,
            "image": viz_image,
            "image_shape": shape,
            "n_real": queries.shape[0],
        }

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "NORM_S2_patch.json")

        if os.path.exists(norm_file):
            print(f"[PASTIS-S2] Loading per-fold normalization from {norm_file}")
            with open(norm_file, "r") as f:
                all_stats = json.load(f)

            # Determine which folds we're using
            fold_mapping = {
                "train": [1, 2, 3],
                "val": [4],
                "test": [5],
            }
            mapped_split = self.split_mapping.get(self.split, self.split)
            folds = fold_mapping.get(mapped_split, [1, 2, 3])

            # Average stats across our folds
            means = []
            stds = []
            for f in folds:
                key = f"Fold_{f}"
                if key in all_stats:
                    means.append(torch.tensor(all_stats[key]["mean"], dtype=torch.float32))
                    stds.append(torch.tensor(all_stats[key]["std"], dtype=torch.float32))

            mean = torch.stack(means).mean(dim=0)
            std = torch.stack(stds).mean(dim=0)

            print(f"[PASTIS-S2] Folds used: {folds}")
            print(f"[PASTIS-S2] mean: {mean.numpy()}")
            print(f"[PASTIS-S2] std:  {std.numpy()}")
            return {"mean": mean, "std": std}

        # Fallback to old method
        print(f"[PASTIS-S2] WARNING: {norm_file} not found, falling back to compute")
        return {"mean": torch.zeros(self.NUM_BANDS), "std": torch.ones(self.NUM_BANDS)}

    def _compute_normalization(self):
        running_sum = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        running_sq = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        running_n = torch.zeros(self.NUM_BANDS, dtype=torch.float64)

        for pid in tqdm(self.patch_ids, desc="S2 normalization"):
            try:
                s2_path = os.path.join(self.root_path, "DATA_S2", f"S2_{pid}.npy")
                s2 = np.load(s2_path).astype(np.float64)
                s2 = np.nan_to_num(s2)
                for c in range(self.NUM_BANDS):
                    valid = s2[:, c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        running_sum[c] += valid.sum()
                        running_sq[c] += (valid ** 2).sum()
                        running_n[c] += len(valid)
            except Exception as e:
                print(f"[PASTIS-S2] Warning: skipping {pid}: {e}")
                continue

        mean = (running_sum / running_n.clamp(min=1)).float()
        std = ((running_sq / running_n.clamp(min=1) - mean.double() ** 2).sqrt()).float()

        return {"mean": mean, "std": std}

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std.clamp(min=1e-6)