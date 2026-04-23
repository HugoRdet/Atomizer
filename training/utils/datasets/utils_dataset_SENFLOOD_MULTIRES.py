"""
Sen1Floods11 Multi-Resolution Dataset — Diagnostic variant
============================================================

Takes the same Sen1Floods11 data (13 S2 + 2 S1 bands, all 512×512 at 10m)
and splits them into 3 resolution groups to simulate MADOS-like multi-resolution input.

Purpose: test whether the multi-resolution pipeline itself is broken,
independent of MADOS data issues (sparsity, NaN, etc.)

Resolution groups:
    10m (512×512):  B02, B03, B04, B08           → 4 bands (native 10m S2)
    20m (256×256):  B05, B06, B07, B8A, B11, B12 → 6 bands (native 20m S2)
                    + VV, VH                      → 2 bands (SAR at 20m)
                    = 8 bands total
    60m (85×85):    B01, B09, B10                 → 3 bands (native 60m S2)

Downsampling: area-average interpolation to simulate coarser resolution.
Labels: always at 10m (512×512), downsampled to 20m/60m for each group's tokens.
Queries: at 10m resolution.

If this converges → multi-res pipeline is fine, MADOS issue is data-specific.
If this fails → multi-res grouping or lookup table is broken.
"""

import os
import csv
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import einops
from tqdm import tqdm

from .token_grouping import *


class Sen1Floods11MultiResDataset(Dataset):
    """
    Sen1Floods11 with MADOS-like multi-resolution groups.

    Returns:
    {
        "groups": {
            10.0: {"tokens": [N_10, 8], "mask": [N_10], "shape": (4, 512, 512)},
            20.0: {"tokens": [N_20, 8], "mask": [N_20], "shape": (8, 256, 256)},
            60.0: {"tokens": [N_60, 8], "mask": [N_60], "shape": (3, 85, 85)},
        },
        "queries":      [M, 8],
        "queries_mask":  [M],
        "label":         [512, 512],
        "target_resolution": 10.0,
        "image":         [15, 512, 512],
    }
    """

    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1

    # ── Band-to-resolution mapping (S2 band order: B01..B12 = indices 0..12) ──
    # S2 native resolutions per band index in the 13-band S2 image:
    #   idx 0:  B01 (443nm)  → 60m
    #   idx 1:  B02 (490nm)  → 10m
    #   idx 2:  B03 (560nm)  → 10m
    #   idx 3:  B04 (665nm)  → 10m
    #   idx 4:  B05 (705nm)  → 20m
    #   idx 5:  B06 (740nm)  → 20m
    #   idx 6:  B07 (783nm)  → 20m
    #   idx 7:  B08 (842nm)  → 10m
    #   idx 8:  B8A (865nm)  → 20m
    #   idx 9:  B09 (945nm)  → 60m
    #   idx 10: B10 (1375nm) → 60m
    #   idx 11: B11 (1610nm) → 20m
    #   idx 12: B12 (2190nm) → 20m
    # S1 (SAR): idx 13 (VV), idx 14 (VH) → 20m (per user request)

    S2_10M_INDICES = [1, 2, 3, 7]              # B02, B03, B04, B08
    S2_20M_INDICES = [4, 5, 6, 8, 11, 12]     # B05, B06, B07, B8A, B11, B12
    S2_60M_INDICES = [0, 9, 10]                # B01, B09, B10
    S1_INDICES = [0, 1]                        # VV, VH → goes to 20m group

    # Spatial dimensions per resolution group
    SIZE_10M = 512
    SIZE_20M = 256   # 512 / 2
    SIZE_60M = 85    # 512 / ~6

    def __init__(
        self,
        root_path: str = "./data/SENFLOOD",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model

        # Config parameters
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        # Split mapping
        self.split_mapping = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.split_mapping[mode]}_data.csv",
        )

        # Load & filter file lists
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        # Band metadata from YAML (same as original SenFlood)
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()

        # Build spectral indices for ALL 15 bands (same order as original)
        self.spectral_indices_all = self._build_spectral_indices()

        # Split spectral indices into resolution groups
        # 10m: S2 bands at indices [1,2,3,7] in the 15-band image
        # 20m: S2 bands at [4,5,6,8,11,12] + S1 at [13,14]
        # 60m: S2 bands at [0,9,10]
        self.spectral_indices_10m = self.spectral_indices_all[self.S2_10M_INDICES]
        s2_20m_in_full = self.S2_20M_INDICES
        s1_in_full = [13, 14]  # S1 bands come after 13 S2 bands
        self.spectral_indices_20m = self.spectral_indices_all[s2_20m_in_full + s1_in_full]
        self.spectral_indices_60m = self.spectral_indices_all[self.S2_60M_INDICES]

        # Resolution indices
        self.res_idx_10m = self.look_up.get_resolution_idx(10.0)
        self.res_idx_20m = self.look_up.get_resolution_idx(20.0)
        self.res_idx_60m = self.look_up.get_resolution_idx(60.0)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[SenFlood-MultiRes] Split={self.split}, {len(self.s1_image_list)} samples")
        print(f"  10m group: {len(self.S2_10M_INDICES)} bands → {self.SIZE_10M}×{self.SIZE_10M}")
        print(f"  20m group: {len(s2_20m_in_full) + len(s1_in_full)} bands → {self.SIZE_20M}×{self.SIZE_20M}")
        print(f"  60m group: {len(self.S2_60M_INDICES)} bands → {self.SIZE_60M}×{self.SIZE_60M}")
        print(f"  Resolution indices: 10m={self.res_idx_10m}, 20m={self.res_idx_20m}, 60m={self.res_idx_60m}")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
        # ── Load ────────────────────────────────────────────
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)    # [13, 512, 512]
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)    # [2, 512, 512]
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)        # [512, 512]

        # ── Clean ───────────────────────────────────────────
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label = torch.from_numpy(label)

        # ── Normalize ───────────────────────────────────────
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)
        image_s2 = torch.clamp(image_s2, -10, 10)
        image_s1 = torch.clamp(image_s1, -10, 10)

        # ── Split into resolution groups ────────────────────
        # 10m: keep at native 512×512
        img_10m = image_s2[self.S2_10M_INDICES]  # [4, 512, 512]

        # 20m: downsample S2 20m bands + SAR to 256×256
        s2_20m = image_s2[self.S2_20M_INDICES]   # [6, 512, 512]
        img_20m = torch.cat([s2_20m, image_s1], dim=0)  # [8, 512, 512]
        img_20m = F.interpolate(
            img_20m.unsqueeze(0),
            size=(self.SIZE_20M, self.SIZE_20M),
            mode="area",
        ).squeeze(0)  # [8, 256, 256]

        # 60m: downsample S2 60m bands to 85×85
        img_60m = image_s2[self.S2_60M_INDICES]  # [3, 512, 512]
        img_60m = F.interpolate(
            img_60m.unsqueeze(0),
            size=(self.SIZE_60M, self.SIZE_60M),
            mode="area",
        ).squeeze(0)  # [3, 85, 85]

        # ── Build token groups ──────────────────────────────
        groups = {}

        groups[10.0] = self._build_group(
            img_10m, label, 10.0,
            self.spectral_indices_10m, self.res_idx_10m,
        )

        groups[20.0] = self._build_group(
            img_20m, label, 20.0,
            self.spectral_indices_20m, self.res_idx_20m,
        )

        groups[60.0] = self._build_group(
            img_60m, label, 60.0,
            self.spectral_indices_60m, self.res_idx_60m,
        )

        # ── Queries at 10m ──────────────────────────────────
        queries = self._build_queries(label, 10.0)

        queries_mask = torch.zeros(queries.shape[0])
        queries, queries_mask = self._shuffle_arrays([queries, queries_mask])
        nb_queries = self.max_tokens_reconstruction
        queries = queries[:nb_queries]
        queries_mask = queries_mask[:nb_queries]

        # ── Combined image for visualization (all at 10m) ───
        img_20m_up = F.interpolate(
            img_20m.unsqueeze(0),
            size=(self.SIZE_10M, self.SIZE_10M),
            mode="nearest",
        ).squeeze(0)
        img_60m_up = F.interpolate(
            img_60m.unsqueeze(0),
            size=(self.SIZE_10M, self.SIZE_10M),
            mode="nearest",
        ).squeeze(0)
        image_combined = torch.cat([img_10m, img_20m_up, img_60m_up], dim=0)

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "label": label,
            "target_resolution": 10.0,
            "image": image_combined,
        }

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_group(self, image, label_10m, resolution, spectral_indices, resolution_idx):
        """
        Build a single resolution group with tokens [N, 8].

        Args:
            image:           [C, H, W] already at target resolution
            label_10m:       [H_10, W_10] labels at 10m
            resolution:      float, meters per pixel
            spectral_indices: [C] spectral lookup indices
            resolution_idx:  int, resolution lookup index

        Returns:
            dict with "tokens", "mask", "shape"
        """
        C, H, W = image.shape

        # Downsample label to this resolution if needed
        if H != label_10m.shape[0] or W != label_10m.shape[1]:
            label_res = F.interpolate(
                label_10m.float().unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        else:
            label_res = label_10m

        # Spectral indices: [C*H*W]
        spectral_coords = spectral_indices.repeat_interleave(H * W)

        # Position coordinates: [C, H, W, 1]
        x_indices, y_indices = self._get_position_coordinates((C, H, W), resolution)

        # Query position indices: [C, H, W, 1]
        query_indices = self._get_query_coordinates((C, H, W), resolution)

        # Expand label: [H, W] → [C, H, W]
        label_expanded = label_res.unsqueeze(0).expand(C, -1, -1)

        # Resolution: same for all bands in this group
        resolution_col = torch.full((C, H, W, 1), resolution_idx, dtype=torch.float32)

        # Time: -1 for all tokens
        time_col = torch.full((C, H, W, 1), self.TIME_IDX_NA, dtype=torch.float32)

        # Stack: [C, H, W, 8]
        image_tokens = torch.cat([
            image.unsqueeze(-1),                                    # col 0: value
            x_indices.float(),                                      # col 1: x
            y_indices.float(),                                      # col 2: y
            spectral_coords.view(C, H, W, 1).float(),              # col 3: spectral_idx
            label_expanded.unsqueeze(-1).float(),                   # col 4: label
            query_indices.float(),                                  # col 5: query_idx
            resolution_col,                                         # col 6: resolution_idx
            time_col,                                               # col 7: time_idx
        ], dim=-1)

        # Flatten: [C*H*W, 8]
        image_tokens = einops.rearrange(image_tokens, "c h w f -> (c h w) f")

        attention_mask = torch.zeros(image_tokens.shape[0])

        return {
            "tokens": image_tokens,
            "mask": attention_mask,
            "shape": tuple(image.shape),
        }

    def _build_queries(self, label, resolution):
        """Build query tokens at 10m resolution → [H*W, 8]."""
        H, W = label.shape

        # Use first spectral index from 10m group
        first_spectral_idx = self.spectral_indices_10m[0]

        x_indices, y_indices = self._get_position_coordinates((1, H, W), resolution)
        query_indices = self._get_query_coordinates((1, H, W), resolution)

        # Resolution index for queries = 10m
        resolution_col = torch.full((1, H, W, 1), self.res_idx_10m, dtype=torch.float32)
        time_col = torch.full((1, H, W, 1), self.TIME_IDX_NA, dtype=torch.float32)

        queries = torch.cat([
            torch.zeros(1, H, W, 1),                                             # col 0: value
            x_indices.float(),                                                    # col 1: x
            y_indices.float(),                                                    # col 2: y
            torch.full((1, H, W, 1), first_spectral_idx, dtype=torch.float),     # col 3: spectral
            label.unsqueeze(0).unsqueeze(-1).float(),                             # col 4: label
            query_indices.float(),                                                # col 5: query_idx
            resolution_col,                                                       # col 6: res_idx
            time_col,                                                             # col 7: time_idx
        ], dim=-1)

        queries = einops.rearrange(queries, "c h w f -> (c h w) f")
        return queries

    def _get_position_coordinates(self, image_shape, resolution):
        """Pixel position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        res_key = int(resolution * 1000)
        global_offset = self.look_up.table[(res_key, H)]

        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")

        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)

        return x_indices, y_indices

    def _get_query_coordinates(self, image_shape, resolution):
        """Query position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        resolution_latents = 10  # m
        res_key = int(resolution_latents * 1000)

        if (res_key, H) in self.look_up.table_queries:
            global_offset = self.look_up.table_queries[(res_key, H)]
        else:
            res_key = int(resolution * 1000)
            global_offset = self.look_up.table_queries.get((res_key, H), 0)

        return torch.full((C, H, W, 1), global_offset, dtype=torch.float32)

    # =========================================================================
    # FILE LOADING (identical to original)
    # =========================================================================

    def _load_file_lists(self):
        s1_images, s2_images, labels = [], [], []
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
        for i in tqdm(range(len(self.label_list)), desc="[SenFlood-MR] Checking labels"):
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
            except Exception:
                skipped += 1
        print(f"[SenFlood-MR] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list = valid_labels

    # =========================================================================
    # NORMALIZATION (reuse same stats as original SenFlood)
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")
        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            return stats
        if self.split != "train":
            return {
                "s2_mean": torch.zeros(13), "s2_std": torch.ones(13),
                "s1_mean": torch.zeros(2),  "s1_std": torch.ones(2),
            }
        print(f"[SenFlood-MR] Computing normalization...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(13, dtype=torch.float64)
        s2_sq  = torch.zeros(13, dtype=torch.float64)
        s2_n   = torch.zeros(13, dtype=torch.float64)
        s1_sum = torch.zeros(2, dtype=torch.float64)
        s1_sq  = torch.zeros(2, dtype=torch.float64)
        s1_n   = torch.zeros(2, dtype=torch.float64)
        for idx in tqdm(range(len(self.s2_image_list)), desc="Normalization"):
            try:
                with rasterio.open(self.s2_image_list[idx]) as src:
                    s2 = src.read().astype(np.float64)
                s2 = np.nan_to_num(s2)
                for c in range(13):
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
                for c in range(2):
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
        return {"s2_mean": s2_mean, "s2_std": s2_std, "s1_mean": s1_mean, "s1_std": s1_std}

    def normalize_image(self, s2, s1):
        s2_mean = self.norm_stats["s2_mean"].view(13, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(13, 1, 1)
        s1_mean = self.norm_stats["s1_mean"].view(2, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(2, 1, 1)
        return (s2 - s2_mean) / s2_std, (s1 - s1_mean) / s1_std

    # =========================================================================
    # BAND METADATA (identical to original)
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
        bw = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]
        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} key={key} not in lookup. "
                    f"Available: {list(self.look_up.table_wave.keys())}"
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