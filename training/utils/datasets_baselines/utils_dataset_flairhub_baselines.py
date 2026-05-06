"""
FLAIR-HUB Baseline Dataset
============================

Standard tensor format for ResNet/ViT/UPerNet baselines on FLAIR-HUB.
All modalities are upsampled to a common 0.2m (512×512) resolution via
bilinear interpolation and concatenated into a single channel dimension.

Output (per sample):
    {
        "image":    {"flairhub": [C_total, H, W]},
        "target":   [H, W]                          (long, 0..18, 255=ignore)
        "metadata": {"patch_id": str, ...},
    }

Channel ordering (deterministic, matches the modality flags' order):
    [VHR R, VHR G, VHR B, VHR NIR,
     DEM DSM, DEM DTM,
     S2 t0_b1, S2 t0_b2, ..., S2 t0_b10, S2 t1_b1, ..., S2 t5_b10,
     S1_ASC t0_VV, S1_ASC t0_VH, ..., S1_ASC t5_VH,
     S1_DESC t0_VV, ..., S1_DESC t5_VH]

With default flags (VHR + DEM + S2 + S1, T=6):
    4 + 2 + (10×6) + (2×6) + (2×6) = 90 channels

Cross-sensor transfer:
    Train with use_vhr=True, use_spot=False  → 4 VHR ch in slots [0..3]
    Test  with use_vhr=False, use_spot=True  → 4 SPOT ch in slots [0..3]
    Same architecture, same channel count, same baseline weights.
    SPOT (1.6m) gets bilinearly upsampled to 512×512 like everything else.

Memory:
    [B, 90, 512, 512] fp32 ≈ 94 MB / sample
    Batch 4 ≈ 376 MB just for inputs. Reasonable on H100.
"""

import os
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import rasterio


class FlairHubBaselineDataset(Dataset):
    """
    Baseline-format FLAIR-HUB dataset (zero-padding/upsample fusion).

    Same patches/splits/subsets as the Atomizer FlairHubDataset, but
    returns a standard 3D image tensor instead of token sets.

    Args mirror FlairHubDataset where applicable (use_vhr, use_spot, etc.)
    so the same CLI flags can drive both Atomizer and baseline runs.
    """

    # Resolutions and sizes per modality (must match what's on disk)
    VHR_SIZE     = 512
    SPOT_SIZE    = 64
    SAT_SIZE     = 10
    TARGET_SIZE  = 512   # everything upsampled to here

    # Bands per modality
    NUM_VHR_BANDS  = 4
    NUM_SPOT_BANDS = 4
    NUM_DEM_BANDS  = 2
    NUM_S2_BANDS   = 10
    NUM_S1_BANDS   = 2

    NUM_CLASSES  = 19
    IGNORE_INDEX = 255

    SPLIT_FILES = {
        "train":      "FLAIR-HUB_TRAIN.csv",
        "validation": "FLAIR-HUB_VALID.csv",
        "test":       "FLAIR-HUB_TEST.csv",
    }

    def __init__(
        self,
        root_path: str = "./data/FLAIR-HUB",
        mode: str = "train",
        use_vhr: bool = True,
        use_spot: bool = False,
        use_dem: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        multi_temporal: int = 6,
    ):
        super().__init__()
        if not (use_vhr or use_spot):
            raise ValueError("At least one of use_vhr / use_spot must be True.")

        self.root_path      = root_path
        self.split          = mode
        self.use_vhr        = use_vhr
        self.use_spot       = use_spot
        self.use_dem        = use_dem
        self.use_s2         = use_s2
        self.use_s1         = use_s1
        self.multi_temporal = multi_temporal

        # ── Read split CSV ──────────────────────────────────────
        split_csv = os.path.join(root_path, self.SPLIT_FILES[mode])
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        self.split_df = pd.read_csv(split_csv, sep=";")
        self.patch_rows = self.split_df.to_dict("records")

        # ── Load normalization stats ────────────────────────────
        self._load_normalization_stats()

        # ── Compute total channel count ─────────────────────────
        self.num_channels = self._compute_channel_count()

        # ── Print summary ───────────────────────────────────────
        print(f"[FLAIR-HUB-Baseline] split={mode}  patches={len(self.patch_rows)}  "
              f"channels={self.num_channels}")
        modality_str = []
        if self.use_vhr:  modality_str.append(f"VHR(4)")
        if self.use_spot: modality_str.append(f"SPOT(4)")
        if self.use_dem:  modality_str.append(f"DEM(2)")
        if self.use_s2:   modality_str.append(f"S2(10×{multi_temporal}={10*multi_temporal})")
        if self.use_s1:   modality_str.append(f"S1×2(2×2×{multi_temporal}={4*multi_temporal})")
        print(f"[FLAIR-HUB-Baseline] modalities: {' + '.join(modality_str)} "
              f"= {self.num_channels} ch @ {self.TARGET_SIZE}×{self.TARGET_SIZE}")

    # ────────────────────────────────────────────────────────────
    # Init helpers
    # ────────────────────────────────────────────────────────────

    def _compute_channel_count(self) -> int:
        c = 0
        if self.use_vhr:  c += self.NUM_VHR_BANDS
        if self.use_spot: c += self.NUM_SPOT_BANDS
        if self.use_dem:  c += self.NUM_DEM_BANDS
        if self.use_s2:   c += self.NUM_S2_BANDS  * self.multi_temporal
        if self.use_s1:   c += 2 * self.NUM_S1_BANDS * self.multi_temporal  # ASC + DESC
        return c

    def _load_normalization_stats(self):
        norm_path = os.path.join(self.root_path, "normalization_stats.json")
        if not os.path.exists(norm_path):
            print(f"[FLAIR-HUB-Baseline] WARNING: norm stats not found.")
            self.norm_stats = {}
            return
        with open(norm_path) as f:
            self.norm_stats = json.load(f)

    def _resolve_path(self, rel_path: str) -> str:
        """Strip leading ../ or ./ and prepend root/extracted/."""
        rel_clean = rel_path
        while rel_clean.startswith(("../", "./")):
            if rel_clean.startswith("../"):
                rel_clean = rel_clean[3:]
            else:
                rel_clean = rel_clean[2:]
        return os.path.join(self.root_path, "extracted", rel_clean)

    # ────────────────────────────────────────────────────────────
    # Dataset interface
    # ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]
        patch_id = row["patch_id"]

        # ── Label (target) ──────────────────────────────────────
        label_path = self._resolve_path(row["AERIAL_LABEL-COSIA"])
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)
        valid = (label >= 0) & (label < self.NUM_CLASSES)
        label = np.where(valid, label, self.IGNORE_INDEX)
        label = torch.from_numpy(label).long()                     # [512, 512]

        # ── Build channel tensors per modality ──────────────────
        # All upsampled to [TARGET_SIZE, TARGET_SIZE] via bilinear.
        # Concatenated along channel dim in this order:
        #   VHR, SPOT, DEM, S2, S1_ASC, S1_DESC
        channels = []

        if self.use_vhr:
            channels.append(self._load_vhr(row))                   # [4, 512, 512]

        if self.use_spot:
            channels.append(self._load_spot(row))                  # [4, 512, 512]

        if self.use_dem:
            channels.append(self._load_dem(row))                   # [2, 512, 512]

        if self.use_s2:
            channels.append(self._load_s2_ts(row))                 # [10*T, 512, 512]

        if self.use_s1:
            channels.append(self._load_s1_ts(row, mode="asc"))     # [2*T, 512, 512]
            channels.append(self._load_s1_ts(row, mode="desc"))    # [2*T, 512, 512]

        image = torch.cat(channels, dim=0)                         # [num_channels, 512, 512]

        return {
            "image":    {"flairhub": image},
            "target":   label,
            "metadata": {"patch_id": patch_id},
        }

    # ────────────────────────────────────────────────────────────
    # Per-modality loaders
    # ────────────────────────────────────────────────────────────

    def _load_vhr(self, row) -> torch.Tensor:
        """Load AERIAL_RGBI as [4, 512, 512] float32, normalized."""
        path = self._resolve_path(row["AERIAL_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)                    # [4, 512, 512]
        img = torch.from_numpy(img)
        img = self._normalize(img, "aerial")
        img = torch.clamp(img, -10, 10)
        img = torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)
        # Already at TARGET_SIZE — no upsample.
        return img

    def _load_spot(self, row) -> torch.Tensor:
        """Load SPOT_RGBI as [4, 64, 64], upsample to [4, 512, 512]."""
        path = self._resolve_path(row["SPOT_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)                    # [4, 64, 64]
        img = torch.from_numpy(img)
        img = self._normalize(img, "spot")
        img = torch.clamp(img, -10, 10)
        img = torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)
        return self._upsample(img)                                  # [4, 512, 512]

    def _load_dem(self, row) -> torch.Tensor:
        """Load DEM (DSM + DTM) as [2, 512, 512]. Already at TARGET_SIZE."""
        path = self._resolve_path(row["DEM_ELEV"])
        with rasterio.open(path) as src:
            dem = src.read().astype(np.float32)                    # [2, 512, 512]
        dem = torch.from_numpy(dem)
        dem = self._normalize_dem(dem)
        dem = torch.clamp(dem, -10, 10)
        dem = torch.nan_to_num(dem, nan=0.0, posinf=10.0, neginf=-10.0)
        return dem

    def _load_s2_ts(self, row) -> torch.Tensor:
        """
        Load S2 time series, sample T timesteps via linspace, normalize,
        upsample to TARGET_SIZE, return [10*T, 512, 512].
        """
        path = self._resolve_path(row["SENTINEL2_TS"])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)                  # [T_total*10, 10, 10]
        T_total = stack.shape[0] // self.NUM_S2_BANDS
        stack = stack.reshape(T_total, self.NUM_S2_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)        # [T, 10, 10, 10]

        # Sample T timesteps
        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]                                   # [T_sel, 10, 10, 10]

        # Normalize each band
        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, "s2")
        stack = torch.clamp(stack, -10, 10)

        # Reshape to [10*T, 10, 10] and upsample
        T_sel = stack.shape[0]
        stack = stack.reshape(T_sel * self.NUM_S2_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        return self._upsample(stack)                                # [10*T, 512, 512]

    def _load_s1_ts(self, row, mode: str) -> torch.Tensor:
        """Load S1 ASC or DESC time series, similar to S2."""
        if mode == "asc":
            col, norm_key = "SENTINEL1-ASC_TS", "s1_asc"
        else:
            col, norm_key = "SENTINEL1-DESC_TS", "s1_des"

        path = self._resolve_path(row[col])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)                  # [T_total*2, 10, 10]
        T_total = stack.shape[0] // self.NUM_S1_BANDS
        stack = stack.reshape(T_total, self.NUM_S1_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)

        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]

        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, norm_key)
        stack = torch.clamp(stack, -10, 10)

        T_sel = stack.shape[0]
        stack = stack.reshape(T_sel * self.NUM_S1_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        return self._upsample(stack)                                # [2*T, 512, 512]

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample a [C, H, W] tensor to [C, TARGET_SIZE, TARGET_SIZE]."""
        if x.shape[-1] == self.TARGET_SIZE and x.shape[-2] == self.TARGET_SIZE:
            return x
        return F.interpolate(
            x.unsqueeze(0),
            size=(self.TARGET_SIZE, self.TARGET_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _linspace_sample(self, T_total: int, n: int) -> np.ndarray:
        if T_total <= n:
            return np.arange(T_total)
        return np.linspace(0, T_total - 1, n, dtype=int)

    def _normalize(self, img: torch.Tensor, mod_key: str) -> torch.Tensor:
        if mod_key not in self.norm_stats:
            return img
        stats = self.norm_stats[mod_key]
        C = img.shape[0]
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)[:C]
        std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1)[:C].clamp(min=1e-6)
        return (img - mean) / std

    def _normalize_per_timestep(self, stack: torch.Tensor, mod_key: str) -> torch.Tensor:
        if mod_key not in self.norm_stats:
            return stack
        stats = self.norm_stats[mod_key]
        B = stack.shape[1]
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)[:, :B]
        std  = torch.tensor(stats["std"],  dtype=torch.float32).view(1, -1, 1, 1)[:, :B].clamp(min=1e-6)
        return (stack - mean) / std

    def _normalize_dem(self, dem: torch.Tensor) -> torch.Tensor:
        """Same convention as Atomizer dataset: per-image mean, fixed std=50m."""
        if "dem" in self.norm_stats:
            stats = self.norm_stats["dem"]
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)
            return (dem - mean) / std
        per_image_mean = dem.mean(dim=(1, 2), keepdim=True)
        return (dem - per_image_mean) / 50.0