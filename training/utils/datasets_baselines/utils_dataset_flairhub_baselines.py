"""
FLAIR-HUB Baseline Dataset (with per-modality support)
========================================================

Returns either a single concatenated tensor (for "one-encoder" baselines)
or a dict of separate per-modality tensors (for "per-modality" baselines).

Output (per sample):

    per_modality=False (default, single-encoder concat fusion):
        {
            "image":    {"flairhub": [C_total, H, W]},
            "target":   [H, W],
            "metadata": {"patch_id": str, ...},
        }

    per_modality=True (per-modality fusion baselines):
        {
            "image":    {
                "optical": [4, 512, 512],         # VHR or SPOT (always 4 ch)
                "dem":     [2, 512, 512],
                "s2":      [T, 10, H_s2, W_s2],   # 5D, kept temporal
                "s1":      [T,  4, H_s1, W_s1],   # 5D, ASC+DESC concat
            },
            "target":   [H, W],
            "metadata": {"patch_id": str, ...},
        }

Notes on per-modality format:
  - Satellite modalities ("s2", "s1") keep their T dimension explicit so
    the per-modality model can use TimeMerge / LTAE for temporal aggregation.
  - "optical" is the same key whether VHR or SPOT is loaded — the
    per-modality model routes either input through one branch with 4
    channels. This makes the cross-sensor transfer a config flag flip.
  - In per-modality mode, satellite modalities are returned at native
    spatial size (10×10). The per-modality model upsamples internally.
    This saves dataloader memory (batches don't carry the upsampled
    versions over the worker→main-process boundary).
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import rasterio


class FlairHubBaselineDataset(Dataset):
    """
    Baseline-format FLAIR-HUB dataset.

    Two output modes (controlled by `per_modality`):

      per_modality=False:
        All modalities upsampled to 0.2m (512×512) and concatenated as
        flat channels → one big tensor under {"image": {"flairhub": ...}}.
        Used by single-encoder baselines (90-channel ResNet, ViT).

      per_modality=True:
        Each modality returned as its own tensor under
        {"image": {"optical": ..., "dem": ..., "s2": ..., "s1": ...}}.
        Used by per-modality fusion baselines.
    """

    # Resolutions and sizes per modality (must match what's on disk)
    VHR_SIZE     = 512
    SPOT_SIZE    = 64
    SAT_SIZE     = 10
    TARGET_SIZE  = 512   # everything upsampled to here in concat mode

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
        spot_norm_as_vhr: bool = False,
        use_dem: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        multi_temporal: int = 6,
        per_modality: bool = False,
    ):
        super().__init__()
        if not (use_vhr or use_spot):
            raise ValueError("At least one of use_vhr / use_spot must be True.")
        if use_vhr and use_spot and not per_modality:
            print("[FLAIR-HUB-Baseline] WARNING: VHR and SPOT both enabled in "
                  "concat mode. They'll occupy separate channel slots.")
        if use_vhr and use_spot and per_modality:
            raise ValueError(
                "per_modality=True with both VHR and SPOT is ambiguous: "
                "both would go to the 'optical' branch. Enable only one."
            )
        if spot_norm_as_vhr and not use_spot:
            print("[FLAIR-HUB-Baseline] WARNING: spot_norm_as_vhr=True but "
                  "use_spot=False, the flag will have no effect.")

        self.root_path        = root_path
        self.split            = mode
        self.use_vhr          = use_vhr
        self.use_spot         = use_spot
        self.spot_norm_as_vhr = spot_norm_as_vhr
        self.use_dem          = use_dem
        self.use_s2           = use_s2
        self.use_s1           = use_s1
        self.multi_temporal   = multi_temporal
        self.per_modality     = per_modality

        # ── Read split CSV ──────────────────────────────────────
        split_csv = os.path.join(root_path, self.SPLIT_FILES[mode])
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        self.split_df = pd.read_csv(split_csv, sep=";")
        self.patch_rows = self.split_df.to_dict("records")

        # ── Load normalization stats ────────────────────────────
        self._load_normalization_stats()

        # ── Compute total channel count (concat mode only) ──────
        self.num_channels = self._compute_channel_count()

        # ── Print summary ───────────────────────────────────────
        mode_str = "per-modality" if per_modality else "concat"
        print(f"[FLAIR-HUB-Baseline] split={mode}  patches={len(self.patch_rows)}  "
              f"output={mode_str}")
        modality_str = []
        if self.use_vhr:  modality_str.append("VHR(4)")
        if self.use_spot: modality_str.append("SPOT(4)")
        if self.use_dem:  modality_str.append("DEM(2)")
        if self.use_s2:   modality_str.append(f"S2(10×{multi_temporal})")
        if self.use_s1:   modality_str.append(f"S1(4×{multi_temporal})")
        if per_modality:
            print(f"[FLAIR-HUB-Baseline] modalities (separate tensors): "
                  f"{' + '.join(modality_str)}")
        else:
            print(f"[FLAIR-HUB-Baseline] modalities (concat): "
                  f"{' + '.join(modality_str)} = {self.num_channels} ch "
                  f"@ {self.TARGET_SIZE}×{self.TARGET_SIZE}")

    # ────────────────────────────────────────────────────────────
    # Init helpers
    # ────────────────────────────────────────────────────────────

    def _compute_channel_count(self) -> int:
        c = 0
        if self.use_vhr:  c += self.NUM_VHR_BANDS
        if self.use_spot: c += self.NUM_SPOT_BANDS
        if self.use_dem:  c += self.NUM_DEM_BANDS
        if self.use_s2:   c += self.NUM_S2_BANDS  * self.multi_temporal
        if self.use_s1:   c += 2 * self.NUM_S1_BANDS * self.multi_temporal
        return c

    def _load_normalization_stats(self):
        norm_path = os.path.join(self.root_path, "normalization_stats.json")
        if not os.path.exists(norm_path):
            print("[FLAIR-HUB-Baseline] WARNING: norm stats not found.")
            self.norm_stats = {}
            return
        with open(norm_path) as f:
            self.norm_stats = json.load(f)

    def _resolve_path(self, rel_path: str) -> str:
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
        label = torch.from_numpy(label).long()

        if self.per_modality:
            return self._getitem_per_modality(row, label, patch_id)
        return self._getitem_concat(row, label, patch_id)

    # ────────────────────────────────────────────────────────────
    # Concat output (existing behavior)
    # ────────────────────────────────────────────────────────────

    def _getitem_concat(self, row, label, patch_id):
        channels = []

        if self.use_vhr:
            channels.append(self._load_vhr(row))
        if self.use_spot:
            channels.append(self._load_spot(row))
        if self.use_dem:
            channels.append(self._load_dem(row))
        if self.use_s2:
            channels.append(self._load_s2_ts_concat(row))
        if self.use_s1:
            channels.append(self._load_s1_ts_concat(row, mode="asc"))
            channels.append(self._load_s1_ts_concat(row, mode="desc"))

        image = torch.cat(channels, dim=0)

        return {
            "image":    {"flairhub": image},
            "target":   label,
            "metadata": {"patch_id": patch_id},
        }

    # ────────────────────────────────────────────────────────────
    # Per-modality output (new)
    # ────────────────────────────────────────────────────────────

    def _getitem_per_modality(self, row, label, patch_id):
        """
        Return modalities as separate tensors keyed by branch name.

        Optical (VHR or SPOT) goes under "optical" — same key whichever
        sensor is enabled, so the per-modality model has one branch
        slot for either source.

        Satellite modalities keep their T dimension explicit and stay at
        native 10×10 spatial size. The per-modality model upsamples
        internally, which avoids carrying upsampled tensors across the
        DataLoader worker boundary (saves shared memory).
        """
        image = {}

        if self.use_vhr:
            image["optical"] = self._load_vhr(row)            # [4, 512, 512]
        elif self.use_spot:
            image["optical"] = self._load_spot(row)           # [4, 512, 512]

        if self.use_dem:
            image["dem"] = self._load_dem(row)                # [2, 512, 512]

        if self.use_s2:
            image["s2"] = self._load_s2_ts_5d(row)            # [T, 10, 10, 10]

        if self.use_s1:
            image["s1"] = self._load_s1_ts_5d(row)            # [T,  4, 10, 10]

        return {
            "image":    image,
            "target":   label,
            "metadata": {"patch_id": patch_id},
        }

    # ────────────────────────────────────────────────────────────
    # Mono-temporal loaders (used by both modes)
    # ────────────────────────────────────────────────────────────

    def _load_vhr(self, row) -> torch.Tensor:
        path = self._resolve_path(row["AERIAL_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = torch.from_numpy(img)
        img = self._normalize(img, "aerial")
        img = torch.clamp(img, -10, 10)
        img = torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)
        return img                                              # [4, 512, 512]

    def _load_spot(self, row) -> torch.Tensor:
        path = self._resolve_path(row["SPOT_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = torch.from_numpy(img)
        # Default: SPOT-specific normalization. Diagnostic mode: VHR
        # ("aerial") statistics, to test whether cross-sensor degradation
        # is driven by pixel-value distribution shift.
        norm_key = "aerial" if self.spot_norm_as_vhr else "spot"
        img = self._normalize(img, norm_key)
        img = torch.clamp(img, -10, 10)
        img = torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)
        return self._upsample(img)                              # [4, 512, 512]

    def _load_dem(self, row) -> torch.Tensor:
        path = self._resolve_path(row["DEM_ELEV"])
        with rasterio.open(path) as src:
            dem = src.read().astype(np.float32)
        dem = torch.from_numpy(dem)
        dem = self._normalize_dem(dem)
        dem = torch.clamp(dem, -10, 10)
        dem = torch.nan_to_num(dem, nan=0.0, posinf=10.0, neginf=-10.0)
        return dem                                              # [2, 512, 512]

    # ────────────────────────────────────────────────────────────
    # Satellite loaders — split into "raw" + "format-specific" wrappers.
    # ────────────────────────────────────────────────────────────

    def _load_s2_raw(self, row) -> torch.Tensor:
        """Load + sample + normalize S2: returns [T_sel, 10, 10, 10] tensor."""
        path = self._resolve_path(row["SENTINEL2_TS"])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)
        T_total = stack.shape[0] // self.NUM_S2_BANDS
        stack = stack.reshape(T_total, self.NUM_S2_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]
        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, "s2")
        stack = torch.clamp(stack, -10, 10)
        return stack                                            # [T_sel, 10, 10, 10]

    def _load_s1_raw_one(self, row, mode: str) -> torch.Tensor:
        """Load + sample + normalize S1 ASC or DESC: [T_sel, 2, 10, 10]."""
        if mode == "asc":
            col, norm_key = "SENTINEL1-ASC_TS", "s1_asc"
        else:
            col, norm_key = "SENTINEL1-DESC_TS", "s1_des"
        path = self._resolve_path(row[col])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)
        T_total = stack.shape[0] // self.NUM_S1_BANDS
        stack = stack.reshape(T_total, self.NUM_S1_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]
        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, norm_key)
        stack = torch.clamp(stack, -10, 10)
        return stack                                            # [T_sel, 2, 10, 10]

    # ── Concat-mode satellite loaders (existing behavior) ────────

    def _load_s2_ts_concat(self, row) -> torch.Tensor:
        """[10*T, 512, 512] — flat channels, upsampled."""
        stack = self._load_s2_raw(row)                          # [T, 10, 10, 10]
        T_sel = stack.shape[0]
        stack = stack.reshape(T_sel * self.NUM_S2_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        return self._upsample(stack)

    def _load_s1_ts_concat(self, row, mode: str) -> torch.Tensor:
        """[2*T, 512, 512] — flat channels, upsampled."""
        stack = self._load_s1_raw_one(row, mode)                # [T, 2, 10, 10]
        T_sel = stack.shape[0]
        stack = stack.reshape(T_sel * self.NUM_S1_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)
        return self._upsample(stack)

    # ── Per-modality satellite loaders (5D, native 10×10) ───────

    def _load_s2_ts_5d(self, row) -> torch.Tensor:
        """[T, 10, 10, 10] — preserves T dim, native spatial size."""
        return self._load_s2_raw(row)

    def _load_s1_ts_5d(self, row) -> torch.Tensor:
        """ASC + DESC fused along channel dim: [T, 4, 10, 10]."""
        asc  = self._load_s1_raw_one(row, mode="asc")           # [T, 2, 10, 10]
        desc = self._load_s1_raw_one(row, mode="desc")          # [T, 2, 10, 10]
        return torch.cat([asc, desc], dim=1)                    # [T, 4, 10, 10]

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
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
        if "dem" in self.norm_stats:
            stats = self.norm_stats["dem"]
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)
            return (dem - mean) / std
        per_image_mean = dem.mean(dim=(1, 2), keepdim=True)
        return (dem - per_image_mean) / 50.0