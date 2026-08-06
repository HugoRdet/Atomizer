"""
BioMassters Baseline Dataset
==============================

Fixed-format dataset for baseline models (ResNet+double-conv, ViT+LTAE, RAMEN).
Returns {"image": {"s2": ..., "s1": ...}, "target": [1, H, W], "metadata": {...}}

No tokenization, no metadata encoding, no lookup tables — just normalized
imagery, mirroring PastisBaselineDataset's structure.

FAIRNESS WITH ATOMIZER: this dataset reuses the SAME fixed-T selection
policy as BioMasstersSkipDataset (last-N months when a chip has more than
num_timesteps present, pad-by-replication when it has fewer -- see
_pad_or_subsample docstring below) and the SAME normalization_stats.pt file,
so baselines see the exact same temporal window and per-band statistics
Atomizer does. If you want baselines to see a genuinely different temporal
policy (e.g. always full 12 months, since some of these architectures --
particularly LTAE -- were designed around full temporal stacks in their
original papers), that's a deliberate experiment design choice, not a bug:
override via multi_temporal / temporal_last at construction time.

Temporal handling (controlled via temporal_mode):
  - "stack":    [C*T, H, W]  — all frames concatenated as channels (ResNet)
  - "sequence": [T, C, H, W] — temporal sequence preserved (ViT+LTAE, RAMEN)

Bands: 10 S2 channels (physical bands only, CLP excluded -- matches PANGAEA's
band set) + 4 S1 channels (VV/VH x asc/desc), matching BioMasstersSkipDataset's
NUM_S2_BANDS/NUM_S1_BANDS exactly.

Directory structure: identical to BioMasstersSkipDataset's (manifest_{split}.json,
{split}_features/, {split}_agbm/) -- this dataset reads the SAME manifests,
so run prepare_biomassters.py once and both datasets share it.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS (must match BioMasstersSkipDataset exactly, for fairness)
# ═══════════════════════════════════════════════════════════════════════

N_MONTHS = 12
NODATA_S1 = -9999.0

NUM_S2_BANDS = 10   # physical bands only -- CLP excluded (matches PANGAEA's band set)
NUM_S1_BANDS = 4    # VV_asc, VH_asc, VV_desc, VH_desc

IGNORE_VALUE = -1.0  # AGB (Mg/ha) is never negative


# ═══════════════════════════════════════════════════════════════════════
# BASELINE DATASET
# ═══════════════════════════════════════════════════════════════════════

class BioMasstersBaselineDataset(Dataset):
    """
    BioMassters dataset for baseline regression models.

    Returns:
        {
            "image": {
                "s2": [C*T, H, W] or [T, C, H, W],
                "s1": [C*T, H, W] or [T, C, H, W],
            },
            "target": [1, H, W] float (AGB, Mg/ha),
            "metadata": {
                "sensors": list[str],
                "n_s2_bands": int,
                "n_s1_bands": int,
                "n_frames": int,
                "temporal_mode": str,
                "chip_id": str,
            },
        }

    Temporal modes:
        "stack":    concatenate all frames as channels → [C*T, H, W]
        "sequence": keep temporal dim → [T, C, H, W]
    """

    def __init__(
        self,
        root_path: str = "./data/biomassters",
        mode: str = "train",
        multi_temporal: int = 3,
        temporal_last: bool = True,
        temporal_mode: str = "stack",
        augment: bool = True,
        val_fraction: float = 0.1,
        val_seed: int = 42,
        band_dropout: bool = True,
        p_dropout_applied: float = 0.5,
        p_whole_modality: float = 0.5,
        p_band_drop: float = 0.15,
    ):
        super().__init__()

        self.root_path = Path(root_path)
        self.split = mode
        self.multi_temporal = multi_temporal
        self.temporal_last = temporal_last
        self.temporal_mode = temporal_mode
        self.augment = augment and (mode == "train")
        self.band_dropout = band_dropout and (mode == "train")
        self.p_dropout_applied = p_dropout_applied
        self.p_whole_modality = p_whole_modality
        self.p_band_drop = p_band_drop

        self.split_mapping = {"train": "train", "validation": "train", "test": "test"}
        mapped_split = self.split_mapping.get(mode, mode)

        self.features_dir = self.root_path / f"{mapped_split}_features"
        self.agbm_dir      = self.root_path / f"{mapped_split}_agbm"

        self._file_index_features = None
        self._file_index_agbm     = None

        manifest_path = self.root_path / f"manifest_{mapped_split}.json"
        with open(manifest_path) as f:
            self.records: List[dict] = json.load(f)

        # Same deterministic val carving as BioMasstersSkipDataset -- shares
        # the identical held-out chip set (same seed/fraction) so Atomizer
        # and baselines are evaluated on the exact same validation chips.
        if mode in ("train", "validation"):
            self.records = self._carve_val_split(self.records, mode, val_fraction, val_seed)

        # ── Normalization stats (SAME file as BioMasstersSkipDataset) ────
        self.norm_stats = self._load_normalization()

        print(f"[BioMassters-BL] {len(self.records)} chips, split='{self.split}'")
        print(f"[BioMassters-BL] S2: {NUM_S2_BANDS} bands (CLP excluded), "
              f"S1: {NUM_S1_BANDS} bands (VV/VH asc+desc)")
        print(f"[BioMassters-BL] Temporal: {self.multi_temporal} frames "
              f"({'last-N' if self.temporal_last else 'uniform'}), mode={self.temporal_mode}")
        print(f"[BioMassters-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        if self.band_dropout:
            print(f"[BioMassters-BL] Band dropout: ON "
                  f"(p_applied={self.p_dropout_applied}, "
                  f"p_whole_modality={self.p_whole_modality}, "
                  f"p_band_drop={self.p_band_drop}, applied consistently "
                  f"across all T timesteps -- a dropped sensor/band is "
                  f"missing for the whole time series, not per-frame)")
        else:
            print(f"[BioMassters-BL] Band dropout: OFF")

    # ═════════════════════════════════════════════════════════════════
    # VAL SPLIT (identical logic to BioMasstersSkipDataset)
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _carve_val_split(records: List[dict], mode: str, val_fraction: float, seed: int):
        chip_ids = sorted(r["chip_id"] for r in records)
        rng = random.Random(seed)
        rng.shuffle(chip_ids)
        n_val = max(1, int(len(chip_ids) * val_fraction))
        val_ids = set(chip_ids[:n_val])
        keep_ids = val_ids if mode == "validation" else set(chip_ids[n_val:])
        return [r for r in records if r["chip_id"] in keep_ids]

    # ═════════════════════════════════════════════════════════════════
    # NORMALIZATION (shared file with BioMasstersSkipDataset)
    # ═════════════════════════════════════════════════════════════════

    def _load_normalization(self):
        norm_file = self.root_path / "normalization_stats.pt"
        if norm_file.exists():
            stats = torch.load(norm_file, weights_only=True)
            if "agb_mean" not in stats:
                print(f"[BioMassters-BL] WARNING: {norm_file} predates target "
                      f"normalization (missing agb_mean/std). Re-run "
                      f"BioMasstersSkipDataset(mode='train', ...) once to "
                      f"recompute it (it auto-invalidates stale cached stats).")
            else:
                print(f"[BioMassters-BL] Loading normalization stats from {norm_file}")
                return stats

        print(f"[BioMassters-BL] WARNING: No usable normalization file at {norm_file}. "
              f"Run BioMasstersSkipDataset(mode='train', ...) once first to "
              f"compute it, or this baseline will use identity normalization "
              f"(including for the AGB target -- NOT the z-score transform "
              f"the trainer expects, so this fallback is a real correctness risk, "
              f"not just a cosmetic warning).")
        return {
            "s2_mean": torch.zeros(NUM_S2_BANDS), "s2_std": torch.ones(NUM_S2_BANDS),
            "s1_mean": torch.zeros(NUM_S1_BANDS), "s1_std": torch.ones(NUM_S1_BANDS),
            "agb_mean": torch.tensor(0.0), "agb_std": torch.tensor(1.0),
        }

    def _normalize(self, data: torch.Tensor, mean, std, n_bands) -> torch.Tensor:
        """data is [T, C, H, W] or [C, H, W]."""
        std = std.clamp(min=1e-6)
        if data.dim() == 4:
            mean = mean.view(1, n_bands, 1, 1)
            std = std.view(1, n_bands, 1, 1)
        else:
            mean = mean.view(n_bands, 1, 1)
            std = std.view(n_bands, 1, 1)
        return (data - mean) / std

    # ═════════════════════════════════════════════════════════════════
    # FILE RESOLUTION (same nested-extraction resilience as the skip dataset)
    # ═════════════════════════════════════════════════════════════════

    def _build_file_index(self, root_dir: Path) -> dict:
        index = {}
        for p in root_dir.rglob("*.tif"):
            index[p.name] = p
        return index

    def _resolve_path(self, root_dir: Path, fname: str, index_attr: str) -> Path:
        direct = root_dir / fname
        if direct.exists():
            return direct
        index = getattr(self, index_attr)
        if index is None:
            index = self._build_file_index(root_dir)
            setattr(self, index_attr, index)
        basename = Path(fname).name
        if basename in index:
            return index[basename]
        raise FileNotFoundError(f"Could not find '{fname}' under {root_dir}")

    # ═════════════════════════════════════════════════════════════════
    # FILE LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_sensor(self, files_dict: dict, n_bands: int, is_s1: bool):
        """
        Loads only PRESENT months. Returns ([T_present,C,H,W], months sorted asc).
        S2 CLP removal: same as BioMasstersSkipDataset -- raw tifs ship 11 S2
        bands (10 physical + CLP as last channel); sliced off here.
        """
        months_sorted = sorted(int(m) for m in files_dict.keys())
        frames = []
        for month in months_sorted:
            fname = files_dict.get(str(month), files_dict.get(month))
            path = self._resolve_path(self.features_dir, fname, "_file_index_features")
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
            if not is_s1 and arr.shape[0] > n_bands:
                arr = arr[:n_bands]  # drop CLP (assumed last channel)
            if is_s1:
                arr = np.where(arr == NODATA_S1, 0.0, arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            frames.append(torch.from_numpy(arr))
        if not frames:
            return torch.zeros(0, n_bands, 256, 256, dtype=torch.float32), []
        return torch.stack(frames, dim=0), months_sorted

    def _load_agbm(self, rec: dict):
        path = self._resolve_path(self.agbm_dir, rec["agbm_file"], "_file_index_agbm")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # [1, H, W]
        return torch.from_numpy(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))

    # ═════════════════════════════════════════════════════════════════
    # TEMPORAL FIXING (identical policy to BioMasstersSkipDataset._pad_or_subsample)
    # ═════════════════════════════════════════════════════════════════

    def _pad_or_subsample(self, data: torch.Tensor, months: List[int], T: int):
        """
        Fixes [T_present, C, H, W] to exactly T frames, tracking which month
        each final frame corresponds to (needed for day-of-year positional
        encoding, e.g. LTAE):
          - T_present > T : last T months if temporal_last, else evenly-spaced.
          - T_present < T : pad by round-robin replication of present frames
            (never zero-fill), matching BioMasstersSkipDataset exactly.
            Replicated frames keep the ORIGINAL month they copied from.
          - T_present == 0: all-zero fallback (degenerate, shouldn't occur
            given S1's 100% month coverage in this dataset). Uses month 0
            as a dummy doy source.

        Returns (data_fixed [T,C,H,W], months_fixed list[int] len T).
        """
        T_present = data.shape[0]
        if T_present == 0:
            C, H, W = data.shape[1], 256, 256
            return torch.zeros(T, C, H, W, dtype=torch.float32), [0] * T

        if T_present > T:
            if self.temporal_last:
                return data[-T:], months[-T:]
            indices = torch.linspace(0, T_present - 1, T, dtype=torch.long)
            return data[indices], [months[i.item()] for i in indices]
        elif T_present < T:
            pad_needed = T - T_present
            pad_source_idx = [i % T_present for i in range(pad_needed)]
            data_fixed = torch.cat([data, data[pad_source_idx]], dim=0)
            months_fixed = months + [months[i] for i in pad_source_idx]
            return data_fixed, months_fixed
        else:
            return data, months

    @staticmethod
    def _month_to_doy(month: int) -> int:
        """Chip-relative month slot -> a day-of-year proxy, matching
        BioMasstersSkipDataset._month_to_doy exactly."""
        return int(month) * 30 + 15

    # ═════════════════════════════════════════════════════════════════
    # AUGMENTATION: band dropout (train only)
    # ═════════════════════════════════════════════════════════════════

    def _band_dropout_augment(self, s2_data: torch.Tensor, s1_data: torch.Tensor):
        """
        Zero out whole modalities or random individual bands, applied to
        the already-normalized, already-fixed-T [T, C, H, W] tensors.

        Same probability structure as Sen1Floods11BaselineDataset's
        _band_dropout_augment, ONE KEY DIFFERENCE for the multi-temporal
        case: a drop is applied CONSISTENTLY ACROSS ALL T TIMESTEPS, not
        per-frame independently. A missing sensor/band in the real world
        stays missing for the whole observation window (e.g. "no SAR data
        for this chip at all"), not flickering in and out frame-to-frame --
        so the augmentation should mirror that, both for realism and
        because it's the meaningful ablation to match at eval time (e.g.
        an "S1-only"/"S2-only" whole-series eval, not a per-frame one).

        With probability (1 - p_dropout_applied): no-op, sample keeps all
        bands (keeps the full-band regime well-represented in training).

        Otherwise, with probability p_whole_modality: zero either all S2
        or all S1 bands, for every timestep. With probability
        (1 - p_whole_modality): zero each band independently with
        probability p_band_drop, same dropped-band set applied to every
        timestep.
        """
        if torch.rand(1).item() >= self.p_dropout_applied:
            return s2_data, s1_data

        s2_data = s2_data.clone()
        s1_data = s1_data.clone()

        if torch.rand(1).item() < self.p_whole_modality:
            if torch.rand(1).item() < 0.5:
                s1_data[:] = 0.0  # drop S1 entirely, all timesteps
            else:
                s2_data[:] = 0.0  # drop S2 entirely, all timesteps
        else:
            s2_band_mask = torch.rand(NUM_S2_BANDS) < self.p_band_drop
            s1_band_mask = torch.rand(NUM_S1_BANDS) < self.p_band_drop
            if s2_band_mask.any():
                s2_data[:, s2_band_mask] = 0.0  # same bands dropped every timestep
            if s1_band_mask.any():
                s1_data[:, s1_band_mask] = 0.0

        return s2_data, s1_data

    # ═════════════════════════════════════════════════════════════════
    # FORMAT OUTPUT
    # ═════════════════════════════════════════════════════════════════

    def _format_temporal(self, data: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "stack":
            T, C, H, W = data.shape
            return data.reshape(T * C, H, W)
        return data

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]

        s2_data, s2_months = self._load_sensor(rec["s2_files"], NUM_S2_BANDS, is_s1=False)
        s1_data, s1_months = self._load_sensor(rec["s1_files"], NUM_S1_BANDS, is_s1=True)
        agbm = self._load_agbm(rec)  # [1, H, W]

        # ── Normalize ───────────────────────────────────────────────
        s2_data = self._normalize(s2_data, self.norm_stats["s2_mean"],
                                   self.norm_stats["s2_std"], NUM_S2_BANDS)
        s1_data = self._normalize(s1_data, self.norm_stats["s1_mean"],
                                   self.norm_stats["s1_std"], NUM_S1_BANDS)
        s2_data = torch.clamp(s2_data, -10, 10)
        s1_data = torch.clamp(s1_data, -10, 10)

        # ── Fix to multi_temporal frames (same policy as skip dataset) ──
        s2_data, s2_months = self._pad_or_subsample(s2_data, s2_months, self.multi_temporal)
        s1_data, s1_months = self._pad_or_subsample(s1_data, s1_months, self.multi_temporal)

        # ── Band-dropout augmentation (training only, AFTER normalize) ──
        if self.band_dropout:
            s2_data, s1_data = self._band_dropout_augment(s2_data, s1_data)

        # ── D4 augmentation ─────────────────────────────────────────
        if self.augment:
            k = random.randint(0, 3)
            do_flip = random.random() > 0.5
            if k > 0:
                s2_data = torch.rot90(s2_data, k, dims=(-2, -1))
                s1_data = torch.rot90(s1_data, k, dims=(-2, -1))
                agbm = torch.rot90(agbm, k, dims=(-2, -1))
            if do_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                s1_data = torch.flip(s1_data, dims=(-1,))
                agbm = torch.flip(agbm, dims=(-1,))

        # ── Format temporal dimension ───────────────────────────────
        s2_out = self._format_temporal(s2_data)
        s1_out = self._format_temporal(s1_data)

        # ── Day-of-year for temporal positional encoding (LTAE etc.) ──
        s2_doy = torch.tensor([self._month_to_doy(m) for m in s2_months], dtype=torch.long)
        s1_doy = torch.tensor([self._month_to_doy(m) for m in s1_months], dtype=torch.long)

        return {
            "image": {"s2": s2_out, "s1": s1_out},
            "dates": {"s2": s2_doy, "s1": s1_doy},
            "target": agbm,  # [1, H, W] float
            "metadata": {
                "sensors": ["s2", "s1"],
                "n_s2_bands": NUM_S2_BANDS,
                "n_s1_bands": NUM_S1_BANDS,
                "n_frames": self.multi_temporal,
                "temporal_mode": self.temporal_mode,
                "chip_id": rec["chip_id"],
            },
        }
