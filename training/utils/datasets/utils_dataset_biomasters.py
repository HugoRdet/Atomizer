"""
BioMassters Dataset — Multi-temporal S1+S2 fusion, AGB regression (SKIP variant)

Mirrors PastisHDDataset's flat multi-temporal token layout (frame-major concat,
S2 block then S1 block, single 10m resolution group) and Sen1Floods11's
zero+mask primitive, adapted for:

  1. FIXED T timesteps per sensor (config: dataset.num_timesteps, default 12),
     SAME for both S1 and S2, for uniform tensor shapes across chips/batches.
     A chip's actual present-month count varies per chip/sensor though, so:
       - if present > T: evenly-spaced subsample (PASTIS's `_sample_temporal`)
       - if present < T: PAD BY REPLICATION -- cycle through the chip's own
         valid frames to fill the remaining slots (never zero-fill), and mark
         the replicated slots with mask=1.0 so cross-attention ignores them.
         This keeps every token a real observed value (better for norm stats
         / feature stability than zero-padding) while giving every chip the
         same fixed T for simple batching.
     Replicated frames carry the ORIGINAL month's time_idx (they're literal
     copies of that month's data), so temporal metadata stays accurate even
     though the slot is padding.

  2. Continuous AGB regression instead of segmentation classes. The target is
     carried in the label slot (col 4) exactly like PASTIS's build_queries
     call, just with a float target instead of a class id — this already
     works today via the "reconstruction" mode in Sen1Floods11SkipDataset,
     where queries[:, 4] holds a continuous value. TASK_NAME follows PASTIS's
     `tasks: {name: {queries, queries_mask}}` convention.

  3. Non-optical band registration for S1 (VV_asc, VH_asc, VV_desc, VH_desc)
     and S2's CLP (cloud probability, not a physical reflectance band):
     registered as pseudo-spectral entries with negative wavelength/bandwidth
     placeholders in `look_up.table_wave`, exactly like PASTIS's
     S1_BANDS_INFO (VV/VH/VV_VH -> idx -1/-2/-3).

  4. MODALITY/BAND DROPPING (NEW), mirroring Sen1Floods11SkipDataset's exact
     mechanism, adapted for the multi-temporal case:
       - STATIC eval-time drop (config: trainer.bands.drop -> band names),
         applied at EVERY split via _apply_drop_mask -- this is what the
         modality-drop ablation script uses (--drop VV_asc,VH_asc,... etc,
         same convention as the baselines' script_test_..._modality_drop.py).
       - STOCHASTIC training-time band-dropout augmentation (config:
         trainer.band_dropout_augmentation), gated to split=="train", giving
         the model exposure to missing bands during training -- matches the
         intent of BioMasstersBaselineDataset's _band_dropout_augment.
       - KEY DIFFERENCE from Sen1Floods11 (single-frame): a dropped band is
         masked across ALL T timesteps for both S2 and S1 blocks, not a
         single frame -- consistent with the baselines' "a missing sensor
         stays missing for the whole time series" convention. Since sat_mask
         is already keyed per-token (not per-frame), this falls out of the
         SAME per-token spectral_idx match used by Sen1Floods11 -- a token's
         spectral_idx doesn't encode which timestep it's from, so matching
         on spectral_idx alone already masks that band in every frame.

POOL LAYOUT (matches PASTIS's sat_tokens convention):
    sat_tokens = cat([ S2_f0(c h w), ..., S2_f(T-1),  S1_f0(c h w), ..., S1_f(T-1) ])
  - within a frame: channel-major (c h w)->row => pixel p={p + c*HW} per sub-block
  - frames FRAME-MAJOR; S2 block precedes S1 block
  - T is FIXED (same for S2 and S1, same across all chips) -- real frames come
    first in chronological order, any replicated (padding) frames follow, per
    the `_pad_or_subsample` construction below.

  pixel p atoms (relative to sat_tokens, the ONLY group for BioMassters):
    S2: { t*C2*HW + c*HW + p : t<T, c<C2 }
    S1: (T*C2*HW) + { t*C1*HW + c*HW + p : t<T, c<C1 }
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .token_builder import TokenBuilder

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


class BioMasstersSkipDataset(Dataset):
    """
    BioMassters Dataset — flat multi-temporal S1+S2 fusion, AGB regression, SKIP variant.

    Token format (same 8 columns as Sen1Floods11/PASTIS):
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2        3          4        5            6            7

    `label` (col 4) holds the continuous AGB value (Mg/ha) for query tokens,
    following the same convention as Sen1Floods11's reconstruction mode.
    """

    SAT_RESOLUTION = 10.0
    N_MONTHS       = 12  # chip-relative acquisition slots 0..11

    NUM_S2_BANDS = 10  # 10 physical bands only -- CLP removed (matches PANGAEA's band set)
    NUM_S1_BANDS = 4   # VV_asc, VH_asc, VV_desc, VH_desc (all pseudo-spectral)

    IGNORE_VALUE = -1.0   # AGB (Mg/ha) is never negative; safe sentinel for invalid pixels
    TIME_IDX_NA  = -1

    NODATA_S1 = -9999.0  # official BioMassters S1 nodata sentinel

    ALL_BAND_NAMES = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        "VV_asc", "VH_asc", "VV_desc", "VH_desc",
    ]
    S1_BAND_NAMES = {"VV_asc", "VH_asc", "VV_desc", "VH_desc"}

    # Physical S2 bands, same wavelengths as Atomizer-IO's S2_BANDS.
    # CLP (cloud probability, band index 10 in the raw tif) is deliberately
    # excluded -- matches PANGAEA's band set for BioMassters, and sliced off
    # at read time in _load_sensor since the source tifs ship it baked in
    # as the 11th channel.
    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 0},
        "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 1},
        "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 2},
        "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 3},
        "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 4},
        "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 5},
        "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 6},
        "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 7},
        "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 8},
        "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
    }

    # SAR bands: all pseudo-spectral. Each of the 4 asc/desc VV/VH combinations
    # is a DISTINCT channel (unlike PASTIS's generic VV/VH, which doesn't
    # distinguish orbit direction) -- these MUST match
    # Lookup_encoding.ABSTRACT_CHANNELS["VV_ASC"/"VH_ASC"/"VV_DESC"/"VH_DESC"].
    S1_BANDS_INFO = {
        "VV_asc":  {"central_wavelength": -41, "bandwidth": -41, "idx": 0},
        "VH_asc":  {"central_wavelength": -42, "bandwidth": -42, "idx": 1},
        "VV_desc": {"central_wavelength": -43, "bandwidth": -43, "idx": 2},
        "VH_desc": {"central_wavelength": -44, "bandwidth": -44, "idx": 3},
    }

    TASK_NAME = "biomassters_regression"

    def __init__(
        self,
        root_path: str = "./data/biomassters",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path    = Path(root_path)
        self.split        = mode
        self.look_up      = look_up
        self.config_model = config_model
        self.augment      = (mode == "train")

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens   = config_model["trainer"]["max_tokens"]
        self.max_queries = config_model["trainer"].get("max_tokens_reconstruction", 100_000)

        self.fixed_T = config_model.get("dataset", {}).get("num_timesteps", self.N_MONTHS)

        self.split_mapping = {"train": "train", "validation": "train", "test": "test"}
        mapped_split = self.split_mapping.get(mode, mode)

        self.features_dir = self.root_path / f"{mapped_split}_features"
        self.agbm_dir      = self.root_path / f"{mapped_split}_agbm"

        self._file_index_features = None
        self._file_index_agbm     = None

        manifest_path = self.root_path / f"manifest_{mapped_split}.json"
        with open(manifest_path) as f:
            self.records: List[dict] = json.load(f)

        if mode in ("train", "validation"):
            self.records = self._carve_val_split(self.records, mode, val_fraction=0.1, seed=42)

        self._setup_band_indices()

        # ── Static eval-time band drop (trainer.bands.drop) ─────────────
        # Applied at EVERY split. This is what the modality-drop ablation
        # script drives (e.g. --drop VV_asc,VH_asc,VV_desc,VH_desc for
        # "S2 only"). Mirrors Sen1Floods11SkipDataset's dropped_spectral_indices.
        bands_cfg = config_model["trainer"].get("bands", {}) or {}
        drop_names = bands_cfg.get("drop", None)
        self.drop_band_names = set(drop_names) if drop_names else set()
        self.dropped_spectral_indices = self._resolve_drop_indices()
        if self.drop_band_names:
            print(f"[BioMassters-SKIP] Static bands dropped (all splits): "
                  f"{sorted(self.drop_band_names)}")

        # ── Stochastic training-time band-dropout augmentation ──────────
        # Matches BioMasstersBaselineDataset's _band_dropout_augment intent,
        # via the SAME zero-value+mask=1.0 primitive as the static drop above
        # (so a stochastic drop and a config-driven drop are indistinguishable
        # to the model). Config-gated, train-only.
        aug_cfg = config_model["trainer"].get("band_dropout_augmentation", {}) or {}
        self.band_dropout_enabled = bool(aug_cfg.get("enabled", True)) and (mode == "train")
        self.p_dropout_applied = float(aug_cfg.get("p_dropout_applied", 0.5))
        self.p_whole_modality  = float(aug_cfg.get("p_whole_modality", 0.5))
        self.p_band_drop       = float(aug_cfg.get("p_band_drop", 0.15))

        self.sat_resolution_idx = self.look_up.get_resolution_idx(self.SAT_RESOLUTION)

        self.norm_stats = self._load_or_compute_normalization()

        print(f"[BioMassters-SKIP] {len(self.records)} chips, split='{self.split}'")
        print(f"[BioMassters-SKIP] S2: {self.NUM_S2_BANDS} bands (CLP excluded) @ {self.SAT_RESOLUTION}m")
        print(f"[BioMassters-SKIP] S1: {self.NUM_S1_BANDS} bands (VV/VH asc+desc) @ {self.SAT_RESOLUTION}m")
        print(f"[BioMassters-SKIP] Fixed T={self.fixed_T} timesteps/sensor "
              f"(pad-by-replication + mask when a chip has fewer present months)")
        print(f"[BioMassters-SKIP] D4 augmentations: {'ON' if self.augment else 'OFF'}")
        if self.band_dropout_enabled:
            print(f"[BioMassters-SKIP] Band-dropout augmentation: ON "
                  f"(p_applied={self.p_dropout_applied}, "
                  f"p_whole_modality={self.p_whole_modality}, "
                  f"p_band_drop={self.p_band_drop}, applied across ALL T "
                  f"timesteps per drop, matching the baselines' convention)")
        else:
            print(f"[BioMassters-SKIP] Band-dropout augmentation: OFF")

    @staticmethod
    def _carve_val_split(records: List[dict], mode: str, val_fraction: float = 0.1, seed: int = 42):
        chip_ids = sorted(r["chip_id"] for r in records)
        rng = random.Random(seed)
        rng.shuffle(chip_ids)
        n_val = max(1, int(len(chip_ids) * val_fraction))
        val_ids = set(chip_ids[:n_val])

        if mode == "validation":
            keep_ids = val_ids
        else:
            keep_ids = set(chip_ids[n_val:])

        return [r for r in records if r["chip_id"] in keep_ids]

    # =========================================================================
    # BAND DROPPING: static (config, all splits) + stochastic (train only)
    # =========================================================================

    def _resolve_drop_indices(self):
        """Band names in trainer.bands.drop -> their spectral_idx in look_up.table_wave."""
        if not self.drop_band_names:
            return set()

        unknown = self.drop_band_names - set(self.ALL_BAND_NAMES)
        if unknown:
            raise ValueError(
                f"trainer.bands.drop contains unknown band names: {unknown}. "
                f"Valid names: {self.ALL_BAND_NAMES}"
            )

        all_bands_info = {**self.S2_BANDS_INFO, **self.S1_BANDS_INFO}
        dropped = set()
        for name in self.drop_band_names:
            info = all_bands_info[name]
            key = (int(info["bandwidth"]), int(info["central_wavelength"]))
            if key not in self.look_up.table_wave:
                raise KeyError(f"Band '{name}' key={key} not found in lookup table.")
            dropped.add(self.look_up.table_wave[key])
        return dropped

    @staticmethod
    def _zero_and_mask_by_spectral_indices(tokens: torch.Tensor, mask: torch.Tensor,
                                            spectral_indices_to_drop):
        """
        Shared primitive (identical to Sen1Floods11SkipDataset's): zero the
        token value and set mask=1.0 for every token whose spectral index
        (col 3) is in `spectral_indices_to_drop`. Since a token's
        spectral_idx doesn't encode WHICH timestep it came from, matching on
        spectral_idx alone already drops that band across every one of the
        fixed_T frames in one pass -- no separate per-timestep loop needed,
        unlike the baselines' explicit "zero every T slice" tensor indexing.
        """
        if not spectral_indices_to_drop:
            return tokens, mask

        tokens = tokens.clone()
        mask   = mask.clone().float()

        spec_idx = tokens[:, 3]
        drop = torch.zeros(tokens.shape[0], dtype=torch.bool)
        for sid in spectral_indices_to_drop:
            drop |= (spec_idx == sid)

        tokens[drop, 0] = 0.0
        mask[drop]      = 1.0

        return tokens, mask

    def _apply_drop_mask(self, tokens: torch.Tensor, mask: torch.Tensor):
        """Static, config-driven band drop (trainer.bands.drop) -- applied at every split."""
        return self._zero_and_mask_by_spectral_indices(tokens, mask, self.dropped_spectral_indices)

    def _sample_band_dropout_indices(self):
        """
        Per-sample stochastic augmentation, mirroring Sen1Floods11's:
          - with prob (1 - p_dropout_applied): no-op, all bands kept
          - else with prob p_whole_modality: drop ALL S1 or ALL S2 spectral
            indices (mirrors the "S2 only"/"S1 only" eval ablations)
          - else: drop each currently-kept spectral index independently with
            probability p_band_drop

        Returns a set of spectral indices to drop this sample. Layered ON TOP
        of the static trainer.bands.drop config, applied separately in
        __getitem__ -- not a replacement for it.
        """
        if torch.rand(1).item() >= self.p_dropout_applied:
            return set()

        if torch.rand(1).item() < self.p_whole_modality:
            pool = (self.s1_spectral_indices if torch.rand(1).item() < 0.5
                    else self.s2_spectral_indices)
            return set(pool.tolist())
        else:
            n_total = self.s2_spectral_indices.shape[0] + self.s1_spectral_indices.shape[0]
            keep_mask = torch.rand(n_total) < self.p_band_drop
            all_indices = torch.cat([self.s2_spectral_indices, self.s1_spectral_indices])
            return set(all_indices[keep_mask].tolist())

    # =========================================================================
    # >>> SKIP: per-query gather index into own band x month atoms
    # =========================================================================

    @staticmethod
    def _build_full_pixel_index(T, C2, C1, H, W):
        HW = H * W
        p = torch.arange(HW)
        blocks = []

        t2 = torch.arange(T).view(T, 1, 1)
        c2 = torch.arange(C2).view(1, C2, 1)
        s2 = (t2 * C2 * HW + c2 * HW).reshape(-1, 1) + p.view(1, -1)
        blocks.append(s2)

        off = T * C2 * HW
        t1 = torch.arange(T).view(T, 1, 1)
        c1 = torch.arange(C1).view(1, C1, 1)
        s1 = (off + t1 * C1 * HW + c1 * HW).reshape(-1, 1) + p.view(1, -1)
        blocks.append(s1)

        return torch.cat(blocks, dim=0).t().contiguous()

    def _build_query_token_index(self, H, W, kept_indices=None):
        full = self._build_full_pixel_index(self.fixed_T, self.NUM_S2_BANDS, self.NUM_S1_BANDS, H, W)
        idx = full if kept_indices is None else full[kept_indices]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        rec = self.records[index]

        s2_data_raw, s2_months_raw = self._load_sensor(rec["s2_files"], self.NUM_S2_BANDS, is_s1=False)
        s1_data_raw, s1_months_raw = self._load_sensor(rec["s1_files"], self.NUM_S1_BANDS, is_s1=True)

        s2_data, s2_time_indices, s2_replicated = self._pad_or_subsample(s2_data_raw, s2_months_raw)
        s1_data, s1_time_indices, s1_replicated = self._pad_or_subsample(s1_data_raw, s1_months_raw)

        agbm_arr = self._load_agbm(rec)
        H, W = s2_data.shape[-2], s2_data.shape[-1]

        s2_data, s1_data = self._normalize(s2_data, s1_data)
        s2_data = torch.clamp(s2_data, -10, 10)
        s1_data = torch.clamp(s1_data, -10, 10)

        if self.augment:
            d4_k    = random.randint(0, 3)
            d4_flip = random.random() > 0.5
            if d4_k > 0:
                s2_data = torch.rot90(s2_data, d4_k, dims=(-2, -1))
                s1_data = torch.rot90(s1_data, d4_k, dims=(-2, -1))
                if agbm_arr is not None:
                    agbm_arr = torch.rot90(agbm_arr, d4_k, dims=(-2, -1))
            if d4_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                s1_data = torch.flip(s1_data, dims=(-1,))
                if agbm_arr is not None:
                    agbm_arr = torch.flip(agbm_arr, dims=(-1,))

        s2_tokens = self._build_temporal_tokens(
            s2_data, s2_time_indices, self.s2_spectral_indices, self.sat_resolution_idx)
        s1_tokens = self._build_temporal_tokens(
            s1_data, s1_time_indices, self.s1_spectral_indices, self.sat_resolution_idx)
        sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)

        sat_mask = self._build_replication_mask(s2_replicated, s1_replicated, H, W)

        # ── Static eval-time band drop (trainer.bands.drop), every split ──
        sat_tokens, sat_mask = self._apply_drop_mask(sat_tokens, sat_mask)

        # ── Stochastic training-time band-dropout augmentation ──────────
        if self.band_dropout_enabled:
            aug_drop_indices = self._sample_band_dropout_indices()
            sat_tokens, sat_mask = self._zero_and_mask_by_spectral_indices(
                sat_tokens, sat_mask, aug_drop_indices
            )

        groups = {
            self.SAT_RESOLUTION: {
                "tokens": sat_tokens,
                "mask":   sat_mask,
                "shape":  (H, W),
            },
        }

        label_for_queries = agbm_arr[0] if agbm_arr is not None else torch.full(
            (H, W), self.IGNORE_VALUE, dtype=torch.float32)
        queries = self.token_builder.build_queries(
            label=label_for_queries,
            resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=s2_time_indices[0],
        )
        queries, kept_indices = self.token_builder.subsample_queries(
            queries, max_queries=self.max_queries,
            ignore_index=self.IGNORE_VALUE, prioritize_valid=True,
            return_indices=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        query_token_idx, query_token_valid = self._build_query_token_index(
            H, W, kept_indices=kept_indices,
        )

        image = torch.cat([s2_data[0], s1_data[0]], dim=0)

        result = {
            "groups": groups,
            "tasks": {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "target_resolution": self.SAT_RESOLUTION,
            "image": image,
            "chip_id": rec["chip_id"],
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }
        if agbm_arr is not None:
            result["label"] = agbm_arr[0]

        return result

    # =========================================================================
    # TEMPORAL FIXING: subsample if too many, pad-by-replication if too few
    # =========================================================================

    def _pad_or_subsample(self, data: torch.Tensor, months: List[int]):
        T_present = data.shape[0]
        T = self.fixed_T

        if T_present == 0:
            C, H, W = data.shape[1], 256, 256
            data_fixed = torch.zeros(T, C, H, W, dtype=torch.float32)
            time_indices = [self.look_up.get_or_register_time_idx(self._month_to_doy(0))] * T
            replicated = [True] * T
            return data_fixed, time_indices, replicated

        if T_present > T:
            data_fixed = data[-T:]
            months_fixed = months[-T:]
            replicated = [False] * T
        elif T_present < T:
            pad_needed = T - T_present
            pad_source_idx = [i % T_present for i in range(pad_needed)]
            data_fixed = torch.cat([data, data[pad_source_idx]], dim=0)
            months_fixed = months + [months[i] for i in pad_source_idx]
            replicated = [False] * T_present + [True] * pad_needed
        else:
            data_fixed = data
            months_fixed = months
            replicated = [False] * T

        time_indices = [self.look_up.get_or_register_time_idx(self._month_to_doy(m)) for m in months_fixed]
        return data_fixed, time_indices, replicated

    def _build_replication_mask(self, s2_replicated: List[bool], s1_replicated: List[bool], H, W):
        HW = H * W
        s2_blocks = [torch.full((self.NUM_S2_BANDS * HW,), 1.0 if r else 0.0) for r in s2_replicated]
        s1_blocks = [torch.full((self.NUM_S1_BANDS * HW,), 1.0 if r else 0.0) for r in s1_replicated]
        return torch.cat(s2_blocks + s1_blocks, dim=0)

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_temporal_tokens(self, data, time_indices, spectral_indices, resolution_idx):
        frames = []
        dummy_label = torch.zeros(data.shape[-2], data.shape[-1])
        for t in range(data.shape[0]):
            frames.append(self.token_builder.build_tokens(
                image=data[t], label=dummy_label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=spectral_indices,
                resolution_idx=resolution_idx,
                time_idx=time_indices[t],
            ))
        return torch.cat(frames, dim=0)

    # =========================================================================
    # MONTH -> DOY
    # =========================================================================

    @staticmethod
    def _month_to_doy(month: int) -> int:
        return int(month) * 30 + 15

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _build_file_index(self, root_dir: Path) -> dict:
        print(f"[BioMassters-SKIP] Direct path lookup failed under {root_dir}; "
              f"building a recursive filename index (one-time cost)...")
        index = {}
        n_dupes = 0
        for p in root_dir.rglob("*.tif"):
            if p.name in index:
                n_dupes += 1
            index[p.name] = p
        print(f"[BioMassters-SKIP] Indexed {len(index)} .tif files under {root_dir}"
              + (f" ({n_dupes} duplicate basenames, kept last seen)" if n_dupes else ""))
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

        raise FileNotFoundError(
            f"Could not find '{fname}' under {root_dir}, either directly or "
            f"via basename '{basename}' in the recursive index "
            f"({len(index)} files indexed). Check that extraction completed "
            f"fully for this split."
        )

    def _load_sensor(self, files_dict: dict, n_bands: int, is_s1: bool):
        H = W = 256
        months_sorted = sorted(int(m) for m in files_dict.keys())
        frames = []
        for month in months_sorted:
            fname = files_dict[str(month)] if str(month) in files_dict else files_dict[month]
            path = self._resolve_path(self.features_dir, fname, "_file_index_features")
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
            if not is_s1 and arr.shape[0] > n_bands:
                arr = arr[:n_bands]
            if is_s1:
                arr = np.where(arr == self.NODATA_S1, 0.0, arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            frames.append(torch.from_numpy(arr))

        if not frames:
            return torch.zeros(0, n_bands, H, W, dtype=torch.float32), []

        return torch.stack(frames, dim=0), months_sorted

    def _load_agbm(self, rec: dict):
        if not rec.get("agbm_file"):
            return None
        path = self._resolve_path(self.agbm_dir, rec["agbm_file"], "_file_index_agbm")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(arr)

    # =========================================================================
    # BAND METADATA / LOOKUP REGISTRATION
    # =========================================================================

    def _setup_band_indices(self):
        self.s2_spectral_indices = []
        for name in sorted(self.S2_BANDS_INFO, key=lambda b: self.S2_BANDS_INFO[b]["idx"]):
            info = self.S2_BANDS_INFO[name]
            key  = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(f"S2 band {name} key={key} not in lookup.")
            self.s2_spectral_indices.append(self.look_up.table_wave[key])
        self.s2_spectral_indices = torch.tensor(self.s2_spectral_indices, dtype=torch.long)

        self.s1_spectral_indices = []
        for name in sorted(self.S1_BANDS_INFO, key=lambda b: self.S1_BANDS_INFO[b]["idx"]):
            info = self.S1_BANDS_INFO[name]
            key  = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"S1 band {name} key={key} not in lookup. "
                    f"Register pseudo-spectral SAR bands the same way PASTIS does "
                    f"before building this dataset."
                )
            self.s1_spectral_indices.append(self.look_up.table_wave[key])
        self.s1_spectral_indices = torch.tensor(self.s1_spectral_indices, dtype=torch.long)

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = self.root_path / "normalization_stats.pt"

        if norm_file.exists():
            stats = torch.load(norm_file, weights_only=True)
            stale = (
                stats["s1_mean"].shape[0] != self.NUM_S1_BANDS
                or stats["s2_mean"].shape[0] != self.NUM_S2_BANDS
                or "agb_mean" not in stats
            )
            if stale:
                print(f"[BioMassters-SKIP] Cached stats at {norm_file} are stale "
                      f"(shape mismatch or missing agb_mean/std) -- recomputing.")
                norm_file.unlink()
            else:
                self._print_norm_stats(stats)
                return stats

        if self.split != "train":
            print(f"[BioMassters-SKIP] WARNING: no normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(self.NUM_S2_BANDS), "s2_std": torch.ones(self.NUM_S2_BANDS),
                "s1_mean": torch.zeros(self.NUM_S1_BANDS), "s1_std": torch.ones(self.NUM_S1_BANDS),
                "agb_mean": torch.tensor(0.0), "agb_std": torch.tensor(1.0),
            }

        print(f"[BioMassters-SKIP] Computing normalization from {len(self.records)} chips...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self, max_chips: int = 200):
        s2_sum = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_sq  = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_n   = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s1_sum = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_sq  = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_n   = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        agb_sum = torch.zeros(1, dtype=torch.float64)
        agb_sq  = torch.zeros(1, dtype=torch.float64)
        agb_n   = torch.zeros(1, dtype=torch.float64)

        for rec in tqdm(self.records[:max_chips], desc="Computing normalization stats"):
            for month_str, fname in rec["s2_files"].items():
                path = self._resolve_path(self.features_dir, fname, "_file_index_features")
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float64)
                if arr.shape[0] > self.NUM_S2_BANDS:
                    arr = arr[:self.NUM_S2_BANDS]
                arr = np.nan_to_num(arr)
                for c in range(self.NUM_S2_BANDS):
                    v = arr[c].flatten()
                    if len(v):
                        s2_sum[c] += v.sum(); s2_sq[c] += (v ** 2).sum(); s2_n[c] += len(v)
            for month_str, fname in rec["s1_files"].items():
                path = self._resolve_path(self.features_dir, fname, "_file_index_features")
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float64)
                arr = np.where(arr == self.NODATA_S1, np.nan, arr)
                for c in range(self.NUM_S1_BANDS):
                    v = arr[c].flatten()
                    v = v[~np.isnan(v)]
                    if len(v):
                        s1_sum[c] += v.sum(); s1_sq[c] += (v ** 2).sum(); s1_n[c] += len(v)

            if rec.get("agbm_file"):
                agb_path = self._resolve_path(self.agbm_dir, rec["agbm_file"], "_file_index_agbm")
                with rasterio.open(agb_path) as src:
                    agb_arr = src.read().astype(np.float64)
                agb_arr = np.nan_to_num(agb_arr, nan=0.0, posinf=0.0, neginf=0.0)
                agb_flat = agb_arr.flatten()
                agb_sum[0] += agb_flat.sum()
                agb_sq[0]  += (agb_flat ** 2).sum()
                agb_n[0]   += len(agb_flat)

        def _stats(s, sq, n):
            mean = (s / n.clamp(min=1)).float()
            std  = ((sq / n.clamp(min=1) - mean.double() ** 2).clamp(min=0).sqrt()).float()
            return mean, std

        s2_mean, s2_std = _stats(s2_sum, s2_sq, s2_n)
        s1_mean, s1_std = _stats(s1_sum, s1_sq, s1_n)
        agb_mean, agb_std = _stats(agb_sum, agb_sq, agb_n)

        return {
            "s2_mean": s2_mean, "s2_std": s2_std,
            "s1_mean": s1_mean, "s1_std": s1_std,
            "agb_mean": agb_mean.squeeze(),
            "agb_std":  agb_std.squeeze(),
        }

    def _print_norm_stats(self, stats):
        print(f"[BioMassters-SKIP] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[BioMassters-SKIP] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[BioMassters-SKIP] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[BioMassters-SKIP] S1 std:  {stats['s1_std'].numpy()}")
        if "agb_mean" in stats:
            print(f"[BioMassters-SKIP] AGB mean: {stats['agb_mean'].item():.4f}")
            print(f"[BioMassters-SKIP] AGB std:  {stats['agb_std'].item():.4f}")
        else:
            print(f"[BioMassters-SKIP] WARNING: no agb_mean/std in cached stats "
                  f"(computed before target normalization was added) -- delete "
                  f"normalization_stats.pt and recompute, or the trainer's target "
                  f"normalization will have nothing to load.")

    def _normalize(self, s2, s1):
        s2 = (s2 - self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)) \
             / self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1).clamp(min=1e-6)
        s1 = (s1 - self.norm_stats["s1_mean"].view(1, self.NUM_S1_BANDS, 1, 1)) \
             / self.norm_stats["s1_std"].view(1, self.NUM_S1_BANDS, 1, 1).clamp(min=1e-6)
        return s2, s1
