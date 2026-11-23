"""
PLANTEUR Dataset — tropical crop-type segmentation (PASTIS file format), SKIP variant.
https://huggingface.co/datasets/allenai/planteur-benchmark

Subclasses PastisHDDataset and reuses its token building, closed-form skip
gather index (query_token_idx / query_token_valid), date handling and band
lookup. Everything that differs from PASTIS-HD is overridden here:

  PASTIS-HD                          PLANTEUR
  ---------------------------------  -------------------------------------------
  DATA_S1A/S1A_<id>.npy, 3 bands dB  DATA_S1/S1_<id>.npy, 2 bands (VV, VH) LINEAR
                                     backscatter, 0 = no observation, both orbits
  dates-S1A                          dates-S1 (+ orbits-S1: "A" | "D" | "AD")
  folds 1,2,3 / 4 / 5                Fold 1 / 4 / 5   (+ leave-one-island-out)
  20 classes, 19 -> void             8 classes (0..7), 255 = void
  ~1 agricultural year               2018-09-01 .. 2019-12-31 (16 months!)
  T >= 38 for every patch            T varies 40..150 per patch and sensor
  SPOT 1 m                           none (Landsat exists but is not loaded here)

Key choices (all configurable):

* temporal_window: the time encoding is day-of-year (365-day period), so
  Sep-Dec 2018 and Sep-Dec 2019 would collide. Default "2019" keeps the
  calendar year 2019, which every patch covers through December. "agri" =
  2018-09-01..2019-08-31 (some patches only start in 2019-09, they fall back
  to their full series). "all" = no filtering (DOY aliasing, not advised).

* Always exactly `multi_temporal` frames per sensor (indices repeat when a
  window holds fewer observations). This keeps a fixed token layout across
  samples, which the U-Net pre-flight and the closed-form skip index need.
  Unlike the PASTIS loader, frames are read through mmap, so only the
  selected dates are loaded from disk.

* S1 is converted to [VV_dB, VH_dB, VV_dB - VH_dB], i.e. the same 3 channels
  (and lookup keys) as the PASTIS-R convention used for PASTIS. No-data
  pixels are set to 0 after normalisation; with mask_s1_nodata=True they are
  additionally flagged in the group mask (ASSUMES mask True = ignore, as for
  padding — check this against collate_multitask / the encoder).

* Normalisation stats are computed on the TRAIN patches of the chosen split
  protocol (S2 raw DN, S1 in dB, zeros/no-data excluded), cached per protocol,
  and used by every split. (The parent falls back to identity stats on
  val/test if the cache is missing; that cannot happen here.)
"""

import os
import re
import json
import random
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import torch

from .utils_dataset_PASTIS import PastisHDDataset

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


TEMPORAL_WINDOWS = {
    "2019": (20190101, 20191231),
    "agri": (20180901, 20190831),
    "all":  (0, 99999999),
}

ISLAND_ALIASES = {
    "guadeloupe": "Guadeloupe",
    "martinique": "Martinique",
    "guyane":     "Guyane",
    "reunion":    "Réunion",
    "réunion":    "Réunion",
    "mayotte":    "Mayotte",
}


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


class PlanteurDataset(PastisHDDataset):

    NUM_S2_BANDS = 10
    NUM_S1_BANDS = 3          # VV_dB, VH_dB, VV_dB - VH_dB (PASTIS lookup keys)
    NUM_CLASSES  = 8
    IGNORE_INDEX = 255

    CLASS_NAMES = [
        "background", "meadow", "fruits_veg_flowers", "orchard",
        "sugarcane", "banana", "pineapple", "tropical_tuber",
    ]

    # Kept identical to PASTIS so any encoder/head keyed on the task name works
    # unchanged (the trainer only takes the first task anyway).
    TASK_NAME = "pastis_segmentation"

    def __init__(
        self,
        root_path: str = "./data/PLANTEUR",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
        use_s1: bool = True,
        use_spot: bool = False,
        split_protocol: str = "default",     # "default" | "loio:<island>"
        temporal_window: str = "2019",       # "2019" | "agri" | "all"
        s1_orbit: str = "all",               # "all" | "A" | "D"
        mask_s1_nodata: bool = False,
        balanced_labels: bool = False,       # ANNOTATIONS_BALANCED for train
        stats_stride: int = 4,               # spatial stride for norm stats
    ):
        if use_spot:
            raise ValueError("PLANTEUR has no SPOT imagery.")
        if temporal_window not in TEMPORAL_WINDOWS:
            raise ValueError(f"temporal_window must be one of {list(TEMPORAL_WINDOWS)}")
        if s1_orbit not in ("all", "A", "D"):
            raise ValueError("s1_orbit must be 'all', 'A' or 'D'")

        # must exist before the parent __init__ calls the overridden hooks
        self.split_protocol  = split_protocol
        self.temporal_window = temporal_window
        self.s1_orbit        = s1_orbit
        self.mask_s1_nodata  = mask_s1_nodata
        self.balanced_labels = balanced_labels
        self.stats_stride    = stats_stride

        super().__init__(
            root_path=root_path, transform=transform, model=model,
            modality_mode=modality_mode, mode=mode,
            dataset_config=dataset_config, config_model=config_model,
            look_up=look_up, use_s1=use_s1, use_spot=False,
        )

        print(f"[PLANTEUR] protocol={self.split_protocol} split={self.split} "
              f"patches={len(self.patch_ids)} window={self.temporal_window} "
              f"frames/sensor={self.multi_temporal} S1={'on' if use_s1 else 'off'} "
              f"orbit={self.s1_orbit} mask_s1_nodata={self.mask_s1_nodata}")

    # =========================================================================
    # SPLITS / METADATA
    # =========================================================================

    def _split_name(self) -> str:
        s = self.split_mapping.get(self.split, self.split)
        if s not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{self.split}'")
        return s

    def _resolve_split_ids(self, meta) -> Dict[str, List[int]]:
        proto = self.split_protocol
        if proto == "default":
            folds = {"train": 1, "val": 4, "test": 5}
            fold_col = meta["Fold"].astype(int)
            return {s: meta.loc[fold_col == f, "ID_PATCH"].astype(int).tolist()
                    for s, f in folds.items()}

        if proto.startswith("loio:"):
            island = proto.split(":", 1)[1]
            island = ISLAND_ALIASES.get(island.lower(), island)
            path = os.path.join(self.root_path, "splits", "loio.json")
            with open(path, encoding="utf-8") as f:
                folds = json.load(f)["folds"]
            if island not in folds:
                raise ValueError(f"LOIO island '{island}' not in {list(folds)}")
            return {s: [int(i) for i in folds[island][s]]
                    for s in ("train", "val", "test")}

        raise ValueError(f"Unknown split_protocol '{proto}' "
                         "(use 'default' or 'loio:<island>')")

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required for PLANTEUR.")
        path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(path):
            raise FileNotFoundError(f"metadata.geojson not found at {path}")

        meta = gpd.read_file(path)
        meta["ID_PATCH"] = meta["ID_PATCH"].astype(int)

        ids = self._resolve_split_ids(meta)
        self._train_ids = ids["train"]                       # for norm stats
        wanted = set(ids[self._split_name()])

        self.metadata = (meta[meta["ID_PATCH"].isin(wanted)]
                         .sort_values("ID_PATCH").reset_index(drop=True))
        self.patch_ids = self.metadata["ID_PATCH"].tolist()
        if len(self.patch_ids) != len(wanted):
            print(f"[PLANTEUR] WARNING: {len(wanted) - len(self.patch_ids)} split ids "
                  f"missing from metadata.geojson")

    # =========================================================================
    # DATES / FRAME SELECTION
    # =========================================================================

    @staticmethod
    def _series(row, key) -> list:
        """{step: value} metadata field -> list ordered by step."""
        if key not in row:
            raise KeyError(f"metadata.geojson has no '{key}' field")
        d = row[key]
        if isinstance(d, str):
            d = json.loads(d)
        return [v for _, v in sorted(d.items(), key=lambda kv: int(kv[0]))]

    def _select_frames(self, dates: List[int], keep: Optional[List[bool]] = None):
        """Exactly `multi_temporal` evenly spaced frames inside the window.

        Falls back to the whole (orbit-filtered) series when the window is
        empty; repeats indices when fewer than `multi_temporal` are available.
        """
        lo, hi = TEMPORAL_WINDOWS[self.temporal_window]
        ok = [keep is None or keep[i] for i in range(len(dates))]
        cand = [i for i, d in enumerate(dates) if ok[i] and lo <= d <= hi]
        if not cand:
            cand = [i for i in range(len(dates)) if ok[i]] or list(range(len(dates)))
        pos = torch.linspace(0, len(cand) - 1, self.multi_temporal,
                             dtype=torch.long).tolist()   # same rule as PASTIS
        idx = [cand[p] for p in pos]
        return idx, [dates[i] for i in idx]

    @staticmethod
    def _load_frames(path: str, idx: List[int], n_dates: int) -> torch.Tensor:
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] != n_dates:
            raise RuntimeError(f"{path}: {arr.shape[0]} frames but {n_dates} dates")
        return torch.from_numpy(np.asarray(arr[np.asarray(idx)], dtype=np.float32))

    # =========================================================================
    # SENTINEL-1: linear -> dB
    # =========================================================================

    @staticmethod
    def _s1_to_db(s1_lin: torch.Tensor):
        """[T, 2, H, W] linear (VV, VH) -> [T, 3, H, W] dB and no-data mask [T, H, W]."""
        vv, vh = s1_lin[:, 0], s1_lin[:, 1]
        nodata = ~(torch.isfinite(vv) & torch.isfinite(vh) & (vv > 0) & (vh > 0))
        vv = torch.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-6)
        vh = torch.nan_to_num(vh, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-6)
        vv_db = 10.0 * torch.log10(vv)
        vh_db = 10.0 * torch.log10(vh)
        return torch.stack([vv_db, vh_db, vv_db - vh_db], dim=1), nodata

    # =========================================================================
    # NORMALISATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(
            self.root_path, f"normstats_atomizer_{_slug(self.split_protocol)}.pt")
        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            if (stats.get("s2_mean", torch.empty(0)).numel() == self.NUM_S2_BANDS
                    and stats.get("s1_mean", torch.empty(0)).numel() == self.NUM_S1_BANDS):
                self._print_norm_stats(stats)
                return stats
            print(f"[PLANTEUR] stale stats file {norm_file}, recomputing")

        print(f"[PLANTEUR] computing normalisation stats on {len(self._train_ids)} "
              f"train patches (stride {self.stats_stride}) ...")
        stats = self._compute_normalization_stats()
        tmp = f"{norm_file}.tmp{os.getpid()}"
        torch.save(stats, tmp)
        os.replace(tmp, norm_file)                       # atomic under DDP
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        st = self.stats_stride
        acc = {k: [np.zeros(c), np.zeros(c), np.zeros(c)]
               for k, c in (("s2", self.NUM_S2_BANDS), ("s1", self.NUM_S1_BANDS))}

        def add(key, x, m):          # x: [C, N] float64, m: [C, N] bool
            s, sq, n = acc[key]
            xm = np.where(m, x, 0.0)
            s += xm.sum(1); sq += (xm * xm).sum(1); n += m.sum(1)

        for pid in tqdm(self._train_ids, desc="PLANTEUR norm stats"):
            s2 = np.load(os.path.join(self.root_path, "DATA_S2", f"S2_{pid}.npy"),
                         mmap_mode="r")[:, :, ::st, ::st].astype(np.float64)
            x = s2.transpose(1, 0, 2, 3).reshape(self.NUM_S2_BANDS, -1)
            add("s2", x, np.isfinite(x) & (x != 0))

            s1 = np.load(os.path.join(self.root_path, "DATA_S1", f"S1_{pid}.npy"),
                         mmap_mode="r")[:, :, ::st, ::st].astype(np.float64)
            vv, vh = s1[:, 0].ravel(), s1[:, 1].ravel()
            ok = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
            vv_db = 10 * np.log10(vv[ok]); vh_db = 10 * np.log10(vh[ok])
            x = np.stack([vv_db, vh_db, vv_db - vh_db])
            add("s1", x, np.ones_like(x, dtype=bool))

        stats = {}
        for k, (s, sq, n) in acc.items():
            n = np.maximum(n, 1)
            mean = s / n
            std = np.sqrt(np.maximum(sq / n - mean ** 2, 0.0))
            stats[f"{k}_mean"] = torch.tensor(mean, dtype=torch.float32)
            stats[f"{k}_std"] = torch.tensor(std, dtype=torch.float32)
        return stats

    def _normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean = self.norm_stats[f"{key}_mean"].view(1, -1, 1, 1)
        std = self.norm_stats[f"{key}_std"].view(1, -1, 1, 1).clamp(min=1e-6)
        return ((x - mean) / std).clamp(-10, 10)

    # =========================================================================
    # LABELS
    # =========================================================================

    def _load_label(self, patch_id):
        folder = ("ANNOTATIONS_BALANCED"
                  if self.balanced_labels and self._split_name() == "train"
                  else "ANNOTATIONS")
        lab = np.load(os.path.join(self.root_path, folder, f"TARGET_{patch_id}.npy"))
        label = torch.from_numpy(lab[0].astype(np.int64))
        label[(label < 0) | (label >= self.NUM_CLASSES)] = self.IGNORE_INDEX
        return label

    # =========================================================================
    # SAMPLE
    # =========================================================================

    def __getitem__(self, index: int) -> Dict:
        return self._build_sample(index, viz=False)

    def get_viz_sample(self, index: int) -> dict:
        return self._build_sample(index, viz=True)

    def _build_sample(self, index: int, viz: bool) -> Dict:
        row = self.metadata.iloc[index]
        pid = int(row["ID_PATCH"])

        # ── S2 ───────────────────────────────────────────────────────
        s2_dates_all = [int(d) for d in self._series(row, "dates-S2")]
        s2_idx, s2_dates = self._select_frames(s2_dates_all)
        s2 = self._load_frames(os.path.join(self.root_path, "DATA_S2", f"S2_{pid}.npy"),
                               s2_idx, len(s2_dates_all))
        s2 = self._normalize(torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0), "s2")
        s2_time = self._doy_to_time_indices(self._dates_to_doy(s2_dates))

        # ── S1 ───────────────────────────────────────────────────────
        if self.use_s1:
            s1_dates_all = [int(d) for d in self._series(row, "dates-S1")]
            keep = None
            if self.s1_orbit != "all":
                keep = [self.s1_orbit in str(o) for o in self._series(row, "orbits-S1")]
            s1_idx, s1_dates = self._select_frames(s1_dates_all, keep)
            s1_lin = self._load_frames(os.path.join(self.root_path, "DATA_S1", f"S1_{pid}.npy"),
                                       s1_idx, len(s1_dates_all))
            s1, s1_nodata = self._s1_to_db(s1_lin)
            s1 = self._normalize(s1, "s1").masked_fill(s1_nodata.unsqueeze(1), 0.0)
            s1_time = self._doy_to_time_indices(self._dates_to_doy(s1_dates))

        label = self._load_label(pid)

        # ── D4 augmentation ──────────────────────────────────────────
        if self.augment and not viz:
            k, flip = random.randint(0, 3), random.random() > 0.5

            def d4(x):
                if k:
                    x = torch.rot90(x, k, dims=(-2, -1))
                if flip:
                    x = torch.flip(x, dims=(-1,))
                return x.contiguous()

            s2, label = d4(s2), d4(label)
            if self.use_s1:
                s1, s1_nodata = d4(s1), d4(s1_nodata)

        # ── Tokens (layout identical to PASTIS: S2 frames, then S1 frames) ──
        _, _, H, W = s2.shape
        s2_tokens = self._build_temporal_tokens(
            s2, label, s2_time, self.s2_spectral_indices, self.sat_resolution_idx)
        Ts2 = s2.shape[0]
        sat_mask = torch.zeros(s2_tokens.shape[0], dtype=torch.bool)

        if self.use_s1:
            s1_tokens = self._build_temporal_tokens(
                s1, label, s1_time, self.s1_spectral_indices, self.sat_resolution_idx)
            Ts1 = s1.shape[0]
            sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)
            if self.mask_s1_nodata:
                # S1 block is frame-major then (c h w): matches build_tokens flatten
                m1 = s1_nodata.unsqueeze(1).expand(Ts1, self.NUM_S1_BANDS, H, W).reshape(-1)
            else:
                m1 = torch.zeros(s1_tokens.shape[0], dtype=torch.bool)
            sat_mask = torch.cat([sat_mask, m1])
        else:
            Ts1, sat_tokens = 0, s2_tokens

        groups = {self.SAT_RESOLUTION: {"tokens": sat_tokens, "mask": sat_mask,
                                        "shape": (H, W)}}

        # ── Queries + skip gather index ──────────────────────────────
        queries = self.token_builder.build_queries(
            label=label, resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=s2_time[0],
        )
        kept_indices = None
        if not viz:
            queries, kept_indices = self.token_builder.subsample_queries(
                queries, max_queries=self.max_queries,
                ignore_index=self.IGNORE_INDEX, prioritize_valid=True,
                return_indices=True,
            )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        C1 = self.NUM_S1_BANDS if self.use_s1 else 0
        query_token_idx, query_token_valid = self._build_query_token_index(
            Ts2, self.NUM_S2_BANDS, Ts1, C1, H, W, kept_indices=kept_indices)

        image = torch.cat([s2[0], s1[0]], dim=0) if self.use_s1 else s2[0]

        sample = {
            "groups": groups,
            "tasks":  {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "label":  label,
            "target_resolution": self.SAT_RESOLUTION,
            "image":  image,
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }
        if viz:
            sample["image_shape"] = (H, W)
            sample["n_queries"] = queries.shape[0]
        return sample
