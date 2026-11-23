"""
AgriPotential Dataset — agricultural-potential segmentation (HDF5 release), SKIP variant.
Zenodo: https://zenodo.org/records/17787322  (dataset.zip -> dataset.h5)

File layout (from the authors' construction script, scripts/dataset.py):

    dataset.h5
      {train, val, test}/
        sentinel2/  01_january_2019 ... 10_october_2019, 12_december_2019   (11 keys,
                    no November)   each: [N, 10, 128, 128] float64
        labels/     viticulture | market | field                  each: [N, 128, 128, 5]
                    one-hot over {very low, low, average, high, very high};
                    all-zero vector = unlabelled

    N = 7095 / 914 / 881 (train / val / test). Patches are 128x128 at 5 m
    (Sentinel-2 super-resolved from 10/20 m with CARN), cut with 50 % overlap
    inside 256x256 tiles; the train/val/test split is done on tiles, so there is
    no overlap ACROSS splits.

Things worth knowing for an atomic model:

* Radiometry is NOT absolute reflectance. Each (patch, date) was divided by
  10 000, clipped to its own 98th percentile (over all bands) and divided by it.
  Band ratios inside a (patch, date) survive; absolute levels across patches and
  dates do not. We z-score per band on top (train stats).
* GSD is 5 m for every band (all super-resolved). Resolution 5.0 must be
  registered in TokenBuilder.REFERENCE_SIZES and the lookup table (the training
  script does it; other entry points such as viz scripts must too).
* Only acquisition months are known; each date is placed on the 15th of its month
  (single year 2019, so day-of-year encoding has no aliasing).
* Labels are ordinal (0..4); unlabelled -> 255. One crop is used as the target
  (`crop`), viticulture by default (the ImageCLEF AI4Agri 2026 subtask).

HDF5 is opened lazily per DataLoader worker (an h5py handle must not be shared
across fork); the full set of 11 dates is used by default (`n_frames`).
"""

import os
import random
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import torch

from .utils_dataset_PASTIS import PastisHDDataset

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


DATE_KEYS = [
    "01_january_2019", "02_february_2019", "03_march_2019", "04_april_2019",
    "05_may_2019", "06_june_2019", "07_july_2019", "08_august_2019",
    "09_september_2019", "10_october_2019", "12_december_2019",
]
CROPS = ("viticulture", "market", "field")


def _key_to_yyyymmdd(key: str) -> int:
    month, year = int(key[:2]), int(key.rsplit("_", 1)[1])
    return int(date(year, month, 15).strftime("%Y%m%d"))


class AgriPotentialDataset(PastisHDDataset):

    SAT_RESOLUTION = 5.0           # used by the parent's token/query builders
    NUM_S2_BANDS   = 10
    NUM_CLASSES    = 5
    IGNORE_INDEX   = 255

    CLASS_NAMES = ["very_low", "low", "average", "high", "very_high"]
    TASK_NAME = "pastis_segmentation"   # kept for encoder/head compatibility

    # Band order IN THE H5 FILE. The paper/tutorial list B2..B7, B8, B8A, B11, B12,
    # but per-band statistics (position 3 ~ NIR level, B5 < B6 < B7 < B8A) show the
    # super-resolution output order: 10 m bands first, then 20 m bands.
    # "idx" = position along the h5 band axis; the parent sorts by it to build
    # s2_spectral_indices, so each atom gets the wavelength of its actual band.
    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 0},
        "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 1},
        "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 2},
        "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 3},
        "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 4},
        "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 5},
        "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 6},
        "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 7},
        "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 8},
        "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
    }

    def __init__(
        self,
        root_path: str = "./data/AGRIPOTENTIAL",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
        use_s1: bool = False,
        use_spot: bool = False,
        crop: str = "viticulture",
        n_frames: Optional[int] = None,       # None -> all 11 dates
        h5_name: str = "dataset.h5",
        stats_patch_stride: int = 20,
        stats_pixel_stride: int = 4,
    ):
        if use_s1 or use_spot:
            raise ValueError("AgriPotential is Sentinel-2 only.")
        if crop not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        if not HAS_H5PY:
            raise ImportError("h5py is required for AgriPotential (pip install h5py).")

        self.crop               = crop
        self.h5_path            = os.path.join(root_path, h5_name)
        self.stats_patch_stride = stats_patch_stride
        self.stats_pixel_stride = stats_pixel_stride
        self._h5                = None
        self._n_frames_arg      = n_frames

        super().__init__(
            root_path=root_path, transform=transform, model=model,
            modality_mode=modality_mode, mode=mode,
            dataset_config=dataset_config, config_model=config_model,
            look_up=look_up, use_s1=False, use_spot=False,
        )

        n = len(self.date_keys) if n_frames is None else int(n_frames)
        pos = torch.linspace(0, len(self.date_keys) - 1, n, dtype=torch.long).tolist()
        self.frame_keys = [self.date_keys[p] for p in pos]
        self.frame_dates = [_key_to_yyyymmdd(k) for k in self.frame_keys]

        print(f"[AGRIPOTENTIAL] split={self.split} patches={len(self)} crop={self.crop} "
              f"frames={len(self.frame_keys)} ({self.frame_keys[0]} .. {self.frame_keys[-1]}) "
              f"GSD={self.SAT_RESOLUTION} m")

    # =========================================================================
    # HDF5 HANDLING
    # =========================================================================

    def _group(self) -> str:
        g = self.split_mapping.get(self.split, self.split)
        if g not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{self.split}'")
        return g

    def _file(self):
        if self._h5 is None:                      # lazily, once per worker
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):                       # never pickle an open handle
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # =========================================================================
    # PARENT HOOKS
    # =========================================================================

    def _load_metadata(self):
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"{self.h5_path} not found (unzip dataset.zip there)")
        with h5py.File(self.h5_path, "r") as f:
            g = f[self._group()]
            keys = list(g["sentinel2"].keys())
            missing = [k for k in DATE_KEYS if k not in keys]
            if missing:
                raise KeyError(f"sentinel2 keys missing from {self.h5_path}: {missing}")
            self.date_keys = DATE_KEYS
            self.n_patches = g["labels"][self.crop].shape[0]
            shp = g["sentinel2"][DATE_KEYS[0]].shape
            if tuple(shp[1:]) != (self.NUM_S2_BANDS, 128, 128):
                raise RuntimeError(f"unexpected sentinel2 shape {shp}")
        self.patch_ids = list(range(self.n_patches))
        self.metadata = None

    def __len__(self):
        return self.n_patches

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normstats_atomizer_agripotential.pt")
        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            if stats.get("s2_mean", torch.empty(0)).numel() == self.NUM_S2_BANDS:
                self._print_norm_stats(stats)
                return stats
        stats = self._compute_normalization_stats()
        tmp = f"{norm_file}.tmp{os.getpid()}"
        torch.save(stats, tmp)
        os.replace(tmp, norm_file)
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        """Per-band mean/std on TRAIN, every `stats_patch_stride`-th patch, all dates."""
        ps, px = self.stats_patch_stride, self.stats_pixel_stride
        s = np.zeros(self.NUM_S2_BANDS); sq = np.zeros(self.NUM_S2_BANDS); n = 0
        print(f"[AGRIPOTENTIAL] computing normalisation stats on train "
              f"(patch stride {ps}, pixel stride {px}) ...")
        with h5py.File(self.h5_path, "r") as f:
            for k in tqdm(DATE_KEYS, desc="AgriPotential norm stats"):
                x = f["train"]["sentinel2"][k][::ps, :, ::px, ::px].astype(np.float64)
                x = x.transpose(1, 0, 2, 3).reshape(self.NUM_S2_BANDS, -1)
                s += x.sum(1); sq += (x * x).sum(1); n += x.shape[1]
        mean = s / n
        std = np.sqrt(np.maximum(sq / n - mean ** 2, 0.0))
        return {"s2_mean": torch.tensor(mean, dtype=torch.float32),
                "s2_std":  torch.tensor(std, dtype=torch.float32)}

    def _print_norm_stats(self, stats):
        print(f"[AGRIPOTENTIAL] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[AGRIPOTENTIAL] S2 std:  {stats['s2_std'].numpy()}")

    @property
    def spectral_indices(self) -> torch.Tensor:
        return self.s2_spectral_indices

    # =========================================================================
    # SAMPLE
    # =========================================================================

    def _load_label(self, idx):
        onehot = self._file()[self._group()]["labels"][self.crop][idx]   # [H, W, 5]
        onehot = torch.from_numpy(np.asarray(onehot))
        label = onehot.argmax(-1).long()
        label[onehot.sum(-1) <= 0] = self.IGNORE_INDEX
        return label

    def __getitem__(self, index: int) -> Dict:
        return self._build_sample(index, viz=False)

    def get_viz_sample(self, index: int) -> dict:
        return self._build_sample(index, viz=True)

    def _build_sample(self, index: int, viz: bool) -> Dict:
        g = self._file()[self._group()]["sentinel2"]
        s2 = np.stack([np.asarray(g[k][index], dtype=np.float32) for k in self.frame_keys])
        s2 = torch.nan_to_num(torch.from_numpy(s2), nan=0.0, posinf=0.0, neginf=0.0)
        mean = self.norm_stats["s2_mean"].view(1, -1, 1, 1)
        std = self.norm_stats["s2_std"].view(1, -1, 1, 1).clamp(min=1e-6)
        s2 = ((s2 - mean) / std).clamp(-10, 10)                      # [T, 10, H, W]

        label = self._load_label(index)
        time_idx = self._doy_to_time_indices(self._dates_to_doy(self.frame_dates))

        if self.augment and not viz:
            k, flip = random.randint(0, 3), random.random() > 0.5
            if k:
                s2, label = torch.rot90(s2, k, dims=(-2, -1)), torch.rot90(label, k, dims=(-2, -1))
            if flip:
                s2, label = torch.flip(s2, dims=(-1,)), torch.flip(label, dims=(-1,))
            s2, label = s2.contiguous(), label.contiguous()

        T, _, H, W = s2.shape
        tokens = self._build_temporal_tokens(
            s2, label, time_idx, self.s2_spectral_indices, self.sat_resolution_idx)
        groups = {self.SAT_RESOLUTION: {
            "tokens": tokens,
            "mask":   torch.zeros(tokens.shape[0], dtype=torch.bool),
            "shape":  (H, W),
        }}

        queries = self.token_builder.build_queries(
            label=label, resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=time_idx[0],
        )
        kept_indices = None
        if not viz:
            queries, kept_indices = self.token_builder.subsample_queries(
                queries, max_queries=self.max_queries,
                ignore_index=self.IGNORE_INDEX, prioritize_valid=True,
                return_indices=True,
            )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        query_token_idx, query_token_valid = self._build_query_token_index(
            T, self.NUM_S2_BANDS, 0, 0, H, W, kept_indices=kept_indices)

        sample = {
            "groups": groups,
            "tasks":  {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "label":  label,
            "target_resolution": self.SAT_RESOLUTION,
            "image":  s2[0],
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }
        if viz:
            sample["image_shape"] = (H, W)
            sample["n_queries"] = queries.shape[0]
        return sample
