"""
EuroSAT Dataset for Atomiser (Classification)
================================================

Single-temporal classification dataset in the grouped-token format,
following the same pattern as ForestNetDataset (dummy queries, scalar
label, task="classification").

Source files (geo-bench-1.0/classification_v1.0/m-eurosat/):
    band_stats.json         — per-band mean/std (13 keys + 'label', same
                               schema as Cashew's band_stats.json)
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
                               sample names match HDF5 filenames directly,
                               e.g. "id_12886" -> "id_12886.hdf5"
    1.00x_train_partition.json — limited-label partition file (same
                               PANGAEA-style scheme ForestNet supports).
                               NOT wired up here per current requirements
                               (default_partition.json only) — see
                               ForestNetDataset's _load_train_fraction_partition
                               if this needs adding later.
    {sample_name}.hdf5      — 13 band datasets, [64, 64] int16 each (raw
                               Sentinel-2 digital numbers, NOT pre-
                               normalized float32 like Cashew/MADOS).
                               NO 'label' dataset key and NO separate
                               label_map.json — the class label is a
                               plain Python int embedded in the HDF5's
                               top-level 'pickle' attribute, itself a
                               pickled dict: {'label': <int 0..9>, '01 -
                               Coastal aerosol': {...band metadata...}, ...}.

Label extraction: f.attrs["pickle"] is stored as the STRING repr of a
bytes literal (e.g. "b'\\x80\\x04...'"), not raw bytes — ast.literal_eval
recovers the actual bytes, which then unpickle to the metadata dict. The
pickled band-metadata entries reference geobench.dataset.Sentinel2, which
this codebase does not depend on and should not need to import just to
read a label int — _PermissiveUnpickler substitutes a harmless stub class
for anything it can't import, so unpickling completes regardless. Every
band's [64,64] array itself is read directly via h5py as normal (this
custom unpickling is ONLY for the label attribute).

Band <-> Sentinel-2 code mapping (all 13 optical S2 bands, INCLUDING B10
cirrus — unlike Cashew, which dropped it). Reuses bands_senflood for
wavelength/bandwidth metadata, same as CashewSkipDataset, since these are
the same physical Sentinel-2 bands under GeoBench's descriptive names.

    EuroSAT HDF5 prefix                 -> Sentinel-2 code (bands_senflood key)
    "01 - Coastal aerosol"              -> "B01"
    "02 - Blue"                         -> "B02"
    "03 - Green"                        -> "B03"
    "04 - Red"                          -> "B04"
    "05 - Vegetation Red Edge"          -> "B05"
    "06 - Vegetation Red Edge"          -> "B06"
    "07 - Vegetation Red Edge"          -> "B07"
    "08 - NIR"                          -> "B08"
    "08A - Vegetation Red Edge"         -> "B08A"
    "09 - Water vapour"                 -> "B09"
    "10 - SWIR - Cirrus"                -> "B10"
    "11 - SWIR"                         -> "B11"
    "12 - SWIR"                         -> "B12"

Native size: 64x64 at 10m (standard EuroSAT patch size) — every band
already arrives at the same [64, 64] shape, no per-band upscaling needed
(unlike MADOS's mixed 10/20/60m native resolutions).

Output format (identical shape to ForestNetDataset's):
    {
        "groups": {
            10.0: {
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (13, H, W),
            },
        },
        "queries":           [1, 8],          # dummy — unused for classification
        "queries_mask":      [1],
        "label":             0-d long tensor (class index in [0, 10)),
        "task":              "classification",
        "target_resolution": 10.0,
        "image":             [13, H, W],
    }

Augmentations (training only):
    D4 group: 4 rotations x 2 flips = 8 transforms (image-only, same as
    ForestNetDataset — classification has no spatial label to keep in
    sync).
"""

import ast
import io
import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .token_grouping import *
from .token_builder import TokenBuilder


class _PermissiveUnpickler(pickle.Unpickler):
    """
    Substitutes a harmless stub class for anything it can't import (e.g.
    geobench.dataset.Sentinel2, which this codebase doesn't depend on),
    so the pickle can still be fully reconstructed. We only ever read the
    plain-int 'label' field out of the result — the band-metadata entries
    (which are what actually reference geobench classes) are read from
    them but never touched.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {})


class EurosatDataset(Dataset):
    """EuroSAT (geo-bench m-eurosat) classification dataset for Atomiser."""

    SENTINEL2_RESOLUTION = 10.0
    NUM_BANDS = 13
    NUM_CLASSES = 10
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE_NATIVE = 64
    TASK_NAME = "classification"

    # HDF5 key prefix -> Sentinel-2 band code (key into bands_senflood).
    # Order here defines channel order in the output image tensor.
    BAND_PREFIX_TO_S2_CODE = {
        "01 - Coastal aerosol":       "B01",
        "02 - Blue":                  "B02",
        "03 - Green":                 "B03",
        "04 - Red":                   "B04",
        "05 - Vegetation Red Edge":   "B05",
        "06 - Vegetation Red Edge":   "B06",
        "07 - Vegetation Red Edge":   "B07",
        "08 - NIR":                   "B08",
        "08A - Vegetation Red Edge":  "B08A",
        "09 - Water vapour":          "B09",
        "10 - SWIR - Cirrus":         "B10",
        "11 - SWIR":                  "B11",
        "12 - SWIR":                  "B12",
    }
    BAND_PREFIXES = list(BAND_PREFIX_TO_S2_CODE.keys())

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-eurosat",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        crop_size: int = PATCH_SIZE_NATIVE,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        assert crop_size <= self.PATCH_SIZE_NATIVE

        self.root_path     = root_path
        self.split          = mode
        self.crop_size       = crop_size
        self.look_up          = look_up
        self.config_model     = config_model
        self.dataset_config   = dataset_config

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens = config_model["trainer"]["max_tokens"]

        # ── Load JSON metadata ──────────────────────────────
        # Per instructions: default_partition.json only, the
        # 1.00x_train_partition.json limited-label scheme is not wired up
        # here.
        with open(os.path.join(root_path, "default_partition.json")) as f:
            default_partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            self.band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(default_partition[split_key])

        # ── Normalization tensors ───────────────────────────
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in self.band_stats:
                raise KeyError(
                    f"[EuroSAT] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[prefix]["mean"])
            stds.append(self.band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        # Reuse bands_senflood (Sentinel-2 wavelength/bandwidth by B0X
        # code) rather than duplicating values — same trick as
        # CashewSkipDataset, but EuroSAT keeps B10 (cirrus), which
        # Cashew dropped.
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.SENTINEL2_RESOLUTION)

        print(f"[EuroSAT] task={self.TASK_NAME}, split={mode} -> "
              f"{len(self.sample_names)} samples")
        print(f"[EuroSAT] bands ({self.NUM_BANDS}): "
              f"{[self.BAND_PREFIX_TO_S2_CODE[p] for p in self.BAND_PREFIXES]}")
        print(f"[EuroSAT] center crop: {crop_size}x{crop_size}")
        print(f"[EuroSAT] resolution idx: {self.resolution_idx}")
        print(f"[EuroSAT] D4 augment: {'ON' if mode == 'train' else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # LABEL EXTRACTION (pickle-embedded, see module docstring)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_label(h5_file: h5py.File) -> int:
        raw = h5_file.attrs["pickle"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        raw_bytes = ast.literal_eval(raw)
        meta = _PermissiveUnpickler(io.BytesIO(raw_bytes)).load()
        return int(meta["label"])

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION (image-only — classification has no spatial label)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    @staticmethod
    def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = image.shape
        if H == size and W == size:
            return image
        top  = (H - size) // 2
        left = (W - size) // 2
        return image[:, top:top + size, left:left + size]

    # ─────────────────────────────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_sample(self, name):
        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(
                        f"[EuroSAT] No key with prefix '{prefix}' in {path}"
                    )
                # Raw digital numbers are int16 -- cast to float32 before
                # any arithmetic (normalization below).
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            label = self._extract_label(f)

        image = torch.from_numpy(np.stack(bands, axis=0))
        image = self._center_crop(image, self.crop_size)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        image, cls_idx = self._load_sample(name)

        if self.split == "train":
            image = self._d4_augment(image)

        C, H, W = image.shape

        # Classification doesn't use per-pixel labels -- dummy
        # IGNORE_INDEX label, same pattern as ForestNetDataset.
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.SENTINEL2_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        dummy_query      = torch.zeros(1, 8, dtype=image_tokens.dtype)
        dummy_query_mask = torch.zeros(1, dtype=torch.float32)

        attention_mask = torch.zeros(image_tokens.shape[0])

        return {
            "groups": {
                self.SENTINEL2_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           dummy_query,
            "queries_mask":      dummy_query_mask,
            "label":             torch.tensor(cls_idx, dtype=torch.long),
            "task":              self.TASK_NAME,
            "target_resolution": self.SENTINEL2_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        name = self.sample_names[index]
        image, cls_idx = self._load_sample(name)

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.SENTINEL2_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        attention_mask   = torch.zeros(image_tokens.shape[0])
        dummy_query      = torch.zeros(1, 8, dtype=image_tokens.dtype)
        dummy_query_mask = torch.zeros(1, dtype=torch.float32)

        return {
            "groups": {
                self.SENTINEL2_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           dummy_query,
            "queries_mask":      dummy_query_mask,
            "label":             torch.tensor(cls_idx, dtype=torch.long),
            "task":              self.TASK_NAME,
            "target_resolution": self.SENTINEL2_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        bw_list, wl_list, names = [], [], []
        for prefix in self.BAND_PREFIXES:
            code = self.BAND_PREFIX_TO_S2_CODE[prefix]
            if code not in self.bands_info:
                raise KeyError(
                    f"[EuroSAT] Sentinel-2 code '{code}' (for band "
                    f"'{prefix}') not found in dataset_config['bands_senflood']. "
                    f"Available: {list(self.bands_info.keys())}"
                )
            data = self.bands_info[code]
            if not ("bandwidth" in data and "central_wavelength" in data):
                raise KeyError(
                    f"[EuroSAT] bands_senflood['{code}'] missing "
                    f"bandwidth/central_wavelength: {data}"
                )
            bw_list.append(int(data["bandwidth"]))
            wl_list.append(int(data["central_wavelength"]))
            names.append(code)

        bw = torch.tensor(bw_list, dtype=torch.float32)
        wl = torch.tensor(wl_list, dtype=torch.float32)

        print(f"[EuroSAT] Band order:")
        for prefix, code, b, w in zip(self.BAND_PREFIXES, names, bw_list, wl_list):
            print(f"  {prefix:28s} -> {code:5s}  bw={b:4d}, wl={w:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[EuroSAT] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)
