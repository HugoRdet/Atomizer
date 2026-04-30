"""
EuroSAT Classification Dataset for Atomiser
=============================================

10-class land cover classification on geo-bench m-eurosat.
13 Sentinel-2 bands at 10m resolution, 64×64 patches.

Source files (geo-bench-1.0/classification_v1.0/m-eurosat/):
    band_stats.json         — per-band mean/std/percentiles
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
    {sample_id}.hdf5        — 13 bands, [64, 64] int16 each + label in attrs['pickle']

Classes (EuroSAT standard, 0-indexed):
    0=AnnualCrop, 1=Forest, 2=HerbaceousVegetation, 3=Highway, 4=Industrial,
    5=Pasture, 6=PermanentCrop, 7=Residential, 8=River, 9=SeaLake

Output format (compatible with Model_ForestNet — classification):
    {
        "groups": {
            10.0: {
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (13, 64, 64),
            },
        },
        "queries":           [1, 8]   (single CLS query at center)
        "queries_mask":      [1],
        "label":             scalar long (0..9),
        "task":              "classification",
        "target_resolution": 10.0,
        "image":             [13, 64, 64],
    }

Augmentations (training only): D4 group on image (label is scalar).
"""

import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .token_grouping import *
from .token_builder import TokenBuilder


def _decode_pickle_attr(attr_value):
    """
    Decode the pickle attribute robustly across h5py versions.

    h5py may return the attribute as: bytes, np.bytes_, np.void,
    np.ndarray of bytes, or (most surprising) a Python str containing
    the LITERAL repr of a bytes object — e.g. the string "b'\\x80\\x04...'"
    rather than the bytes b"\\x80\\x04...".

    Try multiple decoding paths in order. On total failure, raises
    RuntimeError with the first 60 chars of the input for diagnosis.
    """
    import codecs

    errors = []

    # Path 1: direct pickle.loads (handles bytes, np.bytes_)
    try:
        return pickle.loads(attr_value)
    except Exception as e:
        errors.append(f"direct: {type(e).__name__}: {e}")

    # Path 2: .tobytes() — handles np.void, np.ndarray
    if hasattr(attr_value, "tobytes"):
        try:
            return pickle.loads(attr_value.tobytes())
        except Exception as e:
            errors.append(f"tobytes: {type(e).__name__}: {e}")

    # Path 3: ndarray of bytes; pick first element
    if isinstance(attr_value, np.ndarray) and attr_value.size > 0:
        elem = attr_value.flat[0]
        try:
            return pickle.loads(elem)
        except Exception as e:
            errors.append(f"ndarray[0]: {type(e).__name__}: {e}")

    # Path 4: str that's the printed repr of a bytes literal
    # e.g. "b'\\x80\\x04...'" — common geo-bench format
    if isinstance(attr_value, str):
        # Strip b' / b" prefix and trailing quote
        if attr_value.startswith("b'") and attr_value.endswith("'"):
            inner = attr_value[2:-1]
        elif attr_value.startswith('b"') and attr_value.endswith('"'):
            inner = attr_value[2:-1]
        else:
            inner = None

        if inner is not None:
            # Use codecs.escape_decode — more permissive than ast.literal_eval.
            # Decodes \x.., \n, \t, etc. — bytes escapes — without requiring
            # full Python-literal syntax.
            try:
                inner_bytes = inner.encode("latin-1")
                raw, _ = codecs.escape_decode(inner_bytes)
                return pickle.loads(raw)
            except Exception as e:
                errors.append(f"escape_decode: {type(e).__name__}: {e}")

        # Path 5: latin-1 round-trip (last resort)
        try:
            return pickle.loads(attr_value.encode("latin-1"))
        except Exception as e:
            errors.append(f"latin-1: {type(e).__name__}: {e}")

    raise RuntimeError(
        f"Could not decode HDF5 pickle attribute. "
        f"Type: {type(attr_value).__name__}, "
        f"length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}, "
        f"first 60: {repr(attr_value[:60]) if hasattr(attr_value, '__getitem__') else '?'}\n"
        f"Errors: {errors}"
    )


class EuroSATDataset(Dataset):
    """EuroSAT (geo-bench m-eurosat) 10-class classification dataset for Atomiser."""

    OPTICAL_RESOLUTION = 10.0
    NUM_BANDS = 13
    NUM_CLASSES = 10
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE = 64
    TASK_NAME = "classification"

    # Exact HDF5 keys (13 S2 bands; geo-bench keeps all including B10 Cirrus)
    BAND_KEYS = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "08 - NIR",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
        "01 - Coastal aerosol",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
    ]

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
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path     = root_path
        self.split         = mode
        self.look_up       = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens                  = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction  = config_model["trainer"]["max_tokens_reconstruction"]

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            self.band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Build name → label mapping (decode pickle attrs once, cache) ──
        cache_path = os.path.join(root_path, f"_label_cache_{split_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.name_to_label = {k: int(v) for k, v in json.load(f).items()}
        else:
            print(f"[EuroSAT] Building label cache for split '{mode}' "
                  f"({len(self.sample_names)} files)...")
            self.name_to_label = {}
            for name in self.sample_names:
                path = os.path.join(root_path, f"{name}.hdf5")
                with h5py.File(path, "r") as f:
                    meta = _decode_pickle_attr(f.attrs["pickle"])
                    self.name_to_label[name] = int(meta["label"])
            try:
                with open(cache_path, "w") as f:
                    json.dump(self.name_to_label, f)
                print(f"[EuroSAT] Saved label cache to {cache_path}")
            except Exception as e:
                print(f"[EuroSAT] WARN: could not save label cache: {e}")

        # ── Build per-band normalization tensors ──────────────
        means, stds = [], []
        for key in self.BAND_KEYS:
            if key not in self.band_stats:
                raise KeyError(
                    f"[EuroSAT] Band '{key}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[key]["mean"])
            stds.append(self.band_stats[key]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        self.bands_info = dataset_config["bands_eurosat"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)

        # ── Class distribution ──────────────────────────────
        from collections import Counter
        label_counts = Counter(self.name_to_label[n] for n in self.sample_names)

        print(f"[EuroSAT] task={self.TASK_NAME}, split={mode} → "
              f"{len(self.sample_names)} samples")
        print(f"[EuroSAT] bands ({self.NUM_BANDS}): all S2 (including B01/B09/B10)")
        print(f"[EuroSAT] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"[EuroSAT] num_classes: {self.NUM_CLASSES}")
        print(f"[EuroSAT] D4 augment: {'ON' if mode == 'train' else 'OFF'}")
        print(f"[EuroSAT] class distribution: "
              f"{dict(sorted(label_counts.items()))}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor):
        """Image-only D4 (label is scalar for classification)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        # ── Load 13 bands (int16 → float32) ────────────────
        bands = []
        with h5py.File(path, "r") as f:
            for key in self.BAND_KEYS:
                if key not in f:
                    raise KeyError(
                        f"[EuroSAT] Key '{key}' not in {path}. "
                        f"Available: {list(f.keys())}"
                    )
                bands.append(np.asarray(f[key], dtype=np.float32))

        image = torch.from_numpy(np.stack(bands, axis=0))   # [13, 64, 64]
        label = torch.tensor(self.name_to_label[name], dtype=torch.long)

        # ── NaN cleanup, normalize ──────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        # ── D4 augmentation (training only) ─────────────────
        if self.split == "train":
            image = self._d4_augment(image)

        C, H, W = image.shape

        # ── Build per-pixel-per-band tokens ─────────────────
        # token_builder needs a per-pixel label tensor; for classification
        # we use IGNORE — the actual class is carried in the query.
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.OPTICAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Build classification query (single CLS token at center) ─
        first_spectral_idx = int(self.spectral_indices[0].item())
        query = torch.tensor([[
            0.0,                  # value (unused for query)
            (W - 1) / 2.0,        # x — center
            (H - 1) / 2.0,        # y — center
            first_spectral_idx,   # spectral_idx (placeholder)
            int(label.item()),    # label (scalar class)
            0,                    # query_idx
            self.resolution_idx,  # resolution_idx
            self.TIME_IDX_NA,     # time_idx
        ]], dtype=torch.float32)

        # ── Subsample image tokens if too many ──────────────
        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(query.shape[0])

        return {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           query,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE (no augmentation)
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        name = self.sample_names[index]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            for key in self.BAND_KEYS:
                bands.append(np.asarray(f[key], dtype=np.float32))

        image = torch.from_numpy(np.stack(bands, axis=0))
        label = torch.tensor(self.name_to_label[name], dtype=torch.long)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.OPTICAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = int(self.spectral_indices[0].item())
        query = torch.tensor([[
            0.0, (W - 1) / 2.0, (H - 1) / 2.0,
            first_spectral_idx, int(label.item()), 0,
            self.resolution_idx, self.TIME_IDX_NA,
        ]], dtype=torch.float32)

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(query.shape[0])

        return {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           query,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             image,
        }

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw    = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[EuroSAT] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:25s} → "
                  f"bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}")

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