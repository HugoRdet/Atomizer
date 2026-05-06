"""
EuroSAT (geo-bench m-eurosat) dataset for multi-task baseline training.

Sentinel-2, 13 bands, single-frame, 64x64 native, 10-class classification.

Spectral canonicalization: EuroSAT already provides all 13 S2 bands but
in geo-bench's own ordering (B02, B03, B04, B08, B05, B06, B07, B8A, B11,
B12, B01, B09, B10). We use a precomputed permutation matrix to map them
to the canonical S2 order (B01..B12). No interpolation involved — every
target wavelength has an exact source match.

Structure of the output (matches every other multi-task dataset):
    {
        "image": {"input": [15, 512, 512]},   # 13 canonical S2 + 2 zero SAR
        "target": int,                         # class index in [0, 10)
        "valid_mask": [512, 512],              # uint8, 1 where real (top-left 64x64)
        "original_size": [2],                  # long, (64, 64)
        "metadata": {...},
    }

Splits (geo-bench convention):
    train -> 'train'
    val   -> 'valid'
    test  -> 'test'

Native normalization: per-band z-score using the geo-bench band_stats.json,
then clamped (cls tasks don't need clamp; included for symmetry with the
seg datasets which clamp at +/-10).

D4 augmentation in training: 4 rotations x 2 flips, applied to image only
(label is scalar).

Reference: https://github.com/ServiceNow/geo-bench
"""

import io
import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .multitask_utils import (
    CANONICAL_SIZE,
    NUM_OPTICAL,
    apply_interpolation_matrix,
    build_canonical_image,
    pad_to_canonical,
    S2_BAND_NAMES,
)


# ────────────────────────────────────────────────────────────────────
# Pickled-attribute decoder (lifted from the single-task EuroSAT dataset)
# ────────────────────────────────────────────────────────────────────

def _decode_pickle_attr(attr_value):
    """
    Robust decoder for the geo-bench HDF5 'pickle' attribute. The attribute
    may be bytes, np.bytes_, np.void, np.ndarray of bytes, or a Python str
    containing the literal repr of bytes (e.g. "b'\\x80\\x04...'"). Uses a
    lenient unpickler that stubs out any geo-bench classes that aren't
    importable locally.
    """
    import codecs

    class _StubClass:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

    class _LenientUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError, AttributeError):
                return _StubClass

    raw_bytes = None
    if isinstance(attr_value, (bytes, bytearray)):
        raw_bytes = bytes(attr_value)
    elif hasattr(attr_value, "tobytes"):
        try:
            raw_bytes = attr_value.tobytes()
        except Exception:
            pass
    if raw_bytes is None and isinstance(attr_value, np.ndarray) and attr_value.size > 0:
        elem = attr_value.flat[0]
        if isinstance(elem, (bytes, bytearray)):
            raw_bytes = bytes(elem)
        elif hasattr(elem, "tobytes"):
            try:
                raw_bytes = elem.tobytes()
            except Exception:
                pass
    if raw_bytes is None and isinstance(attr_value, str):
        if attr_value.startswith("b'") and attr_value.endswith("'"):
            inner = attr_value[2:-1]
        elif attr_value.startswith('b"') and attr_value.endswith('"'):
            inner = attr_value[2:-1]
        else:
            inner = None
        if inner is not None:
            try:
                inner_latin = inner.encode("latin-1")
                decoded, _ = codecs.escape_decode(inner_latin)
                raw_bytes = decoded
            except Exception:
                pass
        if raw_bytes is None:
            raw_bytes = attr_value.encode("latin-1")
    if raw_bytes is None:
        raise RuntimeError(
            f"Could not extract bytes from HDF5 'pickle' attribute "
            f"(type={type(attr_value).__name__})."
        )

    return _LenientUnpickler(io.BytesIO(raw_bytes)).load()


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

class EuroSATMTDataset(Dataset):
    """EuroSAT (geo-bench m-eurosat) dataset for multi-task baselines."""

    NUM_CLASSES = 10
    NATIVE_SIZE = 64
    NUM_NATIVE_BANDS = 13

    # Order in which bands are stored under the geo-bench HDF5 keys.
    # Maps position-in-file -> canonical S2 name.
    BAND_KEYS = [
        "02 - Blue",                      # B02
        "03 - Green",                     # B03
        "04 - Red",                       # B04
        "08 - NIR",                       # B08
        "05 - Vegetation Red Edge",       # B05
        "06 - Vegetation Red Edge",       # B06
        "07 - Vegetation Red Edge",       # B07
        "08A - Vegetation Red Edge",      # B8A
        "11 - SWIR",                      # B11
        "12 - SWIR",                      # B12
        "01 - Coastal aerosol",           # B01
        "09 - Water vapour",              # B09
        "10 - SWIR - Cirrus",             # B10
    ]
    # Canonical S2 names, matching BAND_KEYS positionally.
    BAND_KEY_TO_S2 = [
        "B02", "B03", "B04", "B08", "B05", "B06", "B07",
        "B8A", "B11", "B12", "B01", "B09", "B10",
    ]

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/classification_v1.0/m-eurosat",
        mode: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split = mode
        self.augment = augment and (mode == "train")

        # ── Load metadata ────────────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # ── Build / load label cache ────────────────────────
        cache_path = os.path.join(root_path, f"_label_cache_{split_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.name_to_label = {k: int(v) for k, v in json.load(f).items()}
        else:
            print(f"[EuroSAT-MT] Building label cache for split '{mode}'...")
            self.name_to_label = {}
            for name in self.sample_names:
                path = os.path.join(root_path, f"{name}.hdf5")
                with h5py.File(path, "r") as f:
                    meta = _decode_pickle_attr(f.attrs["pickle"])
                    self.name_to_label[name] = int(meta["label"])
            try:
                with open(cache_path, "w") as f:
                    json.dump(self.name_to_label, f)
            except Exception:
                pass

        # ── Native normalization stats (per-band) ─────────────
        means, stds = [], []
        for key in self.BAND_KEYS:
            if key not in band_stats:
                raise KeyError(
                    f"[EuroSAT-MT] Band '{key}' missing from band_stats.json."
                )
            means.append(band_stats[key]["mean"])
            stds.append(band_stats[key]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Build the [13, 13] permutation matrix:
        # canonical row t picks the input column where BAND_KEY_TO_S2 == S2_BAND_NAMES[t].
        M = torch.zeros(NUM_OPTICAL, self.NUM_NATIVE_BANDS, dtype=torch.float32)
        for t_idx, name in enumerate(S2_BAND_NAMES):
            src_idx = self.BAND_KEY_TO_S2.index(name)
            M[t_idx, src_idx] = 1.0
        self.interp_matrix = M

        # ── Summary ──────────────────────────────────────────
        from collections import Counter
        label_counts = Counter(self.name_to_label[n] for n in self.sample_names)
        print(f"[EuroSAT-MT] split={mode} -> {len(self.sample_names)} samples")
        print(f"[EuroSAT-MT] {self.NUM_NATIVE_BANDS} native bands -> "
              f"canonical 13 (permutation), SAR zero-filled")
        print(f"[EuroSAT-MT] {self.NATIVE_SIZE}x{self.NATIVE_SIZE} -> "
              f"padded to {CANONICAL_SIZE}x{CANONICAL_SIZE}")
        print(f"[EuroSAT-MT] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[EuroSAT-MT] class distribution: {dict(sorted(label_counts.items()))}")

    # ─────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor) -> torch.Tensor:
        """D4 group on [C, H, W]. No label to align (scalar target)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    # ─────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        label = self.name_to_label[name]

        # ── Load 13 native bands ─────────────────────────
        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            for key in self.BAND_KEYS:
                bands.append(np.asarray(f[key], dtype=np.float32))
        image = torch.from_numpy(np.stack(bands, axis=0))   # [13, 64, 64]

        # ── Defensive clean ─────────────────────────────
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Native per-band z-score ─────────────────────
        image = (image - self.norm_mean) / self.norm_std
        image = torch.clamp(image, -10, 10)

        # ── D4 augmentation (consistent across channels) ─
        if self.augment:
            image = self._d4_augment(image)

        # ── Spectral canonicalization (permutation) ─────
        optical_canonical = apply_interpolation_matrix(image, self.interp_matrix)
        # [13, 64, 64]

        # ── Concat with zero SAR -> [15, 64, 64] ────────
        canonical = build_canonical_image(optical_canonical, sar=None)

        # ── Spatial padding 64 -> 512 ───────────────────
        canonical, _, valid_mask, original_size = pad_to_canonical(
            canonical, target=None, size=CANONICAL_SIZE,
        )

        return {
            "image": {"input": canonical},          # [15, 512, 512]
            "target": int(label),                    # scalar
            "valid_mask": valid_mask,                # [512, 512]
            "original_size": original_size,          # [2] -> (64, 64)
            "metadata": {
                "sample_name": name,
                "n_native_bands": self.NUM_NATIVE_BANDS,
            },
        }