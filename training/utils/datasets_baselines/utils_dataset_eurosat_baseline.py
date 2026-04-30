"""
EuroSAT Baseline Dataset
==========================

Plain tensor classification dataset for non-Atomiser baselines (ResNet/ViT
classifier) on geo-bench m-eurosat (10-class S2 land cover, 64×64).

Output format (compatible with classification BaselineTrainer):
    {
        "image":  {"s2": [13, H, W]},
        "target": scalar long (0..9),
        "metadata": {...},
    }

Splits: from default_partition.json (train/valid/test → 2000/1000/1000).
Native size: 64×64 (no cropping needed, divides cleanly by 16 for ViT).
Bands: 13 S2 (all bands including B10 Cirrus, geo-bench convention).
"""

import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_pickle_attr(attr_value):
    """
    Decode the pickle attribute robustly across h5py versions.

    h5py may return the attribute as: bytes, np.bytes_, np.void,
    np.ndarray of bytes, or (most surprising) a Python str containing
    the LITERAL repr of a bytes object — e.g. the string "b'\\x80\\x04...'"
    rather than the bytes b"\\x80\\x04...".

    The pickled dict contains 'label' (what we want) but also references
    to geobench classes that may not be installed locally. We use a
    custom Unpickler that returns a stub object for missing classes —
    we don't need to actually instantiate the band metadata, just to
    extract the integer label.
    """
    import codecs
    import io

    class _StubClass:
        """Placeholder for any class pickle can't import."""
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

    def _lenient_loads(raw_bytes):
        return _LenientUnpickler(io.BytesIO(raw_bytes)).load()

    errors = []

    # Convert input to bytes (try several paths)
    raw_bytes = None

    # Path 1: already bytes
    if isinstance(attr_value, (bytes, bytearray)):
        raw_bytes = bytes(attr_value)

    # Path 2: numpy void / array — has tobytes()
    if raw_bytes is None and hasattr(attr_value, "tobytes"):
        try:
            raw_bytes = attr_value.tobytes()
        except Exception as e:
            errors.append(f"tobytes: {type(e).__name__}: {e}")

    # Path 3: ndarray of bytes
    if raw_bytes is None and isinstance(attr_value, np.ndarray) and attr_value.size > 0:
        elem = attr_value.flat[0]
        if isinstance(elem, (bytes, bytearray)):
            raw_bytes = bytes(elem)
        elif hasattr(elem, "tobytes"):
            try:
                raw_bytes = elem.tobytes()
            except Exception as e:
                errors.append(f"ndarray[0].tobytes: {type(e).__name__}: {e}")

    # Path 4: str containing the printed repr of bytes (e.g. "b'\\x80\\x04...'")
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
            except Exception as e:
                errors.append(f"escape_decode: {type(e).__name__}: {e}")

        # Path 4b: str as latin-1 (last resort)
        if raw_bytes is None:
            try:
                raw_bytes = attr_value.encode("latin-1")
            except Exception as e:
                errors.append(f"latin-1: {type(e).__name__}: {e}")

    if raw_bytes is None:
        raise RuntimeError(
            f"Could not extract bytes from HDF5 pickle attribute. "
            f"Type: {type(attr_value).__name__}, "
            f"length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}, "
            f"first 60: {repr(attr_value[:60]) if hasattr(attr_value, '__getitem__') else '?'}\n"
            f"Errors: {errors}"
        )

    # Now unpickle with stub-classes for missing imports.
    try:
        return _lenient_loads(raw_bytes)
    except Exception as e:
        raise RuntimeError(
            f"Could not unpickle HDF5 attribute even with lenient unpickler. "
            f"Bytes length: {len(raw_bytes)}, "
            f"first 30 bytes: {raw_bytes[:30]!r}\n"
            f"Final error: {type(e).__name__}: {e}\n"
            f"Earlier errors: {errors}"
        )


class EuroSATBaselineDataset(Dataset):
    """EuroSAT (m-eurosat) 10-class classification baseline dataset."""

    NUM_CHANNELS = 13
    NUM_CLASSES = 10
    IGNORE_INDEX = 255
    PATCH_SIZE = 64

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
        mode: str = "train",
        crop_size: int = None,
        augment: bool = True,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split     = mode
        self.crop_size = crop_size
        self.augment   = augment and (mode == "train")

        with open(os.path.join(root_path, "default_partition.json")) as f:
            partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(partition[split_key])

        # Build label cache
        cache_path = os.path.join(root_path, f"_label_cache_{split_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.name_to_label = {k: int(v) for k, v in json.load(f).items()}
        else:
            print(f"[EuroSAT-BL] Building label cache for split '{mode}'...")
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

        means, stds = [], []
        for key in self.BAND_KEYS:
            if key not in band_stats:
                raise KeyError(
                    f"[EuroSAT-BL] Band '{key}' not in band_stats.json. "
                    f"Available: {list(band_stats.keys())}"
                )
            means.append(band_stats[key]["mean"])
            stds.append(band_stats[key]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        from collections import Counter
        label_counts = Counter(self.name_to_label[n] for n in self.sample_names)

        print(f"[EuroSAT-BL] split={mode}, samples={len(self.sample_names)}")
        print(f"[EuroSAT-BL] channels: {self.NUM_CHANNELS} S2 bands")
        print(f"[EuroSAT-BL] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"[EuroSAT-BL] num_classes: {self.NUM_CLASSES}")
        print(f"[EuroSAT-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        print(f"[EuroSAT-BL] class distribution: {dict(sorted(label_counts.items()))}")

    @staticmethod
    def _d4_augment(image):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        path = os.path.join(self.root_path, f"{name}.hdf5")

        bands = []
        with h5py.File(path, "r") as f:
            for key in self.BAND_KEYS:
                bands.append(np.asarray(f[key], dtype=np.float32))

        image = torch.from_numpy(np.stack(bands, axis=0))
        target = torch.tensor(self.name_to_label[name], dtype=torch.long)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        if self.augment:
            image = self._d4_augment(image)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image":  {"s2": image},
            "target": target,
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_CHANNELS,
                "sample_name": name,
            },
        }