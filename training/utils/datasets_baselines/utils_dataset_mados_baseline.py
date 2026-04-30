"""
MADOS Baseline Dataset
=======================

15-class marine debris segmentation on Sentinel-2 patches at 10m.

Loading pipeline (matches the Atomiser dataset for fair comparison):
    1. Discover samples via splits/{train|val|test}_X.txt
    2. Load each band at native resolution (10m / 20m / 60m) from
       Scene_*/{10|20|60}/{tile}_L2R_rhorc_{wl}_{crop_suffix}
    3. Normalize per-band per-resolution (cached stats)
    4. Upscale all bands to 10m via nearest-neighbor → 240×240
    5. Stack in YAML idx order → [C, 240, 240]
    6. D4 augmentation (training only)

Output format (compatible with BaselineTrainer):
    {
        "image":  {"s2": [C, H, W]}    float32 (post-norm)
        "target": [H, W]                long {0..14, 255}
        "metadata": {...}
    }

Label semantics (matches PANGAEA / Atomiser version):
    - Native labels are 1-indexed in the .tif files
    - We remap: label - 1 (so {1..15} → {0..14})
    - IGNORE_INDEX (255) for the original 0/no-data pixels

Native size: 240×240 at 10m. We don't crop further by default — models
accept the full patch. If a model needs a different size (e.g., ViT
img_size constraint), pass crop_size accordingly.
"""

import os
from glob import glob

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class MADOSBaselineDataset(Dataset):
    """MADOS 15-class marine debris segmentation baseline dataset."""

    NUM_CLASSES = 15
    IGNORE_INDEX = 255
    TARGET_RESOLUTION = 10.0
    FULL_SIZE_10M = (240, 240)

    NATIVE_SIZES = {
        10: (240, 240),
        20: (120, 120),
        60: (40, 40),
    }

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "val",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/MADOS",
        mode: str = "train",
        bands_info: dict = None,
        crop_size: int = None,
    ):
        """
        Args:
            root_path:  Path to MADOS data root.
            mode:       "train", "validation", or "test".
            bands_info: bands_mados dict from bands.yaml (with
                        bandwidth/central_wavelength/idx/resolution per band).
            crop_size:  If set, deterministic center crop to crop_size×crop_size
                        after loading. Default None → use full 240×240.

        D4 augmentation is automatic: applied iff mode == "train".
        """
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        if bands_info is None:
            raise ValueError(
                "[MADOS-BL] bands_info is required (pass dataset_config['bands_mados'])"
            )

        self.root_path = root_path
        self.split     = mode
        self.augment   = (mode == "train")    # D4 always on for train
        self.crop_size = crop_size
        self.bands_info = bands_info

        # ── Parse bands metadata, group by native resolution ──
        self.bands_by_resolution = self._parse_bands_info()
        self.all_bands_sorted    = self._build_all_bands_sorted()
        self.num_channels        = len(self.all_bands_sorted)

        # ── Discover samples ──
        self.samples = self._discover_samples()

        # ── Normalization stats ──
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[MADOS-BL] split={mode}, samples={len(self.samples)}")
        print(f"[MADOS-BL] {self.num_channels} bands total, all upscaled to 10m "
              f"({self.FULL_SIZE_10M[0]}×{self.FULL_SIZE_10M[1]})")
        print(f"[MADOS-BL] num_classes={self.NUM_CLASSES}, "
              f"IGNORE_INDEX={self.IGNORE_INDEX}")
        print(f"[MADOS-BL] D4 augment: {'ON (train)' if self.augment else 'OFF'}")
        if self.crop_size is not None:
            print(f"[MADOS-BL] center crop: {self.crop_size}×{self.crop_size}")

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        """Group bands by native resolution (10/20/60m) and sort by idx."""
        all_bands = []
        for band_key, data in self.bands_info.items():
            if (
                "bandwidth" not in data
                or "central_wavelength" not in data
                or "idx" not in data
                or "resolution" not in data
            ):
                continue
            all_bands.append({
                "band_key":   band_key,
                "idx":        data["idx"],
                "wavelength": data["central_wavelength"],
                "bandwidth":  data["bandwidth"],
                "resolution": data["resolution"],
            })
        all_bands.sort(key=lambda b: b["idx"])

        bands_by_res = {}
        for b in all_bands:
            bands_by_res.setdefault(b["resolution"], []).append(b)
        return bands_by_res

    def _build_all_bands_sorted(self):
        """Flat list of bands sorted by idx, with within-resolution position."""
        all_bands = []
        for res, bands in self.bands_by_resolution.items():
            for i, b in enumerate(bands):
                bc = dict(b)
                bc["idx_within_res"] = i
                all_bands.append(bc)
        all_bands.sort(key=lambda b: b["idx"])
        return all_bands

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 transform applied identically to image [C, H, W] and label [H, W]."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _center_crop(image: torch.Tensor, label: torch.Tensor, size: int):
        """Center crop image [C, H, W] and label [H, W] to size×size."""
        C, H, W = image.shape
        assert H >= size and W >= size, f"crop_size={size} exceeds {H}×{W}"
        top  = (H - size) // 2
        left = (W - size) // 2
        return (
            image[:, top:top + size, left:left + size],
            label[top:top + size, left:left + size],
        )

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # ── Load label (1-indexed in source → 0-indexed; 0 → IGNORE) ──
        with rasterio.open(sample["label_path"]) as src:
            label = src.read(1).astype(np.int64)
        label = label - 1
        label[label == -1] = self.IGNORE_INDEX
        label = torch.from_numpy(label)

        # ── Load + normalize + upscale + merge bands → [C, 240, 240] ──
        image = self._load_and_merge(sample)

        # ── Optional center crop ──
        if self.crop_size is not None:
            image, label = self._center_crop(image, label, self.crop_size)

        # ── D4 augmentation (training only) ──
        if self.augment:
            image, label = self._d4_augment(image, label)

        H, W = image.shape[-2], image.shape[-1]

        return {
            "image":  {"s2": image},                  # [C, H, W]
            "target": label,                           # [H, W]
            "metadata": {
                "name":       sample["name"],
                "n_bands":    self.num_channels,
                "resolution": self.TARGET_RESOLUTION,
                "H": H, "W": W,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # LOADING: native → normalize → upscale → merge
    # ─────────────────────────────────────────────────────────────────────

    def _load_and_merge(self, sample):
        """
        Load all bands at native resolution, normalize, upscale to 10m,
        merge into [C_total, 240, 240]. Band order follows YAML idx.
        """
        target_H, target_W = self.FULL_SIZE_10M

        images_by_res = {}
        for resolution in sorted(sample["bands"].keys()):
            band_paths = sample["bands"][resolution]
            expected_H, expected_W = self.NATIVE_SIZES[resolution]

            band_arrays = []
            for path in band_paths:
                with rasterio.open(path, mode="r") as src:
                    band_data = src.read(1).astype(np.float32)
                assert band_data.shape == (expected_H, expected_W), (
                    f"[MADOS-BL] Expected {expected_H}×{expected_W} at {resolution}m, "
                    f"got {band_data.shape} for {path}"
                )
                band_arrays.append(band_data)

            image_res = np.stack(band_arrays, axis=0)
            image_res = np.nan_to_num(image_res, nan=0.0, posinf=0.0, neginf=0.0)
            image_res = torch.from_numpy(image_res)

            image_res = self._normalize_resolution(image_res, resolution)
            image_res = torch.clamp(image_res, -10, 10)
            image_res = torch.nan_to_num(image_res, nan=0.0, posinf=10.0, neginf=-10.0)

            images_by_res[resolution] = image_res

        # Merge in YAML idx order, upscaling everything to 10m via nearest.
        merged = []
        for band_info in self.all_bands_sorted:
            res = band_info["resolution"]
            idx_in_res = band_info["idx_within_res"]

            if res not in images_by_res:
                merged.append(torch.zeros(target_H, target_W))
                continue

            band_data = images_by_res[res][idx_in_res]
            if band_data.shape[0] != target_H or band_data.shape[1] != target_W:
                band_data = F.interpolate(
                    band_data.unsqueeze(0).unsqueeze(0),
                    size=(target_H, target_W),
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            merged.append(band_data)

        return torch.stack(merged, dim=0)

    # ─────────────────────────────────────────────────────────────────────
    # DATA DISCOVERY
    # ─────────────────────────────────────────────────────────────────────

    def _discover_samples(self):
        """Discover valid samples per split.

        File layout (matches PANGAEA / Atomiser version):
            data/MADOS/
              splits/{train|val|test}_X.txt   # list of crop_name strings
              Scene_*/{10|20|60}/             # per-resolution bands
                Scene_*_L2R_rhorc_{wl}_{crop_suffix}.tif
              Scene_*/10/Scene_*_cl_{crop_suffix}.tif   # labels (always 10m)
        """
        split_key = self.SPLIT_MAPPING[self.split]
        split_file = os.path.join(self.root_path, "splits", f"{split_key}_X.txt")
        rois_split = np.genfromtxt(split_file, dtype="str")

        if rois_split.ndim == 0:
            rois_split = {str(rois_split)}
        else:
            rois_split = set(rois_split.tolist())

        expected_wavelengths = {
            res: [b["wavelength"] for b in bands]
            for res, bands in self.bands_by_resolution.items()
        }

        samples = []
        skipped_no_10m = 0
        tiles = sorted(glob(os.path.join(self.root_path, "Scene_*")))

        for tile in tiles:
            tile_name = os.path.basename(tile)
            cl_files = glob(os.path.join(tile, "10", "*_cl_*"))
            if not cl_files:
                continue

            for cl_file in cl_files:
                crop_suffix = os.path.basename(cl_file).split("_cl_")[-1]
                crop_name = tile_name + "_" + crop_suffix.split(".tif")[0]

                if crop_name not in rois_split:
                    continue

                bands_by_res = {}
                for res, wavelengths in expected_wavelengths.items():
                    res_dir = os.path.join(tile, str(res))
                    if not os.path.isdir(res_dir):
                        continue

                    band_paths = []
                    all_found = True
                    for wl in wavelengths:
                        pattern = os.path.join(
                            res_dir, f"*_L2R_rhorc_{wl}_{crop_suffix}"
                        )
                        matches = glob(pattern)
                        if len(matches) != 1:
                            all_found = False
                            break
                        band_paths.append(matches[0])

                    if all_found:
                        bands_by_res[res] = band_paths

                if 10 not in bands_by_res:
                    skipped_no_10m += 1
                    continue

                samples.append({
                    "name":       crop_name,
                    "label_path": cl_file,
                    "bands":      bands_by_res,
                })

        print(f"[MADOS-BL] Found {len(samples)} samples for split={self.split}")
        if skipped_no_10m > 0:
            print(f"[MADOS-BL] Skipped {skipped_no_10m} (missing 10m bands)")
        return samples

    # ─────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_compute_normalization(self):
        """Load or compute per-band, per-resolution normalization stats.

        Reuses the same normalization_stats.pt as the Atomiser dataset (same
        loading pipeline). If absent on val/test split, falls back to
        zero-mean unit-std (warning).
        """
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)

            # Validate shape against current YAML config
            needs_recompute = False
            for res, bands in self.bands_by_resolution.items():
                if res not in stats:
                    needs_recompute = True
                    break
                if len(stats[res]["mean"]) != len(bands):
                    print(f"[MADOS-BL] Band count mismatch at {res}m: "
                          f"stats={len(stats[res]['mean'])}, YAML={len(bands)}. "
                          f"Recomputing...")
                    needs_recompute = True
                    break

            if not needs_recompute:
                print(f"[MADOS-BL] Loaded normalization stats from {norm_file}")
                self._validate_norm_stats(stats)
                return stats
            else:
                os.remove(norm_file)

        if self.split != "train":
            print(f"[MADOS-BL] WARNING: No normalization file at {norm_file}. "
                  f"Using zero-mean / unit-std on val/test.")
            stats = {}
            for res, bands in self.bands_by_resolution.items():
                n = len(bands)
                stats[res] = {"mean": torch.zeros(n), "std": torch.ones(n)}
            return stats

        print(f"[MADOS-BL] Computing normalization from {len(self.samples)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[MADOS-BL] Saved normalization stats to {norm_file}")
        return stats

    def _compute_normalization_stats(self):
        accum = {}
        for res, bands in self.bands_by_resolution.items():
            n = len(bands)
            accum[res] = {
                "sum":   torch.zeros(n, dtype=torch.float64),
                "sq":    torch.zeros(n, dtype=torch.float64),
                "count": torch.zeros(n, dtype=torch.float64),
            }

        for sample in tqdm(self.samples, desc="[MADOS-BL] Computing norm"):
            for res, paths in sample["bands"].items():
                if res not in accum:
                    continue
                for c, path in enumerate(paths):
                    try:
                        with rasterio.open(path) as src:
                            data = src.read(1).astype(np.float64)
                        data = np.nan_to_num(data)
                        valid = data.flatten()
                        valid = valid[valid != 0]
                        if len(valid):
                            accum[res]["sum"][c]   += valid.sum()
                            accum[res]["sq"][c]    += (valid ** 2).sum()
                            accum[res]["count"][c] += len(valid)
                    except Exception as e:
                        print(f"[MADOS-BL] Warning reading {path}: {e}")

        stats = {}
        for res, acc in accum.items():
            mean = (acc["sum"] / acc["count"].clamp(min=1)).float()
            var  = (acc["sq"] / acc["count"].clamp(min=1)) - mean.double() ** 2
            std  = torch.sqrt(var.clamp(min=1e-8)).float()
            mean = torch.nan_to_num(mean, nan=0.0)
            std  = torch.nan_to_num(std, nan=1.0)
            std  = std.clamp(min=1e-6)
            stats[res] = {"mean": mean, "std": std}
        return stats

    def _validate_norm_stats(self, stats):
        for res, s in stats.items():
            m, st = s["mean"], s["std"]
            if m.isnan().any() or st.isnan().any():
                s["mean"] = torch.nan_to_num(m,  nan=0.0)
                s["std"]  = torch.nan_to_num(st, nan=1.0)
                st = s["std"]
            if (st < 1e-6).any():
                s["std"] = st.clamp(min=1e-6)

    def _normalize_resolution(self, image, resolution):
        if resolution not in self.norm_stats:
            return image
        C = image.shape[0]
        mean = self.norm_stats[resolution]["mean"][:C].view(C, 1, 1)
        std  = self.norm_stats[resolution]["std"][:C].view(C, 1, 1)
        std  = std.clamp(min=1e-6)
        out  = (image - mean) / std
        return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)