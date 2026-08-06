import os
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset
import einops
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from .token_grouping import *
from .sliding_window import compute_crop_positions
from .token_builder import TokenBuilder


class MADOSDataset(Dataset):
    """
    MADOS Dataset — all bands upscaled to 10m (matching PANGAEA protocol).

    All bands are loaded at native resolution, normalized, then upscaled
    to 10m (240×240) via nearest-neighbor. Single resolution group at 10m.

    D4 augmentation (random flip + 90° rotation) is applied on the train
    split only, after the focus crop and before token building. Val/test
    (sliding window) remain unaugmented for fair benchmarking.

    SKIP: emits a per-query gather index `query_token_idx` of shape
    [N_q, bands_per_pixel] (and `query_token_valid`), where each row holds
    the row indices (into this sample's `image_tokens` pool) of that
    query-pixel's own band-tokens. This lets a decoder skip cross-attention
    and read each pixel's own raw tokens directly (Atomiser_Senflood_Skip).
    Join key is (x, y) = cols 1,2, shared by a pixel's band-tokens and its
    query. Indices are RELATIVE TO THIS SAMPLE's image_tokens pool — the
    collate function must offset them if it concatenates samples.

    Uses reference grid indexing: all crops (including edge crops from
    sliding window) extract coordinates from a shared 512×512 reference grid,
    ensuring consistent coordinate space across different crop sizes.

    YAML bands_mados must use NATIVE resolutions for file discovery:
        B01: resolution: 60   (files in Scene_*/60/)
        B05: resolution: 20   (files in Scene_*/20/)
        B02: resolution: 10   (files in Scene_*/10/)
    The model sees everything as 10m after upscaling.

    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7
    """

    NUM_CLASSES = 15
    IGNORE_INDEX = 255
    TARGET_RESOLUTION = 10.0
    TIME_IDX_NA = -1

    FULL_SIZE_10M = (240, 240)

    NATIVE_SIZES = {
        10: (240, 240),
        20: (120, 120),
        60: (40, 40),
    }

    FOCUS_CROP_SIZE = 240
    SLIDING_STRIDE = 80

    def __init__(
        self,
        root_path: str = "./data/MADOS",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model

        # Initialize TokenBuilder with reference grid system
        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]
        self.use_sliding = config_model["trainer"].get("slide", False)

        self.split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

        # ── Band metadata from YAML ────────────────────────
        self.bands_info = dataset_config["bands_mados"]
        self.bands_by_resolution = self._parse_bands_info()

        # All bands sorted by idx (for merging order)
        self.all_bands_sorted = self._build_all_bands_sorted()

        # Spectral indices for all bands (merged order)
        self.all_spectral_indices = self._build_all_spectral_indices()

        # Resolution index: everything is 10m after upscale
        self.resolution_idx_10m = self.look_up.get_resolution_idx(10.0)

        # ── Discover data ───────────────────────────────────
        self.samples = self._discover_samples()

        # ── Normalization ───────────────────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        total_bands = len(self.all_bands_sorted)
        print(f"[MADOS] Loaded {len(self.samples)} samples, split={self.split}")
        print(f"[MADOS] {total_bands} bands total, all upscaled to 10m (240×240)")
        print(f"[MADOS] Using reference grid indexing (512×512 @ 10m)")
        for b in self.all_bands_sorted:
            print(f"  idx={b['idx']:2d}: {b['band_key']:4s} (native {b['resolution']}m) → "
                  f"wl={b['wavelength']}nm, bw={b['bandwidth']}nm")
        print(f"[MADOS] D4 augmentations: {'ON' if self.split == 'train' else 'OFF'}")
        if self.split != "train" and self.use_sliding:
            print(f"[MADOS] Val/test: sliding window, crop={self.FOCUS_CROP_SIZE}, "
                  f"stride={self.SLIDING_STRIDE}")

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _parse_bands_info(self):
        """Parse bands_mados from YAML, group by native resolution (for file discovery)."""
        all_bands = []
        for band_key, data in self.bands_info.items():
            if "bandwidth" not in data or "central_wavelength" not in data or "idx" not in data:
                continue
            all_bands.append({
                "band_key": band_key,
                "idx": data["idx"],
                "wavelength": data["central_wavelength"],
                "bandwidth": data["bandwidth"],
                "resolution": data["resolution"],
            })

        all_bands.sort(key=lambda b: b["idx"])

        bands_by_res = {}
        for band in all_bands:
            res = band["resolution"]
            if res not in bands_by_res:
                bands_by_res[res] = []
            bands_by_res[res].append(band)

        return bands_by_res

    def _build_all_bands_sorted(self):
        """All bands sorted by idx, with index-within-resolution for loading."""
        all_bands = []
        for res, bands in self.bands_by_resolution.items():
            for i, band in enumerate(bands):
                band_copy = dict(band)
                band_copy["idx_within_res"] = i
                all_bands.append(band_copy)
        all_bands.sort(key=lambda b: b["idx"])
        return all_bands

    def _build_all_spectral_indices(self):
        """Spectral lookup indices for all bands in merged order."""
        indices = []
        for band in self.all_bands_sorted:
            key = (int(band["bandwidth"]), int(band["wavelength"]))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[MADOS] Band {band['band_key']} key={key} not in lookup. "
                    f"Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # D4 AUGMENTATION
    # =========================================================================

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """Random horizontal flip + random 90° rotation (train split only)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    # =========================================================================
    # SKIP: per-query gather index into own band-tokens
    # =========================================================================

    def _build_full_pixel_index(self, C, H, W):
        """
        Closed-form gather index for ALL pixels, in pixel order p = h*W + w.

        TokenBuilder.build_tokens flattens as `(c h w) -> row`, i.e. channel-
        major: pixel p's band-tokens live at rows {p + c*H*W : c in 0..C-1},
        strided by H*W (NOT contiguous). Same TokenBuilder as Sen1Floods11Skip,
        so the ordering carries over unchanged.

        Returns [H*W, C] long.
        """
        HW = H * W
        p = torch.arange(HW)                                   # [HW]
        c = torch.arange(C)                                    # [C]
        return p.unsqueeze(1) + c.unsqueeze(0) * HW            # [HW, C]

    def _build_query_token_index(self, C, H, W, kept_indices=None):
        """
        Vectorized per-query gather index into own band-tokens.

        idx[i] = the C row indices (into this sample's image_tokens) of the
        band-tokens for query i's pixel.

        Args:
            C, H, W      : image dims used to build the token pool
            kept_indices : [N_q] long or None.
                           None  -> queries are the full pixel grid in order
                                    (sliding crops: queries == seg_queries).
                           tensor-> the row positions (into the full pixel
                                    grid) that subsample_queries kept, in the
                                    SAME order as the returned queries
                                    (train / non-sliding eval). Obtained via
                                    subsample_queries(..., return_indices=True).

        Returns:
            idx   : [N_q, C] long  -- rows into image_tokens
            valid : [N_q] bool     -- all True (closed form always resolves)

        NOTE: indices are RELATIVE TO THIS SAMPLE's image_tokens pool. The
        collate function must offset them if it concatenates samples; no
        offset needed if it pads to [B, N, 8] and the model gathers per-sample.
        """
        full = self._build_full_pixel_index(C, H, W)          # [H*W, C]
        if kept_indices is None:
            idx = full
        else:
            idx = full[kept_indices]                          # [N_q, C]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with rasterio.open(sample["label_path"]) as src:
            label_full = src.read(1).astype(np.int64)

        label_full = label_full - 1
        label_full[label_full == -1] = self.IGNORE_INDEX
        label_full = torch.from_numpy(label_full)

        image = self._load_and_merge(sample)  # [C, 240, 240]

        if self.split == "train":
            return self._getitem_train(image, label_full)
        elif self.use_sliding:
            return self._getitem_sliding(image, label_full)
        else:
            return self._getitem_train(image, label_full)

    # =========================================================================
    # TRAIN: focus crop → D4 augment → token building
    # =========================================================================

    def _getitem_train(self, image, label_full):
        """Focus crop + D4 augmentation (train split only) + token building."""

        crop_coords = self._get_focus_crop(label_full)

        if crop_coords is not None:
            y0, x0, crop_h, crop_w = crop_coords
            label = label_full[y0:y0+crop_h, x0:x0+crop_w]
            image = image[:, y0:y0+crop_h, x0:x0+crop_w]
        else:
            label = label_full

        if self.split == "train":
            image, label = self._d4_augment(image, label)

        # Build single token group — everything at 10m
        C, H, W = image.shape
        image_tokens = self._build_tokens_for_group(
            image, label, self.TARGET_RESOLUTION,
            self.all_spectral_indices, self.resolution_idx_10m,
        )
        attention_mask = torch.zeros(image_tokens.shape[0])

        groups = {
            self.TARGET_RESOLUTION: {
                "tokens": image_tokens,
                "mask": attention_mask,
                "shape": (C, H, W),
            }
        }

        # Build and subsample queries using TokenBuilder
        queries = self._build_queries(label, self.TARGET_RESOLUTION)
        # SKIP: capture which queries were kept so the gather index can be
        #       selected to match (subsample shuffles/truncates order).
        queries, kept_indices = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_tokens_reconstruction,
            ignore_index=self.IGNORE_INDEX,
            prioritize_valid=True,
            return_indices=True,
        )
        queries_mask = torch.zeros(queries.shape[0])

        # SKIP: vectorized per-query gather index (closed form) into this
        # sample's own image_tokens pool.
        query_token_idx, query_token_valid = self._build_query_token_index(
            C, H, W, kept_indices=kept_indices
        )

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "label": label,
            "target_resolution": self.TARGET_RESOLUTION,
            "image": image,
            "query_token_idx": query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # =========================================================================
    # VAL/TEST: sliding window (no augmentation)
    # =========================================================================

    def _getitem_sliding(self, image, label_full):
        """
        Sliding window over full tile. No augmentation (fair benchmarking).

        Edge crops may be smaller than FOCUS_CROP_SIZE, but all extract
        coordinates from the same 512×512 reference grid, ensuring
        consistent coordinate space.
        """
        full_h, full_w = label_full.shape
        crop_size = self.FOCUS_CROP_SIZE
        stride = self.SLIDING_STRIDE

        if full_h <= crop_size and full_w <= crop_size:
            crop_size_h, crop_size_w = full_h, full_w
            positions = [(0, 0)]
        else:
            crop_size_h, crop_size_w = crop_size, crop_size
            positions = compute_crop_positions(
                full_h, full_w, crop_size_h, crop_size_w, stride, stride,
            )

        all_crops = []
        for (y0, x0) in positions:
            crop_dict = self._build_single_crop(
                image, label_full, y0, x0, crop_size_h, crop_size_w,
            )
            all_crops.append(crop_dict)

        return {
            "sliding": True,
            "crops": all_crops,
            "crop_positions": positions,
            "crop_size": (crop_size_h, crop_size_w),
            "full_size": (full_h, full_w),
            "label": label_full,
            "target_resolution": self.TARGET_RESOLUTION,
            "image": image,
        }

    def _build_single_crop(self, image, label_full, y0, x0, crop_h, crop_w):
        """
        Build tokens for one sliding window crop.

        TokenBuilder's reference grid system handles edge crops automatically:
        - 240×240 crop extracts window [136:376, 136:376] from 512×512 reference
        - 160×240 edge crop extracts [176:336, 136:376] from same reference
        - All crops share the same coordinate space
        """
        label_crop = label_full[y0:y0+crop_h, x0:x0+crop_w]
        image_crop = image[:, y0:y0+crop_h, x0:x0+crop_w]

        C, H, W = image_crop.shape

        # TokenBuilder handles coordinate extraction from reference grid
        image_tokens = self._build_tokens_for_group(
            image_crop, label_crop, self.TARGET_RESOLUTION,
            self.all_spectral_indices, self.resolution_idx_10m,
        )
        attention_mask = torch.zeros(image_tokens.shape[0])

        groups = {
            self.TARGET_RESOLUTION: {
                "tokens": image_tokens,
                "mask": attention_mask,
                "shape": (C, H, W),
            }
        }

        queries = self._build_queries(label_crop, self.TARGET_RESOLUTION)
        queries_mask = torch.zeros(queries.shape[0])

        # SKIP: sliding crops use the full pixel grid in order (no
        # subsampling), so kept_indices=None.
        query_token_idx, query_token_valid = self._build_query_token_index(
            C, H, W, kept_indices=None
        )

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "query_token_idx": query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # =========================================================================
    # LOADING: native → normalize → upscale to 10m → merge
    # =========================================================================

    def _load_and_merge(self, sample):
        """
        Load all bands at native resolution, normalize, upscale to 10m,
        merge into [C_total, 240, 240]. Band order follows YAML idx.
        """
        target_H, target_W = self.FULL_SIZE_10M

        # Load and normalize per native resolution
        images_by_res = {}
        for resolution in sorted(sample["bands"].keys()):
            band_paths = sample["bands"][resolution]
            expected_H, expected_W = self.NATIVE_SIZES[resolution]

            band_arrays = []
            for path in band_paths:
                with rasterio.open(path, mode="r") as src:
                    band_data = src.read(1).astype(np.float32)
                assert band_data.shape == (expected_H, expected_W), (
                    f"[MADOS] Expected {expected_H}x{expected_W} at {resolution}m, "
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

        # Merge all bands in idx order, upscaling to 10m
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

    # =========================================================================
    # FOCUS CROPPING
    # =========================================================================

    def _get_focus_crop(self, label):
        """Crop centered on annotated pixels. Returns (y0, x0, h, w) or None."""
        if self.FOCUS_CROP_SIZE is None:
            return None

        crop_size = self.FOCUS_CROP_SIZE
        H, W = label.shape

        if H <= crop_size and W <= crop_size:
            return None

        valid_yx = torch.where(label != self.IGNORE_INDEX)

        if len(valid_yx[0]) == 0:
            y0 = torch.randint(0, H - crop_size + 1, (1,)).item()
            x0 = torch.randint(0, W - crop_size + 1, (1,)).item()
        else:
            anchor_idx = torch.randint(0, len(valid_yx[0]), (1,)).item()
            anchor_y = valid_yx[0][anchor_idx].item()
            anchor_x = valid_yx[1][anchor_idx].item()
            y0 = max(0, min(anchor_y - crop_size // 2, H - crop_size))
            x0 = max(0, min(anchor_x - crop_size // 2, W - crop_size))

        # No grid alignment - crop can start anywhere
        return (y0, x0, crop_size, crop_size)

    # =========================================================================
    # TOKEN BUILDING (using centralized TokenBuilder)
    # =========================================================================

    def _build_tokens_for_group(self, image, label, resolution,
                                spectral_indices, resolution_idx):
        """
        Build [N, 8] tokens using TokenBuilder with reference grid indexing.

        All crops (including edge crops) extract coordinates from the
        512×512 reference grid, ensuring consistent coordinate space.
        """
        return self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=resolution,
            spectral_indices=spectral_indices,
            resolution_idx=resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

    def _build_queries(self, label, resolution):
        """Build query tokens using TokenBuilder with reference grid indexing."""
        first_spectral_idx = self.all_spectral_indices[0]

        return self.token_builder.build_queries(
            label=label,
            resolution=resolution,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx_10m,
            time_idx=self.TIME_IDX_NA,
        )

    # =========================================================================
    # DATA DISCOVERY
    # =========================================================================

    def _discover_samples(self):
        """Discover all valid samples. Uses YAML native resolutions for file paths."""
        split_key = self.split_mapping[self.split]
        split_file = os.path.join(self.root_path, "splits", f"{split_key}_X.txt")
        rois_split = np.genfromtxt(split_file, dtype="str")

        if rois_split.ndim == 0:
            rois_split = {str(rois_split)}
        else:
            rois_split = set(rois_split.tolist())

        expected_wavelengths = {}
        for res, bands in self.bands_by_resolution.items():
            expected_wavelengths[res] = [b["wavelength"] for b in bands]

        samples = []
        resolution_stats = {res: 0 for res in expected_wavelengths}
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
                        resolution_stats[res] += 1

                if 10 not in bands_by_res:
                    skipped_no_10m += 1
                    continue

                samples.append({
                    "name": crop_name,
                    "label_path": cl_file,
                    "bands": bands_by_res,
                })

        print(f"[MADOS] Found {len(samples)} samples for split={self.split}")
        if skipped_no_10m > 0:
            print(f"[MADOS] Skipped {skipped_no_10m} samples (missing 10m bands)")
        for res in sorted(resolution_stats.keys()):
            count = resolution_stats[res]
            pct = 100 * count / len(samples) if samples else 0
            print(f"[MADOS]   {res:3d}m bands: {count}/{len(samples)} samples ({pct:.1f}%)")

        return samples

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        """Load or compute per-band, per-resolution normalization stats."""
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)

            needs_recompute = False
            for res, bands in self.bands_by_resolution.items():
                if res not in stats:
                    needs_recompute = True
                    break
                if len(stats[res]["mean"]) != len(bands):
                    print(f"[MADOS] Band count mismatch at {res}m: "
                          f"stats={len(stats[res]['mean'])}, YAML={len(bands)}. Recomputing...")
                    needs_recompute = True
                    break

            if not needs_recompute:
                print(f"[MADOS] Loading normalization stats from {norm_file}")
                self._validate_norm_stats(stats)
                self._print_norm_stats(stats)
                return stats
            else:
                os.remove(norm_file)

        if self.split != "train":
            print(f"[MADOS] WARNING: No normalization file at {norm_file}")
            stats = {}
            for res, bands in self.bands_by_resolution.items():
                n = len(bands)
                stats[res] = {"mean": torch.zeros(n), "std": torch.ones(n)}
            return stats

        print(f"[MADOS] Computing normalization from {len(self.samples)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[MADOS] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        accum = {}
        for res, bands in self.bands_by_resolution.items():
            n = len(bands)
            accum[res] = {
                "sum": torch.zeros(n, dtype=torch.float64),
                "sq": torch.zeros(n, dtype=torch.float64),
                "count": torch.zeros(n, dtype=torch.float64),
            }

        for sample in tqdm(self.samples, desc="[MADOS] Computing normalization"):
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
                        if len(valid) > 0:
                            accum[res]["sum"][c] += valid.sum()
                            accum[res]["sq"][c] += (valid ** 2).sum()
                            accum[res]["count"][c] += len(valid)
                    except Exception as e:
                        print(f"[Warning] Could not read {path}: {e}")

        stats = {}
        for res, acc in accum.items():
            mean = (acc["sum"] / acc["count"].clamp(min=1)).float()
            var = (acc["sq"] / acc["count"].clamp(min=1)) - mean.double() ** 2
            std = torch.sqrt(var.clamp(min=1e-8)).float()
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0)
            std = std.clamp(min=1e-6)
            stats[res] = {"mean": mean, "std": std}

        return stats

    def _validate_norm_stats(self, stats):
        for res, s in stats.items():
            m, st = s['mean'], s['std']
            if m.isnan().any() or st.isnan().any():
                s['mean'] = torch.nan_to_num(m, nan=0.0)
                s['std'] = torch.nan_to_num(st, nan=1.0)
                st = s['std']
            if (st < 1e-6).any():
                s['std'] = st.clamp(min=1e-6)

    def _normalize_resolution(self, image, resolution):
        if resolution not in self.norm_stats:
            return image
        C = image.shape[0]
        mean = self.norm_stats[resolution]["mean"][:C].view(C, 1, 1)
        std = self.norm_stats[resolution]["std"][:C].view(C, 1, 1)
        std = std.clamp(min=1e-6)
        result = (image - mean) / std
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
        return result

    def _print_norm_stats(self, stats):
        for res in sorted(stats.keys()):
            s = stats[res]
            bands = self.bands_by_resolution.get(res, [])
            print(f"[MADOS] {res}m normalization:")
            for i, band in enumerate(bands):
                if i < len(s["mean"]):
                    print(f"  {band['band_key']} ({band['wavelength']}nm): "
                          f"mean={s['mean'][i]:.6f}, std={s['std'][i]:.6f}")

    @staticmethod
    def _shuffle_arrays(arrays: list):
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]
