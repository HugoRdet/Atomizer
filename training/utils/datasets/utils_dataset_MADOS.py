import os
import numpy as np
import rasterio
import torch
from glob import glob
from torch.utils.data import Dataset
import einops
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from .token_grouping import *


class MADOSDataset(Dataset):
    """
    MADOS Dataset — grouped token format with multi-resolution support (8 columns).

    Marine debris detection from Sentinel-2 L2R imagery.
    Bands at native resolution: 10m (4 bands), 20m (5 bands), 60m (1 band).
    Each resolution forms a separate token group — no resampling needed.

    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7

    For MADOS:
        - resolution_idx: per-group, looked up from GSD (10/20/60 m/px) (≥0)
        - time_idx: -1 (no temporal info, zeroed out by encoder)

    Convention:
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features

    Returns:
    {
        "groups": {
            10.0: {"tokens": [N_10, 8], "mask": [N_10], "shape": (4, 240, 240)},
            20.0: {"tokens": [N_20, 8], "mask": [N_20], "shape": (5, 120, 120)},  # when available
            60.0: {"tokens": [N_60, 8], "mask": [N_60], "shape": (1, 40, 40)},    # when available
        },
        "queries":           [M, 8],
        "queries_mask":      [M],
        "label":             [240, 240],
        "target_resolution": 10.0,
        "image":             [C_avail, 240, 240],  # available bands upscaled to 10m
    }

    Directory structure:
    ./data/MADOS/
    ├── Scene_0/
    │   ├── 10/   (492, 560, 665, 833 nm + cl/conf/rep files)
    │   ├── 20/   (704, 783, 865, 1614, 2202 nm)
    │   └── 60/   (443 nm)
    ├── ...
    ├── Scene_173/
    └── splits/
        ├── train_X.txt
        ├── val_X.txt
        └── test_X.txt

    Classes (after remapping):
        0  = Marine Debris         8  = Foam
        1  = Dense Sargassum       9  = Turbid Water
        2  = Sparse Floating Algae 10 = Shallow Water
        3  = Natural Organic Mat.  11 = Waves & Wakes
        4  = Ship                  12 = Oil Platform
        5  = Oil Spill             13 = Jellyfish
        6  = Marine Water          14 = Sea Snot
        7  = Sediment-Laden Water
        255 = Non-annotated (ignore)
    """

    NUM_CLASSES = 15
    IGNORE_INDEX = 255
    TARGET_RESOLUTION = 10.0  # labels are at 10m
    TIME_IDX_NA = -1          # No temporal info → -1 → zeroed by encoder

    # Fixed spatial dimensions per resolution (full tile)
    RESOLUTION_SIZES = {
        10: (240, 240),
        20: (120, 120),
        60: (40, 40),
    }

    # Focus crop size at 10m (must be divisible by 6 for 60m alignment)
    # Set to None to disable focus cropping (use full tile)
    FOCUS_CROP_SIZE = 120  # 120×120 at 10m → 60×60 at 20m → 20×20 at 60m

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

        # Config parameters
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        # Split mapping
        self.split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

        # ── Band metadata from YAML ────────────────────────
        self.bands_info = dataset_config["bands_mados"]
        self.bands_by_resolution = self._parse_bands_info()
        self.spectral_indices_by_resolution = self._build_spectral_indices()

        # ── Resolution indices (one per resolution group) ───
        self.resolution_idx_by_resolution = {
            res: self.look_up.get_resolution_idx(float(res))
            for res in self.bands_by_resolution.keys()
        }

        # ── Discover data ───────────────────────────────────
        self.samples = self._discover_samples()

        # ── Normalization ───────────────────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[MADOS] Loaded {len(self.samples)} samples, split={self.split}")
        for res, bands in sorted(self.bands_by_resolution.items()):
            res_idx = self.resolution_idx_by_resolution[res]
            names = [b['band_key'] for b in bands]
            print(f"  {res}m: {len(bands)} bands → {names}, resolution_idx={res_idx}")
        print(f"[MADOS] Time idx: -1 (no temporal info, zeroed by encoder)")

    # =========================================================================
    # BAND METADATA (from YAML)
    # =========================================================================

    def _parse_bands_info(self):
        """
        Parse bands_mados from YAML and group by resolution.

        Returns:
            dict: {
                10: [{"band_key": "B02", "idx": 1, "wavelength": 492,
                      "bandwidth": 65, "resolution": 10}, ...],
                20: [...],
                60: [...],
            }
        """
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

        # Sort by idx to ensure consistent ordering
        all_bands.sort(key=lambda b: b["idx"])

        # Group by resolution
        bands_by_res = {}
        for band in all_bands:
            res = band["resolution"]
            if res not in bands_by_res:
                bands_by_res[res] = []
            bands_by_res[res].append(band)

        print(f"[MADOS] Band order (from YAML):")
        for res in sorted(bands_by_res.keys()):
            for b in bands_by_res[res]:
                print(f"  idx={b['idx']:2d}: {b['band_key']:4s} @ {res}m → "
                      f"bw={b['bandwidth']}nm, wl={b['wavelength']}nm")

        return bands_by_res

    def _build_spectral_indices(self):
        """
        Build spectral lookup indices per resolution group.

        Returns:
            dict: {resolution: torch.Tensor([idx_band0, idx_band1, ...])}
        """
        indices_by_res = {}
        for resolution, bands in self.bands_by_resolution.items():
            indices = []
            for band in bands:
                key = (int(band["bandwidth"]), int(band["wavelength"]))
                if key not in self.look_up.table_wave:
                    raise KeyError(
                        f"[MADOS] Band {band['band_key']} key={key} not in lookup. "
                        f"Available: {list(self.look_up.table_wave.keys())}"
                    )
                indices.append(self.look_up.table_wave[key])
            indices_by_res[resolution] = torch.tensor(indices, dtype=torch.long)
        return indices_by_res

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # ── Load label (10m, full tile) ─────────────────────
        with rasterio.open(sample["label_path"]) as src:
            label_full = src.read(1).astype(np.int64)

        # Remap: original 0 (non-annotated) → ignore, classes 1–15 → 0–14
        label_full = label_full - 1
        label_full[label_full == -1] = self.IGNORE_INDEX

        label_full = torch.from_numpy(label_full)

        # ── Focus cropping (at 10m) ─────────────────────────
        # Crop around annotated pixels to increase label density
        crop_coords = self._get_focus_crop(label_full)  # (y0, x0, crop_h, crop_w) at 10m

        if crop_coords is not None:
            y0, x0, crop_h, crop_w = crop_coords
            label = label_full[y0:y0+crop_h, x0:x0+crop_w]
        else:
            label = label_full
            y0, x0 = 0, 0
            crop_h, crop_w = label.shape

        # ── Load bands per resolution & build groups ────────
        groups = {}
        all_images_10m = []  # for the combined "image" field

        # Only iterate over resolutions available for THIS sample
        for resolution in sorted(sample["bands"].keys()):
            res_float = float(resolution)
            band_paths = sample["bands"][resolution]
            expected_H, expected_W = self.RESOLUTION_SIZES[resolution]

            # Read all bands at this resolution (ordered by YAML idx)
            band_arrays = []
            for path in band_paths:
                with rasterio.open(path, mode="r") as src:
                    band_data = src.read(1).astype(np.float32)
                assert band_data.shape == (expected_H, expected_W), (
                    f"[MADOS] Expected {expected_H}x{expected_W} at {resolution}m, "
                    f"got {band_data.shape} for {path}"
                )
                band_arrays.append(band_data)

            image_res = np.stack(band_arrays, axis=0)  # [C_res, H_res, W_res]

            # Track invalid pixels BEFORE replacing NaN
            invalid_mask = np.isnan(image_res) | np.isinf(image_res)  # [C, H, W]

            image_res = np.nan_to_num(image_res, nan=0.0, posinf=0.0, neginf=0.0)
            image_res = torch.from_numpy(image_res)
            invalid_mask = torch.from_numpy(invalid_mask)

            # Normalize
            image_res = self._normalize_resolution(image_res, resolution)
            image_res = torch.clamp(image_res, -10, 10)

            # Zero out invalid pixels AFTER normalization
            image_res[invalid_mask] = 0.0

            # ── Apply focus crop (scaled to this resolution) ──
            if crop_coords is not None:
                # Scale 10m crop coordinates to this resolution
                ry0 = int(y0 * 10 / resolution)
                rx0 = int(x0 * 10 / resolution)
                rh = int(crop_h * 10 / resolution)
                rw = int(crop_w * 10 / resolution)
                image_res = image_res[:, ry0:ry0+rh, rx0:rx0+rw]

            C_res, H_res, W_res = image_res.shape

            # Build tokens for this resolution group
            spectral_indices = self.spectral_indices_by_resolution[resolution]
            resolution_idx = self.resolution_idx_by_resolution[resolution]
            image_tokens = self._build_tokens_for_group(
                image_res, label, res_float, spectral_indices, resolution_idx,
            )

            # No attention masking — matches pangaea behavior
            attention_mask = torch.zeros(image_tokens.shape[0])

            groups[res_float] = {
                "tokens": image_tokens,
                "mask": attention_mask,
                "shape": tuple(image_res.shape),
            }

            # Upscale to 10m for the combined image field
            if resolution != 10:
                image_up = torch.nn.functional.interpolate(
                    image_res.unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode="nearest",
                ).squeeze(0)
            else:
                image_up = image_res
            all_images_10m.append(image_up)

        # ── Combined image (all bands at 10m) ───────────────
        image_combined = torch.cat(all_images_10m, dim=0)

        # ── Queries (at target resolution = 10m) ────────────
        queries = self._build_queries(label, self.TARGET_RESOLUTION)

        # ── Smart query subsampling ─────────────────────────
        nb_queries = self.max_tokens_reconstruction

        query_labels = queries[:, 4]
        valid_mask = (query_labels != self.IGNORE_INDEX)
        valid_indices = torch.where(valid_mask)[0]
        ignore_indices = torch.where(~valid_mask)[0]

        if len(valid_indices) > 0:
            if len(valid_indices) >= nb_queries:
                perm = torch.randperm(len(valid_indices))[:nb_queries]
                selected = valid_indices[perm]
            else:
                n_ignore_needed = nb_queries - len(valid_indices)
                n_ignore_needed = min(n_ignore_needed, len(ignore_indices))
                ignore_perm = torch.randperm(len(ignore_indices))[:n_ignore_needed]
                selected = torch.cat([valid_indices, ignore_indices[ignore_perm]])
                selected = selected[torch.randperm(len(selected))]
        else:
            n_take = min(nb_queries, queries.shape[0])
            perm = torch.randperm(queries.shape[0])[:n_take]
            selected = perm

        queries = queries[selected]
        queries_mask = torch.zeros(queries.shape[0])

        return {
            "groups": groups,
            "queries": queries,
            "queries_mask": queries_mask,
            "label": label,
            "target_resolution": self.TARGET_RESOLUTION,
            "image": image_combined,
        }

    # =========================================================================
    # FOCUS CROPPING
    # =========================================================================

    def _get_focus_crop(self, label):
        """
        Compute crop coordinates centered on annotated pixels.

        Following pangaea's approach: "focus cropping, which only selects
        crops with at least one valid label."

        Only applied during TRAINING. Validation/test use full tiles.

        Picks a random annotated pixel as anchor, crops a fixed-size window
        around it. Crop size must be divisible by 6 (for 60m alignment).

        Args:
            label: [H, W] with ignore_index=255 for unlabeled pixels

        Returns:
            (y0, x0, crop_h, crop_w) at 10m resolution, or None if no cropping
        """
        if self.FOCUS_CROP_SIZE is None:
            return None

        # Only crop during training
        if self.split != "train":
            return None

        crop_size = self.FOCUS_CROP_SIZE
        H, W = label.shape

        # Don't crop if image is already smaller than crop
        if H <= crop_size and W <= crop_size:
            return None

        # Find annotated pixel locations
        valid_yx = torch.where(label != self.IGNORE_INDEX)

        if len(valid_yx[0]) == 0:
            # No valid labels — random crop
            y0 = torch.randint(0, H - crop_size + 1, (1,)).item()
            x0 = torch.randint(0, W - crop_size + 1, (1,)).item()
        else:
            # Pick a random annotated pixel as anchor
            anchor_idx = torch.randint(0, len(valid_yx[0]), (1,)).item()
            anchor_y = valid_yx[0][anchor_idx].item()
            anchor_x = valid_yx[1][anchor_idx].item()

            # Center crop on anchor, clamp to image bounds
            y0 = max(0, min(anchor_y - crop_size // 2, H - crop_size))
            x0 = max(0, min(anchor_x - crop_size // 2, W - crop_size))

        # Ensure alignment with 60m (÷6)
        y0 = (y0 // 6) * 6
        x0 = (x0 // 6) * 6

        return (y0, x0, crop_size, crop_size)

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_tokens_for_group(self, image, label_10m, resolution, spectral_indices,
                                resolution_idx):
        """
        Build [N, 8] tokens for a single resolution group.

        Token format:
            [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
             col 0  1  2       3          4        5            6            7

        Args:
            image:            [C, H_res, W_res] normalized values at this resolution
            label_10m:        [H_10, W_10] labels at 10m resolution
            resolution:       meters per pixel for this group
            spectral_indices: [C] spectral lookup indices for each band
            resolution_idx:   int, lookup index for this resolution (≥0)

        Returns:
            image_tokens: [C * H_res * W_res, 8]
        """
        C, H, W = image.shape

        # Downsample label to this resolution if needed
        if H != label_10m.shape[0] or W != label_10m.shape[1]:
            label_res = torch.nn.functional.interpolate(
                label_10m.float().unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        else:
            label_res = label_10m

        # Spectral indices: [C * H * W]
        spectral_coords = spectral_indices.repeat_interleave(H * W)

        # Position coordinates: [C, H, W, 1]
        x_indices, y_indices = self._get_position_coordinates(
            (C, H, W), resolution
        )

        # Query position indices: [C, H, W, 1]
        query_indices = self._get_query_coordinates((C, H, W), resolution)

        # Expand label: [H, W] → [C, H, W]
        label_expanded = label_res.unsqueeze(0).expand(C, -1, -1)

        # Resolution: real GSD index (≥0)
        resolution_col = torch.full((C, H, W, 1), resolution_idx, dtype=torch.float32)

        # Time: -1 for all tokens (no temporal info)
        time_col = torch.full((C, H, W, 1), self.TIME_IDX_NA, dtype=torch.float32)

        # Stack: [C, H, W, 8]
        image_tokens = torch.cat([
            image.unsqueeze(-1),                                    # col 0: value
            x_indices.float(),                                      # col 1: x
            y_indices.float(),                                      # col 2: y
            spectral_coords.view(C, H, W, 1).float(),              # col 3: spectral_idx
            label_expanded.unsqueeze(-1).float(),                   # col 4: label
            query_indices.float(),                                  # col 5: query_idx
            resolution_col,                                         # col 6: resolution_idx
            time_col,                                               # col 7: time_idx
        ], dim=-1)

        # Flatten: [C * H * W, 8]
        image_tokens = einops.rearrange(image_tokens, "c h w f -> (c h w) f")

        return image_tokens

    def _build_queries(self, label, resolution):
        """
        Build query tokens at target resolution (10m).

        Args:
            label:      [H, W] at 10m
            resolution: target resolution (10m)

        Returns:
            queries: [H * W, 8]
        """
        H, W = label.shape

        # Use first spectral index from 10m bands
        first_spectral_idx = self.spectral_indices_by_resolution[10][0]

        # Resolution index for queries = target resolution (10m)
        query_resolution_idx = self.resolution_idx_by_resolution[10]

        x_indices, y_indices = self._get_position_coordinates(
            (1, H, W), resolution
        )
        query_indices = self._get_query_coordinates((1, H, W), resolution)

        queries = torch.cat([
            torch.zeros(1, H, W, 1),                                             # col 0: value
            x_indices.float(),                                                    # col 1: x
            y_indices.float(),                                                    # col 2: y
            torch.full((1, H, W, 1), first_spectral_idx, dtype=torch.float),     # col 3: spectral_idx
            label.unsqueeze(0).unsqueeze(-1).float(),                             # col 4: label
            query_indices.float(),                                                # col 5: query_idx
            torch.full((1, H, W, 1), query_resolution_idx, dtype=torch.float),   # col 6: resolution_idx
            torch.full((1, H, W, 1), self.TIME_IDX_NA, dtype=torch.float),       # col 7: time_idx
        ], dim=-1)

        queries = einops.rearrange(queries, "c h w f -> (c h w) f")
        return queries

    def _get_position_coordinates(self, image_shape, resolution):
        """Pixel position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        res_key = int(resolution * 1000)
        global_offset = self.look_up.table[(res_key, H)]

        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")

        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)

        return x_indices, y_indices

    def _get_query_coordinates(self, image_shape, resolution):
        """Query position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        resolution_latents = 10  # m
        res_key = int(resolution_latents * 1000)

        if (res_key, H) in self.look_up.table_queries:
            global_offset = self.look_up.table_queries[(res_key, H)]
        else:
            res_key = int(resolution * 1000)
            global_offset = self.look_up.table_queries.get((res_key, H), 0)

        return torch.full((C, H, W, 1), global_offset, dtype=torch.float32)

    # =========================================================================
    # DATA DISCOVERY
    # =========================================================================

    def _discover_samples(self):
        """
        Discover all valid samples matching the current split.

        Band files are matched using wavelength from the YAML config.
        Resolution groups are OPTIONAL — only 10m is required (labels live there).
        20m and 60m bands are included when available.

        Each sample is a dict:
        {
            "name":       "Scene_X_CROP",
            "label_path": ".../Scene_X/10/Scene_X_L2R_cl_CROP.tif",
            "bands": {
                10: [path_492, path_560, path_665, path_833],
                20: [path_704, path_783, path_865, path_1614, path_2202],  # optional
                60: [path_443],                                             # optional
            },
        }
        """
        split_key = self.split_mapping[self.split]
        split_file = os.path.join(self.root_path, "splits", f"{split_key}_X.txt")
        rois_split = np.genfromtxt(split_file, dtype="str")

        # Handle single-element case (np.genfromtxt returns scalar for 1 line)
        if rois_split.ndim == 0:
            rois_split = {str(rois_split)}
        else:
            rois_split = set(rois_split.tolist())

        # Build expected wavelengths per resolution from YAML
        expected_wavelengths = {}
        for res, bands in self.bands_by_resolution.items():
            expected_wavelengths[res] = [b["wavelength"] for b in bands]

        samples = []
        resolution_stats = {res: 0 for res in expected_wavelengths}
        skipped_no_10m = 0

        tiles = sorted(glob(os.path.join(self.root_path, "Scene_*")))

        for tile in tiles:
            tile_name = os.path.basename(tile)

            # Find all crops in this tile via classification files
            cl_files = glob(os.path.join(tile, "10", "*_cl_*"))
            if not cl_files:
                continue

            for cl_file in cl_files:
                # Extract crop suffix: "Scene_0_L2R_cl_CROP.tif" → "CROP.tif"
                crop_suffix = os.path.basename(cl_file).split("_cl_")[-1]
                crop_name = tile_name + "_" + crop_suffix.split(".tif")[0]

                if crop_name not in rois_split:
                    continue

                # For each resolution, try to find band files
                bands_by_res = {}

                for res, wavelengths in expected_wavelengths.items():
                    res_dir = os.path.join(tile, str(res))
                    if not os.path.isdir(res_dir):
                        continue  # skip this resolution, not mandatory

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

                # 10m is mandatory (labels come from 10m cl file)
                if 10 not in bands_by_res:
                    skipped_no_10m += 1
                    continue

                samples.append({
                    "name": crop_name,
                    "label_path": cl_file,
                    "bands": bands_by_res,
                })

        # Print summary
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
            print(f"[MADOS] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[MADOS] WARNING: No normalization file at {norm_file}")
            stats = {}
            for res, bands in self.bands_by_resolution.items():
                n = len(bands)
                stats[res] = {
                    "mean": torch.zeros(n),
                    "std": torch.ones(n),
                }
            return stats

        print(f"[MADOS] Computing normalization from {len(self.samples)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[MADOS] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        """Compute per-band mean/std for each resolution group."""
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
            stats[res] = {"mean": mean, "std": std}

        return stats

    def _normalize_resolution(self, image, resolution):
        """Normalize image tensor for a given resolution group."""
        if resolution not in self.norm_stats:
            return image
        C = image.shape[0]
        mean = self.norm_stats[resolution]["mean"][:C].view(C, 1, 1)
        std = self.norm_stats[resolution]["std"][:C].view(C, 1, 1)
        return (image - mean) / std

    def _print_norm_stats(self, stats):
        for res in sorted(stats.keys()):
            s = stats[res]
            bands = self.bands_by_resolution.get(res, [])
            print(f"[MADOS] {res}m normalization:")
            for i, band in enumerate(bands):
                if i < len(s["mean"]):
                    print(f"  {band['band_key']} ({band['wavelength']}nm): "
                          f"mean={s['mean'][i]:.6f}, std={s['std'][i]:.6f}")

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _shuffle_arrays(arrays: list):
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]