import os
import csv
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import einops
from tqdm import tqdm

from .token_grouping import *
from .token_builder import TokenBuilder


class Sen1Floods11SkipDataset(Dataset):
    """
    Sen1Floods11 Dataset — grouped token format (8 columns), SKIP variant.

    Identical to Sen1Floods11Dataset, with ONE addition: it emits a
    per-query gather index `query_token_idx` of shape [N_q, bands_per_pixel],
    where each row holds the row indices (into this sample's `image_tokens`
    pool) of that query-pixel's own band-tokens. This lets a decoder skip
    cross-attention read each pixel's own raw tokens directly.

    Everything tagged with  # >>> SKIP  is new relative to the base dataset.

    S2 (13 optical bands) + S1 (2 SAR bands), all at 10m, 512×512.
    Both modalities share the same resolution -> single group.

    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2        3          4        5            6            7

    Join key for the skip index is (x, y) = cols 1,2, shared by a pixel's
    band-tokens and its query.
    """

    OPTICAL_RESOLUTION = 10.0
    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1

    ALL_BAND_NAMES = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B08A", "B09", "B10", "B11", "B12",
        "VV", "VH",
    ]

    def __init__(
        self,
        root_path: str = "./data/SENFLOOD",
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

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction = config_model["trainer"].get("mode", "segmentation") == "reconstruction"

        # ── Band selection ──────────────────────────────────────────────────
        bands_cfg = config_model["trainer"].get("bands", {}) or {}
        keep_names = bands_cfg.get("keep", None)
        drop_names = bands_cfg.get("drop", None)

        if keep_names is None:
            sc = config_model["trainer"].get("single_channel", -1)
            if isinstance(sc, list):
                keep_names = [self.ALL_BAND_NAMES[i] for i in sorted(sc)]
            elif isinstance(sc, int) and sc >= 0:
                keep_names = [self.ALL_BAND_NAMES[sc]]

        self.selected_channels = self._resolve_band_names(keep_names)
        self.drop_band_names = set(drop_names) if drop_names else set()

        self._print_band_selection()

        self.split_mapping = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.split_mapping[mode]}_data.csv",
        )

        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        if self.selected_channels is not None:
            self.bandwidths       = self.bandwidths[self.selected_channels]
            self.wavelengths      = self.wavelengths[self.selected_channels]
            self.band_names       = [self.band_names[i] for i in self.selected_channels]
            self.spectral_indices = self.spectral_indices[self.selected_channels]

        self.dropped_spectral_indices = self._resolve_drop_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)
        self.norm_stats = self._load_or_compute_normalization()

        if self.reconstruction:
            print(f"[Sen1Floods11Skip] Mode: RECONSTRUCTION")
        else:
            print(f"[Sen1Floods11Skip] Mode: SEGMENTATION")
        print(f"[Sen1Floods11Skip] Loaded {len(self.bandwidths)} bands")
        print(f"[Sen1Floods11Skip] Resolution idx: {self.resolution_idx}")
        print(f"[Sen1Floods11Skip] D4 augmentations: {'ON' if self.split == 'train' else 'OFF'}")

    # =========================================================================
    # BAND SELECTION HELPERS
    # =========================================================================

    def _resolve_band_names(self, names):
        if names is None:
            return None
        invalid = set(names) - set(self.ALL_BAND_NAMES)
        if invalid:
            raise ValueError(
                f"Unknown band names: {invalid}. Valid names: {self.ALL_BAND_NAMES}"
            )
        return [self.ALL_BAND_NAMES.index(n) for n in names]

    def _resolve_drop_indices(self):
        if not self.drop_band_names:
            return set()

        kept = set(self.band_names)
        unknown = self.drop_band_names - set(self.ALL_BAND_NAMES)
        if unknown:
            raise ValueError(f"bands.drop contains unknown names: {unknown}")
        not_kept = self.drop_band_names - kept
        if not_kept:
            raise ValueError(
                f"bands.drop {not_kept} are not in bands.keep {kept}. "
                f"You can only drop bands that were kept."
            )

        dropped = set()
        for name in self.drop_band_names:
            data = self.bands_info[name]
            key = (int(data["bandwidth"]), int(data["central_wavelength"]))
            if key in self.look_up.table_wave:
                dropped.add(self.look_up.table_wave[key])
            else:
                raise KeyError(f"Band '{name}' key={key} not found in lookup table.")
        return dropped

    def _print_band_selection(self):
        if self.selected_channels is None:
            kept_str = "ALL"
        else:
            kept_str = str([self.ALL_BAND_NAMES[i] for i in self.selected_channels])
        drop_str = str(sorted(self.drop_band_names)) if self.drop_band_names else "none"
        print(f"[Sen1Floods11Skip] Bands kept    : {kept_str}")
        print(f"[Sen1Floods11Skip] Bands dropped : {drop_str} (padding tokens, grid unchanged)")

    # =========================================================================
    # D4 AUGMENTATION
    # =========================================================================

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _random_crop(image: torch.Tensor, label: torch.Tensor, size: int = 256):
        C, H, W = image.shape
        assert H >= size and W >= size
        top  = torch.randint(0, H - size + 1, (1,)).item()
        left = torch.randint(0, W - size + 1, (1,)).item()
        return image[:, top:top + size, left:left + size], label[top:top + size, left:left + size]

    # =========================================================================
    # CHANNEL SELECTION
    # =========================================================================

    def _select_channels(self, image):
        if self.selected_channels is None:
            return image
        return image[self.selected_channels]

    def _apply_drop_mask(self, tokens: torch.Tensor, mask: torch.Tensor):
        if not self.dropped_spectral_indices:
            return tokens, mask

        tokens = tokens.clone()
        mask   = mask.clone().float()

        spec_idx = tokens[:, 3]
        drop = torch.zeros(tokens.shape[0], dtype=torch.bool)
        for sid in self.dropped_spectral_indices:
            drop |= (spec_idx == sid)

        tokens[drop, 0] = 0.0
        mask[drop]      = 1.0

        return tokens, mask

    # =========================================================================
    # >>> SKIP: per-query gather index into own band-tokens
    # =========================================================================

    def _build_full_pixel_index(self, C, H, W):
        """
        Closed-form gather index for ALL pixels, in pixel order p = h*W + w.

        TokenBuilder.build_tokens flattens as `(c h w) -> row`, i.e. channel-
        major: pixel p's band-tokens live at rows {p + c*H*W : c in 0..C-1},
        strided by H*W (NOT contiguous). Verified numerically against the
        builder's einops.rearrange ordering.

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
                                    (validation: queries == seg_queries).
                           tensor-> the row positions (into the full pixel
                                    grid) that subsample_queries kept, in the
                                    SAME order as the returned queries
                                    (training). Obtained via
                                    subsample_queries(..., return_indices=True).

        Returns:
            idx   : [N_q, C] long  -- rows into image_tokens
            valid : [N_q] bool     -- all True (closed form always resolves)

        NOTE: indices are RELATIVE TO THIS SAMPLE's image_tokens pool. Your
        collate must offset them if it concatenates samples; no offset needed
        if it pads to [B, N, 8] and the model gathers per-sample.
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
        return len(self.s1_image_list)

    def __getitem__(self, index):

        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)

        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label    = torch.from_numpy(label)

        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)

        image_full = torch.cat([image_s2, image_s1], dim=0)  # [15, H, W]
        image      = self._select_channels(image_full)        # [C', H, W]

        if self.split == "train":
            image, label = self._d4_augment(image, label)

        resolution = self.OPTICAL_RESOLUTION
        image_tokens, seg_queries = self._build_tokens(image, label, resolution)

        attention_mask = torch.zeros(image_tokens.shape[0])
        image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)

        if self.reconstruction:
            queries = image_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()
            perm    = torch.randperm(queries.shape[0])[:self.max_tokens_reconstruction]
            queries = queries[perm]
            kept_indices = None  # reconstruction skip not used in this experiment
        else:
            if self.split == "train":
                # >>> SKIP: capture which queries were kept so the gather index
                #           can be selected to match (subsample shuffles order).
                queries, kept_indices = self.token_builder.subsample_queries(
                    seg_queries,
                    max_queries=self.max_tokens_reconstruction,
                    ignore_index=self.IGNORE_INDEX,
                    prioritize_valid=True,
                    return_indices=True,
                )
            else:
                queries = seg_queries
                kept_indices = None  # full pixel grid in order

        queries_mask = torch.zeros(queries.shape[0])

        # >>> SKIP: vectorized per-query gather index (closed form).
        C_img, H_img, W_img = image.shape
        query_token_idx, query_token_valid = self._build_query_token_index(
            C_img, H_img, W_img, kept_indices=kept_indices
        )

        result = {
            "groups": {
                resolution: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "target_resolution": resolution,
            "image":             image,
            # >>> SKIP
            "query_token_idx":   query_token_idx,    # [N_q, bands_per_pixel]
            "query_token_valid": query_token_valid,  # [N_q] bool
        }

        if not self.reconstruction:
            result["label"] = label

        return result

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_tokens(self, image, label, resolution):
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=resolution,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=resolution,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        return image_tokens, queries

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)

        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label    = torch.from_numpy(label)
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)

        image_full = torch.cat([image_s2, image_s1], dim=0)
        image      = self._select_channels(image_full)
        C, H, W    = image.shape

        if self.reconstruction:
            dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
            tokens = self.token_builder.build_tokens(
                image=image, label=dummy_label,
                resolution=self.OPTICAL_RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )
            tokens[:, 4] = tokens[:, 0].clone()
            attention_mask = torch.zeros(tokens.shape[0])
            tokens, attention_mask = self._apply_drop_mask(tokens, attention_mask)
            queries      = tokens.clone()
            queries_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

            # >>> SKIP: reconstruction queries == all tokens; index not
            # meaningful for the seg skip. Provide full-grid index for shape
            # consistency (one entry per pixel), harmless if unused.
            query_token_idx, query_token_valid = self._build_query_token_index(
                C, H, W, kept_indices=None
            )

            return {
                "groups": {self.OPTICAL_RESOLUTION: {
                    "tokens": tokens, "mask": attention_mask, "shape": (C, H, W),
                }},
                "queries": queries, "queries_mask": queries_mask,
                "target_resolution": self.OPTICAL_RESOLUTION,
                "image": image, "image_shape": (C, H, W),
                "n_real": (attention_mask == 0).sum().item(),
                "query_token_idx": query_token_idx,
                "query_token_valid": query_token_valid,
            }
        else:
            image_tokens, queries = self._build_tokens(image, label, self.OPTICAL_RESOLUTION)
            attention_mask = torch.zeros(image_tokens.shape[0])
            image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

            # >>> SKIP: viz uses full seg_queries in pixel order
            query_token_idx, query_token_valid = self._build_query_token_index(
                C, H, W, kept_indices=None
            )

            return {
                "groups": {self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens, "mask": attention_mask, "shape": (C, H, W),
                }},
                "queries": queries, "queries_mask": queries_mask,
                "label": label,
                "target_resolution": self.OPTICAL_RESOLUTION,
                "image": image,
                "query_token_idx": query_token_idx,
                "query_token_valid": query_token_valid,
            }

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_file_lists(self):
        s1_images, s2_images, labels = [], [], []
        print(f"[Sen1Floods11Skip] Loading split file: {self.split_file}")
        with open(self.split_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                s1_filename    = row[0].replace("S1Hand/", "")
                label_filename = row[1].replace("LabelHand/", "")
                s2_filename    = s1_filename.replace("_S1Hand", "_S2Hand")
                s1_images.append(os.path.join(self.data_root, "S1Hand",    s1_filename))
                s2_images.append(os.path.join(self.data_root, "S2Hand",    s2_filename))
                labels.append(   os.path.join(self.data_root, "LabelHand", label_filename))
        return s1_images, s2_images, labels

    def _filter_invalid_samples(self):
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0
        print(f"[Sen1Floods11Skip] Filtering invalid samples...")
        for i in tqdm(range(len(self.label_list)), desc="Checking labels"):
            try:
                with rasterio.open(self.label_list[i]) as src:
                    lbl = src.read(1)
                lbl[lbl == -1] = 255
                if (lbl != 255).sum() > 100:
                    valid_s1.append(self.s1_image_list[i])
                    valid_s2.append(self.s2_image_list[i])
                    valid_labels.append(self.label_list[i])
                else:
                    skipped += 1
            except Exception as e:
                print(f"[Warning] Could not read {self.label_list[i]}: {e}")
                skipped += 1
        print(f"[Sen1Floods11Skip] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list    = valid_labels

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")
        if os.path.exists(norm_file):
            print(f"[Sen1Floods11Skip] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats
        if self.split != "train":
            print(f"[Sen1Floods11Skip] WARNING: No normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(13), "s2_std": torch.ones(13),
                "s1_mean": torch.zeros(2),  "s1_std": torch.ones(2),
            }
        print(f"[Sen1Floods11Skip] Computing normalization from {len(self.s1_image_list)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(13, dtype=torch.float64)
        s2_sq  = torch.zeros(13, dtype=torch.float64)
        s2_n   = torch.zeros(13, dtype=torch.float64)
        s1_sum = torch.zeros(2,  dtype=torch.float64)
        s1_sq  = torch.zeros(2,  dtype=torch.float64)
        s1_n   = torch.zeros(2,  dtype=torch.float64)

        for idx in tqdm(range(len(self.s2_image_list)), desc="Computing normalization"):
            try:
                with rasterio.open(self.s2_image_list[idx]) as src:
                    s2 = src.read().astype(np.float64)
                s2 = np.nan_to_num(s2)
                for c in range(13):
                    valid = s2[c].flatten()
                    valid = valid[valid > 0]
                    if len(valid):
                        s2_sum[c] += valid.sum()
                        s2_sq[c]  += (valid ** 2).sum()
                        s2_n[c]   += len(valid)
            except Exception:
                continue
            try:
                with rasterio.open(self.s1_image_list[idx]) as src:
                    s1 = src.read().astype(np.float64)
                s1 = np.nan_to_num(s1)
                for c in range(2):
                    valid = s1[c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        s1_sum[c] += valid.sum()
                        s1_sq[c]  += (valid ** 2).sum()
                        s1_n[c]   += len(valid)
            except Exception:
                continue

        s2_mean = (s2_sum / s2_n.clamp(min=1)).float()
        s2_std  = ((s2_sq / s2_n.clamp(min=1) - s2_mean.double() ** 2).sqrt()).float()
        s1_mean = (s1_sum / s1_n.clamp(min=1)).float()
        s1_std  = ((s1_sq / s1_n.clamp(min=1) - s1_mean.double() ** 2).sqrt()).float()
        return {"s2_mean": s2_mean, "s2_std": s2_std, "s1_mean": s1_mean, "s1_std": s1_std}

    def _print_norm_stats(self, stats):
        print(f"[Sen1Floods11Skip] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[Sen1Floods11Skip] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[Sen1Floods11Skip] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[Sen1Floods11Skip] S1 std:  {stats['s1_std'].numpy()}")

    def normalize_image(self, s2, s1):
        s2_mean = self.norm_stats["s2_mean"].view(13, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(13, 1, 1)
        s1_mean = self.norm_stats["s1_mean"].view(2, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(2, 1, 1)
        return (s2 - s2_mean) / s2_std, (s1 - s1_mean) / s1_std

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _parse_bands_info(self):
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": data["bandwidth"],
                    "central_wavelength": data["central_wavelength"],
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw    = torch.tensor([b["bandwidth"]          for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]
        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} key={key} not in lookup. "
                    f"Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _shuffle_arrays(arrays: list):
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]
