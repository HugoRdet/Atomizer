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


class Sen1Floods11Dataset(Dataset):
    """
    Sen1Floods11 Dataset — grouped token format (8 columns).
    
    S2 (13 optical bands) + S1 (2 SAR bands), all at 10m, 512×512.
    Both modalities share the same resolution → single group.
    
    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7
    
    Modes:
        - segmentation (default): queries = 1 per pixel, col 4 = class label
        - reconstruction:         queries = 1 per pixel-band, col 4 = reflectance
    
    Single-channel mode:
        When `single_channel` is set in trainer config, only the specified band
        index (0-based into the merged S2+S1 stack) is used. Useful for debugging
        reconstruction and isolating per-band issues.
        Set to -1 or omit to use all bands (default).
    """

    OPTICAL_RESOLUTION = 10.0
    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1

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

        # Initialize TokenBuilder
        self.token_builder = TokenBuilder(look_up)

        # Config parameters
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction = config_model["trainer"].get("mode", "segmentation") == "reconstruction"

        # ── Single-channel mode ─────────────────────────────
        # single_channel: index into the merged [S2(13) + S1(2)] band stack
        #   -1 or absent → all 15 bands (default)
        #    0..14       → only that band
        #   list[int]    → only those bands
        sc = config_model["trainer"].get("single_channel", -1)
        if isinstance(sc, list):
            self.selected_channels = sorted(sc)
        elif isinstance(sc, int) and sc >= 0:
            self.selected_channels = [sc]
        else:
            self.selected_channels = None  # all bands

        if self.selected_channels is not None:
            total = self.NUM_S2_BANDS + self.NUM_S1_BANDS
            for ch in self.selected_channels:
                assert 0 <= ch < total, (
                    f"single_channel index {ch} out of range [0, {total})"
                )
            print(f"[Sen1Floods11] SINGLE-CHANNEL MODE: using band indices {self.selected_channels}")

        if self.reconstruction:
            print(f"[Sen1Floods11] Mode: RECONSTRUCTION (queries = image tokens, col 4 = reflectance)")
        else:
            print(f"[Sen1Floods11] Mode: SEGMENTATION (queries = pixels, col 4 = class label)")

        # Split mapping
        self.split_mapping = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.split_mapping[mode]}_data.csv",
        )

        # Load & filter file lists
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        # Band metadata
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        # Apply single-channel filtering to band metadata
        if self.selected_channels is not None:
            self.bandwidths = self.bandwidths[self.selected_channels]
            self.wavelengths = self.wavelengths[self.selected_channels]
            self.band_names = [self.band_names[i] for i in self.selected_channels]
            self.spectral_indices = self.spectral_indices[self.selected_channels]
            print(f"[Sen1Floods11] After channel selection: {len(self.bandwidths)} bands → "
                  f"{[self.band_names[i] if i < len(self.band_names) else '?' for i in range(len(self.band_names))]}")

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[Sen1Floods11] Loaded {len(self.bandwidths)} bands")
        print(f"[Sen1Floods11] Resolution idx: {self.resolution_idx} "
              f"(GSD={self.OPTICAL_RESOLUTION} m/px, all bands)")
        print(f"[Sen1Floods11] Time idx: -1 (no temporal info, zeroed by encoder)")

    # =========================================================================
    # CHANNEL SELECTION HELPER
    # =========================================================================

    def _select_channels(self, image):
        """
        Apply single-channel selection to a [C, H, W] image tensor.
        Returns the selected subset of channels.
        """
        if self.selected_channels is None:
            return image
        return image[self.selected_channels]  # [len(selected), H, W]

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
 
        # ── Load ────────────────────────────────────────────
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)

        # ── Clean ───────────────────────────────────────────
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label = torch.from_numpy(label)

        # ── Normalize ───────────────────────────────────────
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)
        image_s2 = torch.clamp(image_s2, -10, 10)
        image_s1 = torch.clamp(image_s1, -10, 10)

        # ── Merge & select channels ─────────────────────────
        image_full = torch.cat([image_s2, image_s1], dim=0)  # [15, H, W]
        image = self._select_channels(image_full)              # [C', H, W]

        # ── Build tokens [N, 8] ─────────────────────────────
        resolution = self.OPTICAL_RESOLUTION
        image_tokens, seg_queries = self._build_tokens(image, label, resolution)

        # ── Build queries (mode-dependent) ──────────────────
        if self.reconstruction:
            queries = image_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()  # reflectance → label col
            # Subsample
            perm = torch.randperm(queries.shape[0])[:self.max_tokens_reconstruction]
            queries = queries[perm]
        else:
            queries = self.token_builder.subsample_queries(
                seg_queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )

        # ── Masks ───────────────────────────────────────────
        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask = torch.zeros(queries.shape[0])

        # ── Return ──────────────────────────────────────────
        result = {
            "groups": {
                resolution: {
                    "tokens": image_tokens,
                    "mask": attention_mask,
                    "shape": tuple(image.shape),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": resolution,
            "image": image,
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
        """
        Viz sample — mode-aware.
        Reconstruction: all tokens as queries, col 4 = reflectance.
        Segmentation: all pixels as queries, includes label + image.
        """
    
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
        label = torch.from_numpy(label)

        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)
        image_s2 = torch.clamp(image_s2, -10, 10)
        image_s1 = torch.clamp(image_s1, -10, 10)

        image_full = torch.cat([image_s2, image_s1], dim=0)
        image = self._select_channels(image_full)
        C, H, W = image.shape

        if self.reconstruction:
            # All image tokens as queries
            dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
            tokens = self.token_builder.build_tokens(
                image=image,
                label=dummy_label,
                resolution=self.OPTICAL_RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )
            tokens[:, 4] = tokens[:, 0].clone()

            queries = tokens.clone()
            queries_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)
            attention_mask = torch.zeros(tokens.shape[0])

            return {
                "groups": {
                    self.OPTICAL_RESOLUTION: {
                        "tokens": tokens,
                        "mask": attention_mask,
                        "shape": (C, H, W),
                    },
                },
                "queries": queries,
                "queries_mask": queries_mask,
                "target_resolution": self.OPTICAL_RESOLUTION,
                "image": image,
                "image_shape": (C, H, W),
                "n_real": tokens.shape[0],
            }
        else:
            # Segmentation: all pixels as queries, no subsampling
            image_tokens, queries = self._build_tokens(image, label, self.OPTICAL_RESOLUTION)
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
            attention_mask = torch.zeros(image_tokens.shape[0])

            return {
                "groups": {
                    self.OPTICAL_RESOLUTION: {
                        "tokens": image_tokens,
                        "mask": attention_mask,
                        "shape": (C, H, W),
                    },
                },
                "queries": queries,
                "queries_mask": queries_mask,
                "label": label,
                "target_resolution": self.OPTICAL_RESOLUTION,
                "image": image,
            }

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_file_lists(self):
        s1_images, s2_images, labels = [], [], []
        print(f"[Sen1Floods11] Loading split file: {self.split_file}")

        with open(self.split_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                s1_filename = row[0].replace("S1Hand/", "")
                label_filename = row[1].replace("LabelHand/", "")
                s2_filename = s1_filename.replace("_S1Hand", "_S2Hand")

                s1_images.append(os.path.join(self.data_root, "S1Hand", s1_filename))
                s2_images.append(os.path.join(self.data_root, "S2Hand", s2_filename))
                labels.append(os.path.join(self.data_root, "LabelHand", label_filename))

        return s1_images, s2_images, labels

    def _filter_invalid_samples(self):
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0

        print(f"[Sen1Floods11] Filtering invalid samples...")
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

        print(f"[Sen1Floods11] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list = valid_labels

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[Sen1Floods11] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[Sen1Floods11] WARNING: No normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(13), "s2_std": torch.ones(13),
                "s1_mean": torch.zeros(2),  "s1_std": torch.ones(2),
            }

        print(f"[Sen1Floods11] Computing normalization from {len(self.s1_image_list)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[Sen1Floods11] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(13, dtype=torch.float64)
        s2_sq  = torch.zeros(13, dtype=torch.float64)
        s2_n   = torch.zeros(13, dtype=torch.float64)
        s1_sum = torch.zeros(2, dtype=torch.float64)
        s1_sq  = torch.zeros(2, dtype=torch.float64)
        s1_n   = torch.zeros(2, dtype=torch.float64)

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

        return {
            "s2_mean": s2_mean, "s2_std": s2_std,
            "s1_mean": s1_mean, "s1_std": s1_std,
        }

    def _print_norm_stats(self, stats):
        print(f"[Sen1Floods11] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[Sen1Floods11] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[Sen1Floods11] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[Sen1Floods11] S1 std:  {stats['s1_std'].numpy()}")

    def normalize_image(self, s2, s1):
        """Normalize S2 and S1 separately using precomputed stats."""
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

        bw = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[Sen1Floods11] Band order:")
        for b in all_bands:
            tag = " (SAR)" if b["bandwidth"] < 0 or b["central_wavelength"] < 0 else ""
            print(f"  idx={b['idx']:2d}: {b['name']:4s} → bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}{tag}")

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