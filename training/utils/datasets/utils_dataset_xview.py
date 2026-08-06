"""
xView2 Damage Assessment — Atomiser Dataset
=============================================

5-class damage segmentation matching PANGAEA's xView2 setup:
    0 = background (no building)
    1 = no-damage   (also: un-classified)
    2 = minor-damage
    3 = major-damage
    4 = destroyed

Multi-temporal segmentation with T=2 frames (pre + post disaster),
3 RGB bands at 0.5 m native resolution.

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

Output format (compatible with Model_SenFlood — segmentation):
    {
        "groups": {
            0.5: {
                "tokens": [N, 8],          # T*C*H*W tokens (subsampled)
                "mask":   [N],
                "shape":  (T*C, H, W),     # for reference
            },
        },
        "queries":           [M, 8]        # per-pixel queries (M = H*W or subsampled)
        "queries_mask":      [M],
        "label":             [H, W]        # 5-class target
        "target_resolution": 0.5,
        "image":             [T, C, H, W]  # for viz/debug
    }

Augmentations (training only): D4 group + random crop with building preference.
Val: deterministic center crop. Test: full 1024×1024 (sliding-window evaluation
handled by trainer / inference loop).

Splits (matching PANGAEA's xView2.py):
    - Train: stratified 90% of (train + tier3) by disaster name
    - Val:   stratified 10% of (train + tier3) by disaster name
    - Test:  the test/ directory
    - random_state=23 to match PANGAEA's split exactly
"""

import json
import os
import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .token_grouping import *
from .token_builder import TokenBuilder


# Constants matching the baseline dataset
NUM_CLASSES = 5
IGNORE_INDEX = 255
NUM_BANDS = 3
NUM_FRAMES = 2

# PANGAEA's BGR statistics (cv2 imread convention)
DATA_MEAN_BGR = [66.7703, 88.4452, 85.1047]
DATA_STD_BGR  = [48.3066, 51.9129, 62.7612]


class XView2Dataset(Dataset):
    """xView2 damage assessment dataset for Atomiser."""

    OPTICAL_RESOLUTION = 0.5     # native xView2 GSD (sub-meter aerial)
    NATIVE_SIZE = 1024
    PATCH_SIZE = 512             # default training crop
    TASK_NAME = "segmentation"

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "validation",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/xview",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        crop_size: int = 512,
        oversample_building_damage: bool = True,
        random_state: int = 23,
        full_image: bool = False,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        root_path="./data/xview"

        # NOTE: was hardcoded to "./data/xview" in a prior edit, which
        # silently ignored the root_path argument (and thus --data_dir).
        # Reverted to actually use the constructor argument.
        self.root_path     = root_path
        self.split         = mode
        self.look_up       = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config
        self.crop_size     = crop_size
        self.oversample_building_damage = (
            oversample_building_damage and (mode == "train")
        )
        self.random_state  = random_state
        self.full_image    = full_image  # if True, skip cropping (sliding-window test)

        self.token_builder = TokenBuilder(look_up)

        # Trainer config
        self.nb_tokens                  = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction  = config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction = (
            config_model["trainer"].get("mode", "segmentation") == "reconstruction"
        )

        # Normalization tensors (BGR ordering, matching PANGAEA)
        self.norm_mean = torch.tensor(DATA_MEAN_BGR, dtype=torch.float32).view(3, 1, 1)
        self.norm_std  = torch.tensor(DATA_STD_BGR,  dtype=torch.float32).view(3, 1, 1)

        # Band metadata (3 RGB bands, reusing standard S2 RGB wavelengths)
        self.bands_info = dataset_config["bands_xview"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)

        # Sample list
        self.all_files = self._get_all_files()

        if self.full_image:
            print(f"[xView2-Atom] split={mode}, samples={len(self.all_files)} "
                  f"(FULL IMAGE mode, 1024×1024)")
        else:
            print(f"[xView2-Atom] split={mode}, samples={len(self.all_files)}")
        print(f"[xView2-Atom] crop_size={self.crop_size}, "
              f"oversample={self.oversample_building_damage}")
        print(f"[xView2-Atom] num_classes={NUM_CLASSES} (PANGAEA setup)")
        print(f"[xView2-Atom] T=2 frames (pre, post) at {self.OPTICAL_RESOLUTION}m")
        print(f"[xView2-Atom] resolution_idx: {self.resolution_idx}")

    # ─────────────────────────────────────────────────────────────────────
    # SPLIT HANDLING — matches PANGAEA's xView2.py exactly
    # ─────────────────────────────────────────────────────────────────────

    def _get_all_files(self):
        if self.split == "test":
            data_dirs = [os.path.join(self.root_path, "test")]
        else:
            data_dirs = [os.path.join(self.root_path, d) for d in ["train", "tier3"]]

        all_files = []
        for d in data_dirs:
            images_dir = os.path.join(d, "images")
            if not os.path.isdir(images_dir):
                print(f"[xView2-Atom] WARN: {images_dir} not found")
                continue
            for f in sorted(os.listdir(images_dir)):
                if "_pre_disaster.png" in f:
                    all_files.append(os.path.join(images_dir, f))

        # Filter out samples with zero building pixels (cached)
        all_files = self._filter_no_building_samples(all_files)

        if self.split != "test":
            train_idxs, val_idxs = self._stratified_split(all_files)
            chosen = train_idxs if self.split == "train" else val_idxs

            if self.split == "train" and self.oversample_building_damage:
                chosen = self._oversample_building_files(all_files, chosen)

            all_files = [all_files[i] for i in chosen]

        return all_files

    def _filter_no_building_samples(self, all_files):
        """Drop samples whose pre-disaster mask has zero building pixels.

        Reuses the same cache as the baseline dataset
        (_xview_buildings_cache.json at the data root).
        """
        cache_path = os.path.join(self.root_path, "_xview_buildings_cache.json")

        cached = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
            except Exception:
                cached = {}

        kept = []
        to_scan = []
        for fn in all_files:
            key = os.path.relpath(fn, self.root_path)
            if key in cached:
                if cached[key]:
                    kept.append(fn)
            else:
                to_scan.append((fn, key))

        if to_scan:
            print(f"[xView2-Atom] Scanning {len(to_scan)} masks for buildings "
                  f"(one-time, cached after)...")
            from tqdm import tqdm
            for fn, key in tqdm(to_scan, desc="Filter"):
                msk_pre_path = fn.replace("/images/", "/masks/")
                msk = cv2.imread(msk_pre_path, cv2.IMREAD_UNCHANGED)
                has_building = bool(msk is not None and (msk > 0).any())
                cached[key] = has_building
                if has_building:
                    kept.append(fn)
            try:
                with open(cache_path, "w") as f:
                    json.dump(cached, f)
                print(f"[xView2-Atom] Saved building-filter cache to {cache_path}")
            except Exception as e:
                print(f"[xView2-Atom] WARN: could not save cache: {e}")

        n_dropped = len(all_files) - len(kept)
        if n_dropped > 0:
            print(f"[xView2-Atom] Filtered out {n_dropped} samples with zero "
                  f"building pixels ({len(kept)} retained).")
        return kept

    def _stratified_split(self, all_files):
        disaster_names = [pathlib.Path(p).name.split("_")[0] for p in all_files]
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=self.random_state,
            stratify=disaster_names,
        )
        return train_idxs, val_idxs

    def _oversample_building_files(self, all_files, train_idxs):
        train_idx_set = set(train_idxs.tolist())
        file_classes = []
        for i, fn in enumerate(all_files):
            fl = np.zeros((4,), dtype=bool)
            if i in train_idx_set:
                msk_path = (fn.replace("/images/", "/masks/")
                              .replace("_pre_disaster", "_post_disaster"))
                msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
                if msk is not None:
                    for c in range(1, 5):
                        fl[c - 1] = (msk == c).any()
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)

        new_train_idxs = []
        for i in train_idxs:
            new_train_idxs.append(i)
            if file_classes[i, 1:].max():
                new_train_idxs.append(i)
            if file_classes[i, 1:3].max():
                new_train_idxs.append(i)
        return np.asarray(new_train_idxs)

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION + CROP
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 transform applied identically to image and label.

        image: [T, C, H, W]   label: [H, W]
        """
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[3])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[2, 3])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _crop_prefer_buildings(
        image: torch.Tensor, label: torch.Tensor,
        crop_size: int, random: bool, max_retries: int = 10,
    ):
        """Crop biased toward regions containing building pixels.

        image: [T, C, H, W], label: [H, W].
        Building pixel = label > 0 (any damage class).
        """
        T, C, H, W = image.shape
        assert H >= crop_size and W >= crop_size

        def _take(top, left):
            img = image[:, :, top:top + crop_size, left:left + crop_size]
            lbl = label[top:top + crop_size, left:left + crop_size]
            return img, lbl

        def _has_building(lbl):
            return (lbl > 0).any().item()

        if random:
            for _ in range(max_retries):
                top  = torch.randint(0, H - crop_size + 1, (1,)).item()
                left = torch.randint(0, W - crop_size + 1, (1,)).item()
                img, lbl = _take(top, left)
                if _has_building(lbl):
                    return img, lbl
        else:
            top  = (H - crop_size) // 2
            left = (W - crop_size) // 2
            img, lbl = _take(top, left)
            if _has_building(lbl):
                return img, lbl

        # Fallback: center on a building pixel
        building_mask = (label > 0)
        if building_mask.any():
            ys, xs = torch.where(building_mask)
            mid = ys.shape[0] // 2
            cy, cx = int(ys[mid].item()), int(xs[mid].item())
            top  = max(0, min(H - crop_size, cy - crop_size // 2))
            left = max(0, min(W - crop_size, cx - crop_size // 2))
            return _take(top, left)
        return _take((H - crop_size) // 2, (W - crop_size) // 2)

    # ─────────────────────────────────────────────────────────────────────
    # >>> SKIP: per-query gather index into own band x time atoms
    # Mirrors BioMasstersSkipDataset's _build_full_pixel_index /
    # _build_query_token_index, simplified to a single spectral block: xView2
    # has one modality (RGB) rather than BioMassters' S2+S1 split, so there's
    # no second "off" block to concatenate -- just T*C atoms per pixel.
    #
    # Token layout produced above in __getitem__/get_viz_sample:
    #   image_tokens = cat([ frame_0(c h w), frame_1(c h w) ])
    # i.e. frame-major, and within a frame TokenBuilder.build_tokens orders
    # tokens channel-major (pixel p -> row p + c*HW), the SAME convention
    # BioMasstersSkipDataset's docstring documents for its S2/S1 blocks
    # (shared TokenBuilder, so the ordering convention is identical here).
    # For pixel p, band c, frame t: index = t*C*HW + c*HW + p.
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_full_pixel_index(T, C, H, W):
        """
        For every pixel p in [0, H*W), returns the indices into the flat
        `image_tokens` array of every (band, time) atom at that pixel --
        i.e. everything the SKIP decoder needs to gather to reconstruct
        pixel p directly from its own input atoms.

        Returns: LongTensor [H*W, T*C]
        """
        HW = H * W
        p = torch.arange(HW)
        t = torch.arange(T).view(T, 1, 1)
        c = torch.arange(C).view(1, C, 1)
        idx = (t * C * HW + c * HW).reshape(-1, 1) + p.view(1, -1)  # [T*C, HW]
        return idx.t().contiguous()  # [HW, T*C]

    def _build_query_token_index(self, T, C, H, W, kept_indices=None):
        """
        kept_indices=None means queries were NOT subsampled (val/test: all
        H*W pixels), so the full per-pixel index is used as-is. Otherwise
        kept_indices are row indices into the pre-subsample, row-major
        [0, H*W) pixel ordering returned by _subsample_queries_building_priority
        (or, previously, subsample_queries(..., return_indices=True)) -- same
        ordering _build_full_pixel_index is built over, so full[kept_indices]
        lines up 1:1 with the surviving queries.

        `valid` is unconditionally True: every index here always points to
        a real position in image_tokens (this dataset never pads/replicates
        frames the way BioMassters' variable-length time series does), so
        there's nothing for a per-atom validity flag to encode here -- kept
        only for interface parity with BioMasstersSkipDataset's SKIP fields.
        """
        full = self._build_full_pixel_index(T, C, H, W)
        idx = full if kept_indices is None else full[kept_indices]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # ─────────────────────────────────────────────────────────────────────
    # QUERY SUBSAMPLING: building-priority
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _subsample_queries_building_priority(queries: torch.Tensor,
                                              target: torch.Tensor,
                                              max_queries: int):
        """
        Training-time query subsampling that prioritizes building/damage
        pixels (label > 0) over background (label == 0), filling any
        remaining budget with background once all building pixels are kept.

        Replaces token_builder.subsample_queries(..., prioritize_valid=True)
        for this dataset: that flag only distinguishes ignore_index from
        everything else, but xView2's target NEVER contains IGNORE_INDEX
        (see module docstring -- invalid mask values are mapped to 0/
        background in __getitem__), so prioritize_valid=True was a no-op
        here. This gives an actual class-aware priority instead, consistent
        with the building bias already applied elsewhere in this pipeline
        (_crop_prefer_buildings, _oversample_building_files).

        queries: [H*W, 8], row-major over pixels (row p == target.flatten()[p]).
        target:  [H, W] class labels for this sample.

        Returns (queries_subsampled, kept_indices) where kept_indices are
        row indices into the pre-subsample [0, H*W) ordering -- same
        convention _build_query_token_index expects for query_token_idx.
        """
        flat_labels = target.reshape(-1)  # [H*W], row-major, matches queries' row order
        building_idx = (flat_labels > 0).nonzero(as_tuple=True)[0]
        background_idx = (flat_labels == 0).nonzero(as_tuple=True)[0]

        n_building = building_idx.shape[0]
        if n_building >= max_queries:
            # More building pixels than budget: subsample among them only.
            perm = torch.randperm(n_building)[:max_queries]
            kept_indices = building_idx[perm]
        else:
            # Keep every building pixel, fill the rest with random background.
            n_background_needed = max_queries - n_building
            if background_idx.shape[0] > n_background_needed:
                perm = torch.randperm(background_idx.shape[0])[:n_background_needed]
                background_kept = background_idx[perm]
            else:
                background_kept = background_idx  # fewer background px than budget: keep all
            kept_indices = torch.cat([building_idx, background_kept])

        return queries[kept_indices], kept_indices

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        fn_pre = self.all_files[index]
        fn_post = fn_pre.replace("_pre_", "_post_")

        # ── Load images (cv2: BGR, [H, W, 3], uint8) ──
        img_pre  = cv2.imread(fn_pre,  cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn_post, cv2.IMREAD_COLOR)
        if img_pre is None or img_post is None:
            raise RuntimeError(f"[xView2-Atom] Could not read image: {fn_pre}")

        # ── Load post mask (5-class target) ──
        msk_post_path = (fn_pre.replace("/images/", "/masks/")
                              .replace("_pre_disaster", "_post_disaster"))
        msk_post = cv2.imread(msk_post_path, cv2.IMREAD_UNCHANGED)
        if msk_post is None:
            raise RuntimeError(
                f"[xView2-Atom] Could not read post mask: {msk_post_path}"
            )

        # ── Stack frames: [T=2, H, W, C=3] then permute to [T, C, H, W] ──
        img = np.stack([img_pre, img_post], axis=0)
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float()  # [2, 3, H, W]

        # ── Build target (PANGAEA setup: msk_post directly as 5-class) ──
        msk_post = msk_post.astype(np.int64)
        target = np.where(
            (msk_post >= 0) & (msk_post < NUM_CLASSES),
            msk_post,
            0,
        )
        target = torch.from_numpy(target).long()  # [H, W]

        # ── Normalize ──
        img = (img - self.norm_mean.unsqueeze(0)) / self.norm_std.unsqueeze(0)

        # ── Crop ──
        if not self.full_image:
            img, target = self._crop_prefer_buildings(
                img, target, self.crop_size, random=(self.split == "train"),
            )

        # ── Augment (training only) ──
        if self.split == "train":
            img, target = self._d4_augment(img, target)

        T, C, H, W = img.shape

        # ── Build tokens for each (T, C, H, W) slot ──
        # Atomiser needs per-(pixel, band, time) tokens. We loop over T frames
        # and concatenate, assigning time_idx = 0 for pre, 1 for post.
        all_tokens = []
        for t in range(T):
            # token_builder.build_tokens expects [C, H, W] image and [H, W] label.
            # We build tokens with a per-frame time_idx.
            frame_tokens = self.token_builder.build_tokens(
                image=img[t],                     # [C, H, W]
                label=target,                     # [H, W] — same target for all frames
                resolution=self.OPTICAL_RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=t,                       # 0 = pre, 1 = post
            )
            all_tokens.append(frame_tokens)
        image_tokens = torch.cat(all_tokens, dim=0)   # [T*C*H*W, 8]

        # ── Build queries (per-pixel, mode-dependent) ──
        # Queries are pixel positions; for segmentation, col 4 = class label.
        # We use the first spectral_idx as a placeholder (queries don't carry
        # spectral info — they're position queries that retrieve from tokens).
        first_spectral_idx = int(self.spectral_indices[0].item())
        seg_queries = self.token_builder.build_queries(
            label=target,
            resolution=self.OPTICAL_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=-1,                          # query has no specific time
        )

        # ── No token subsampling at dataset level ──
        # Atomiser's geographic Voronoi pruning (in the model) handles
        # token selection per latent. Passing the full token stream is
        # what working datasets (BurnScars, Sen1Floods11, PASTIS, MADOS)
        # do. Subsampling at dataset level discards information the
        # pruning step is designed to use.

        # ── Subsample queries (training only) ──
        # kept_indices tracks which of the seg_queries' H*W rows survived
        # subsampling -- needed below to build query_token_idx so the SKIP
        # gather index stays aligned with the (possibly subsampled) queries.
        # None means "not subsampled" (reconstruction mode, or val/test).
        kept_indices = None
        if self.reconstruction:
            queries = image_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()
            perm = torch.randperm(queries.shape[0])[:self.max_tokens_reconstruction]
            queries = queries[perm]
        else:
            if self.split == "train":
                # Building-priority subsampling: keep every building/damage
                # pixel first, fill remaining budget with background. See
                # _subsample_queries_building_priority's docstring for why
                # this replaces token_builder.subsample_queries(...,
                # prioritize_valid=True) for this dataset specifically.
                queries, kept_indices = self._subsample_queries_building_priority(
                    seg_queries, target, self.max_tokens_reconstruction,
                )
            else:
                # Val/test: all pixels for accurate evaluation
                queries = seg_queries

        # ── Masks ──
        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(queries.shape[0])

        result = {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (T * C, H, W),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             img,
        }

        if not self.reconstruction:
            result["label"] = target
            # SKIP decoder pixel-gather index. NOT built for reconstruction
            # mode: there, each "query" IS one atom (one band/time/pixel,
            # copied straight from image_tokens), not a per-pixel query
            # needing all of that pixel's atoms gathered together -- the
            # concept doesn't map onto that branch, so it's intentionally
            # skipped rather than built with wrong semantics.
            query_token_idx, query_token_valid = self._build_query_token_index(
                T, C, H, W, kept_indices=kept_indices,
            )
            result["query_token_idx"]   = query_token_idx
            result["query_token_valid"] = query_token_valid

        return result

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE (no augmentation, deterministic)
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        fn_pre = self.all_files[index]
        fn_post = fn_pre.replace("_pre_", "_post_")

        img_pre  = cv2.imread(fn_pre,  cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn_post, cv2.IMREAD_COLOR)
        msk_post_path = (fn_pre.replace("/images/", "/masks/")
                              .replace("_pre_disaster", "_post_disaster"))
        msk_post = cv2.imread(msk_post_path, cv2.IMREAD_UNCHANGED)

        img = np.stack([img_pre, img_post], axis=0)
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        msk_post = msk_post.astype(np.int64)
        target = np.where(
            (msk_post >= 0) & (msk_post < NUM_CLASSES),
            msk_post, 0,
        )
        target = torch.from_numpy(target).long()
        img = (img - self.norm_mean.unsqueeze(0)) / self.norm_std.unsqueeze(0)

        # Deterministic center crop (no augmentation)
        if not self.full_image:
            img, target = self._crop_prefer_buildings(
                img, target, self.crop_size, random=False,
            )
        T, C, H, W = img.shape

        all_tokens = []
        for t in range(T):
            frame_tokens = self.token_builder.build_tokens(
                image=img[t], label=target,
                resolution=self.OPTICAL_RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=t,
            )
            all_tokens.append(frame_tokens)
        image_tokens = torch.cat(all_tokens, dim=0)

        first_spectral_idx = int(self.spectral_indices[0].item())
        queries = self.token_builder.build_queries(
            label=target,
            resolution=self.OPTICAL_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=-1,
        )

        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask   = torch.zeros(queries.shape[0], dtype=torch.bool)

        # No subsampling here (viz uses all H*W pixels), so kept_indices=None
        # -> the full, unsubsampled per-pixel gather index.
        query_token_idx, query_token_valid = self._build_query_token_index(
            T, C, H, W, kept_indices=None,
        )

        return {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (T * C, H, W),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             target,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             img,
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        """Parse bands_xview entries from the dataset config.

        Note on band ordering: cv2.imread returns BGR. So channel 0 is Blue,
        1 is Green, 2 is Red. The bands_xview YAML must match this order
        (idx=0 → Blue, idx=1 → Green, idx=2 → Red) for spectral_idx assignment
        to be correct.
        """
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

        if len(all_bands) != NUM_BANDS:
            raise ValueError(
                f"[xView2-Atom] Expected {NUM_BANDS} bands in bands_xview, "
                f"got {len(all_bands)}: {[b['name'] for b in all_bands]}"
            )

        bw    = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[xView2-Atom] Band order (BGR convention from cv2):")
        for b in all_bands:
            print(f"  idx={b['idx']}: {b['name']:8s} → "
                  f"bw={b['bandwidth']:3d}, wl={b['central_wavelength']:3d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[xView2-Atom] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)
