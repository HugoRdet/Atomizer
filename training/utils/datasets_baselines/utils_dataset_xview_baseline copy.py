"""
xView2 Damage Assessment — Baseline Dataset
=============================================

4-class damage classification at known building locations (Variant B).

Task framing:
    - Input:  pre + post RGB images, [T=2, C=3, H, W]
    - Target: 4-class damage at building pixels, IGNORE elsewhere
              {0=no-damage, 1=minor, 2=major, 3=destroyed, 255=ignore}
    - The pre-disaster building footprint mask is used to gate predictions —
      only pixels with buildings (msk_pre == 1) get a damage label;
      non-building pixels are masked with IGNORE_INDEX.

This is *damage classification given known building locations*, which isolates
the change-detection capability of the model from the building-detection task.

Splits (matching PANGAEA's xView2.py):
    - Train: stratified 90% of (train + tier3) by disaster name
    - Val:   stratified 10% of (train + tier3) by disaster name
    - Test:  the test/ directory
    - Stratification uses sklearn.train_test_split(random_state=23)

Output format:
    {
        "image":  [T=2, C=3, H, W]  float32 (BGR ordering, post-norm)
        "target": [H, W]             long {0..3, 255}
        "metadata": {...}
    }

Augmentations (training only): D4 group on image (T,H,W rotations/flips)
+ random 512×512 crop. Val/test: deterministic 512×512 center crop.

Notes on PANGAEA compatibility:
    - cv2.imread returns BGR (not RGB). We keep BGR and use PANGAEA's
      precomputed BGR mean/std for normalization.
    - PANGAEA shape: [C, T, H, W]. Ours: [T, C, H, W] (matches our other
      multi-temporal datasets like PASTIS).
"""

import os
import pathlib
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# Damage classes after subtracting 1 (we drop background, build target with IGNORE)
NUM_CLASSES = 4
IGNORE_INDEX = 255

# PANGAEA's BGR statistics (post-disaster RGB images, computed on training set)
DATA_MEAN_BGR = [66.7703, 88.4452, 85.1047]
DATA_STD_BGR  = [48.3066, 51.9129, 62.7612]

# PANGAEA's class distribution (full image, all 5 classes)
# 0=Background, 1=NoDamage, 2=Minor, 3=Major, 4=Destroyed
PANGAEA_DISTRIBUTION = [0.9415, 0.0448, 0.0049, 0.0057, 0.0031]


class XView2BaselineDataset(Dataset):
    """xView2 damage assessment baseline dataset (4-class, building-gated)."""

    NATIVE_SIZE = 1024

    def __init__(
        self,
        root_path: str = "./data/xView2",
        mode: str = "train",
        crop_size: int = 512,
        augment: bool = True,
        oversample_building_damage: bool = True,
        random_state: int = 23,
    ):
        super().__init__()
        assert mode in ("train", "validation", "test"), f"Unknown mode: {mode}"

        self.root_path = root_path
        self.split     = mode
        self.crop_size = crop_size
        self.augment   = augment and (mode == "train")
        self.oversample_building_damage = oversample_building_damage and (mode == "train")
        self.random_state = random_state

        # Normalization tensors
        self.norm_mean = torch.tensor(DATA_MEAN_BGR, dtype=torch.float32).view(3, 1, 1)
        self.norm_std  = torch.tensor(DATA_STD_BGR,  dtype=torch.float32).view(3, 1, 1)

        # Build file list
        self.all_files = self._get_all_files()

        print(f"[xView2-BL] split={mode}, samples={len(self.all_files)}")
        print(f"[xView2-BL] crop_size={self.crop_size}, augment={self.augment}, "
              f"oversample={self.oversample_building_damage}")
        print(f"[xView2-BL] num_classes={NUM_CLASSES} "
              f"(0=NoDamage, 1=Minor, 2=Major, 3=Destroyed)")
        print(f"[xView2-BL] IGNORE_INDEX={IGNORE_INDEX} (non-building pixels)")

    # ─────────────────────────────────────────────────────────────────────
    # SPLIT HANDLING — matches PANGAEA's xView2.py
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
                print(f"[xView2-BL] WARN: {images_dir} not found")
                continue
            for f in sorted(os.listdir(images_dir)):
                if "_pre_disaster.png" in f:
                    all_files.append(os.path.join(images_dir, f))

        # Filter out samples with zero building pixels — these produce
        # all-IGNORE targets which give NaN cross-entropy loss.
        # Cache the filter result so we only scan masks once.
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

        These samples would produce all-IGNORE targets at any crop, leading
        to NaN cross-entropy loss. Caches result to avoid rescanning ~10k
        masks on every dataset construction.
        """
        cache_path = os.path.join(self.root_path, "_xview_buildings_cache.json")

        # Try to use cache
        cached = {}
        if os.path.exists(cache_path):
            try:
                import json
                with open(cache_path) as f:
                    cached = json.load(f)
            except Exception:
                cached = {}

        # Determine which files need scanning
        kept = []
        to_scan = []
        for fn in all_files:
            key = os.path.relpath(fn, self.root_path)
            if key in cached:
                if cached[key]:
                    kept.append(fn)
                # else: dropped — has no buildings
            else:
                to_scan.append((fn, key))

        # Scan uncached files
        if to_scan:
            print(f"[xView2-BL] Scanning {len(to_scan)} masks for buildings "
                  f"(one-time, cached after)...")
            from tqdm import tqdm
            for fn, key in tqdm(to_scan, desc="Filter"):
                msk_pre_path = fn.replace("/images/", "/masks/")
                msk = cv2.imread(msk_pre_path, cv2.IMREAD_UNCHANGED)
                has_building = bool(msk is not None and (msk > 0).any())
                cached[key] = has_building
                if has_building:
                    kept.append(fn)
            # Save cache (best-effort)
            try:
                import json
                with open(cache_path, "w") as f:
                    json.dump(cached, f)
                print(f"[xView2-BL] Saved building-filter cache to {cache_path}")
            except Exception as e:
                print(f"[xView2-BL] WARN: could not save cache: {e}")

        n_dropped = len(all_files) - len(kept)
        if n_dropped > 0:
            print(f"[xView2-BL] Filtered out {n_dropped} samples with zero "
                  f"building pixels ({len(kept)} retained).")
        return kept

    def _stratified_split(self, all_files):
        """Stratify by disaster name (e.g. "hurricane-harvey", "guatemala-volcano").

        Disaster name = first underscore-separated chunk of the filename.
        Same stratification as PANGAEA (random_state=23).
        """
        disaster_names = [pathlib.Path(p).name.split("_")[0] for p in all_files]
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=self.random_state,
            stratify=disaster_names,
        )
        return train_idxs, val_idxs

    def _oversample_building_files(self, all_files, train_idxs):
        """Replicate PANGAEA's oversampling:
           - Image with any building damage → included 2×
           - Image with minor or major damage → included 3× (these are
             hardest classes per the xView2 first-place solution).
        """
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
            # Any building damage → 2× total
            if file_classes[i, 1:].max():
                new_train_idxs.append(i)
            # Minor or major → 3× total (hardest classes)
            if file_classes[i, 1:3].max():
                new_train_idxs.append(i)
        return np.asarray(new_train_idxs)

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 transform applied identically to image and label.

        image: [T, C, H, W]
        label: [H, W]
        """
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[3])  # flip W
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[2, 3])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _crop(image: torch.Tensor, label: torch.Tensor,
              crop_size: int, random: bool):
        """Spatial crop. image: [T, C, H, W], label: [H, W]."""
        T, C, H, W = image.shape
        assert H >= crop_size and W >= crop_size
        if random:
            top  = torch.randint(0, H - crop_size + 1, (1,)).item()
            left = torch.randint(0, W - crop_size + 1, (1,)).item()
        else:
            top  = (H - crop_size) // 2
            left = (W - crop_size) // 2
        image = image[:, :, top:top + crop_size, left:left + crop_size]
        label = label[top:top + crop_size, left:left + crop_size]
        return image, label

    @staticmethod
    def _crop_prefer_buildings(
        image: torch.Tensor, label: torch.Tensor,
        crop_size: int, random: bool,
        max_retries: int = 10,
    ):
        """
        Crop with a preference for crops that contain building pixels.

        At training (random=True): retry random crops up to max_retries
        times until a crop contains at least one valid (non-IGNORE) pixel.
        If all retries fail, fall back to a building-centered crop.

        At val/test (random=False): start with center crop. If it has no
        valid pixels, fall back to a building-centered crop.

        This prevents NaN losses from samples where the random/center crop
        lands entirely on background (no buildings → all IGNORE → CE loss
        is NaN over an empty set).
        """
        T, C, H, W = image.shape
        assert H >= crop_size and W >= crop_size

        def _take(top, left):
            img = image[:, :, top:top + crop_size, left:left + crop_size]
            lbl = label[top:top + crop_size, left:left + crop_size]
            return img, lbl

        def _has_valid(lbl):
            return (lbl != IGNORE_INDEX).any().item()

        if random:
            for _ in range(max_retries):
                top  = torch.randint(0, H - crop_size + 1, (1,)).item()
                left = torch.randint(0, W - crop_size + 1, (1,)).item()
                img, lbl = _take(top, left)
                if _has_valid(lbl):
                    return img, lbl
        else:
            top  = (H - crop_size) // 2
            left = (W - crop_size) // 2
            img, lbl = _take(top, left)
            if _has_valid(lbl):
                return img, lbl

        # Fallback: find ANY building pixel in the full image and center crop on it.
        building_mask = (label != IGNORE_INDEX)
        if building_mask.any():
            ys, xs = torch.where(building_mask)
            # Pick a building pixel near the median to avoid edge cases
            mid = ys.shape[0] // 2
            cy, cx = int(ys[mid].item()), int(xs[mid].item())
            top  = max(0, min(H - crop_size, cy - crop_size // 2))
            left = max(0, min(W - crop_size, cx - crop_size // 2))
            return _take(top, left)

        # No building anywhere in the source image — fall back to original crop.
        # Trainer must handle all-IGNORE batches.
        return _take((H - crop_size) // 2, (W - crop_size) // 2)

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        fn_pre = self.all_files[idx]
        fn_post = fn_pre.replace("_pre_", "_post_")

        # ── Load images (cv2 returns BGR, [H, W, 3], uint8) ──
        img_pre  = cv2.imread(fn_pre,  cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn_post, cv2.IMREAD_COLOR)
        if img_pre is None or img_post is None:
            raise RuntimeError(f"[xView2-BL] Could not read image: {fn_pre}")

        # ── Load masks ──
        msk_pre_path  = fn_pre.replace("/images/", "/masks/")
        msk_post_path = (fn_pre.replace("/images/", "/masks/")
                              .replace("_pre_disaster", "_post_disaster"))
        msk_pre  = cv2.imread(msk_pre_path,  cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(msk_post_path, cv2.IMREAD_UNCHANGED)
        if msk_pre is None or msk_post is None:
            raise RuntimeError(
                f"[xView2-BL] Could not read masks for {fn_pre}\n"
                f"  pre:  {msk_pre_path}\n"
                f"  post: {msk_post_path}"
            )

        # ── Stack frames: [T=2, H, W, C=3] then permute to [T, C, H, W] ──
        img = np.stack([img_pre, img_post], axis=0)            # [2, H, W, 3]
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float() # [2, 3, H, W]

        # ── Build target: 4-class damage at building pixels, IGNORE elsewhere ──
        # msk_post in {0..4}: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
        # msk_pre  in {0, 1}: building footprint
        # We want target in {0..3, 255}: 0..3 = damage classes, 255 = ignore
        msk_pre  = msk_pre.astype(np.int64)
        msk_post = msk_post.astype(np.int64)

        # Start with damage-1 (so 1..4 → 0..3); fill non-building with IGNORE
        target = msk_post.copy() - 1
        # Mask out everything that's not a building according to the pre mask.
        target = np.where(msk_pre > 0, target, IGNORE_INDEX)
        # Also mask any post pixel that was 0 (background per post mask),
        # in case msk_post and msk_pre disagree (shouldn't happen but defensive).
        target = np.where(msk_post > 0, target, IGNORE_INDEX)
        # Clip any unexpected values (defensive)
        target = np.where((target >= 0) & (target < NUM_CLASSES),
                          target, IGNORE_INDEX)
        target = torch.from_numpy(target).long()

        # ── Normalize ──
        img = (img - self.norm_mean.unsqueeze(0)) / self.norm_std.unsqueeze(0)

        # ── Crop ──
        img, target = self._crop_prefer_buildings(
            img, target, self.crop_size, random=self.augment,
        )

        # ── Augment ──
        if self.augment:
            img, target = self._d4_augment(img, target)

        return {
            "image":    img,       # [T, C, H, W]
            "target":   target,    # [H, W]
            "metadata": {"filename": fn_pre},
        }