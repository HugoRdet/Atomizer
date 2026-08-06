"""
EuroSAT-SAR Baseline Dataset
===============================

Plain tensor classification dataset for non-Atomiser baselines (ResNet/ViT
classifier), fusing Sentinel-2 optical (13 bands) and Sentinel-1 SAR
(VV, VH), paired by identical filename across two class-folder trees.

Output format (compatible with ClassificationBaselineTrainer, which does
`image = batch["image"][self.modality]` then `model(image) -> logits`,
so a single modality key is required — pass modality="fused"):
    {
        "image":  {"fused": [15, H, W]},   # S2 (13 bands) ++ S1 (VV, VH)
        "target": scalar long (0..9),
        "metadata": {...},
    }

A concat-style baseline (plain ResNet/ViT, in_channels=15) consumes this
directly. A per-modality/two-branch baseline instead slices the fused
tensor inside its own forward() (image[:, :13] vs image[:, 13:]) — that
architectural choice lives in the model, not this dataset, since the
trainer's contract only allows one modality key per batch.

IMPORTANT — shares split & normalization with EuroSATSARDataset:
    This dataset reads (or, if absent, generates with the identical seeded
    algorithm) the same `split_cache.json` and `normalization_stats.pt`
    cached under root_ms by EuroSATSARDataset (the Atomiser dataset). This
    is deliberate: baseline vs. Atomizer comparisons are only meaningful if
    both see the same train/val/test split and the same per-band
    normalization. Whichever dataset class runs first creates the cache;
    the other reads it.

Splits: stratified 80/10/10 per class, seeded (see SPLIT_SEED/SPLIT_RATIOS),
cached to `{root_ms}/split_cache.json`.
Native size: 64×64.
Bands: 13 optical (bands_eurosat idx order) + VV + VH (raw tif is [VV, VH]).

Band dropping (optional, for modality-drop baselines):
    Pass `bands={"keep": [...], "drop": [...]}` using names from
    ALL_BAND_NAMES. Unlike Atomiser's token masking, a fixed-channel-count
    baseline can't add/remove input channels per ablation, so dropped/
    non-kept channels are ZEROED rather than removed — same effective
    semantics (the model sees no information from that channel), same
    input shape across ablations, so ablations on a single trained
    checkpoint are meaningful.

Band-dropout augmentation (train only):
    Zeroes whole modalities or random individual bands during training
    (on top of, not instead of, the static bands.keep/drop config above),
    matching the semantics of the test-time modality-drop ablation script
    and Sen1Floods11BaselineDataset's identical augmentation. Applied
    AFTER normalization, for the same reason as there: zeroing raw pixel
    values before normalization would leave a nonzero value post-z-score
    ((0-mean)/std != 0), a different signal than what the model actually
    sees at ablation eval time (a literal zero). Sampled per-sample,
    independently of the fixed eval ablations — NOT limited to replaying
    the exact eval-time combinations, since always training on only those
    exact combinations would be a soft form of eval-set leakage into
    training. Baseline-only compensation for these architectures' lack of
    a native way to represent "this band is absent" (unlike Atomiser's
    token masking) — not neutral, should be reported as such in any
    writeup.
"""

import glob
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class EuroSATSARBaselineDataset(Dataset):
    """EuroSAT-SAR (+ paired EuroSAT_MS) 10-class classification baseline dataset."""

    NUM_S2_CHANNELS = 13
    NUM_S1_CHANNELS = 2
    NUM_CLASSES     = 10
    IGNORE_INDEX    = 255
    PATCH_SIZE      = 64

    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    ]

    # Raw .tif channel order for EuroSAT_MS (standard Sentinel-2 "all bands" order)
    RAW_MS_BAND_ORDER = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B08A", "B09", "B10", "B11", "B12",
    ]
    RAW_SAR_BAND_ORDER = ["VV", "VH"]

    # config-key name -> raw S2 band code (matches bands_eurosat idx order)
    NAME_TO_S2CODE = {
        "Blue": "B02", "Green": "B03", "Red": "B04", "NIR": "B08",
        "RedEdge1": "B05", "RedEdge2": "B06", "RedEdge3": "B07", "RedEdge4": "B08A",
        "SWIR1": "B11", "SWIR2": "B12",
        "CoastalAerosol": "B01", "WaterVapour": "B09", "Cirrus": "B10",
    }
    # bands_eurosat idx order (0-12), matches EuroSATSARDataset
    S2_NAME_ORDER = [
        "Blue", "Green", "Red", "NIR", "RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4",
        "SWIR1", "SWIR2", "CoastalAerosol", "WaterVapour", "Cirrus",
    ]
    ALL_BAND_NAMES = S2_NAME_ORDER + ["VV", "VH"]

    SPLIT_SEED   = 42                                   # must match EuroSATSARDataset
    SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    MS_SUBDIR  = "EuroSAT_MS"
    SAR_SUBDIR = "EuroSAT-SAR"

    def __init__(
        self,
        root_path: str = "./data",
        mode: str = "train",
        crop_size: int = None,
        augment: bool = True,
        bands: dict = None,
        root_ms: str = None,
        root_sar: str = None,
        band_dropout: bool = True,
        p_dropout_applied: float = 0.5,
        p_whole_modality: float = 0.5,
        p_band_drop: float = 0.15,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"

        self.root_ms  = root_ms  if root_ms  is not None else os.path.join(root_path, self.MS_SUBDIR)
        self.root_sar = root_sar if root_sar is not None else os.path.join(root_path, self.SAR_SUBDIR)
        self.split    = self.SPLIT_MAPPING[mode]
        self.crop_size = crop_size
        self.augment   = augment and (mode == "train")
        self.band_dropout = band_dropout and (mode == "train")
        self.p_dropout_applied = p_dropout_applied
        self.p_whole_modality = p_whole_modality
        self.p_band_drop = p_band_drop

        # ── Band selection (zero-out, not remove — fixed channel count) ──
        bands = bands or {}
        keep_names = bands.get("keep", None)
        drop_names = bands.get("drop", None)
        self.active_mask = self._build_active_mask(keep_names, drop_names)  # [15] bool

        # ── Pair MS + SAR samples by filename, build/load shared split ──
        self.samples = self._build_sample_list()
        self.split_assignment = self._load_or_build_split()
        self.sample_list = [
            (cls, fn) for (cls, fn) in self.samples
            if self.split_assignment[fn] == self.split
        ]

        self.ms_reorder_idx = [
            self.RAW_MS_BAND_ORDER.index(self.NAME_TO_S2CODE[name])
            for name in self.S2_NAME_ORDER
        ]

        self.norm_stats = self._load_or_compute_normalization()

        label_counts = Counter(cls for cls, _ in self.sample_list)
        print(f"[EuroSAT-SAR-BL] split={mode}, samples={len(self.sample_list)}")
        print(f"[EuroSAT-SAR-BL] channels: {self.NUM_S2_CHANNELS} optical + "
              f"{self.NUM_S1_CHANNELS} SAR (VV, VH)")
        print(f"[EuroSAT-SAR-BL] active bands: "
              f"{[n for n, m in zip(self.ALL_BAND_NAMES, self.active_mask) if m]}")
        print(f"[EuroSAT-SAR-BL] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"[EuroSAT-SAR-BL] num_classes: {self.NUM_CLASSES}")
        print(f"[EuroSAT-SAR-BL] D4 augment: {'ON' if self.augment else 'OFF'}")
        if self.band_dropout:
            print(f"[EuroSAT-SAR-BL] Band dropout: ON "
                  f"(p_applied={self.p_dropout_applied}, "
                  f"p_whole_modality={self.p_whole_modality}, "
                  f"p_band_drop={self.p_band_drop})")
        else:
            print(f"[EuroSAT-SAR-BL] Band dropout: OFF")
        print(f"[EuroSAT-SAR-BL] class distribution: "
              f"{ {self.CLASS_NAMES[k]: v for k, v in sorted(label_counts.items())} }")

    # =========================================================================
    # BAND SELECTION (zero-out semantics)
    # =========================================================================

    def _build_active_mask(self, keep_names, drop_names):
        if keep_names is None:
            active = set(self.ALL_BAND_NAMES)
        else:
            invalid = set(keep_names) - set(self.ALL_BAND_NAMES)
            if invalid:
                raise ValueError(f"Unknown band names in keep: {invalid}")
            active = set(keep_names)
        if drop_names:
            invalid = set(drop_names) - set(self.ALL_BAND_NAMES)
            if invalid:
                raise ValueError(f"Unknown band names in drop: {invalid}")
            not_kept = set(drop_names) - active
            if not_kept:
                raise ValueError(
                    f"bands.drop {not_kept} are not in bands.keep {active}. "
                    f"You can only drop bands that were kept."
                )
            active -= set(drop_names)
        return [name in active for name in self.ALL_BAND_NAMES]

    # =========================================================================
    # D4 AUGMENTATION (applied identically to both modalities)
    # =========================================================================

    @staticmethod
    def _d4_augment(s2: torch.Tensor, s1: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            s2 = torch.flip(s2, dims=[2])
            s1 = torch.flip(s1, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            s2 = torch.rot90(s2, k, dims=[1, 2])
            s1 = torch.rot90(s1, k, dims=[1, 2])
        return s2, s1

    # =========================================================================
    # BAND-DROPOUT AUGMENTATION (train only, applied AFTER normalization)
    # =========================================================================

    @staticmethod
    def _band_dropout_augment(
        fused: torch.Tensor,
        p_dropout_applied: float,
        p_whole_modality: float,
        p_band_drop: float,
        num_s2_bands: int,
        num_s1_bands: int,
    ) -> torch.Tensor:
        """
        Zero out whole modalities or random individual bands, applied to
        the already-normalized, already-merged [15, H, W] "fused" tensor
        (S2 first num_s2_bands channels, then S1's VV/VH).

        With probability (1 - p_dropout_applied): no-op, sample keeps all
        bands (keeps the full-band regime well-represented in training).

        Otherwise, with probability p_whole_modality: zero either all S2
        or all S1 bands (mirrors the "S2 only" / "S1 only" eval
        ablations). With probability (1 - p_whole_modality): zero each of
        the 15 bands independently with probability p_band_drop (covers
        the RGB-only / no-SWIR / no-red-edge style subset ablations
        without hardcoding to those exact combinations).
        """
        if torch.rand(1).item() >= p_dropout_applied:
            return fused

        fused = fused.clone()

        if torch.rand(1).item() < p_whole_modality:
            if torch.rand(1).item() < 0.5:
                fused[num_s2_bands:] = 0.0                    # drop S1
            else:
                fused[:num_s2_bands] = 0.0                    # drop S2
        else:
            total_bands = num_s2_bands + num_s1_bands
            band_mask = torch.rand(total_bands) < p_band_drop
            fused[band_mask] = 0.0

        return fused

    # =========================================================================
    # SAMPLE LIST / SPLIT  (identical algorithm to EuroSATSARDataset, so the
    # cache is bit-for-bit reusable regardless of which dataset builds it first)
    # =========================================================================

    def _build_sample_list(self):
        samples = []
        for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
            ms_files  = {os.path.basename(p) for p in
                         glob.glob(os.path.join(self.root_ms, cls_name, "*.tif"))}
            sar_files = {os.path.basename(p) for p in
                         glob.glob(os.path.join(self.root_sar, cls_name, "*.tif"))}
            paired = sorted(ms_files & sar_files)
            for fn in paired:
                samples.append((cls_idx, fn))
        if not samples:
            raise RuntimeError(
                f"[EuroSAT-SAR-BL] No paired MS/SAR samples found under "
                f"{self.root_ms} / {self.root_sar}"
            )
        return samples

    def _load_or_build_split(self):
        cache_path = os.path.join(self.root_ms, "split_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"[EuroSAT-SAR-BL] Loaded shared split cache from {cache_path} "
                  f"({len(cached)} entries)")
            return cached

        print(f"[EuroSAT-SAR-BL] No split cache found — building stratified "
              f"{self.SPLIT_RATIOS} split (seed={self.SPLIT_SEED})...")
        by_class = defaultdict(list)
        for cls_idx, fn in self.samples:
            by_class[cls_idx].append(fn)

        rng = random.Random(self.SPLIT_SEED)
        assignment = {}
        for cls_idx, filenames in by_class.items():
            filenames = sorted(filenames)
            rng.shuffle(filenames)
            n = len(filenames)
            n_train = int(round(n * self.SPLIT_RATIOS["train"]))
            n_valid = int(round(n * self.SPLIT_RATIOS["valid"]))
            for fn in filenames[:n_train]:
                assignment[fn] = "train"
            for fn in filenames[n_train:n_train + n_valid]:
                assignment[fn] = "valid"
            for fn in filenames[n_train + n_valid:]:
                assignment[fn] = "test"

        try:
            with open(cache_path, "w") as f:
                json.dump(assignment, f)
            print(f"[EuroSAT-SAR-BL] Saved split cache to {cache_path}")
        except Exception as e:
            print(f"[EuroSAT-SAR-BL] WARN: could not save split cache: {e}")
        return assignment

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.sample_list)

    def _load_pair(self, cls_idx, filename):
        cls_name = self.CLASS_NAMES[cls_idx]
        ms_path  = os.path.join(self.root_ms,  cls_name, filename)
        sar_path = os.path.join(self.root_sar, cls_name, filename)

        with rasterio.open(ms_path) as src:
            ms = src.read().astype(np.float32)        # [13, 64, 64], raw tif order
        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)        # [2, 64, 64], [VV, VH]

        ms  = np.nan_to_num(ms, nan=0.0, posinf=0.0, neginf=0.0)
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        ms  = torch.from_numpy(ms)[self.ms_reorder_idx]   # -> bands_eurosat idx order
        sar = torch.from_numpy(sar)

        return ms, sar

    def __getitem__(self, index):
        cls_idx, filename = self.sample_list[index]
        ms, sar = self._load_pair(cls_idx, filename)

        ms  = (ms  - self.norm_stats["ms_mean"].view(-1, 1, 1))  / self.norm_stats["ms_std"].view(-1, 1, 1)
        sar = (sar - self.norm_stats["sar_mean"].view(-1, 1, 1)) / self.norm_stats["sar_std"].view(-1, 1, 1)

        if self.augment:
            ms, sar = self._d4_augment(ms, sar)

        # zero-out dropped/non-kept channels (fixed input shape across ablations)
        active_mask = torch.tensor(self.active_mask, dtype=torch.float32).view(-1, 1, 1)
        fused = torch.cat([ms, sar], dim=0)   # [15, H, W] — S2(13) then S1(VV,VH),
                                                # same channel order as EuroSATSARDataset
        fused = fused * active_mask

        # ── Band-dropout augmentation (training only, AFTER active_mask) ────
        # Layered on top of the static keep/drop config above, not instead
        # of it — a band already zeroed by active_mask just stays zeroed
        # (idempotent), and this adds STOCHASTIC additional dropout on top
        # for training-time robustness exposure.
        if self.band_dropout:
            fused = self._band_dropout_augment(
                fused, self.p_dropout_applied, self.p_whole_modality,
                self.p_band_drop, self.NUM_S2_CHANNELS, self.NUM_S1_CHANNELS,
            )

        target = torch.tensor(cls_idx, dtype=torch.long)
        H, W = fused.shape[-2], fused.shape[-1]

        return {
            "image":  {"fused": fused},
            "target": target,
            "metadata": {
                "H": H, "W": W,
                "n_bands": self.NUM_S2_CHANNELS + self.NUM_S1_CHANNELS,
                "sample_name": f"{self.CLASS_NAMES[cls_idx]}/{filename}",
                "active_bands": [n for n, m in zip(self.ALL_BAND_NAMES, self.active_mask) if m],
            },
        }

    # =========================================================================
    # NORMALIZATION  (shared cache with EuroSATSARDataset)
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_ms, "normalization_stats.pt")
        if os.path.exists(norm_file):
            print(f"[EuroSAT-SAR-BL] Loading shared normalization stats from {norm_file}")
            return torch.load(norm_file, weights_only=True)

        if self.split != "train":
            print(f"[EuroSAT-SAR-BL] WARNING: No normalization file at {norm_file}; "
                  f"using identity stats. Run the train split (this dataset or "
                  f"EuroSATSARDataset) first to build shared stats.")
            return {
                "ms_mean":  torch.zeros(self.NUM_S2_CHANNELS),  "ms_std":  torch.ones(self.NUM_S2_CHANNELS),
                "sar_mean": torch.zeros(self.NUM_S1_CHANNELS),  "sar_std": torch.ones(self.NUM_S1_CHANNELS),
            }

        print(f"[EuroSAT-SAR-BL] No normalization cache found — computing from "
              f"{len(self.sample_list)} train samples "
              f"(consider running EuroSATSARDataset's batched/parallel version "
              f"first if this is slow)...")
        from tqdm import tqdm

        ms_sum = np.zeros(self.NUM_S2_CHANNELS, dtype=np.float64)
        ms_sq  = np.zeros(self.NUM_S2_CHANNELS, dtype=np.float64)
        ms_n   = np.zeros(self.NUM_S2_CHANNELS, dtype=np.float64)
        sar_sum = np.zeros(self.NUM_S1_CHANNELS, dtype=np.float64)
        sar_sq  = np.zeros(self.NUM_S1_CHANNELS, dtype=np.float64)
        sar_n   = np.zeros(self.NUM_S1_CHANNELS, dtype=np.float64)

        for cls_idx, filename in tqdm(self.sample_list, desc="Computing normalization"):
            try:
                ms, sar = self._load_pair(cls_idx, filename)
            except Exception:
                continue
            ms_np, sar_np = ms.double().numpy(), sar.double().numpy()
            valid = ms_np > 0
            ms_sum += np.where(valid, ms_np, 0.0).sum(axis=(1, 2))
            ms_sq  += np.where(valid, ms_np ** 2, 0.0).sum(axis=(1, 2))
            ms_n   += valid.sum(axis=(1, 2))
            valid_s = np.isfinite(sar_np) & (sar_np != 0)
            sar_sum += np.where(valid_s, sar_np, 0.0).sum(axis=(1, 2))
            sar_sq  += np.where(valid_s, sar_np ** 2, 0.0).sum(axis=(1, 2))
            sar_n   += valid_s.sum(axis=(1, 2))

        ms_mean  = torch.from_numpy(ms_sum  / np.clip(ms_n, 1, None)).float()
        ms_std   = torch.from_numpy(np.sqrt(np.clip(ms_sq / np.clip(ms_n, 1, None) - (ms_sum / np.clip(ms_n, 1, None)) ** 2, 0, None))).float()
        sar_mean = torch.from_numpy(sar_sum / np.clip(sar_n, 1, None)).float()
        sar_std  = torch.from_numpy(np.sqrt(np.clip(sar_sq / np.clip(sar_n, 1, None) - (sar_sum / np.clip(sar_n, 1, None)) ** 2, 0, None))).float()

        stats = {"ms_mean": ms_mean, "ms_std": ms_std.clamp(min=1e-6),
                 "sar_mean": sar_mean, "sar_std": sar_std.clamp(min=1e-6)}
        try:
            torch.save(stats, norm_file)
            print(f"[EuroSAT-SAR-BL] Saved shared normalization stats to {norm_file}")
        except Exception as e:
            print(f"[EuroSAT-SAR-BL] WARN: could not save normalization stats: {e}")
        return stats
