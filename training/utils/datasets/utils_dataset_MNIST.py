import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .token_builder import TokenBuilder


class MNISTSparseCanvas(Dataset):
    """
    MNIST Dataset — grouped token format (8 columns).

    Each 28×28 grayscale image is tokenized into one token per pixel-band.
    With NUM_BANDS=1 this gives 784 tokens per image. The dataset mirrors
    the structure of Sen1Floods11Dataset:

    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7

    Modes:
        - classification (default): queries = pixel positions, col 4 = digit class
                                    (broadcast per pixel for parity with the
                                    segmentation pipeline; the scalar `label`
                                    key is what the classification head should use).
        - reconstruction:           queries = image tokens, col 4 = reflectance.

    Splits:
        - "train" -> torchvision's 60k train set, MINUS a stratified held-out
                     val slice (see _load_mnist_with_val_split). Forced to
                     subsample_keep_rate=1.0 regardless of config (see below).
        - "val"   -> the stratified held-out slice carved out of the 60k
                     train set. Disjoint from "train", untouched by "test".
        - "test"  -> torchvision's real, untouched 10k test set.

    MNIST-specific notes:
        - D4 augmentation is INTENTIONALLY OFF. Digits are not symmetric:
            6 ↔ 9 under 180° rotation; 2/4/5/7 lose meaning under horizontal flip.
          Sen1Floods11 satellite imagery is roughly D4-invariant; MNIST is not.
        - Single grayscale band. We borrow the spectral metadata of a real
          optical band (S2 B02, 490 nm / bw 65 nm) so the lookup table stays
          consistent across datasets.
        - The `label` key is always present (scalar digit class), even in
          reconstruction mode — it's a per-image property, unlike the per-pixel
          labels in Sen1Floods11 that get folded into col 4.
        - subsample_keep_rate / subsample_mode are FORCED to (1.0, n/a) when
          split=="train", regardless of what config_model says. This guarantees
          the checkpoint is trained on full 784-token inputs, so any reduced-
          token evaluation on "val"/"test" is a genuine zero-shot generalization
          test rather than something the model saw during training.
    """

    RESOLUTION = 0.2          # m/px (arbitrary; must match train script registration)
    NUM_BANDS = 1
    NUM_CLASSES = 10
    CANVAS_SIZE = 28
    IGNORE_INDEX = 255        # unused for MNIST (no per-pixel labels), kept for symmetry
    TIME_IDX_NA = -1
    VAL_SPLIT_SEED = 42

    def __init__(
        self,
        root_path: str = "./data",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config=None,
        config_model: dict = None,
        look_up=None,
        num_samples: int = None,
        **kwargs,
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
        self.reconstruction = (
            config_model["trainer"].get("mode", "classification") == "reconstruction"
        )

        # ── Single-channel mode (degenerate for MNIST but kept for parity) ─
        sc = config_model["trainer"].get("single_channel", -1)
        if isinstance(sc, list):
            self.selected_channels = sorted(sc)
        elif isinstance(sc, int) and sc >= 0:
            self.selected_channels = [sc]
        else:
            self.selected_channels = None  # all bands

        if self.selected_channels is not None:
            for ch in self.selected_channels:
                assert 0 <= ch < self.NUM_BANDS, (
                    f"single_channel index {ch} out of range [0, {self.NUM_BANDS})"
                )
            print(f"[MNIST] SINGLE-CHANNEL MODE: using band indices {self.selected_channels}")

        if self.reconstruction:
            print(f"[MNIST] Mode: RECONSTRUCTION (queries = image tokens, col 4 = reflectance)")
        else:
            print(f"[MNIST] Mode: CLASSIFICATION (queries = pixels, scalar digit `label`)")

        # Band metadata (single virtual band borrowed from S2 B02)
        self.spectral_indices = self._build_spectral_indices()
        if self.selected_channels is not None:
            self.spectral_indices = self.spectral_indices[self.selected_channels]

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # ── Load MNIST with stratified train/val split ──────────────
        assert self.split in ("train", "val", "test"), (
            f"mode must be 'train', 'val', or 'test'; got {self.split!r}"
        )
        self._load_mnist_with_val_split(config_model)
        self.num_samples = (
            min(num_samples, len(self.indices)) if num_samples else len(self.indices)
        )

        # Normalization (computed only from the "train" split's own indices,
        # never touching "val" or "test" data — see _compute_normalization_stats)
        self.norm_stats = self._load_or_compute_normalization()

        # ── Random subsampling ───────────────────────────────────────
        # At each __getitem__ call, keep a random subset of the 784 pixel
        # tokens. The kept positions are *irregular per sample* but remain
        # uniformly distributed across the canvas (unlike foreground-only
        # filtering, which would cluster latents in the digit region).
        #
        # Use this to test the architecture's flexibility to arbitrary
        # sparse inputs: train with subsample_keep_rate=1.0 (ENFORCED, see
        # below), then evaluate checkpoints ("val"/"test" splits) with
        # reduced rates to measure graceful degradation.
        #
        # Companion option `latent_layout`:
        #   "grid"      (default) — encoder builds its usual regular grid.
        #                Latent count is derived from total_tokens, so it
        #                still adapts to the dropped count, but positions
        #                stay on a regular lattice.
        #   "at_tokens" — encoder places one latent at each kept token's
        #                position (irregular, sparse layout matching the
        #                kept pixel set).
        #
        # batch_size MUST be 1 since token count varies per sample.
        self.subsample_keep_rate = float(
            config_model["trainer"].get("subsample_keep_rate", 1.0)
        )
        self.subsample_mode = config_model["trainer"].get(
            "subsample_mode", "uniform"
        )
        self.pixel_threshold = float(
            config_model["trainer"].get("pixel_threshold", 0.5)
        )
        self.latent_layout = config_model["trainer"].get("latent_layout", "grid")
        assert 0.0 < self.subsample_keep_rate <= 1.0, (
            f"subsample_keep_rate must be in (0, 1]; got {self.subsample_keep_rate}"
        )
        assert self.subsample_mode in ("uniform", "background_only"), (
            f"subsample_mode must be 'uniform' or 'background_only'; "
            f"got {self.subsample_mode!r}"
        )
        assert self.latent_layout in ("grid", "at_tokens"), (
            f"latent_layout must be 'grid' or 'at_tokens'; "
            f"got {self.latent_layout!r}"
        )

        # ── Force full-token training ────────────────────────────────
        # The checkpoint used for any token-removal ablation table must be
        # trained on the complete 784-token input; otherwise "1.00 tokens
        # kept" at eval time would itself be an out-of-distribution
        # condition relative to training, undermining the "zero-shot
        # generalization to unseen token layouts" framing. "val" and "test"
        # splits are NOT touched by this override — they use whatever
        # subsample_keep_rate / subsample_mode the config (or an external
        # sweep script) requests.
        if self.split == "train" and self.subsample_keep_rate != 1.0:
            print(
                f"[MNIST] WARNING: subsample_keep_rate={self.subsample_keep_rate} "
                f"requested for split='train', but training must use the full "
                f"784-token input so any downstream ablation's zero-shot claim "
                f"holds. Forcing subsample_keep_rate=1.0 for training. "
                f"(The requested rate/mode still apply normally for 'val'/'test'.)"
            )
            self.subsample_keep_rate = 1.0

        print(f"[MNIST] Split: {self.split}, Samples: {self.num_samples}")
        print(f"[MNIST] Loaded {len(self.spectral_indices)} band(s)")
        print(f"[MNIST] Resolution idx: {self.resolution_idx} "
              f"(GSD={self.RESOLUTION} m/px)")
        print(f"[MNIST] Time idx: -1 (no temporal info, zeroed by encoder)")
        print(f"[MNIST] D4 augmentations: OFF (digits are not symmetric)")
        print(f"[MNIST] subsample_mode={self.subsample_mode!r}, "
              f"subsample_keep_rate={self.subsample_keep_rate:.3f} "
              f"(threshold={self.pixel_threshold} on raw [0,1] values) "
              f"| latent_layout={self.latent_layout!r}")

        # One-shot debug print fires on the first __getitem__ call per
        # DataLoader worker, so you get concrete evidence that the pixel
        # filter is actually changing the attention mask (and not silently
        # leaving every token valid).
        self._filter_debug_logged = False

    # =========================================================================
    # DATA SPLITTING
    # =========================================================================

    def _load_mnist_with_val_split(self, config_model):
        """
        mode="train" -> torchvision train split, MINUS the held-out val slice
        mode="val"   -> the held-out slice carved OUT of torchvision's train split
        mode="test"  -> torchvision's real, untouched test split (10k images)

        The val slice is chosen via sklearn's stratified train_test_split on
        digit label, with a fixed random_state, so:
          - "train" and "val" are always disjoint and reproducible across runs
            (independent of global RNG state set elsewhere, e.g.
            seed_everything(42) in the training script)
          - each digit class (0-9) is represented in "val" at the same
            proportion as in the full 60k train set, rather than leaving
            class balance to chance -- with only a few thousand held out, a
            plain random split could plausibly skew a class by a few points.
        """
        val_size = int(config_model["trainer"].get("val_size", 5000))

        if self.split == "test":
            self.mnist = torchvision.datasets.MNIST(
                root=self.root_path, train=False, download=True
            )
            self.indices = list(range(len(self.mnist)))
            return

        # split in {"train", "val"} both draw from torchvision's TRAIN set
        self.mnist = torchvision.datasets.MNIST(
            root=self.root_path, train=True, download=True
        )
        full_len = len(self.mnist)

        # ── val_size == 0: no held-out slice at all ──────────────────
        # Used for the controlled "old-style" comparison: train on the
        # full 60k, no validation carve-out. "val" split becomes a
        # (trivially small) placeholder so DataLoader/Lightning plumbing
        # that expects a val split doesn't have to be restructured --
        # in this mode the training script should skip passing a real
        # val_dataloaders to trainer.fit() and use save_last / last-epoch
        # checkpointing instead of monitor-based selection.
        if val_size == 0:
            if self.split == "val":
                print(f"[MNIST] val_size=0: 'val' split is EMPTY. "
                      f"Checkpoint selection must not use val_loss in this mode "
                      f"(use save_last / last-epoch checkpointing instead).")
                self.indices = []
            else:  # "train"
                self.indices = list(range(full_len))
                print(f"[MNIST] val_size=0: 'train' split uses the FULL "
                      f"{full_len} train images (no held-out slice).")
            return

        all_indices = np.arange(full_len)
        # .targets is a Tensor of digit labels, available without touching
        # __getitem__ / any transform.
        labels = self.mnist.targets.numpy()

        assert val_size < full_len, (
            f"val_size={val_size} must be smaller than the full train set "
            f"({full_len})"
        )

        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=val_size,
            random_state=self.VAL_SPLIT_SEED,
            stratify=labels,
        )

        if self.split == "val":
            self.indices = val_idx.tolist()
        else:  # "train"
            self.indices = train_idx.tolist()

        print(f"[MNIST] '{self.split}' split: {len(self.indices)} samples "
              f"(val_size={val_size}, stratified, seed={self.VAL_SPLIT_SEED})")
        if self.split == "val":
            class_counts = np.bincount(labels[val_idx], minlength=10)
            print(f"[MNIST] val class counts (0-9): {class_counts.tolist()}")

    # =========================================================================
    # AUGMENTATION
    # =========================================================================
    # NOTE: D4 (rotations + flips) and random crop are intentionally OMITTED
    # for MNIST. They corrupt digit identity. If you want augmentation, prefer
    # small random shifts / elastic deformations, applied here before tokenization.

    # =========================================================================
    # CHANNEL SELECTION HELPER (degenerate but kept for API parity)
    # =========================================================================

    def _select_channels(self, image):
        if self.selected_channels is None:
            return image
        return image[self.selected_channels]

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # ── Load ────────────────────────────────────────────
        real_idx = self.indices[index % len(self.indices)]
        digit_img, digit_label = self.mnist[real_idx]
        image = torch.tensor(np.array(digit_img), dtype=torch.float32) / 255.0  # [H, W]
        image = image.unsqueeze(0)                                              # [1, H, W]

        # ── Clean (MNIST has no NaNs but kept for symmetry) ─
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────
        image = self.normalize_image(image)

        # ── Channel selection (no-op for 1 band) ────────────
        image = self._select_channels(image)
        C, H, W = image.shape

        # Per-pixel label map filled with the digit class. For Sen1Floods11
        # this column carries the segmentation label; for MNIST we fold the
        # global class in so col 4 stays semantically meaningful, but the
        # trainer should still use the scalar `label` key for the loss.
        label_map = torch.full((H, W), int(digit_label), dtype=torch.long)

        # ── Build tokens [N, 8] ─────────────────────────────
        resolution = self.RESOLUTION
        image_tokens, seg_queries = self._build_tokens(image, label_map, resolution)

        # ── Build queries (mode-dependent) ──────────────────
        if self.reconstruction:
            queries = image_tokens.clone()
            queries[:, 4] = queries[:, 0].clone()  # reflectance → label col
            if queries.shape[0] > self.max_tokens_reconstruction:
                perm = torch.randperm(queries.shape[0])[:self.max_tokens_reconstruction]
                queries = queries[perm]
        else:
            if self.split == "train":
                # Training: subsample for memory/speed.
                # prioritize_valid is moot here (MNIST has no IGNORE_INDEX pixels)
                # but we use the same call as Sen1Floods11 for API parity.
                queries = self.token_builder.subsample_queries(
                    seg_queries,
                    max_queries=self.max_tokens_reconstruction,
                    ignore_index=self.IGNORE_INDEX,
                    prioritize_valid=True,
                )
            else:
                # Val/test: all pixels (only 784 for MNIST, no chunking needed)
                queries = seg_queries

        # ── Masks (0 = valid, matches Sen1Floods11 __getitem__) ─
        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask = torch.zeros(queries.shape[0])

        # ── Random subsampling (HARD-DROP) ─
        # Two modes:
        #   "uniform"         — keep a uniformly-random fraction of all 784
        #                       tokens. The kept set covers the canvas
        #                       uniformly per sample.
        #   "background_only" — always keep the foreground (above-threshold)
        #                       pixels; subsample only the background pixels
        #                       at the configured rate. Isolates how much
        #                       background context the model actually needs.
        # batch_size=1 is required since N varies per sample.
        #
        # NOTE: for split=="train", subsample_keep_rate is forced to 1.0 in
        # __init__, so this block is always a no-op during training.
        if self.subsample_mode == "uniform":
            if self.subsample_keep_rate < 1.0:
                N = image_tokens.shape[0]
                n_keep = max(1, int(round(N * self.subsample_keep_rate)))
                perm = torch.randperm(N)[:n_keep]
                # Sort to keep token order spatially monotonic.
                perm, _ = torch.sort(perm)
                image_tokens   = image_tokens[perm]
                attention_mask = torch.zeros(image_tokens.shape[0])

                if not self._filter_debug_logged:
                    col0_min = float(image_tokens[:, 0].min())
                    col0_max = float(image_tokens[:, 0].max())
                    print(
                        f"[MNIST/{self.split}] uniform subsample "
                        f"(pid={os.getpid()}, idx={index}): "
                        f"keep_rate={self.subsample_keep_rate:.3f} | "
                        f"col0 range=[{col0_min:+.3f}, {col0_max:+.3f}] | "
                        f"kept={n_keep}/{N} | "
                        f"latent_layout={self.latent_layout!r}"
                    )
                    self._filter_debug_logged = True

        else:  # "background_only"
            if self.subsample_keep_rate < 1.0:
                mean = float(self.norm_stats["mean"][0])
                std  = float(self.norm_stats["std"][0])
                threshold_norm = (self.pixel_threshold - mean) / std

                is_fg = image_tokens[:, 0] >= threshold_norm   # white pixels
                fg_idx = torch.where(is_fg)[0]
                bg_idx = torch.where(~is_fg)[0]

                n_bg = bg_idx.numel()
                n_bg_keep = max(0, int(round(n_bg * self.subsample_keep_rate)))
                if n_bg_keep < n_bg:
                    perm = torch.randperm(n_bg)[:n_bg_keep]
                    bg_idx = bg_idx[perm]

                # Safety: ensure at least one token survives.
                if fg_idx.numel() == 0 and bg_idx.numel() == 0:
                    bg_idx = torch.tensor([image_tokens[:, 0].argmin().item()])

                # Combine, sort, slice.
                all_idx = torch.cat([fg_idx, bg_idx])
                all_idx, _ = torch.sort(all_idx)
                image_tokens   = image_tokens[all_idx]
                attention_mask = torch.zeros(image_tokens.shape[0])

                if not self._filter_debug_logged:
                    n_fg = int(fg_idx.numel())
                    n_bg_kept = int(bg_idx.numel())
                    print(
                        f"[MNIST/{self.split}] bg-only subsample "
                        f"(pid={os.getpid()}, idx={index}): "
                        f"thr_raw={self.pixel_threshold:.3f} → "
                        f"thr_norm={threshold_norm:+.3f} | "
                        f"keep_rate={self.subsample_keep_rate:.3f} | "
                        f"fg={n_fg} (all kept) + bg={n_bg_kept}/{n_bg} "
                        f"= {image_tokens.shape[0]} total | "
                        f"latent_layout={self.latent_layout!r}"
                    )
                    self._filter_debug_logged = True

        # ── Return ──────────────────────────────────────────
        result = {
            "groups": {
                resolution: {
                    "tokens": image_tokens,
                    "mask": attention_mask,
                    "shape": tuple(image.shape),     # (C, H, W), like Sen1Floods11
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "target_resolution": resolution,
            "latent_layout": self.latent_layout,    # "grid" or "at_tokens"
            "image": image,
            # Always include digit class — it's a per-image property, unlike
            # Sen1Floods11 where the label is per-pixel and folded into col 4.
            "label": torch.tensor(int(digit_label), dtype=torch.long),
        }

        return result

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_tokens(self, image, label_map, resolution):
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label_map,
            resolution=resolution,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label_map,
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
        Viz sample — mode-aware. No augmentation applied (deterministic).
        """
        real_idx = self.indices[index % len(self.indices)]
        digit_img, digit_label = self.mnist[real_idx]
        image = torch.tensor(np.array(digit_img), dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = self.normalize_image(image)
        image = self._select_channels(image)
        C, H, W = image.shape

        label_map = torch.full((H, W), int(digit_label), dtype=torch.long)
        digit_label_t = torch.tensor(int(digit_label), dtype=torch.long)

        if self.reconstruction:
            tokens = self.token_builder.build_tokens(
                image=image,
                label=label_map,
                resolution=self.RESOLUTION,
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
                    self.RESOLUTION: {
                        "tokens": tokens,
                        "mask": attention_mask,
                        "shape": (C, H, W),
                    },
                },
                "queries": queries,
                "queries_mask": queries_mask,
                "target_resolution": self.RESOLUTION,
                "image": image,
                "image_shape": (C, H, W),
                "n_real": tokens.shape[0],
                "label": digit_label_t,
            }
        else:
            image_tokens, queries = self._build_tokens(image, label_map, self.RESOLUTION)
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
            attention_mask = torch.zeros(image_tokens.shape[0])

            return {
                "groups": {
                    self.RESOLUTION: {
                        "tokens": image_tokens,
                        "mask": attention_mask,
                        "shape": (C, H, W),
                    },
                },
                "queries": queries,
                "queries_mask": queries_mask,
                "label": digit_label_t,
                "target_resolution": self.RESOLUTION,
                "image": image,
            }

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "mnist_norm_stats.pt")

        if os.path.exists(norm_file):
            print(f"[MNIST] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[MNIST] WARNING: No normalization file at {norm_file}, "
                  f"using identity (mean=0, std=1)")
            return {
                "mean": torch.zeros(self.NUM_BANDS),
                "std":  torch.ones(self.NUM_BANDS),
            }

        print(f"[MNIST] Computing normalization from {len(self.indices)} train samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[MNIST] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        """
        Streaming mean/std over the TRAIN split ONLY (self.indices, which
        excludes the held-out val slice), in [0, 1] reflectance space.
        Single channel → returns 1-element tensors.
        """
        total_sum = 0.0
        total_sq = 0.0
        total_n = 0
        for real_idx in tqdm(self.indices, desc="Computing MNIST normalization"):
            img, _ = self.mnist[real_idx]
            arr = np.array(img, dtype=np.float64) / 255.0
            total_sum += arr.sum()
            total_sq += (arr ** 2).sum()
            total_n += arr.size
        mean = total_sum / max(total_n, 1)
        var = total_sq / max(total_n, 1) - mean ** 2
        std = float(np.sqrt(max(var, 0.0)))
        return {
            "mean": torch.tensor([mean], dtype=torch.float32),
            "std":  torch.tensor([std],  dtype=torch.float32),
        }

    def _print_norm_stats(self, stats):
        print(f"[MNIST] mean: {stats['mean'].numpy()}")
        print(f"[MNIST] std:  {stats['std'].numpy()}")

    def normalize_image(self, image):
        """Per-channel z-score normalization."""
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _build_spectral_indices(self):
        """
        MNIST has a single grayscale band. We borrow the spectral metadata of a
        real optical band (Sentinel-2 B02, 490 nm / bw 65 nm) so the shared
        lookup table stays consistent. If that key isn't registered, we fall
        back to the first available entry.
        """
        key = (65, 490)  # Sentinel-2 B02 (bandwidth=65, wavelength=490)
        if key in self.look_up.table_wave:
            idx = self.look_up.table_wave[key]
        else:
            first_key = next(iter(self.look_up.table_wave))
            idx = self.look_up.table_wave[first_key]
            print(f"[MNIST] Warning: spectral key {key} missing from lookup, "
                  f"falling back to {first_key}")
        return torch.tensor([idx], dtype=torch.long)
