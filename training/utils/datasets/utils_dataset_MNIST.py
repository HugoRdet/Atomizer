"""
MNIST Dataset — New format (8-column tokens, batch dict output)
================================================================

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

Returns:
{
    "groups": {
        0.2: {
            "tokens": [N, 8],
            "mask":   [N],
            "shape":  (28, 28),
        },
    },
    "queries":      [M, 8],
    "queries_mask":  [M],
    "label":         scalar (digit class),
}
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from .token_builder import TokenBuilder


class MNISTSparseCanvas(Dataset):
    """
    MNIST digit classification dataset in Atomiser token format.

    Each 28×28 grayscale image is tokenized into 784 tokens (1 band).
    Resolution = 0.2 m/px (arbitrary but consistent with previous setup).
    No temporal info (time_idx = -1).
    """

    RESOLUTION = 0.2          # m/px
    NUM_BANDS = 1
    NUM_CLASSES = 10
    CANVAS_SIZE = 28
    TIME_IDX_NA = -1

    def __init__(
        self,
        canvas_size: int = 28,
        num_bands: int = 1,
        mode: str = "train",
        config_model: dict = None,
        look_up=None,
        num_samples: int = None,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.look_up = look_up
        self.config_model = config_model

        # Token builder for consistent coordinate encoding
        self.token_builder = TokenBuilder(look_up)

        # Config
        self.max_queries = config_model["trainer"].get("max_tokens_reconstruction", 784)

        # Band metadata — single grayscale band (use arbitrary wavelength)
        # We register it as a "virtual" optical band
        self.spectral_indices = self._build_spectral_indices()
        self.resolution_idx = look_up.get_resolution_idx(self.RESOLUTION)

        # Load MNIST
        self.mnist = torchvision.datasets.MNIST(
            root="./data", train=(mode == "train"), download=True
        )
        self.num_samples = (
            min(num_samples, len(self.mnist)) if num_samples else len(self.mnist)
        )

        print(f"[MNIST] Mode: {mode}, Samples: {self.num_samples}")
        print(f"[MNIST] Resolution: {self.RESOLUTION} m/px, "
              f"Resolution idx: {self.resolution_idx}")
        print(f"[MNIST] Spectral indices: {self.spectral_indices.tolist()}")

    def _build_spectral_indices(self):
        """
        Register a single grayscale band in the lookup table.
        Uses the first available optical band (e.g., B02 at 490nm/65bw).
        """
        # Use B02-like wavelength for the single grayscale channel
        key = (65, 490)  # bandwidth=65, wavelength=490 (Sentinel-2 B02)
        if key in self.look_up.table_wave:
            idx = self.look_up.table_wave[key]
        else:
            # Fallback: use first available
            first_key = next(iter(self.look_up.table_wave))
            idx = self.look_up.table_wave[first_key]
            print(f"[MNIST] Warning: using fallback spectral key {first_key}")

        return torch.tensor([idx], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        digit_img, digit_label = self.mnist[idx % len(self.mnist)]
        digit_tensor = torch.tensor(
            np.array(digit_img), dtype=torch.float32
        ) / 255.0

        # [1, 28, 28] — single band
        image = digit_tensor.unsqueeze(0)
        H, W = self.CANVAS_SIZE, self.CANVAS_SIZE

        # Dummy label map for token builder (not used for classification)
        dummy_label = torch.zeros(H, W, dtype=torch.long)

        # ── Build tokens [N, 8] ─────────────────────────────
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Queries (same as image tokens for classification) ──
        queries = self.token_builder.build_queries(
            label=dummy_label,
            resolution=self.RESOLUTION,
            first_spectral_idx=self.spectral_indices[0],
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # Subsample queries if needed
        if queries.shape[0] > self.max_queries:
            perm = torch.randperm(queries.shape[0])[:self.max_queries]
            queries = queries[perm]

        # ── Masks ───────────────────────────────────────────
        token_mask = torch.zeros(image_tokens.shape[0], dtype=torch.bool)
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": image_tokens,
                    "mask": token_mask,
                    "shape": (H, W),
                },
            },
            "queries": queries,
            "queries_mask": queries_mask,
            "label": torch.tensor(digit_label, dtype=torch.long),
        }