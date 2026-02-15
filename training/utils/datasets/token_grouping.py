"""
Grouped token utilities for Atomizer datasets.

Return contract from __getitem__ (normal):
{
    "groups": {
        <resolution_m>: {
            "tokens":  [N, 8]   — [value, x, y, spectral_idx, label, query_idx, res_idx, time_idx]
            "mask":    [N]      — 0.0 = valid, 1.0 = ignore
            "shape":   (C, H, W)
        },
        ...  # one entry per native resolution
    },
    "queries":           [M, 8]   — sub-sampled query tokens
    "queries_mask":      [M]
    "label":             [H, W]   — at target resolution
    "target_resolution": float    — resolution of the label grid (meters)
    "image":             [C, H, W] — raw normalized image for visualization
}

Return contract from __getitem__ (sliding window, val/test):
{
    "sliding":           True
    "crops": [
        {"groups": {...}, "queries": [M, 8], "queries_mask": [M]},
        ...  # one per sliding window crop
    ],
    "crop_positions":    [(y0, x0), ...]   — at 10m
    "crop_size":         (crop_h, crop_w)  — at 10m
    "full_size":         (full_h, full_w)  — at 10m
    "label":             [H, W]            — full tile
    "target_resolution": float
    "image":             [C, H, W]         — full tile
}

After collate_grouped, sliding samples become:
{
    "sliding":           True
    "groups": {
        <res>: {
            "tokens":  [num_crops, N, 8]   — sliding dim replaces batch dim
            "mask":    [num_crops, N]
            "shape":   (C, H, W)
        },
    },
    "queries":           [num_crops, M, 8]
    "queries_mask":      [num_crops, M]
    "crop_positions":    [(y0, x0), ...]
    "crop_size":         (crop_h, crop_w)
    "full_size":         (full_h, full_w)
    "label":             [H, W]             — NOT batched (single tile)
    "target_resolution": float
    "image":             [C, H, W]          — NOT batched
}

The val/test step can then iterate over dim 0 of groups/queries,
treating each slice as a B=1 mini-batch.
"""

import torch


# =============================================================================
# COLLATE
# =============================================================================

def collate_grouped(batch: list[dict]) -> dict:
    """
    Collate a batch of grouped-format samples.

    Two modes:
    1. Normal (train): all samples have same shape → stack into [B, ...].
    2. Sliding window (val/test): batch_size=1, sample contains a list of
       crops. Stack crops along dim 0 → [num_crops, ...]. Metadata (label,
       image, positions) is kept as-is (single tile).

    The downstream code sees the same tensor shapes in both cases:
      groups[res]["tokens"] is [B_or_crops, N, 8]
      queries is [B_or_crops, M, 8]
    So the encoder/decoder can process them identically.
    """

    # ── Sliding window path ─────────────────────────────────
    if batch[0].get("sliding", False):
        assert len(batch) == 1, (
            f"Sliding window requires batch_size=1, got {len(batch)}"
        )
        return _collate_sliding(batch[0])

    # ── Normal path (training) ──────────────────────────────
    return _collate_normal(batch)


def _collate_normal(batch: list[dict]) -> dict:
    """Stack normal samples along batch dimension."""
    all_resolutions = list(batch[0]["groups"].keys())

    groups = {}
    for res in all_resolutions:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in batch]),
            "mask":   torch.stack([s["groups"][res]["mask"] for s in batch]),
            "shape":  batch[0]["groups"][res]["shape"],
        }

    return {
        "groups":            groups,
        "queries":           torch.stack([s["queries"] for s in batch]),
        "queries_mask":      torch.stack([s["queries_mask"] for s in batch]),
        "label":             torch.stack([s["label"] for s in batch]),
        "target_resolution": batch[0]["target_resolution"],
        "image":             torch.stack([s["image"] for s in batch]),
    }


def _collate_sliding(sample: dict) -> dict:
    """
    Collate a single sliding-window sample.

    Stacks the list of crop dicts into tensors where dim 0 = num_crops.
    This makes the output shape-compatible with the normal collated batch
    (dim 0 is just num_crops instead of batch_size).
    """
    crops = sample["crops"]
    num_crops = len(crops)
    all_resolutions = list(crops[0]["groups"].keys())

    # Stack crop tokens along dim 0 → [num_crops, N, 8]
    groups = {}
    for res in all_resolutions:
        groups[res] = {
            "tokens": torch.stack([c["groups"][res]["tokens"] for c in crops]),
            "mask":   torch.stack([c["groups"][res]["mask"] for c in crops]),
            "shape":  crops[0]["groups"][res]["shape"],
        }

    return {
        "sliding":           True,
        "groups":            groups,
        "queries":           torch.stack([c["queries"] for c in crops]),
        "queries_mask":      torch.stack([c["queries_mask"] for c in crops]),
        # ── Per-tile metadata (NOT stacked) ──
        "crop_positions":    sample["crop_positions"],
        "crop_size":         sample["crop_size"],
        "full_size":         sample["full_size"],
        "label":             sample["label"],
        "target_resolution": sample["target_resolution"],
        "image":             sample["image"],
    }


# =============================================================================
# LATENT GRID CONFIG (computed from data, not from yaml)
# =============================================================================

def compute_grid_config(
    resolution: float,
    shape: tuple,
    pixels_per_latent: int = 50,
    sigma_factor: float = 1.5,
    max_k: int = 2000,
    hexagonal: bool = False,
) -> dict:
    """
    Derive latent grid parameters from resolution + image shape.

    The ONLY tunable hyperparameter is pixels_per_latent:
      - pixels_per_latent=50, image=512×512 → 10×10 latent grid
      - pixels_per_latent=50, image=120×120 → 2×2  latent grid
      - pixels_per_latent=50, image=60×60   → 1×1  latent grid

    Everything else (sigma, k, span) follows automatically.
    """
    C, H, W = shape

    latents_x = max(1, round(W / pixels_per_latent))
    latents_y = max(1, round(H / pixels_per_latent))

    span_x = W * resolution
    span_y = H * resolution

    cell_x_m = span_x / latents_x
    cell_y_m = span_y / latents_y
    cell_m = max(cell_x_m, cell_y_m)

    geo_sigma = cell_m * sigma_factor

    total_tokens = C * H * W
    tokens_per_latent = total_tokens / (latents_x * latents_y)
    geo_k = min(int(tokens_per_latent * 2), max_k)

    train_k = min(500, geo_k)
    val_k = min(1000, geo_k)

    return {
        "latents_x": latents_x,
        "latents_y": latents_y,
        "L_spatial": latents_x * latents_y,
        "geo_sigma": geo_sigma,
        "geo_k": geo_k,
        "train_k": train_k,
        "val_k": val_k,
        "span_x": span_x,
        "span_y": span_y,
        "cell_size_m": cell_m,
        "resolution": resolution,
        "pixels_per_latent": pixels_per_latent,
        "hexagonal": hexagonal,
    }