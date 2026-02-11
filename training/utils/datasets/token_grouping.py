"""
Grouped token utilities for Atomizer datasets.

Return contract from __getitem__:
{
    "groups": {
        <resolution_m>: {
            "tokens":  [N, 6]   — [value, x, y, spectral_idx, label, query_idx]
            "mask":    [N]      — 0.0 = valid, 1.0 = ignore
            "shape":   (C, H, W)
        },
        ...  # one entry per native resolution
    },
    "queries":           [M, 6]   — sub-sampled query tokens
    "queries_mask":      [M]
    "label":             [H, W]   — at target resolution
    "target_resolution": float    — resolution of the label grid (meters)
    "image":             [C, H, W] — raw normalized image for visualization
}

Token format (unchanged):
    [0] value        — normalized reflectance / backscatter
    [1] x            — pixel x position (lookup-indexed)
    [2] y            — pixel y position (lookup-indexed)
    [3] spectral_idx — global spectral index (lookup table for RBF encoding)
    [4] label        — per-pixel class label
    [5] query_idx    — query position index (for decoder)
"""

import torch


# =============================================================================
# COLLATE
# =============================================================================

def collate_grouped(batch: list[dict]) -> dict:
    """
    Collate a batch of grouped-format samples.
    
    All samples from the same dataset have identical resolution keys
    and shapes, so we just stack — no padding needed.
    
    Args:
        batch: list of dicts from __getitem__
    
    Returns:
        Same structure with batch dimension prepended to all tensors.
    """
    all_resolutions = list(batch[0]["groups"].keys())

    groups = {}
    for res in all_resolutions:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in batch]),   # [B, N, 6]
            "mask":   torch.stack([s["groups"][res]["mask"] for s in batch]),     # [B, N]
            "shape":  batch[0]["groups"][res]["shape"],                           # shared
        }

    return {
        "groups":            groups,
        "queries":           torch.stack([s["queries"] for s in batch]),           # [B, M, 6]
        "queries_mask":      torch.stack([s["queries_mask"] for s in batch]),      # [B, M]
        "label":             torch.stack([s["label"] for s in batch]),             # [B, H, W]
        "target_resolution": batch[0]["target_resolution"],                        # scalar
        "image":             torch.stack([s["image"] for s in batch]),             # [B, C, H, W]
    }


# =============================================================================
# LATENT GRID CONFIG (computed from data, not from yaml)
# =============================================================================

# ============================================================================
# In token_grouping.py
# ============================================================================

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
    
    Args:
        resolution: meters per pixel
        shape: (C, H, W)
        pixels_per_latent: pixels between latent centers (one axis)
        sigma_factor: geo_sigma = cell_size_meters * factor
        max_k: upper bound on tokens per latent for geographic pruning
        hexagonal: if True, use staggered rows for hexagonal Voronoi cells
    
    Returns:
        dict with grid parameters
    """
    C, H, W = shape
    
    # Latent grid dimensions (at least 1 per side)
    latents_x = max(1, round(W / pixels_per_latent))
    latents_y = max(1, round(H / pixels_per_latent))
    
    # Physical extents (meters)
    span_x = W * resolution
    span_y = H * resolution
    
    # Cell size in meters
    cell_x_m = span_x / latents_x
    cell_y_m = span_y / latents_y
    cell_m = max(cell_x_m, cell_y_m)
    
    # Sigma in meters (used by geographic bias: -dist_m² / 2σ_m²)
    geo_sigma = cell_m * sigma_factor
    
    # k: how many tokens each latent should attend to
    total_tokens = C * H * W
    tokens_per_latent = total_tokens / (latents_x * latents_y)
    geo_k = min(int(tokens_per_latent * 2), max_k)
    
    # Sub-sampling for train/val cross-attention
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