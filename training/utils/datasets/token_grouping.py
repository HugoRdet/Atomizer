"""
Grouped token format for Atomizer datasets.

Return contract from __getitem__:
{
    "groups": {
        <resolution_m>: {
            "tokens":  [N, 6]   — [value, pixel_x, pixel_y, spectral_idx]
            "mask":    [N]      — 0.0 = valid, 1.0 = ignore
            "shape":   (C, H, W)
        },
        ...  # one entry per native resolution
    },
    "label":             [H, W]   at target resolution
    "target_resolution": float    resolution of the label grid (meters)
    "image":             [C, H, W] raw normalized image for visualization
}

Design principles:
  - Tokens carry raw pixel coordinates (0..H-1, 0..W-1), NOT lookup indices.
    The model scales to physical coords: physical = (pixel - size/2) * resolution.
  - Resolution is the dict KEY, not packed into tokens.
  - Spectral identity comes from spectral_idx (lookup table for RBF encoding).
  - Labels live outside the tokens.
  - No modality string needed — the model derives grid config from 
    resolution + shape + pixels_per_latent hyperparameter.
  - SAR / non-optical channels get a hardcoded resolution and distinct 
    spectral_idx values.
"""

import torch
import numpy as np
import einops


# =============================================================================
# TOKEN BUILDING HELPERS
# =============================================================================

def build_group_tokens(
    image: torch.Tensor,           # [C, H, W] normalized values
    spectral_indices: torch.Tensor, # [C] global spectral idx per band
    nodata_value: float = None,     # value that marks invalid pixels
) -> dict:
    """
    Build tokens for a single resolution group.
    
    Args:
        image: [C, H, W] band values (normalized)
        spectral_indices: [C] one spectral lookup index per band
        nodata_value: if set, pixels with this value get mask=1.0
    
    Returns:
        {
            "tokens": [C*H*W, 6]  — [value, pixel_x, pixel_y, spectral_idx]
            "mask":   [C*H*W]     — 0.0 valid, 1.0 ignore
            "shape":  (C, H, W)
        }
    """
    C, H, W = image.shape

    # Pixel coordinates: same for every band
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
    # [H, W] each → expand to [C, H, W]
    px = einops.repeat(grid_x, "h w -> c h w", c=C)
    py = einops.repeat(grid_y, "h w -> c h w", c=C)

    # Spectral index: [C] → [C, H, W]
    spec = einops.repeat(spectral_indices, "c -> c h w", h=H, w=W).float()

    # Stack: [C, H, W, 4]
    tokens = torch.stack([image, px, py, spec], dim=-1)

    # Flatten: [C*H*W, 4]
    tokens = einops.rearrange(tokens, "c h w f -> (c h w) f")

    # Attention mask
    mask = torch.zeros(C * H * W)
    if nodata_value is not None:
        flat_vals = einops.rearrange(image, "c h w -> (c h w)")
        mask[flat_vals == nodata_value] = 1.0

    return {
        "tokens": tokens,
        "mask": mask,
        "shape": (C, H, W),
    }


def collate_grouped(batch: list[dict]) -> dict:
    """
    Collate a batch of grouped-format samples.
    
    Since all samples from the same dataset have identical resolution keys 
    and shapes, we just stack along batch dim — no padding needed.
    
    Args:
        batch: list of dicts from __getitem__
    
    Returns:
        {
            "groups": {
                res: {
                    "tokens": [B, N, 4],
                    "mask":   [B, N],
                    "shape":  (C, H, W),  # shared across batch
                },
            },
            "label": [B, H, W],
            "target_resolution": float,
            "image": [B, C, H, W],
        }
    """
    all_resolutions = list(batch[0]["groups"].keys())

    groups = {}
    for res in all_resolutions:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in batch]),
            "mask": torch.stack([s["groups"][res]["mask"] for s in batch]),
            "shape": batch[0]["groups"][res]["shape"],  # same for all
        }

    return {
        "groups": groups,
        "label": torch.stack([s["label"] for s in batch]),
        "target_resolution": batch[0]["target_resolution"],
        "image": torch.stack([s["image"] for s in batch]),
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
    min_k: int = 200,
) -> dict:
    """
    Derive latent grid parameters from resolution + image shape.
    
    The ONLY hyperparameter is pixels_per_latent:
      - pixels_per_latent=50, image=512×512 → 10×10 latent grid
      - pixels_per_latent=50, image=120×120 → 2×2  latent grid
      - pixels_per_latent=50, image=60×60   → 1×1  latent grid
    
    Everything else (sigma, k, span) follows automatically.
    
    Args:
        resolution: meters per pixel
        shape: (C, H, W)
        pixels_per_latent: pixels per latent grid cell (one side)
        sigma_factor: sigma = cell_size_physical * factor
        max_k, min_k: bounds on tokens per latent
    
    Returns:
        dict with: latents_x, latents_y, L_spatial, geo_sigma, 
                   geo_k, span_x, span_y, resolution
    """
    C, H, W = shape
    
    # Latent grid dimensions (at least 1 per side)
    latents_x = max(1, round(W / pixels_per_latent))
    latents_y = max(1, round(H / pixels_per_latent))
    
    # Physical extents
    span_x = W * resolution   # meters
    span_y = H * resolution   # meters
    
    # Cell size in meters (for sigma)
    cell_x = span_x / latents_x
    cell_y = span_y / latents_y
    cell_size = max(cell_x, cell_y)
    
    # Sigma: receptive field in normalized coordinates
    # (the model works in resolution-normalized space where 1 unit = G meters)
    geo_sigma = (cell_size / resolution) * sigma_factor  # in pixel units
    
    # k: how many tokens each latent should attend to
    total_tokens = C * H * W
    tokens_per_latent = total_tokens / (latents_x * latents_y)
    geo_k = int(np.clip(tokens_per_latent * 2, min_k, max_k))
    
    # Sub-sampling for training/val (as fraction of geo_k)
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
        "cell_size": cell_size,
        "resolution": resolution,
        "pixels_per_latent": pixels_per_latent,
    }