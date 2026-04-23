"""
Query Augmentation — Boundary-Focused Upsampled Queries
========================================================

During training, augments the query resolution to produce sub-pixel
predictions focused on class boundaries and small objects.

Strategy:
  1. Upsample the 10m label to a finer resolution (e.g., 5m) via
     nearest-neighbor interpolation.
  2. Detect boundary zones: pixels where at least one neighbor belongs
     to a different class.
  3. Sample queries with a configurable mix of boundary (~70%) and
     interior (~30%) pixels.
  4. Build queries at the augmented resolution with proper positional
     and resolution encoding.

This teaches the decoder to:
  - Produce predictions at sub-pixel resolution
  - Focus on class transitions where segmentation quality matters most
  - Learn spatially precise features that transfer better across cities

Usage in C2SegDataset.__getitem__:
    if self.augment and self.query_augment_config is not None:
        queries = build_augmented_queries(
            label_10m=label,
            token_builder=self.token_builder,
            look_up=self.look_up,
            query_info=query_info,
            config=self.query_augment_config,
        )
"""

import torch
import torch.nn.functional as F
import random
from typing import Optional


IGNORE_INDEX = 255


def detect_boundaries(label: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    Detect boundary pixels: pixels where at least one 4-connected neighbor
    has a different class.

    Parameters
    ----------
    label : Tensor [H, W]
        Class labels.
    ignore_index : int
        Index to treat as invalid.

    Returns
    -------
    Tensor [H, W] : bool mask, True at boundary pixels.
    """
    H, W = label.shape
    valid = (label != ignore_index)

    # Pad label for neighbor comparison
    padded = F.pad(label.unsqueeze(0).float(), (1, 1, 1, 1), mode="replicate").squeeze(0).long()

    # 4-connected neighbors
    center = label
    up = padded[0:H, 1:W+1]
    down = padded[2:H+2, 1:W+1]
    left = padded[1:H+1, 0:W]
    right = padded[1:H+1, 2:W+2]

    # Boundary = any neighbor differs AND both are valid
    diff_up = (center != up) & valid & (up != ignore_index)
    diff_down = (center != down) & valid & (down != ignore_index)
    diff_left = (center != left) & valid & (left != ignore_index)
    diff_right = (center != right) & valid & (right != ignore_index)

    boundary = diff_up | diff_down | diff_left | diff_right
    return boundary


def dilate_boundary(boundary: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """
    Dilate boundary mask to include a band of pixels around boundaries.
    This captures the context around transitions, not just the edge pixels.

    Parameters
    ----------
    boundary : Tensor [H, W] bool
    dilation : int
        Number of pixels to dilate on each side.

    Returns
    -------
    Tensor [H, W] : dilated bool mask.
    """
    if dilation <= 0:
        return boundary

    # Use max_pool2d as morphological dilation
    kernel = 2 * dilation + 1
    dilated = F.max_pool2d(
        boundary.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=dilation,
    ).squeeze(0).squeeze(0)

    return dilated > 0


def upsample_label(label: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Upsample label via nearest-neighbor interpolation.

    Parameters
    ----------
    label : Tensor [H, W] int64
    factor : int
        Upsampling factor (e.g., 2 for 10m → 5m).

    Returns
    -------
    Tensor [H*factor, W*factor] int64
    """
    # F.interpolate expects [B, C, H, W] float
    upsampled = F.interpolate(
        label.float().unsqueeze(0).unsqueeze(0),
        scale_factor=factor,
        mode="nearest",
    ).squeeze(0).squeeze(0).long()

    return upsampled


def sample_boundary_focused_indices(
    label: torch.Tensor,
    max_queries: int,
    boundary_fraction: float = 0.7,
    dilation: int = 2,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Sample pixel indices focused on class boundaries.

    Parameters
    ----------
    label : Tensor [H, W]
        Class labels (potentially upsampled).
    max_queries : int
        Total number of queries to sample.
    boundary_fraction : float
        Fraction of queries to draw from boundary zones (default: 0.7).
    dilation : int
        Dilation radius around boundaries (default: 2 pixels).
    ignore_index : int
        Index to treat as invalid.

    Returns
    -------
    Tensor [max_queries, 2] : (row, col) indices into the label grid.
    """
    H, W = label.shape
    valid = (label != ignore_index)

    # Detect and dilate boundaries
    boundary = detect_boundaries(label, ignore_index)
    boundary_zone = dilate_boundary(boundary, dilation)

    # Separate boundary and interior valid pixels
    boundary_valid = boundary_zone & valid
    interior_valid = (~boundary_zone) & valid

    boundary_indices = boundary_valid.nonzero(as_tuple=False)  # [N_b, 2]
    interior_indices = interior_valid.nonzero(as_tuple=False)  # [N_i, 2]

    n_boundary = boundary_indices.shape[0]
    n_interior = interior_indices.shape[0]

    # Compute target counts
    n_boundary_target = int(max_queries * boundary_fraction)
    n_interior_target = max_queries - n_boundary_target

    # If not enough boundary pixels, take what's available and fill with interior
    if n_boundary < n_boundary_target:
        n_boundary_target = n_boundary
        n_interior_target = max_queries - n_boundary_target

    # If not enough interior pixels, take what's available and fill with boundary
    if n_interior < n_interior_target:
        n_interior_target = n_interior
        n_boundary_target = min(n_boundary, max_queries - n_interior_target)

    total_available = n_boundary_target + n_interior_target
    if total_available == 0:
        # Fallback: sample from all valid pixels
        all_valid = valid.nonzero(as_tuple=False)
        if all_valid.shape[0] == 0:
            # No valid pixels at all — return zeros
            return torch.zeros(max_queries, 2, dtype=torch.long)
        perm = torch.randperm(all_valid.shape[0])[:max_queries]
        return all_valid[perm]

    # Sample from each pool
    selected = []

    if n_boundary_target > 0:
        perm = torch.randperm(n_boundary)[:n_boundary_target]
        selected.append(boundary_indices[perm])

    if n_interior_target > 0:
        perm = torch.randperm(n_interior)[:n_interior_target]
        selected.append(interior_indices[perm])

    indices = torch.cat(selected, dim=0)

    # Shuffle so boundary and interior are mixed
    perm = torch.randperm(indices.shape[0])
    indices = indices[perm]

    return indices


def build_augmented_queries(
    label_10m: torch.Tensor,
    token_builder,
    look_up,
    first_spectral_idx: int,
    max_queries: int = 16384,
    upsample_factor: int = 2,
    target_gsd: float = 5.0,
    boundary_fraction: float = 0.7,
    boundary_dilation: int = 2,
    time_idx: int = -1,
    prob: float = 0.5,
) -> Optional[torch.Tensor]:
    """
    Build boundary-focused queries at an upsampled resolution.

    With probability `prob`, returns augmented queries at `target_gsd`.
    Otherwise returns None (caller should use standard queries).

    Parameters
    ----------
    label_10m : Tensor [H, W]
        Original label at 10m resolution.
    token_builder : TokenBuilder
        Builds query tokens with positional/spectral encoding.
    look_up : Lookup_encoding
        Lookup table for resolution indices.
    first_spectral_idx : int
        Spectral index for the query (from the reference sensor).
    max_queries : int
        Total number of queries to generate.
    upsample_factor : int
        Spatial upsampling factor (2 = 10m→5m, 4 = 10m→2.5m).
    target_gsd : float
        Target ground sampling distance in meters.
    boundary_fraction : float
        Fraction of queries from boundary zones.
    boundary_dilation : int
        Dilation radius around boundaries in upsampled pixels.
    time_idx : int
        Time index (-1 for static).
    prob : float
        Probability of applying augmentation (0 = never, 1 = always).

    Returns
    -------
    Tensor [max_queries, 8] or None
        Augmented query tokens, or None if not applied this sample.
    """
    # Stochastic: only apply with probability `prob`
    if random.random() > prob:
        return None

    # Upsample label
    label_up = upsample_label(label_10m, upsample_factor)
    H_up, W_up = label_up.shape

    # Sample boundary-focused pixel indices
    indices = sample_boundary_focused_indices(
        label=label_up,
        max_queries=max_queries,
        boundary_fraction=boundary_fraction,
        dilation=boundary_dilation,
    )

    # Get resolution index for target GSD
    res_idx = look_up.get_resolution_idx(target_gsd)

    # Build query tokens for selected pixels
    # Each query needs: [value=0, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
    n_queries = indices.shape[0]
    queries = torch.zeros(n_queries, 8)

    rows = indices[:, 0]  # row indices in upsampled grid
    cols = indices[:, 1]  # col indices in upsampled grid

    # Get position indices from token_builder's reference grid
    # The token builder maps (row, col) at a given resolution to position indices
    if hasattr(token_builder, 'reference_grids') and target_gsd in token_builder.reference_grids:
        offset = token_builder.reference_grids[target_gsd]
    else:
        # Fallback: register the resolution and get offset
        from .token_builder import TokenBuilder
        if target_gsd not in TokenBuilder.REFERENCE_SIZES:
            TokenBuilder.REFERENCE_SIZES[target_gsd] = 2048
        offset = look_up.get_or_register_modality(target_gsd, 2048)

    # Fill query tokens
    queries[:, 0] = 0.0  # value (unused for queries)
    queries[:, 1] = (cols + offset).float()  # x position
    queries[:, 2] = (rows + offset).float()  # y position
    queries[:, 3] = float(first_spectral_idx)  # spectral index
    queries[:, 4] = label_up[rows, cols].float()  # label at upsampled position
    queries[:, 5] = 0.0  # query index (unused)
    queries[:, 6] = float(res_idx)  # resolution index
    queries[:, 7] = float(time_idx)  # time index

    return queries