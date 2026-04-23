"""
Spatial Block Masking for MAE Pre-Training
==========================================

Generates spatially coherent block masks for remote sensing tokens.
Masking is applied independently per modality to enable cross-modal
reconstruction learning.

When a pixel (x, y) is masked, ALL spectral bands at that pixel are masked.
This prevents trivial same-pixel interpolation.

Usage:
    mask = generate_spatial_block_mask(H=128, W=128, mask_ratio=0.75)
    # mask: [H, W] bool, True = masked

    token_mask = expand_mask_to_tokens(mask, num_bands=10, num_times=1)
    # token_mask: [num_times * num_bands * H * W] bool
"""

import torch
import torch.nn.functional as F


def generate_spatial_block_mask(
    H: int,
    W: int,
    mask_ratio: float = 0.75,
    block_size: int = 8,
) -> torch.Tensor:
    """
    Generate a spatially coherent block mask.

    Strategy:
        1. Generate low-resolution random noise (block_size × block_size)
        2. Upsample to (H, W) via bicubic interpolation → smooth blocks
        3. Threshold to achieve EXACT mask_ratio

    The exact threshold is computed by sorting the noise values and
    picking the cutoff that gives precisely the desired ratio. This
    guarantees a deterministic token count after masking.

    Args:
        H: Image height in pixels
        W: Image width in pixels
        mask_ratio: Fraction of pixels to mask (0.0–1.0)
        block_size: Size of the low-res noise grid. Smaller = larger blocks.
                    8 works well for 64–128px images.

    Returns:
        mask: [H, W] bool tensor. True = masked, False = visible.
    """
    # Low-res random noise
    noise_h = min(block_size, H)
    noise_w = min(block_size, W)
    noise = torch.rand(1, 1, noise_h, noise_w)

    # Upsample to full resolution — smooth blocks
    if noise_h < H or noise_w < W:
        noise = F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=False)

    noise = noise.squeeze(0).squeeze(0)  # [H, W]

    # Exact threshold: sort and pick cutoff for precise mask_ratio
    flat = noise.flatten()
    n_masked = int(H * W * mask_ratio)
    n_masked = max(1, min(n_masked, H * W - 1))  # at least 1 visible

    # Tokens with noise value >= threshold are masked
    threshold = torch.kthvalue(flat, H * W - n_masked).values
    mask = noise > threshold

    return mask


def expand_mask_to_tokens(
    spatial_mask: torch.Tensor,
    num_bands: int,
    num_times: int = 1,
) -> torch.Tensor:
    """
    Expand a spatial mask [H, W] to cover all bands and time steps.

    Token ordering follows TokenBuilder convention:
        For each time step t:
            For each band c:
                For each pixel (flattened H*W):
                    token index = t * (C * H * W) + c * (H * W) + pixel_idx

    When a pixel is masked, ALL bands at that pixel (for that time step)
    are masked. This prevents trivial same-pixel interpolation.

    Args:
        spatial_mask: [H, W] bool, True = masked
        num_bands: Number of spectral bands (C)
        num_times: Number of temporal frames (T)

    Returns:
        token_mask: [T * C * H * W] bool, True = masked
    """
    H, W = spatial_mask.shape
    # Expand: [H, W] → [C, H, W] (same mask for all bands)
    band_mask = spatial_mask.unsqueeze(0).expand(num_bands, -1, -1)  # [C, H, W]
    # Flatten: [C * H * W]
    frame_mask = band_mask.reshape(-1)
    # Repeat for time steps: [T * C * H * W]
    if num_times > 1:
        return frame_mask.repeat(num_times)
    return frame_mask


def apply_mask_to_tokens(
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    target_visible: int = None,
) -> tuple:
    """
    Split tokens into visible and masked sets, with padding to fixed size.

    Args:
        tokens: [N, 8] all tokens for one modality
        token_mask: [N] bool, True = masked
        target_visible: If set, pad/trim visible tokens to this exact count.
                       If None, return variable-length visible tokens.

    Returns:
        visible_tokens: [N_vis, 8] (or [target_visible, 8] if padded)
        visible_pad_mask: [N_vis] bool, True = padded (invalid)
        masked_tokens: [N_masked, 8] (candidates for reconstruction queries)
    """
    visible_tokens = tokens[~token_mask]
    masked_tokens = tokens[token_mask]

    if target_visible is None:
        visible_pad_mask = torch.zeros(visible_tokens.shape[0], dtype=torch.bool)
        return visible_tokens, visible_pad_mask, masked_tokens

    n_visible = visible_tokens.shape[0]

    if n_visible == target_visible:
        visible_pad_mask = torch.zeros(target_visible, dtype=torch.bool)
    elif n_visible < target_visible:
        # Pad with zeros, mark as invalid
        pad = torch.zeros(target_visible - n_visible, 8, dtype=tokens.dtype)
        visible_tokens = torch.cat([visible_tokens, pad], dim=0)
        visible_pad_mask = torch.zeros(target_visible, dtype=torch.bool)
        visible_pad_mask[n_visible:] = True
    else:
        # Too many visible — randomly drop excess
        perm = torch.randperm(n_visible)[:target_visible]
        visible_tokens = visible_tokens[perm]
        visible_pad_mask = torch.zeros(target_visible, dtype=torch.bool)

    return visible_tokens, visible_pad_mask, masked_tokens


def build_mae_queries(
    masked_tokens: torch.Tensor,
    max_queries: int = 200_000,
) -> tuple:
    """
    Build MAE reconstruction queries from masked tokens.

    Each query is a copy of the masked token with value (column 0) zeroed.
    The ground truth is the original value from column 0.

    Args:
        masked_tokens: [N_masked, 8] tokens that were spatially masked
        max_queries: Maximum number of queries to sample

    Returns:
        queries: [M, 8] query tokens (column 0 = 0)
        ground_truth: [M] target scalar values (original column 0)
        queries_mask: [M] bool, all False (all valid)
    """
    N = masked_tokens.shape[0]

    if max_queries is not None and N > max_queries:
        perm = torch.randperm(N)[:max_queries]
        sampled = masked_tokens[perm]
    elif N == 0:
        # Edge case: nothing masked
        queries = torch.zeros(1, 8, dtype=masked_tokens.dtype)
        return queries, torch.zeros(1), torch.ones(1, dtype=torch.bool)
    else:
        sampled = masked_tokens

    # Ground truth = column 0 (normalized reflectance/backscatter)
    ground_truth = sampled[:, 0].clone()

    # Query = same metadata, but value zeroed out
    queries = sampled.clone()
    queries[:, 0] = 0.0

    queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

    return queries, ground_truth, queries_mask