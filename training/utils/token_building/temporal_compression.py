"""
Temporal Token Compression — Pack Approach
=============================================

Compresses T tokens per (pixel, band) into 1 wider token:

    Before: [C * T * H * W, 8]        — one per observation
    After:  [C * H * W, 8 + 2*K]      — one per (pixel, band)

    Columns 0-7:   standard metadata (col 0 = most recent reflectance)
    Columns 8..8+K-1:    temporal profile  (K values)
    Columns 8+K..8+2K-1: temporal support  (K values)

Why pack into the token tensor:
    - collate (_pad_tokens) handles any D automatically → no change
    - geographic pruning reads col 1,2 for position → no change
    - token sampling subsets along k dim → no change
    - ONLY the TokenProcessor needs to split out profiles

Integration:
    1. Dataset: call compress_temporal_tokens() → [N, 8+2K] tokens
    2. TokenProcessor: detect wider tokens, split, use profile

    Everything else (collate, geo pruning, sampling, encoder) unchanged.
"""

import torch
from typing import List
from training.atomiser.temporal_profile import TemporalReflectanceProfile


def compress_temporal_tokens(
    frames: List[torch.Tensor],
    label: torch.Tensor,
    delta_days: List[float],
    spectral_indices: torch.Tensor,
    resolution_idx: int,
    token_builder,
    temporal_encoder: TemporalReflectanceProfile,
    resolution: float = 10.0,
    time_idx_default: int = 0,
) -> torch.Tensor:
    """
    Compress T temporal frames into profile-packed tokens.

    Args:
        frames:           list of [C, H, W] tensors
        label:            [H, W] label map
        delta_days:       list of T floats
        spectral_indices: [C] long tensor
        resolution_idx:   int
        token_builder:    TokenBuilder
        temporal_encoder: TemporalReflectanceProfile
        resolution:       GSD in meters
        time_idx_default: time_idx for col 7

    Returns:
        tokens: [C*H*W, 8 + 2*K] — packed with profiles
    """
    T = len(frames)
    C, H, W = frames[0].shape
    N = C * H * W
    K = temporal_encoder.n_centers

    # ── Stack and sort by Δt (most recent first) ─────────────────
    refl_stack = torch.stack(frames, dim=1)  # [C, T, H, W]
    dt_tensor = torch.tensor(delta_days, dtype=torch.float32)
    sort_idx = torch.argsort(dt_tensor)
    dt_sorted = dt_tensor[sort_idx]
    refl_sorted = refl_stack[:, sort_idx]    # [C, T, H, W]

    # ── Reshape for temporal encoder: [C*H*W, T] ────────────────
    refl_flat = refl_sorted.permute(0, 2, 3, 1).reshape(N, T)
    dt_flat = dt_sorted.unsqueeze(0).expand(N, T)

    # ── Compute profile and support ──────────────────────────────
    profiles, supports = temporal_encoder(refl_flat, dt_flat)  # [N, K] each

    # ── Build base tokens from most recent frame ─────────────────
    most_recent_frame = refl_sorted[:, 0]  # [C, H, W]
    base_tokens = token_builder.build_tokens(
        image=most_recent_frame,
        label=label,
        resolution=resolution,
        spectral_indices=spectral_indices,
        resolution_idx=resolution_idx,
        time_idx=time_idx_default,
    )  # [C*H*W, 8]

    # col 7 = mean Δt (temporal center)
    base_tokens[:, 7] = dt_sorted.mean().item()

    # ── Pack: [N, 8] + [N, K] + [N, K] → [N, 8 + 2K] ───────────
    packed = torch.cat([base_tokens, profiles, supports], dim=1)

    return packed


def unpack_temporal_tokens(tokens: torch.Tensor, n_centers: int):
    """
    Split packed tokens back into base tokens + profiles + supports.

    Args:
        tokens:    [*, 8 + 2*K] packed tokens
        n_centers: K (number of temporal Gaussian centers)

    Returns:
        base:     [*, 8]
        profiles: [*, K]
        supports: [*, K]
    """
    K = n_centers
    base = tokens[..., :8]
    profiles = tokens[..., 8:8 + K]
    supports = tokens[..., 8 + K:8 + 2 * K]
    return base, profiles, supports


def has_temporal_profile(tokens: torch.Tensor) -> bool:
    """Check if tokens have packed temporal profiles (width > 8)."""
    return tokens.shape[-1] > 8