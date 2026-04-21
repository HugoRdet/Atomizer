"""
Error Supervision for Latent Refinement
========================================

Computes per-latent soft cross-entropy error from segmentation predictions.
Used to supervise the error predictor MLP and to guide latent refinement.

The key insight: each query pixel is assigned to its k nearest latents
(the same k used in the decoder). The per-pixel CE error is distributed
across those k latents using inverse-distance weighting, producing a
smooth error field over the latent grid.

Usage (inside Atomiser forward):
    from .error_supervision import compute_latent_errors

    zone_error, valid_mask = compute_latent_errors(
        logits       = y_hat,          # [B, M, C]  raw segmentation logits
        labels       = labels,          # [B, M]     int class labels
        topk_indices = topk_indices,    # [B, M, k]  from reconstruct()
        topk_dists_sq= topk_dists_sq,   # [B, M, k]  from reconstruct()
        num_latents  = L,
        ignore_index = 255,
    )
    # zone_error:  [B, L]  mean weighted CE per latent zone (detached)
    # valid_mask:  [B, L]  True where latent has ≥1 labeled pixel

    error_pred_loss = compute_error_predictor_loss(
        predicted_errors = predicted_errors,  # [B, L]  from error_predictor MLP
        zone_error       = zone_error,         # [B, L]  target (detached)
        valid_mask       = valid_mask,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


# =============================================================================
# LATENT ZONE ERROR
# =============================================================================

def compute_latent_errors(
    logits:        torch.Tensor,   # [B, M, C]
    labels:        torch.Tensor,   # [B, M]
    topk_indices:  torch.Tensor,   # [B, M, k]
    topk_dists_sq: torch.Tensor,   # [B, M, k]
    num_latents:   int,
    ignore_index:  int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft cross-entropy per latent zone using inverse-distance weighting.

    Each query pixel contributes its CE error to its k nearest latents,
    weighted by inverse squared distance. This produces a smoother error
    field than hard Voronoi assignment (k=1), especially at zone boundaries.

    Args:
        logits:        [B, M, C]  raw (unnormalized) segmentation logits
        labels:        [B, M]     integer class labels
        topk_indices:  [B, M, k]  indices of k nearest latents per pixel
                                   (reused directly from reconstruct())
        topk_dists_sq: [B, M, k]  squared distances to those latents
        num_latents:   L           total number of latents
        ignore_index:              label value to exclude from error

    Returns:
        zone_error:  [B, L]  inverse-distance-weighted mean CE per latent
                              detached — no gradient flows to segmentation head
        valid_mask:  [B, L]  True where latent received ≥1 labeled pixel
    """
    B, M, C = logits.shape
    k       = topk_indices.shape[-1]
    L       = num_latents
    device  = logits.device

    # ── Per-pixel soft CE ─────────────────────────────────────────────
    valid_pixels = (labels != ignore_index)   # [B, M]  bool

    # Replace ignore labels with 0 to avoid indexing errors in cross_entropy
    labels_safe = labels.clone()
    labels_safe[~valid_pixels] = 0

    ce_per_pixel = F.cross_entropy(
        rearrange(logits,      "b m c -> (b m) c"),
        rearrange(labels_safe, "b m   -> (b m)"),
        reduction="none",
    ).reshape(B, M)                            # [B, M]

    # Zero out ignored pixels — they contribute nothing
    ce_per_pixel = ce_per_pixel * valid_pixels.float()

    # ── Inverse-distance weights ──────────────────────────────────────
    # Closer latents get more of the pixel's error signal.
    # Add small epsilon to avoid division by zero for exactly coincident points.
    inv_dists = 1.0 / (topk_dists_sq + 1e-8)                       # [B, M, k]
    weights   = inv_dists / inv_dists.sum(dim=-1, keepdim=True)     # [B, M, k]

    # Weighted CE contribution per (pixel, latent) pair
    weighted_ce = ce_per_pixel.unsqueeze(-1) * weights              # [B, M, k]

    # ── Scatter to latent zones ───────────────────────────────────────
    # Flatten M × k → single dimension for scatter_add
    flat_indices  = topk_indices.reshape(B, M * k)                  # [B, M*k]
    flat_ce       = weighted_ce.reshape(B, M * k)                   # [B, M*k]

    # For counting: each valid pixel contributes 1/k to each of its k latents
    flat_valid = (
        valid_pixels
        .unsqueeze(-1)
        .expand(-1, -1, k)
        .reshape(B, M * k)
        .float()
    ) / k                                                            # [B, M*k]

    zone_ce    = torch.zeros(B, L, device=device)
    zone_count = torch.zeros(B, L, device=device)

    zone_ce.scatter_add_(1, flat_indices, flat_ce)
    zone_count.scatter_add_(1, flat_indices, flat_valid)

    # ── Normalize and mask ────────────────────────────────────────────
    valid_mask = zone_count > 0                                      # [B, L]
    zone_error = zone_ce / zone_count.clamp(min=1.0)                # [B, L]

    # Detach: the error predictor must not backprop through seg predictions
    return zone_error.detach(), valid_mask


# =============================================================================
# ERROR PREDICTOR LOSS
# =============================================================================

def compute_error_predictor_loss(
    predicted_errors: torch.Tensor,   # [B, L]
    zone_error:       torch.Tensor,   # [B, L]  detached target
    valid_mask:       torch.Tensor,   # [B, L]  bool
) -> torch.Tensor:
    """
    MSE loss between predicted and actual per-latent zone error.

    Only supervises latents that had at least one labeled pixel in
    the current batch (valid_mask). Empty zones are excluded.

    Args:
        predicted_errors: [B, L]  output of error_predictor MLP
        zone_error:       [B, L]  target from compute_latent_errors() — detached
        valid_mask:       [B, L]  which latents have valid supervision signal

    Returns:
        scalar MSE loss, or 0.0 if no valid zones
    """
    if not valid_mask.any():
        return predicted_errors.sum() * 0.0

    return F.mse_loss(
        predicted_errors[valid_mask],
        zone_error[valid_mask],
    )