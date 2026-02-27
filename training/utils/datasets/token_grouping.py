"""
Token grouping utilities: collate and grid configuration.

Handles the grouped-token batch format:
    batch = {
        "groups": {
            res: {
                "tokens": [B, N_max, 8],   # padded across batch
                "mask":   [B, N_max],       # bool, True=padded
                "shape":  (H, W) or (C, H, W),  # spatial geometry
            },
        },
        "queries":           [B, M_max, 8],
        "queries_mask":      [B, M_max],
        "label":             [B, H, W],
        "target_resolution": float,
        "metadata":          {...},
    }

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

All tokens are flat — no temporal dimension. Temporal info is in column 7.
"""

import math
from collections import defaultdict

import torch


# =============================================================================
# COLLATE
# =============================================================================

def collate_grouped(batch: list) -> dict:
    """
    Collate a list of samples into a batched dict.

    Each sample has:
        groups[res]["tokens"]: [N_i, 8]   — flat tokens (variable N across samples)
        groups[res]["mask"]:   [N_i]      — bool
        groups[res]["shape"]:  tuple      — spatial geometry

    Output:
        groups[res]["tokens"]: [B, N_max, 8]  — zero-padded
        groups[res]["mask"]:   [B, N_max]     — True for padded positions
        groups[res]["shape"]:  tuple          — from first sample (assumed constant)

    Queries, labels, etc. are similarly padded and stacked.
    """
    B = len(batch)

    # ── Collect all resolution keys ─────────────────────────
    all_resolutions = set()
    for sample in batch:
        all_resolutions.update(sample["groups"].keys())

    # ── Build groups ────────────────────────────────────────
    groups = {}
    for res in sorted(all_resolutions):
        # Gather per-sample tokens and masks for this resolution
        sample_tokens = []
        sample_masks = []
        shape = None

        for sample in batch:
            if res in sample["groups"]:
                g = sample["groups"][res]
                sample_tokens.append(g["tokens"])   # [N_i, 8]
                sample_masks.append(g["mask"])       # [N_i] bool
                if shape is None:
                    shape = g["shape"]
            else:
                # Sample doesn't have this resolution — will be fully masked
                sample_tokens.append(torch.zeros(0, 8))
                sample_masks.append(torch.zeros(0, dtype=torch.bool))

        # Find max N across batch
        N_max = max(t.shape[0] for t in sample_tokens)
        N_max = max(N_max, 1)  # at least 1 token

        # Pad and stack
        padded_tokens = []
        padded_masks = []
        for tokens, mask in zip(sample_tokens, sample_masks):
            n = tokens.shape[0]
            if n < N_max:
                pad_t = torch.zeros(N_max - n, 8, dtype=tokens.dtype)
                pad_m = torch.ones(N_max - n, dtype=torch.bool)
                padded_tokens.append(torch.cat([tokens, pad_t], dim=0))
                padded_masks.append(torch.cat([mask, pad_m], dim=0))
            else:
                padded_tokens.append(tokens)
                padded_masks.append(mask)

        groups[res] = {
            "tokens": torch.stack(padded_tokens, dim=0),  # [B, N_max, 8]
            "mask": torch.stack(padded_masks, dim=0),      # [B, N_max] bool
            "shape": shape,                                 # tuple, from first sample
        }

    # ── Queries ─────────────────────────────────────────────
    query_list = [s["queries"] for s in batch]
    qmask_list = [s["queries_mask"] for s in batch]
    M_max = max(q.shape[0] for q in query_list)

    padded_queries = []
    padded_qmasks = []
    for q, qm in zip(query_list, qmask_list):
        m = q.shape[0]
        if m < M_max:
            pad_q = torch.zeros(M_max - m, 8, dtype=q.dtype)
            pad_m = torch.ones(M_max - m, dtype=torch.bool)
            padded_queries.append(torch.cat([q, pad_q], dim=0))
            padded_qmasks.append(torch.cat([qm, pad_m], dim=0))
        else:
            padded_queries.append(q)
            padded_qmasks.append(qm)

    # ── Labels ──────────────────────────────────────────────
    #labels = torch.stack([s["label"] for s in batch], dim=0)  # [B, H, W]

    
    


    result = {
        "groups": groups,
        "queries": torch.stack(padded_queries, dim=0),       # [B, M_max, 8]
        "queries_mask": torch.stack(padded_qmasks, dim=0),   # [B, M_max] bool
    }

    # Pass through optional keys
    if "image" in batch[0]:
        result["image"] = torch.stack([s["image"] for s in batch], dim=0)

    if "label" in batch[0]:
        labels = torch.stack([s["label"] for s in batch], dim=0)
        result["label"] = labels

    if "task" in batch[0]:
        result["task"] = batch[0]["task"]

    
    # ── Metadata ────────────────────────────────────────────
    if "target_resolution" in batch[0]:
        result["target_resolution"] = batch[0]["target_resolution"]

    return result


# =============================================================================
# COLLATE — MAE (handles ground_truth, no label/image)
# =============================================================================

def collate_mae(batch: list) -> dict:
    """
    Collate MAE pre-training samples into a batch.

    Groups are fixed-size (thanks to padding in dataset __getitem__),
    so collation is just stacking. Queries may vary slightly in count
    across samples — pad to max M.

    Expected per-sample format:
        {
            "groups": {res: {"tokens": [N, 8], "mask": [N], "shape": tuple}},
            "queries":      [M_i, 8],
            "queries_mask": [M_i],
            "ground_truth": [M_i],
        }

    Output:
        {
            "groups": {res: {"tokens": [B, N, 8], "mask": [B, N], "shape": tuple}},
            "queries":      [B, M_max, 8],
            "queries_mask": [B, M_max],
            "ground_truth": [B, M_max],
        }
    """
    B = len(batch)

    # ── Groups: fixed size per dataset, just stack ──────────
    all_resolutions = set()
    for sample in batch:
        all_resolutions.update(sample["groups"].keys())

    groups = {}
    for res in sorted(all_resolutions):
        groups[res] = {
            "tokens": torch.stack(
                [s["groups"][res]["tokens"] for s in batch], dim=0
            ),
            "mask": torch.stack(
                [s["groups"][res]["mask"] for s in batch], dim=0
            ),
            "shape": batch[0]["groups"][res]["shape"],
        }

    # ── Queries + ground_truth: pad to max M ────────────────
    query_list = [s["queries"] for s in batch]
    gt_list = [s["ground_truth"] for s in batch]
    qmask_list = [s["queries_mask"] for s in batch]
    M_max = max(q.shape[0] for q in query_list)

    padded_queries = []
    padded_gt = []
    padded_qmasks = []

    for q, gt, qm in zip(query_list, gt_list, qmask_list):
        m = q.shape[0]
        if m < M_max:
            pad_n = M_max - m
            padded_queries.append(
                torch.cat([q, torch.zeros(pad_n, 8, dtype=q.dtype)], dim=0)
            )
            padded_gt.append(
                torch.cat([gt, torch.zeros(pad_n, dtype=gt.dtype)], dim=0)
            )
            padded_qmasks.append(
                torch.cat([qm, torch.ones(pad_n, dtype=torch.bool)], dim=0)
            )
        else:
            padded_queries.append(q)
            padded_gt.append(gt)
            padded_qmasks.append(qm)

    return {
        "groups": groups,
        "queries": torch.stack(padded_queries, dim=0),       # [B, M_max, 8]
        "queries_mask": torch.stack(padded_qmasks, dim=0),   # [B, M_max] bool
        "ground_truth": torch.stack(padded_gt, dim=0),       # [B, M_max] float
    }


# =============================================================================
# GRID CONFIGURATION
# =============================================================================

def compute_grid_config(
    resolution: float,
    shape: tuple,
    tokens_per_latent: int = 2000,
    total_tokens: int = None,
    sigma_factor: float = 1.5,
    max_k: int = 2000,
) -> dict:
    """
    Compute latent grid configuration from total token count.

    The grid is sized by dividing the total number of tokens by
    tokens_per_latent, then arranging latents on a spatial grid
    preserving the image's aspect ratio.

    This replaces the old pixels_per_latent approach, which only
    counted spatial pixels and required temporal chunking.

    Args:
        resolution:        Ground sampling distance (m/px)
        shape:             Spatial geometry — (H, W) or (C, H, W)
        tokens_per_latent: Target token budget per latent
        total_tokens:      Actual token count from groups[res]["tokens"].shape[1]
        sigma_factor:      Multiplier for geographic attention sigma
        max_k:             Maximum tokens per latent in geographic pruning

    Returns:
        dict with grid parameters:
            latents_x, latents_y, L_spatial,
            span_x, span_y,
            geo_k, geo_sigma,
            train_k, val_k,
            tokens_per_latent, total_tokens

    Example:
        PASTIS 128x128, 10 S2 bands × 10 timesteps + 3 S1 bands × 10 timesteps
        = 128 * 128 * (100 + 30) = 2,129,920 tokens
        tokens_per_latent = 2000 → ~1065 latents → ~33x33 grid
    """
    # Extract spatial dims from shape
    if len(shape) == 2:
        H, W = shape
    elif len(shape) == 3:
        _, H, W = shape
    else:
        raise ValueError(f"Expected shape (H,W) or (C,H,W), got {shape}")

    if total_tokens is None:
        raise ValueError("total_tokens must be provided (from tokens.shape[1])")

    # Number of latents from token budget
    num_latents = max(1, total_tokens // tokens_per_latent)

    # Arrange on spatial grid preserving aspect ratio
    aspect = W / H
    latents_y = max(1, int(math.sqrt(num_latents / aspect)))
    latents_x = max(1, int(latents_y * aspect))

    # Grow grid to reach target count
    while latents_x * latents_y < num_latents:
        if latents_x / latents_y < aspect:
            latents_x += 1
        else:
            latents_y += 1

    L_spatial = latents_x * latents_y

    # Physical span (meters)
    span_x = W * resolution
    span_y = H * resolution

    # Sigma from cell size
    cell_x_m = span_x / latents_x
    cell_y_m = span_y / latents_y
    geo_sigma = sigma_factor * max(cell_x_m, cell_y_m) / 2.0

    # geo_k bounded by tokens_per_latent and max_k
    geo_k = min(max_k, tokens_per_latent)

    return {
        "latents_x": latents_x,
        "latents_y": latents_y,
        "L_spatial": L_spatial,
        "span_x": span_x,
        "span_y": span_y,
        "geo_k": geo_k,
        "geo_sigma": geo_sigma,
        "train_k": min(500, geo_k),
        "val_k": min(500, geo_k),
        "tokens_per_latent": tokens_per_latent,
        "total_tokens": total_tokens,
    }