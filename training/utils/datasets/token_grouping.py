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
        "dataset_name":      str,
        "metadata":          {...},
    }

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

All tokens are flat -- no temporal dimension. Temporal info is in column 7.

DALES ADDITIONS (see collate_grouped):
    "token_latent_assignment": [B, N_max] long -- per-token nearest-latent
        index, precomputed offline (see precompute_dales_latent_assignment.py)
        and selected for whichever D4 variant DalesDataset sampled for each
        sample. Padded in lockstep with groups[res]["tokens"] (same N_max),
        padding value 0 (harmless -- padded positions are already masked in
        groups[res]["mask"], and GeographicPruningDales gathers that input
        mask too, so a padded token landing in latent 0's cell never
        actually contributes).
    "patch_id": List[str], length B -- passed through as-is (not stacked
        into a tensor), used as the GeographicPruningDales FALLBACK cache
        key when token_latent_assignment isn't available.
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

# =============================================================================
# ROUND-ROBIN BATCH SAMPLER
# =============================================================================

class RoundRobinDistributedBatchSampler(Sampler):
    """
    Yields batches alternating between datasets in a ConcatDataset.

    All ranks see the same dataset at every step, preventing DDP
    deadlocks from mismatched head execution.

    Shorter datasets are cycled (with re-shuffling) to match the
    longest, so no data is wasted from larger datasets.

    Uses deterministic shuffling (seed + epoch) so all ranks generate
    identical orderings, then each rank takes its own slice.
    """

    def __init__(
        self,
        dataset_lengths: list,
        batch_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_datasets = len(dataset_lengths)

        # Distributed batch: each step consumes batch_size * num_replicas samples
        self.global_batch_size = self.batch_size * self.num_replicas

        # Per-dataset: how many full distributed batches each dataset provides
        self.batches_per_dataset = [
            d_len // self.global_batch_size for d_len in dataset_lengths
        ]

        # Round-robin runs for as many rounds as the longest dataset has batches.
        # Shorter datasets cycle.
        self.max_batches = max(self.batches_per_dataset)

        # Total batches yielded per epoch (per rank):
        # max_batches rounds x num_datasets batches per round
        self.total_batches = self.max_batches * self.num_datasets

    def _build_indices(self, d_len, num_needed, offset, generator):
        """
        Build a list of `num_needed` global indices for one dataset,
        cycling through the dataset with fresh shuffles as needed.
        Indices are offset for ConcatDataset addressing.
        """
        indices = []
        while len(indices) < num_needed:
            if self.shuffle:
                perm = torch.randperm(d_len, generator=generator).tolist()
            else:
                perm = list(range(d_len))
            indices.extend(perm)

        # Truncate to exactly what we need and apply ConcatDataset offset
        indices = indices[:num_needed]
        return [i + offset for i in indices]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Each dataset needs max_batches * global_batch_size samples.
        # Shorter datasets will cycle through their data.
        needed_per_dataset = self.max_batches * self.global_batch_size

        dataset_indices = []
        current_offset = 0
        for d_len in self.dataset_lengths:
            # Each dataset gets its own sub-generator for reproducibility
            # (order of randperm calls must be deterministic across ranks)
            sub_seed = g.initial_seed() + current_offset
            sub_g = torch.Generator()
            sub_g.manual_seed(sub_seed)

            indices = self._build_indices(
                d_len, needed_per_dataset, current_offset, sub_g
            )
            dataset_indices.append(indices)
            current_offset += d_len

        # Yield batches in round-robin order
        for b in range(self.max_batches):
            for d_idx in range(self.num_datasets):
                start = b * self.global_batch_size
                end = start + self.global_batch_size
                global_batch = dataset_indices[d_idx][start:end]

                # Each rank takes its slice
                local_start = self.rank * self.batch_size
                local_end = local_start + self.batch_size
                local_batch = global_batch[local_start:local_end]
                yield local_batch

    def __len__(self):
        """Number of batches this rank will yield per epoch."""
        return self.total_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


# =============================================================================
# PAD HELPERS
# =============================================================================

def _pad_tokens(tensors, pad_value=0.0):
    """Pad a list of [N_i, D] tensors to [B, N_max, D]."""
    max_len = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    B = len(tensors)
    padded = torch.full((B, max_len, D), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
    return padded


def _pad_masks(masks, pad_value=True):
    """Pad a list of [N_i] bool tensors to [B, N_max]. Padded = True."""
    max_len = max(m.shape[0] for m in masks)
    B = len(masks)
    padded = torch.full((B, max_len), pad_value, dtype=torch.bool)
    for i, m in enumerate(masks):
        padded[i, :m.shape[0]] = m
    return padded


def _pad_index_2d(idx_list, pad_value=0):
    """
    Pad a list of [M_i, A_i] long tensors to [B, M_max, A_max].
    Padded entries get pad_value (0 -> harmless row 0; masked via valid).
    """
    M_max = max(t.shape[0] for t in idx_list)
    A_max = max(t.shape[1] for t in idx_list)
    B = len(idx_list)
    out = torch.full((B, M_max, A_max), pad_value, dtype=idx_list[0].dtype)
    for i, t in enumerate(idx_list):
        out[i, :t.shape[0], :t.shape[1]] = t
    return out


def _pad_valid_1d(valid_list, pad_value=False):
    """Pad a list of [M_i] bool tensors to [B, M_max]. Padded = False (skip)."""
    M_max = max(v.shape[0] for v in valid_list)
    B = len(valid_list)
    out = torch.full((B, M_max), pad_value, dtype=torch.bool)
    for i, v in enumerate(valid_list):
        out[i, :v.shape[0]] = v
    return out


def _pad_assignment_1d(assign_list, pad_value=0):
    """Pad a list of [N_i] long tensors to [B, N_max]. Padded value is
    harmless (see module docstring) -- those positions are already masked
    in groups[res]['mask'].
    """
    N_max = max(a.shape[0] for a in assign_list)
    B = len(assign_list)
    out = torch.full((B, N_max), pad_value, dtype=torch.long)
    for i, a in enumerate(assign_list):
        out[i, :a.shape[0]] = a
    return out

# =============================================================================
# MULTI-TASK COLLATE (DYNAMIC VERSION)
# =============================================================================

def collate_multitask(samples: list) -> dict:
    """
    SKIP-aware multitask collate. Pads groups + per-task queries, AND carries
    query_token_idx / query_token_valid padded in lockstep with the queries.
    No batch offset on indices: per-sample pools live in separate batch rows.
    """
    B = len(samples)

    # 1. Groups (unchanged)
    all_resolutions = set()
    for s in samples:
        all_resolutions.update(s["groups"].keys())

    groups = {}
    for res in sorted(all_resolutions):
        tokens_list, masks_list, shape = [], [], None
        for s in samples:
            if res in s["groups"]:
                tokens_list.append(s["groups"][res]["tokens"])
                masks_list.append(s["groups"][res]["mask"])
                if shape is None:
                    shape = s["groups"][res]["shape"]
            else:
                tokens_list.append(torch.zeros(0, 8))
                masks_list.append(torch.zeros(0, dtype=torch.bool))
        if shape is None:
            shape = (1, 1)
        groups[res] = {
            "tokens": _pad_tokens(tokens_list),
            "mask":   _pad_masks(masks_list),
            "shape":  shape,
        }

    # 2. Tasks (queries) — unchanged padding to M_max
    all_task_names = set()
    for s in samples:
        all_task_names.update(s.get("tasks", {}).keys())

    tasks = {}
    for task_name in sorted(all_task_names):
        queries_list = [s["tasks"][task_name]["queries"] for s in samples]
        masks_list   = [s["tasks"][task_name]["queries_mask"] for s in samples]
        tasks[task_name] = {
            "queries":      _pad_tokens(queries_list),
            "queries_mask": _pad_masks(masks_list),
        }

    target_resolution = samples[0].get("target_resolution", 10.0)

    result = {
        "groups": groups,
        "tasks":  tasks,
        "target_resolution": target_resolution,
    }

    # 3. >>> SKIP: carry query_token_idx / query_token_valid, padded in lockstep
    #    with the queries (same M_max). Padded query rows -> valid=False so the
    #    model ignores them; their index value is 0 (never read once masked).
    if "query_token_idx" in samples[0]:
        qti_list = [s["query_token_idx"]   for s in samples]   # [M_i, A_i] long
        qtv_list = [s["query_token_valid"] for s in samples]   # [M_i] bool
        result["query_token_idx"]   = _pad_index_2d(qti_list, pad_value=0)
        result["query_token_valid"] = _pad_valid_1d(qtv_list, pad_value=False)

    if "dataset_name" in samples[0]:
        result["dataset_name"] = samples[0]["dataset_name"]

    return result
# =============================================================================
# SINGLE-TASK COLLATE (GROUPED)
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

    Passes through dataset_name from the first sample (homogeneous batches
    guaranteed by round-robin sampling).

    DALES additions (see module docstring):
        "token_latent_assignment": [B, N_max] long, padded in lockstep with
            groups[res]["tokens"] (assumes a SINGLE resolution group, true
            for DALES's LIDAR-only setup — if a sample ever has multiple
            resolution groups this only covers the token dimension shared
            by the assignment's own N, which must match the LIDAR group's
            N specifically).
        "patch_id": List[str], passed through unstacked.
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

    # Pass through dataset_name (homogeneous batch from round-robin)
    if "dataset_name" in batch[0]:
        result["dataset_name"] = batch[0]["dataset_name"]

    # ── DALES: token_latent_assignment (padded in lockstep with tokens) ──
    if "token_latent_assignment" in batch[0]:
        assign_list = [s["token_latent_assignment"] for s in batch]
        result["token_latent_assignment"] = _pad_assignment_1d(
            assign_list, pad_value=0
        )

    # ── DALES: patch_id (fallback cache key, passed through unstacked) ───
    if "patch_id" in batch[0]:
        result["patch_id"] = [s["patch_id"] for s in batch]

    # ── Decoder-skip cascade: query_token_idx / query_token_valid, padded
    # in lockstep with queries (same M_max) ───────────────────────────
    if "query_token_idx" in batch[0]:
        qti_list = [s["query_token_idx"]   for s in batch]   # [M_i, A_i] long
        qtv_list = [s["query_token_valid"] for s in batch]   # [M_i] bool
        result["query_token_idx"]   = _pad_index_2d(qti_list, pad_value=0)
        result["query_token_valid"] = _pad_valid_1d(qtv_list, pad_value=False)

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

    Passes through dataset_name from the first sample (homogeneous batches
    guaranteed by round-robin sampling).

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

    result = {
        "groups": groups,
        "queries": torch.stack(padded_queries, dim=0),       # [B, M_max, 8]
        "queries_mask": torch.stack(padded_qmasks, dim=0),   # [B, M_max] bool
        "ground_truth": torch.stack(padded_gt, dim=0),       # [B, M_max] float
    }

    # Pass through dataset_name (homogeneous batch from round-robin)
    if "dataset_name" in batch[0]:
        result["dataset_name"] = batch[0]["dataset_name"]

    return result


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
    min_pixels_per_latent: int = 1,
) -> dict:
    """
    Compute latent grid configuration from total token count.

    The grid is sized by dividing the total number of tokens by
    tokens_per_latent, then arranging latents on a spatial grid
    preserving the image's aspect ratio.

    For temporal modalities (e.g. S1/S2 time series), total_tokens can
    vastly exceed the spatial extent (H×W) because each timestamp x
    band generates separate tokens at the same spatial positions.
    The latent grid is capped so it never exceeds the spatial pixel
    count, preventing empty Voronoi cells and downstream NaN.

    Args:
        resolution:           Ground sampling distance (m/px)
        shape:                Spatial geometry — (H, W) or (C, H, W)
        tokens_per_latent:    Target token budget per latent
        total_tokens:         Actual token count from groups[res]["tokens"].shape[1]
        sigma_factor:         Multiplier for geographic attention sigma
        max_k:                Maximum tokens per latent in geographic pruning
        min_pixels_per_latent: Minimum spatial pixels per latent (caps grid density).
                              Prevents more latents than spatial positions for
                              temporal data. Default 4.

    Returns:
        dict with grid parameters:
            latents_x, latents_y, L_spatial,
            span_x, span_y,
            geo_k, geo_sigma,
            train_k, val_k,
            tokens_per_latent, total_tokens
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


    # Cap: never more latents than spatial positions allow.
    # Temporal tokens share spatial locations, so the latent grid
    # must fit within the H×W spatial extent.
    max_spatial_latents = max(1, (H * W) // min_pixels_per_latent)
    num_latents = min(num_latents, max_spatial_latents)


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
