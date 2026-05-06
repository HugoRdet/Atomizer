"""
Atomiser Multi-Task pipeline
============================

Glue between single-task Atomiser datasets and the MT trainer.

Key components:

  TaggedAtomiserDataset    Thin wrapper around any single-task Atomiser
                           dataset. Adds per-task identifier (task_name)
                           and task type (seg/cls). Normalizes queries
                           placement (lifts from "tasks" dict if PASTIS-
                           style nested format).

  make_atomiser_mt_collate Collate factory matching the make_tagged_collate
                           pattern from the ResNet/ViT MT pipeline. Wraps
                           collate_multitask (which already handles per-
                           resolution token batching), then propagates the
                           task tag to the top-level batch dict.

  RoundRobinLoader         Reused from training/utils/datasets_baselines_MT/
                           mt_pipeline.py — same round-robin semantics as
                           the ResNet/ViT MT setup. Imported below.

Why a wrapper instead of rewriting datasets:
  The single-task Atomiser datasets (Sen1Floods11, EuroSAT, PASTIS-HD, ...)
  already produce the canonical Atomiser sample format:
      {
        "groups": {res: {"tokens": [N, 8], "mask": [N], "shape": (H, W)}},
        "queries": [M, 8],           # top-level (Sen1Floods11, EuroSAT, ...)
        "queries_mask": [M],
        "label": ...,                # [H, W] for seg, scalar for cls
        ...
      }
  PASTIS-HD is the only outlier — it nests queries under
  "tasks": {TASK_NAME: {"queries": ..., "queries_mask": ...}}. We unify
  this in the wrapper.

  The single-task datasets are tested and working. Wrapping them keeps
  the MT path additive — no risk of breaking single-task code paths.
"""

import torch
from torch.utils.data import Dataset

# Reuse the existing RoundRobinLoader from the ResNet/ViT MT pipeline.
# Same semantics here: cycle through per-task DataLoaders deterministically,
# yielding one batch per task per pass.
from training.utils.datasets_baselines_MT.mt_pipeline import RoundRobinLoader


# ────────────────────────────────────────────────────────────────────
# Tagged dataset wrapper
# ────────────────────────────────────────────────────────────────────

class TaggedAtomiserDataset(Dataset):
    """
    Wraps a single-task Atomiser dataset. Adds task tag, normalizes
    sample shape so the MT trainer/collate sees a consistent format
    across all 5 tasks.

    Output sample (consistent across all wrapped tasks):
        {
            "groups":            {res: {"tokens", "mask", "shape"}},
            "queries":           [M, 8]   (top-level, lifted if needed)
            "queries_mask":      [M],
            "label":             [H, W] for seg, scalar long for cls,
            "task":              str   — per-task name (e.g. "burnscars")
            "task_type":         "seg" | "cls",
            "target_resolution": float,
            ...other passthrough fields...
        }

    Args:
        base:       Any single-task Atomiser dataset instance.
                    Must produce the standard format described above.
        task_name:  Per-task identifier the model uses for head dispatch.
                    Examples: "burnscars", "senflood", "pastis",
                    "eurosat", "forestnet". Must be a key in the
                    MultiTaskAtomiser's `task_specs`.
        task_type:  "seg" or "cls". Determines downstream loss/metric
                    handling and collate behavior (label stacking).
    """

    def __init__(self, base: Dataset, task_name: str, task_type: str):
        if task_type not in ("seg", "cls"):
            raise ValueError(f"task_type must be 'seg' or 'cls', got {task_type!r}")
        self.base = base
        self.task_name = task_name
        self.task_type = task_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        # ── Lift queries from "tasks" dict to top-level ──────────────
        # PASTIS-HD-style format nests queries under "tasks":
        #     {"tasks": {TASK_NAME: {"queries": ..., "queries_mask": ...}}}
        # Other datasets (Sen1Floods11, EuroSAT) already have queries at
        # top-level. Normalize so the MT trainer/model sees one shape.
        if "queries" not in sample and "tasks" in sample:
            task_data = next(iter(sample["tasks"].values()))
            sample["queries"] = task_data["queries"]
            sample["queries_mask"] = task_data["queries_mask"]
            # Keep "tasks" intact for backwards compat with anything
            # downstream that might still read it. Drop only if it's
            # actively confusing — currently nothing reads it post-lift.

        # ── Tag with per-task identifiers ────────────────────────────
        sample["task"] = self.task_name
        sample["task_type"] = self.task_type
        return sample


# ────────────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────────────

def _stack_or_pass(values):
    """
    Stack a list of tensor-or-scalar values into a batched tensor when
    they're stackable. Falls back to a list for things that aren't
    (mixed dtypes, ragged shapes, non-tensor metadata).
    """
    if not values:
        return values
    first = values[0]
    if isinstance(first, torch.Tensor):
        try:
            return torch.stack(values, dim=0)
        except RuntimeError:
            # Ragged shapes (e.g. variable-size labels for seg) — keep as list.
            return values
    if isinstance(first, (int, float)):
        return torch.tensor(values)
    return values


def _atomiser_mt_collate(samples: list, task_name: str, task_type: str):
    """
    Batch a list of TaggedAtomiserDataset samples for Atomiser MT.

    Round-robin assumption: all samples in the batch share the same task,
    so we tag the batch (not per-sample) with `batch["task"]` and
    `batch["task_type"]`.

    Per-resolution token batching:
      For each resolution present in the samples (e.g. 10.0, possibly 1.0
      for SPOT or others), stack tokens and masks. Different samples may
      have different N (token counts per resolution); we pad to the max
      and append a corresponding mask entry (mask=1 means "padding,
      ignore in attention"). Same logic as collate_multitask.

    Per-sample fields:
      - queries / queries_mask: padded across the batch
      - label:
          seg → stacked as [B, H, W] (assumes consistent spatial size)
          cls → stacked as [B] (scalar per sample)
      - target_resolution, image, ...: best-effort stacking via
        _stack_or_pass; non-stackable fields end up as lists.

    Returns:
        batch dict with the same top-level keys as the input samples,
        plus batch["task"] and batch["task_type"].
    """
    if not samples:
        return {}

    # ── Per-resolution token batching ─────────────────────────────────
    # Build the union of resolutions across the batch (in practice all
    # samples in a single MT batch are from the same task → same set of
    # resolutions, but be defensive).
    all_resolutions = set()
    for s in samples:
        all_resolutions.update(s["groups"].keys())

    groups_batched = {}
    for res in sorted(all_resolutions, key=str):
        per_sample_tokens = []
        per_sample_masks = []
        per_sample_shapes = []
        for s in samples:
            if res not in s["groups"]:
                # Sample missing this resolution — emit an empty 0-token
                # entry. Encoder handles empty groups gracefully via mask.
                per_sample_tokens.append(torch.zeros(0, 8, dtype=torch.float32))
                per_sample_masks.append(torch.zeros(0, dtype=torch.bool))
                per_sample_shapes.append((0, 0))
                continue
            g = s["groups"][res]
            per_sample_tokens.append(g["tokens"])
            per_sample_masks.append(g["mask"].bool() if g["mask"].dtype != torch.bool
                                    else g["mask"])
            per_sample_shapes.append(g["shape"])

        # Pad to max N along the token dim.
        max_n = max(t.shape[0] for t in per_sample_tokens)
        padded_tokens = []
        padded_masks = []
        for t, m in zip(per_sample_tokens, per_sample_masks):
            n = t.shape[0]
            if n < max_n:
                pad_n = max_n - n
                t_pad = torch.cat([t, torch.zeros(pad_n, 8, dtype=t.dtype)], dim=0)
                # mask=True means "ignore" — pad entries are ignored.
                m_pad = torch.cat([m, torch.ones(pad_n, dtype=torch.bool)], dim=0)
            else:
                t_pad, m_pad = t, m
            padded_tokens.append(t_pad)
            padded_masks.append(m_pad)

        # Take the first sample's shape as canonical for the batch
        # (single-task batches share spatial size; for MT-within-task this
        # holds because each batch is one task).
        groups_batched[res] = {
            "tokens": torch.stack(padded_tokens, dim=0),       # [B, N_max, 8]
            "mask":   torch.stack(padded_masks, dim=0),         # [B, N_max]
            "shape":  per_sample_shapes[0],
        }

    # ── Queries (only relevant for seg, but cls samples include a single
    #          CLS query as a placeholder) ──────────────────────────────
    queries_list = [s["queries"] for s in samples]
    queries_mask_list = [s.get("queries_mask",
                                torch.zeros(s["queries"].shape[0], dtype=torch.bool))
                         for s in samples]
    max_m = max(q.shape[0] for q in queries_list)
    padded_q = []
    padded_qm = []
    for q, qm in zip(queries_list, queries_mask_list):
        m = q.shape[0]
        if m < max_m:
            q_pad = torch.cat([q, torch.zeros(max_m - m, 8, dtype=q.dtype)], dim=0)
            qm_pad = torch.cat([qm, torch.ones(max_m - m, dtype=torch.bool)], dim=0)
        else:
            q_pad, qm_pad = q, qm
        padded_q.append(q_pad)
        padded_qm.append(qm_pad)

    # ── Label (shape depends on task_type) ────────────────────────────
    label_list = [s.get("label") for s in samples]
    if task_type == "seg":
        # Seg labels: [H, W] each. Stack into [B, H, W].
        # Assumes consistent spatial size within a batch (true for all
        # tasks where the dataset returns full-image labels).
        try:
            label_batched = torch.stack(label_list, dim=0)
        except (RuntimeError, TypeError):
            # Inconsistent shapes — fall back to list (should not happen).
            label_batched = label_list
    elif task_type == "cls":
        # Cls labels: scalar long each. Stack into [B].
        label_batched = torch.stack([
            l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.long)
            for l in label_list
        ], dim=0)
    else:
        label_batched = label_list  # unreachable given __init__ validation

    # ── Other passthrough fields ─────────────────────────────────────
    # target_resolution, image, etc. — best-effort stack, fall back to list.
    extra_keys = set()
    for s in samples:
        extra_keys.update(s.keys())
    extra_keys -= {
        "groups", "queries", "queries_mask", "label",
        "task", "task_type", "tasks",
    }

    extras = {}
    for k in extra_keys:
        vals = [s.get(k) for s in samples]
        extras[k] = _stack_or_pass(vals)

    # ── Assemble final batch ─────────────────────────────────────────
    batch = {
        "groups":       groups_batched,
        "queries":      torch.stack(padded_q, dim=0),    # [B, M_max, 8]
        "queries_mask": torch.stack(padded_qm, dim=0),    # [B, M_max]
        "label":        label_batched,
        "task":         task_name,
        "task_type":    task_type,
        **extras,
    }
    return batch


def make_atomiser_mt_collate(task_name: str, task_type: str):
    """
    Build a per-task collate function for Atomiser MT DataLoaders.

    Mirrors the make_tagged_collate pattern from the ResNet/ViT MT
    pipeline — closes over (task_name, task_type) and returns a callable
    suitable for `DataLoader(..., collate_fn=...)`.

    Args:
        task_name:  Per-task identifier (e.g. "burnscars").
        task_type:  "seg" or "cls".
    """
    if task_type not in ("seg", "cls"):
        raise ValueError(f"task_type must be 'seg' or 'cls', got {task_type!r}")

    def _collate(samples):
        return _atomiser_mt_collate(samples, task_name, task_type)
    return _collate


# Re-export RoundRobinLoader at the module level so the launch script
# imports a single namespace.
__all__ = [
    "TaggedAtomiserDataset",
    "make_atomiser_mt_collate",
    "RoundRobinLoader",
]