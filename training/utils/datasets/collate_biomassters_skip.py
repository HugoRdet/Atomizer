"""
BioMassters collate function.

Wraps PASTIS's collate_multitask (groups + tasks + target_resolution) and
adds what it's missing for the SKIP variant:

  1. query_token_idx / query_token_valid padding -- collate_multitask has
     NO knowledge of these keys (BioMasstersSkipDataset puts them at the
     TOP LEVEL of each sample, not under "groups" or "tasks"), so calling
     it alone would silently DROP them from the batch. Since
     use_decoder_skip=True in the BioMassters config, this would silently
     fall back to global_query (skip disabled) with no error -- exactly
     the kind of invalidation Model_BioMassters_Skip's __init__ guard was
     built to catch, except this happens upstream of that guard, in
     collation, where it can't see it. This collate fixes that.

  2. tasks -> top-level queries/queries_mask bridge, same as PASTIS's own
     pastis_collate() wrapper -- Atomiser_Senflood_Skip.forward() reads
     batch["queries"] directly, not batch["tasks"][name]["queries"].

  3. label / chip_id passthrough for viz/eval convenience.

PADDING ASSUMPTION: query_token_idx's last dim (A = fixed_T * (C2+C1)) is
CONSTANT across every BioMassters sample by construction (pad-by-replication
in the dataset always produces the same fixed_T for every chip), so no
padding is needed along that axis -- only the query count (M, second dim)
varies per sample and needs padding, matching however collate_multitask
pads "queries" (assumed right-padding: original rows first, padding
appended after -- the near-universal convention, but not verified against
collate_multitask's actual _pad_tokens/_pad_masks implementation since
that wasn't shown). Padded rows get query_token_idx=0 (dummy, unused) and
query_token_valid=False, which the pixel-skip cascade's key_padding_mask
already handles safely via its own invalid_q masking.
"""

import torch

from training.utils.datasets.token_grouping import collate_multitask


def collate_biomassters_skip(samples: list) -> dict:
    batch = collate_multitask(samples)

    # ── Bridge: tasks[TASK_NAME] -> top-level queries/queries_mask ──────
    # (same pattern as PASTIS's own pastis_collate wrapper)
    if "queries" not in batch and "tasks" in batch and batch["tasks"]:
        task_name = next(iter(batch["tasks"]))
        batch["queries"]      = batch["tasks"][task_name]["queries"]
        batch["queries_mask"] = batch["tasks"][task_name]["queries_mask"]

    # ── SKIP: pad query_token_idx / query_token_valid to match the
    # padded queries tensor (collate_multitask drops these entirely
    # since it only knows about "groups"/"tasks"). ───────────────────
    B     = len(samples)
    max_M = batch["queries"].shape[1]
    A     = samples[0]["query_token_idx"].shape[1]

    device = samples[0]["query_token_idx"].device
    qti_padded = torch.zeros(B, max_M, A, dtype=torch.long, device=device)
    qtv_padded = torch.zeros(B, max_M, dtype=torch.bool, device=device)

    for i, s in enumerate(samples):
        m = s["query_token_idx"].shape[0]
        if m > max_M:
            raise ValueError(
                f"Sample {i} has {m} query tokens but padded batch width is "
                f"{max_M} -- query_token_idx and queries disagree on count. "
                f"This means collate_multitask's queries padding and this "
                f"function's query_token_idx padding diverged; check that "
                f"both pad the same way (right-padding, original rows first)."
            )
        qti_padded[i, :m] = s["query_token_idx"]
        qtv_padded[i, :m] = s["query_token_valid"]

    batch["query_token_idx"]   = qti_padded
    batch["query_token_valid"] = qtv_padded

    # ── Passthrough: label (full-grid AGB, for viz/eval) + chip_id ──────
    if "label" in samples[0]:
        batch["label"] = torch.stack([s["label"] for s in samples], dim=0)
    if "chip_id" in samples[0]:
        batch["chip_id"] = [s.get("chip_id") for s in samples]

    return batch
