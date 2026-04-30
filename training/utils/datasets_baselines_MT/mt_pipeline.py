"""
Multi-task pipeline utilities.

Provides:
    - `seg_collate`: stacks per-sample dicts into a batch dict for
      single-frame segmentation tasks (BurnScars, Sen1Floods11, MADOS).
      Will need separate collates later for multi-temporal (PASTIS) and
      classification (EuroSAT, ForestNet).

    - `make_tagged_collate(base_collate, task_name)`: wraps any collate
      so the resulting batch carries `batch["task"] = task_name`. This
      is how the trainer dispatches per-batch.

    - `RoundRobinLoader`: a length-bounded iterable that cycles through
      per-task DataLoaders, yielding one batch from each in turn.
      Compatible with Lightning's `train_dataloader` interface.

Batch shape after `seg_collate`:
    {
        "image": {"input": [B, 15, 512, 512]},
        "target": [B, 512, 512],
        "valid_mask": [B, 512, 512],   # uint8
        "original_size": [B, 2],       # long
        "metadata": [list of B dicts],
    }
"""

import torch


# ────────────────────────────────────────────────────────────────────
# Collate functions
# ────────────────────────────────────────────────────────────────────

def seg_collate(batch):
    """
    Collate single-frame segmentation samples (T=1 tasks).

    Each sample is a dict produced by a `*MTDataset.__getitem__`:
        {"image": {"input": [15, H, W]}, "target": [H, W],
         "valid_mask": [H, W], "original_size": [2], "metadata": {...}}

    Returns:
        {"image": {"input": [B, 15, H, W]}, "target": [B, H, W],
         "valid_mask": [B, H, W], "original_size": [B, 2],
         "metadata": [...]}
    """
    images        = torch.stack([s["image"]["input"] for s in batch], dim=0)
    targets       = torch.stack([s["target"]         for s in batch], dim=0)
    valid_masks   = torch.stack([s["valid_mask"]     for s in batch], dim=0)
    original_size = torch.stack([s["original_size"]  for s in batch], dim=0)
    metadata      = [s["metadata"] for s in batch]

    return {
        "image": {"input": images},
        "target": targets,
        "valid_mask": valid_masks,
        "original_size": original_size,
        "metadata": metadata,
    }


def make_tagged_collate(base_collate, task_name: str):
    """
    Wrap a base collate function to inject the task tag into every batch.
    The trainer reads `batch["task"]` to dispatch to the right head.
    """
    def tagged(batch):
        out = base_collate(batch)
        out["task"] = task_name
        return out
    return tagged


# ────────────────────────────────────────────────────────────────────
# Round-robin loader
# ────────────────────────────────────────────────────────────────────

class RoundRobinLoader:
    """
    Length-bounded iterable that cycles through per-task DataLoaders.

    On step `i` it yields a batch from the loader at index `i % num_tasks`.
    When a per-task loader is exhausted, it is restarted (its `iter(...)`
    is called again, which reshuffles if the loader's sampler shuffles).

    The `length` parameter controls how many micro-batches make up one
    Lightning "training epoch" — for multi-task with N tasks and
    `accumulate_grad_batches=N`, set this to
    `optimizer_steps_per_epoch * N`.

    Args:
        task_loaders: dict {task_name: DataLoader}. Iteration order is
                      the dict's insertion order.
        length:       number of micro-batches yielded per __iter__().

    DDP note: each rank instantiates its own RoundRobinLoader with the
    same task ordering, so all ranks select the same task at each step.
    The per-task DataLoaders should be set up with DistributedSampler
    (and `use_distributed_sampler=False` on `pl.Trainer`) so each rank
    sees a disjoint shard of each task's data.
    """

    def __init__(self, task_loaders: dict, length: int):
        assert len(task_loaders) > 0, "RoundRobinLoader needs at least one task loader."
        self.task_loaders = task_loaders
        self.task_names = list(task_loaders.keys())
        self.num_tasks = len(self.task_names)
        self.length = length

    def __iter__(self):
        iters = {name: iter(loader) for name, loader in self.task_loaders.items()}

        for step in range(self.length):
            task = self.task_names[step % self.num_tasks]
            try:
                batch = next(iters[task])
            except StopIteration:
                # Restart this task's loader. With shuffle=True this reshuffles.
                iters[task] = iter(self.task_loaders[task])
                batch = next(iters[task])
            yield batch

    def __len__(self):
        return self.length