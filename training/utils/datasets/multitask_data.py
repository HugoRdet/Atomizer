"""
Multi-task data pipeline for Atomiser_UnetMulti
===============================================

* PerDateView      wraps a PASTIS-family FLAT dataset (PlanteurDataset,
                   AgriPotentialDataset) into per-date samples: tokens [T, N_date, 8],
                   mask [T, N_date], time_positions [T] (S2 day of year). Date t =
                   S2 frame t + S1 frame t (both series are evenly spaced over the same
                   window; PastisHDTemporalDataset instead pairs each S2 date with the
                   nearest-calendar S1 date).
* TaggedView       any dataset, adds sample["dataset"] = name (single-date datasets).
* MultiTaskDataset one index space over several datasets (offsets per dataset).
* MultiTaskBatchSampler
                   one dataset per batch; each epoch draws, per dataset, either all
                   its samples or a fixed number of random ones (redrawn every epoch);
                   batches are scheduled round robin (or shuffled). The schedule comes
                   from a seeded generator shared by every rank, so all DDP ranks train
                   the SAME dataset at the same step (required: the per-task heads and
                   loss weights must be the ones synchronised); each rank takes its
                   own disjoint slice of every global batch.
* collate_multitask_batch
                   stacks a single-dataset batch; queries are padded (label 255,
                   mask True) if their count differs between samples.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


def _queries(sample):
    if "queries" in sample:
        return sample["queries"], sample["queries_mask"]
    task = next(iter(sample["tasks"].values()))
    return task["queries"], task["queries_mask"]


class PerDateView(Dataset):
    """Flat PASTIS-family sample [S2 frames 0..T-1 | S1 frames 0..T-1] -> per date."""

    def __init__(self, dataset, name, n_dates, n_s2, n_s1, lookup_table):
        self.ds, self.name, self.T = dataset, name, int(n_dates)
        self.C2, self.C1 = int(n_s2), int(n_s1)
        self.inv_time = {v: k for k, v in lookup_table.table_time.items()}   # time_idx -> day of year
        if not self.inv_time:
            raise RuntimeError("PerDateView: the lookup table has no registered days; build the "
                               "model (which registers days 1..366) before the datasets are used")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        res, g = next(iter(s["groups"].items()))
        tok, msk = g["tokens"], torch.as_tensor(g["mask"]).bool()
        H, W = g["shape"][-2:]
        HW, T = H * W, self.T
        n2, n1 = T * self.C2 * HW, T * self.C1 * HW
        if tok.shape[0] != n2 + n1:
            raise ValueError(f"[{self.name}] {tok.shape[0]} tokens, expected {T}x({self.C2}+{self.C1})"
                             f"x{HW} = {n2 + n1}: check multi_temporal / n_frames / use_s1")
        s2t, s2m = tok[:n2].view(T, self.C2 * HW, -1), msk[:n2].view(T, self.C2 * HW)
        if self.C1:
            tokens = torch.cat([s2t, tok[n2:].view(T, self.C1 * HW, -1)], 1)
            mask = torch.cat([s2m, msk[n2:].view(T, self.C1 * HW)], 1)
        else:
            tokens, mask = s2t, s2m
        q, qm = _queries(s)
        doy = [float(self.inv_time.get(int(s2t[t, 0, 7]), 0.0)) for t in range(T)]
        return {"groups": {res: {"tokens": tokens, "mask": mask, "shape": g["shape"]}},
                "queries": q, "queries_mask": torch.as_tensor(qm).bool(),
                "time_positions": torch.tensor(doy, dtype=torch.float32),
                "dataset": self.name}


class TaggedView(Dataset):
    def __init__(self, dataset, name):
        self.ds, self.name = dataset, name

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        q, qm = _queries(s)
        res, g = next(iter(s["groups"].items()))
        return {"groups": {res: {"tokens": g["tokens"], "mask": torch.as_tensor(g["mask"]).bool(),
                                 "shape": g["shape"]}},
                "queries": q, "queries_mask": torch.as_tensor(qm).bool(), "dataset": self.name}


class MultiTaskDataset(Dataset):
    def __init__(self, views: Dict[str, Dataset]):
        self.names = list(views)
        self.views = views
        self.offsets, o = {}, 0
        for n in self.names:
            self.offsets[n] = o
            o += len(views[n])
        self.total = o

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for n in reversed(self.names):
            if idx >= self.offsets[n]:
                return self.views[n][idx - self.offsets[n]]
        raise IndexError(idx)


class MultiTaskBatchSampler(Sampler):
    def __init__(self, sizes: Dict[str, int], offsets: Dict[str, int], batch_sizes: Dict[str, int],
                 samples_per_epoch: Dict[str, Optional[int]], order="round_robin", seed=0,
                 rank=0, world_size=1):
        assert order in ("round_robin", "shuffle"), order
        self.sizes, self.offsets, self.bs = sizes, offsets, batch_sizes
        self.per_epoch, self.order, self.seed = samples_per_epoch, order, int(seed)
        self.rank, self.world = int(rank), int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _n_batches(self, name):
        n = self.per_epoch.get(name) or self.sizes[name]
        return n // (self.bs[name] * self.world)

    def __len__(self):
        return sum(self._n_batches(n) for n in self.sizes)

    def schedule(self):
        """[(dataset, global indices of this rank's slice)] for the current epoch
        (identical dataset order on every rank)."""
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        per = {}
        for name, size in self.sizes.items():
            n = self.per_epoch.get(name) or size
            reps = math.ceil(n / size)                      # n > size: repeat full permutations
            idx = torch.cat([torch.randperm(size, generator=g) for _ in range(reps)])[:n]
            gb = self.bs[name] * self.world
            nb = n // gb
            per[name] = [idx[b * gb:(b + 1) * gb] for b in range(nb)]
        if self.order == "round_robin":
            sched, cursors, names = [], {n: 0 for n in per}, list(per)
            while any(cursors[n] < len(per[n]) for n in names):
                for n in names:
                    if cursors[n] < len(per[n]):
                        sched.append((n, per[n][cursors[n]]))
                        cursors[n] += 1
        else:
            sched = [(n, b) for n in per for b in per[n]]
            order = torch.randperm(len(sched), generator=g).tolist()
            sched = [sched[i] for i in order]
        out = []
        for name, gidx in sched:
            b = self.bs[name]
            mine = gidx[self.rank * b:(self.rank + 1) * b]
            out.append((name, (mine + self.offsets[name]).tolist()))
        return out

    def __iter__(self):
        for _, idx in self.schedule():
            yield idx


def collate_multitask_batch(samples: List[dict]) -> dict:
    names = {s["dataset"] for s in samples}
    if len(names) != 1:
        raise ValueError(f"collate_multitask_batch: one dataset per batch, got {names}")
    res = next(iter(samples[0]["groups"]))
    tokens = torch.stack([s["groups"][res]["tokens"] for s in samples])
    mask = torch.stack([s["groups"][res]["mask"] for s in samples])
    M = max(s["queries"].shape[0] for s in samples)
    q = torch.zeros(len(samples), M, samples[0]["queries"].shape[-1])
    qm = torch.ones(len(samples), M, dtype=torch.bool)
    q[..., 4] = 255.0                                              # padded queries are ignored
    for i, s in enumerate(samples):
        n = s["queries"].shape[0]
        q[i, :n], qm[i, :n] = s["queries"], s["queries_mask"]
    batch = {"groups": {res: {"tokens": tokens, "mask": mask, "shape": samples[0]["groups"][res]["shape"]}},
             "queries": q, "queries_mask": qm, "dataset": samples[0]["dataset"]}
    if "time_positions" in samples[0]:
        batch["time_positions"] = torch.stack([s["time_positions"] for s in samples])
    return batch
