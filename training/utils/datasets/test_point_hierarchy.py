#!/usr/bin/env python3
"""
test_point_hierarchy.py
=======================

Checks for point_hierarchy.py.

Synthetic checks (always run):
  1. columns: assignment vs brute-force nearest anchor (differences allowed
     only within half a virtual site of a Voronoi boundary)
  2. vertical cells: z-ordered within each column, sizes <= kz (or the cap
     share in capped columns), count <= vmax, every valid point in one cell,
     cell_atoms point back to their cell
  3. batching: a padded batch gives the same result as each sample alone
  4. queries: own cell found for every context query; for absent queries the
     z-boundary lookup agrees with the point's own cell
  5. local ground: on a sloped scene with a building, height above local
     ground stays small on the ground and on the roof edge columns' ground
  6. occupancy masks per level

Real tiles (with --tiled_dir): runs the dataset + collate + hierarchy on a few
tiles, prints statistics and timing (GPU if available).

    # from anywhere (the repo root is found from this file's location):
    python training/utils/datasets/test_point_hierarchy.py              # synthetic only
    python training/utils/datasets/test_point_hierarchy.py \
        --tiled_dir /path/to/DALES/train --n_tiles 4 --batch 2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# training/utils/datasets/test_point_hierarchy.py -> repo root is 3 levels up
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils.datasets.point_hierarchy import PointHierarchy
from training.utils.datasets.dataset_dales_points import (
    PointLayout as PL, DalesPointDataset, collate_dales)

C = PL.N_COLS


def make_cloud(n, rng, half=25.0, slope=0.0, building=False):
    x = rng.uniform(-half, half, n)
    y = rng.uniform(-half, half, n)
    ground = slope * x
    z = ground.copy()
    veg = rng.random(n) < 0.3
    z[veg] += rng.exponential(4.0, veg.sum())          # vegetation above ground
    if building:
        inb = (np.abs(x - 5) < 6) & (np.abs(y + 3) < 6)
        z[inb] = ground[inb] + 8.0                      # flat roof, no ground below
    pts = np.zeros((n, C), np.float32)
    pts[:, PL.X], pts[:, PL.Y], pts[:, PL.Z] = x, y, z
    return pts, ground


def batch_of(clouds):
    B, N = len(clouds), max(len(c) for c in clouds)
    pts = torch.zeros(B, N, C)
    mask = torch.ones(B, N, dtype=torch.bool)
    for b, c in enumerate(clouds):
        pts[b, :len(c)] = torch.from_numpy(c)
        mask[b, :len(c)] = False
    return pts, mask


def check_columns(ph, pts, mask, pg):
    g = ph.static(pts.device)
    c = g.centers[0].double()
    worst, n_diff, n_tot = 0.0, 0, 0
    for b in range(pts.shape[0]):
        v = ~mask[b]
        xy = pts[b, v][:, [PL.X, PL.Y]].double()
        d = torch.cdist(xy, c)
        nn = d.argmin(1)
        got = pg.point_col[b, v]
        diff = got != nn
        n_diff += int(diff.sum()); n_tot += int(v.sum())
        if diff.any():
            # extra distance of the chosen anchor vs the true nearest one
            extra = d[diff, got[diff]] - d[diff, nn[diff]]
            worst = max(worst, float(extra.max()))
    tol = ph.s * np.sqrt(2)                              # one virtual-site diagonal
    assert worst <= tol + 1e-6, f"column assignment off by {worst:.3f} m (> {tol:.3f})"
    print(f"  [1] columns: {n_diff}/{n_tot} points differ from brute force "
          f"({100 * n_diff / max(n_tot, 1):.3f}%), worst extra distance {worst:.3f} m "
          f"(<= site diagonal {tol:.3f} m)")


def check_vertical(ph, pts, mask, pg):
    B, N, _ = pts.shape
    valid = ~mask
    assert ((pg.point_cell >= 0) == valid).all(), "every valid point must be in exactly one cell"
    assert (pg.col_ncells <= ph.vmax).all()
    flat_z = pts[..., PL.Z].reshape(-1)
    # z-order and sizes, per (b, column)
    zlo = pg.v_zlo.view(B, pg.L0, ph.vmax)
    zhi = pg.v_zhi.view(B, pg.L0, ph.vmax)
    vm = pg.vmask
    ok_order = True
    for b in range(B):
        for col in torch.nonzero(pg.col_ncells[b] > 1).squeeze(1).tolist():
            n = int(pg.col_ncells[b, col])
            assert not vm[b, col, :n].any() and vm[b, col, n:].all(), "cells must be contiguous from 0"
            if (zhi[b, col, :n - 1] > zlo[b, col, 1:n]).any():
                ok_order = False
    assert ok_order, "vertical cells must be ordered in z"
    cap = (pg.col_count + ph.vmax - 1) // ph.vmax                 # per-cell share if capped
    limit = torch.maximum(torch.full_like(cap, ph.kz), cap)
    per_col_max = torch.zeros_like(pg.col_count)
    cc = pg.cell_batch * pg.L0 + pg.cell_col
    cnt = pg.v_count.reshape(-1)[pg.cell_slot]
    per_col_max.view(-1).scatter_reduce_(0, cc, cnt, reduce="amax")
    assert (per_col_max <= limit).all(), "cell larger than kz (or the cap share)"
    # cell_atoms point back to their own cell
    atoms = pg.cell_atoms[~pg.cell_atoms_mask]
    owner = torch.arange(pg.num_cells, device=pts.device).unsqueeze(1).expand_as(pg.cell_atoms)[~pg.cell_atoms_mask]
    assert (pg.point_cell.reshape(-1)[atoms] == owner).all(), "cell_atoms inconsistent"
    read = (~pg.cell_atoms_mask).sum(1)
    assert (read == torch.minimum(cnt, torch.full_like(cnt, ph.m))).all()
    nc = pg.col_ncells[pg.col_count > 0].float()
    print(f"  [2] vertical: {pg.num_cells} cells, per column mean {nc.mean():.2f} max {int(nc.max())}, "
          f"cell size max {int(cnt.max())}, capped columns {int((pg.col_count > ph.kz * ph.vmax).sum())}")


def check_batching(ph, clouds, pg):
    for b, c in enumerate(clouds):
        p1, m1 = batch_of([c])
        g1 = ph(p1, m1)
        n = len(c)
        assert torch.equal(g1.point_col[0, :n], pg.point_col[b, :n])
        assert torch.equal(g1.point_v[0, :n], pg.point_v[b, :n])
        assert torch.equal(g1.col_count[0], pg.col_count[b])
        assert torch.allclose(g1.v_zmean[0], pg.v_zmean[b])
    print(f"  [3] batching: {len(clouds)} samples alone == in a padded batch")


def check_queries(ph, pts, mask, pg):
    B, N, _ = pts.shape
    idx = torch.arange(N).unsqueeze(0).expand(B, N).clone()
    ctx = torch.where(mask, torch.full_like(idx, -1), idx)
    q = ph.lookup_queries(pg, pts, mask, ctx, K=9)
    v = ~mask
    assert (q["col"][v] == pg.point_col[v]).all() and (q["v"][v] == pg.point_v[v]).all()
    own_slot = pg.point_col * ph.vmax + pg.point_v
    has_own = ((q["nbr"] == own_slot.unsqueeze(-1)) & ~q["nbr_mask"]).any(-1)
    # boundary lookup without the context link
    # boundary lookup without the context link. With ties in z (quantised z,
    # flat ground) several cells share the same z range, and any of them is a
    # valid answer: we require the found cell's [z_lo, z_hi] to contain z
    # (or z to fall in the gap just above it).
    q2 = ph.lookup_queries(pg, pts, mask, None, K=9)
    exact = (q2["v"][v] == pg.point_v[v]).float().mean()
    col, vv, z = q2["col"][v], q2["v"][v], pts[..., PL.Z][v]
    bb = torch.arange(B, device=pts.device).unsqueeze(1).expand(B, N)[v]
    zlo, zhi = pg.v_zlo[bb, col, vv], pg.v_zhi[bb, col, vv]
    nxt = (vv + 1).clamp(max=ph.vmax - 1)
    last = (vv + 1) >= pg.col_ncells[bb, col]
    zlo_next = torch.where(last, torch.full_like(z, float("inf")), pg.v_zlo[bb, col, nxt])
    compatible = (zlo <= z) & ((z <= zhi) | (z < zlo_next))
    agree = compatible.float().mean()
    assert agree > 0.999, f"z-boundary lookup compatible for only {agree:.4f}"
    assert not q["nbr_mask"][v][:, 0].any(), "every query must have at least one neighbour cell"
    assert has_own[v].all(), "own cell must be the first neighbour"
    print(f"  [4] queries: own cell found for all; own cell among K neighbours for "
          f"{100 * has_own[v].float().mean():.1f}%; boundary lookup: z-compatible {100 * agree:.2f}%, "
          f"same cell {100 * exact:.1f}% (differences = ties in z)")


def check_ground(ph, rng):
    pts, ground = make_cloud(120_000, rng, slope=0.2, building=True)
    p, m = batch_of([pts])
    pg = ph(p, m)
    h = pg.point_h[0].numpy()
    x, y, z = pts[:, PL.X], pts[:, PL.Y], pts[:, PL.Z]
    on_ground = np.isclose(z, ground)
    roof = (np.abs(x - 5) < 6) & (np.abs(y + 3) < 6)
    raw_err = np.abs(z - 0.0)[on_ground].mean()          # tile-level reference: slope
    print(f"  [5] local ground: |h| on ground points mean {np.abs(h[on_ground]).mean():.2f} m "
          f"(tile reference would give {raw_err:.2f} m on this slope); roof height "
          f"median {np.median(h[roof]):.2f} m (true 8 m, interior columns see no ground)")


def check_levels(ph, pts, mask, pg):
    for lvl, e in enumerate(pg.level_empty):
        assert e.shape[1] == ph.static(pts.device).sizes[lvl]
    print(f"  [6] occupancy: empty cells per level (sample 0) "
          f"{[int(e[0].sum()) for e in pg.level_empty]} of {ph.static(pts.device).sizes}")


def synthetic(device):
    print(f"[synthetic] device={device}")
    rng = np.random.default_rng(0)
    ph = PointHierarchy(verbose=True)
    clouds = [make_cloud(n, rng, slope=s, building=bd)[0]
              for n, s, bd in [(60_000, 0.0, False), (115_000, 0.1, True), (30_000, 0.0, False)]]
    # a dense patch to trigger the vmax cap
    dense = make_cloud(20_000, rng)[0]
    dense[:, PL.X] = rng.uniform(-1, 1, len(dense)); dense[:, PL.Y] = rng.uniform(-1, 1, len(dense))
    clouds[2] = np.concatenate([clouds[2], dense])
    pts, mask = batch_of(clouds)
    pts, mask = pts.to(device), mask.to(device)
    t = time.time()
    pg = ph(pts, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  forward: {time.time() - t:.3f} s for B={len(clouds)}, N={pts.shape[1]}")
    check_columns(ph, pts, mask, pg)
    check_vertical(ph, pts, mask, pg)
    check_batching(ph, [c for c in clouds], pg) if device.type == "cpu" else None
    check_queries(ph, pts, mask, pg)
    check_ground(ph, rng)
    check_levels(ph, pts, mask, pg)
    off = ph.atom_offsets(pg, pts)
    assert off.shape == (pg.num_cells, ph.m, 3) and torch.isfinite(off).all()
    print(f"  atom offsets |dxy| max {off[..., :2].abs().max():.2f}, |dz| max {off[..., 2].abs().max():.2f}")
    print("[synthetic] all checks passed")


def real(args, device):
    from torch.utils.data import DataLoader
    ds = DalesPointDataset(args.tiled_dir, "val")
    ds.rows = ds.rows[:args.n_tiles]
    dl = DataLoader(ds, batch_size=args.batch, collate_fn=collate_dales, num_workers=0)
    ph = PointHierarchy(verbose=True)
    for i, batch in enumerate(dl):
        pts, mask = batch["points"].to(device), batch["point_mask"].to(device)
        t = time.time()
        pg = ph(pts, mask)
        q = ph.lookup_queries(pg, batch["query_points"].to(device), batch["query_mask"].to(device),
                              batch["query_ctx_idx"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t
        nc = pg.col_ncells[pg.col_count > 0].float()
        cnt = pg.v_count.reshape(-1)[pg.cell_slot].float()
        print(f"[real] batch {i}: {batch['patch_id']} N={pts.shape[1]} | geometry+queries {dt:.3f} s | "
              f"empty columns {int(pg.col_empty.sum())} | cells/column mean {nc.mean():.2f} "
              f"p99 {torch.quantile(nc, 0.99):.0f} max {int(nc.max())} | cell size mean {cnt.mean():.1f} "
              f"max {int(cnt.max())} | h_local p1/p50/p99 "
              f"{[round(float(torch.quantile(pg.point_h[~mask], p)), 2) for p in (0.01, 0.5, 0.99)]}")
        check_columns(ph, pts, mask, pg)
        check_vertical(ph, pts, mask, pg)
        check_queries(ph, pts, mask, pg)
    print("[real] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiled_dir", default=None)
    ap.add_argument("--n_tiles", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    synthetic(torch.device("cpu"))
    if device.type == "cuda":
        synthetic(device)
    if args.tiled_dir:
        real(args, device)
