"""
voxel_hierarchy.py -- multi-level voxel hierarchy inside the columns (GPU, per batch)
====================================================================================

Location: training/utils/datasets/voxel_hierarchy.py

For the hierarchical 3D stage (unet_vertical: hier). Built per batch, after
augmentation, from the points and their column (PointHierarchy.point_col).

Levels l = 0..S-1 with voxel sizes s_l (default 0.5, 1, 2 m, each twice the
previous one). A node of level l = one occupied voxel of size s_l INSIDE one
column (column id is part of the node key), so:
    * merges never cross a column: the parent of a level-l node is the
      level-(l+1) voxel of the same column that contains it (octree nesting:
      same origin, size doubling, so every node has exactly one parent);
    * the parent of a top-level node is its column (b * L0 + col), which the
      2D U-Net receives;
    * attention neighbours DO cross columns: the K nodes nearest (centroid
      distance) among the nodes in the 27 voxel positions around a node, in
      any column of the same sample (self always first).

The voxel grid is anchored per sample (x, y at the patch corner, z at the
sample's lowest point), so a sample gives the same hierarchy alone or in a batch.

Everything is flat over the batch: level l has n_l nodes, with node_batch / node_col.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class VoxelLevel:
    size: float
    n: int
    batch: torch.Tensor          # [n] sample of each node
    col: torch.Tensor            # [n] column inside its sample (0..L0-1)
    pos: torch.Tensor            # [n, 3] centroid (sample frame, metres)
    cnt: torch.Tensor            # [n] points per node
    nbr: torch.Tensor            # [n, K] flat node index, self first
    nbr_mask: torch.Tensor       # [n, K] True = invalid
    parent: torch.Tensor         # [n] index into level l+1, or flat column (b * L0 + col) for the top
    children: Optional[torch.Tensor] = None       # [n, Cmax] index into level l-1 (None at level 0)
    children_mask: Optional[torch.Tensor] = None  # [n, Cmax] True = invalid


@dataclass
class VoxelHierarchy:
    levels: List[VoxelLevel]
    B: int
    L0: int
    point_node: torch.Tensor     # [B * N] level-0 node of each point (-1 = padding)
    node_atoms: torch.Tensor     # [n0, m] flat point index (into B * N)
    node_atoms_mask: torch.Tensor  # [n0, m] True = padding
    col_children: torch.Tensor   # [B * L0, Ccol] top-level nodes of each column
    col_children_mask: torch.Tensor
    col_pos: torch.Tensor        # [B * L0, 3] column centre (x, y) and mean z of its points
    zmin: torch.Tensor           # [B] grid anchor in z
    half: float
    stats: dict = field(default_factory=dict)


def _pad_groups(group, n_groups, members, cap=None):
    """members grouped by `group` (sorted or not) -> [n_groups, Cmax] index + mask."""
    dev = group.device
    order = torch.argsort(group, stable=True)
    g = group[order]; mem = members[order]
    cnt = torch.bincount(g, minlength=n_groups)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(g.numel(), device=dev) - start[g]
    cmax = int(cnt.max()) if cnt.numel() else 1
    if cap is not None:
        cmax = min(cmax, cap)
    cmax = max(cmax, 1)
    keep = rank < cmax
    idx = torch.zeros(n_groups, cmax, dtype=torch.long, device=dev)
    msk = torch.ones(n_groups, cmax, dtype=torch.bool, device=dev)
    idx[g[keep], rank[keep]] = mem[keep]
    msk[g[keep], rank[keep]] = False
    return idx, msk, int((~keep).sum())


class VoxelHierarchyBuilder:
    """
    Args:
        sizes:     voxel size per level (each twice the previous one).
        half:      half patch size (x, y in [-half, half]).
        K:         attention neighbours per node (self included).
        per_pos:   at most this many nodes considered per voxel position (columns
                   meeting at a position: usually 1-3).
        m:         atoms read per level-0 node.
    """

    def __init__(self, sizes=(0.5, 1.0, 2.0), half=25.0, K=16, per_pos=3, m=16):
        sizes = [float(s) for s in sizes]
        for a, b in zip(sizes[:-1], sizes[1:]):
            assert abs(b - 2 * a) < 1e-9, f"voxel sizes must double between levels, got {sizes}"
        self.sizes, self.half, self.K, self.per_pos, self.m = sizes, float(half), int(K), int(per_pos), int(m)

    # ------------------------------------------------------------------ keys
    def _ijk(self, rel, size):
        return torch.floor(rel / size).long()

    @torch.no_grad()
    def build(self, xyz, b, col, valid, B, L0, generator=None, point_h=None) -> VoxelHierarchy:
        """xyz [BN, 3] (x, y centred on the patch, z metres), b [BN] sample, col [BN]
        column (-1 for padding), valid [BN] bool."""
        dev = xyz.device
        BN = xyz.shape[0]
        vi = torch.nonzero(valid).squeeze(1)                      # real points only
        P, pb, pc = xyz[vi].double(), b[vi], col[vi]
        zmin = torch.full((B,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(
            0, pb, P[:, 2], reduce="amin")
        zmin = torch.where(torch.isfinite(zmin), zmin, torch.zeros_like(zmin))
        rel = torch.stack([P[:, 0] + self.half, P[:, 1] + self.half, P[:, 2] - zmin[pb]], -1).clamp(min=0)
        levels, prev_node = [], None
        pnode0 = None
        for li, s in enumerate(self.sizes):
            ijk = self._ijk(rel, s) + 1                           # +1 pad: offsets -1 never wrap
            R = (ijk.amax(0) + 2).tolist() if ijk.numel() else [3, 3, 3]
            pos_key = ((pb * R[0] + ijk[:, 0]) * R[1] + ijk[:, 1]) * R[2] + ijk[:, 2]  # (sample, voxel)
            RR = R[0] * R[1] * R[2]
            node_key = (pb * L0 + pc) * RR + (pos_key - pb * RR)                    # (sample, column, voxel)
            uk, pnode = torch.unique(node_key, sorted=True, return_inverse=True)
            n = uk.numel()
            nb_ = uk // RR // L0
            ncol = (uk // RR) % L0
            npos = nb_ * RR + uk % RR                                               # (sample, voxel) key
            cnt = torch.bincount(pnode, minlength=n)
            cen = torch.zeros(n, 3, dtype=torch.float64, device=dev).index_add_(0, pnode, P) / cnt.unsqueeze(-1)
            # ---- neighbours: nodes at the 27 positions around, any column, nearest K ----
            so = torch.argsort(npos, stable=True); sp = npos[so]
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        t = npos + (dx * R[1] + dy) * R[2] + dz
                        lo = torch.searchsorted(sp, t, right=False)
                        hi = torch.searchsorted(sp, t, right=True)
                        for j in range(self.per_pos):
                            k = lo + j
                            ok = k < hi
                            cand.append(torch.where(ok, so[k.clamp(max=n - 1)], torch.full_like(k, -1)))
            ar = torch.arange(n, device=dev).unsqueeze(1)
            cand = torch.stack(cand, 1)                                             # [n, 27 * per_pos]
            cand = torch.where(cand == ar, torch.full_like(cand, -1), cand)        # self is added explicitly
            cand = torch.cat([ar, cand], 1)
            cv = cand >= 0
            cc = cand.clamp(min=0)
            d2 = ((cen[cc] - cen.unsqueeze(1)) ** 2).sum(-1)
            d2 = torch.where(cv, d2, torch.full_like(d2, float("inf")))
            d2[:, 0] = -1.0                                                         # self always first
            k = min(self.K, cand.shape[1])
            dd, sel = d2.topk(k, dim=1, largest=False)
            nbr = cc.gather(1, sel)
            nmask = ~torch.isfinite(dd)
            nbr = torch.where(nmask, torch.arange(n, device=dev).unsqueeze(1).expand_as(nbr), nbr)
            lev = VoxelLevel(size=s, n=n, batch=nb_, col=ncol, pos=cen.float(), cnt=cnt,
                             nbr=nbr, nbr_mask=nmask, parent=torch.empty(0, dtype=torch.long, device=dev))
            if prev_node is not None:                                               # parent of previous level
                par = torch.zeros(levels[-1].n, dtype=torch.long, device=dev).scatter_(0, prev_node, pnode)
                levels[-1].parent = par
                lev.children, lev.children_mask, _ = _pad_groups(par, n, torch.arange(levels[-1].n, device=dev))
            else:
                pnode0 = pnode
            levels.append(lev)
            prev_node = pnode
        top = levels[-1]
        top.parent = top.batch * L0 + top.col
        col_children, col_mask, _ = _pad_groups(top.parent, B * L0, torch.arange(top.n, device=dev))
        # column position: (column centre is added by the caller) mean z of the column's points
        ccnt = torch.zeros(B * L0, dtype=torch.float64, device=dev).index_add_(0, top.parent, top.cnt.double())
        cz = torch.zeros(B * L0, dtype=torch.float64, device=dev).index_add_(
            0, top.parent, top.pos[:, 2].double() * top.cnt.double()) / ccnt.clamp(min=1)
        col_pos = torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), cz], -1).float()
        # ---- level-0 atoms: up to m points per node (random subset) ----
        point_node = torch.full((BN,), -1, dtype=torch.long, device=dev)
        point_node[vi] = pnode0
        n0 = levels[0].n
        rnd = torch.rand(vi.numel(), device=dev, generator=generator)
        o = torch.argsort(rnd); o = o[torch.argsort(pnode0[o], stable=True)]
        atoms, amask, dropped = _pad_groups(pnode0[o], n0, vi[o], cap=self.m)
        # _pad_groups re-sorts stably by group, keeping the random order inside each node
        stats = {f"nodes_{s:g}m": lv.n for s, lv in zip(self.sizes, levels)}
        stats["atoms_dropped"] = dropped
        stats["children_max"] = [int(lv.children.shape[1]) for lv in levels[1:]]
        stats["col_children_max"] = int(col_children.shape[1])
        return VoxelHierarchy(levels=levels, B=B, L0=L0, point_node=point_node, node_atoms=atoms,
                              node_atoms_mask=amask, col_children=col_children, col_children_mask=col_mask,
                              col_pos=col_pos, zmin=zmin.float(), half=self.half, stats=stats)

    # ------------------------------------------------------------------ queries
    @torch.no_grad()
    def query_neighbors(self, vh: VoxelHierarchy, qxyz, qb, qvalid, K=9, own=None, chunk=65536):
        """K nearest level-0 nodes (centroid distance) among the 27 voxel positions
        around each query, any column. qxyz [M, 3], qb [M]; own [M] (optional): the
        query's own level-0 node (-1 if unknown), always put first. Returns nbr [M, K]
        (flat level-0 index), mask [M, K] (True = invalid), offsets [M, K, 3] in units of s0."""
        lv = vh.levels[0]; s = lv.size; dev = qxyz.device
        rel = torch.stack([qxyz[:, 0] + vh.half, qxyz[:, 1] + vh.half,
                           qxyz[:, 2] - vh.zmin[qb]], -1).double().clamp(min=0)
        # rebuild the level-0 position keys of the nodes from their centroids' voxels
        nrel = torch.stack([lv.pos[:, 0] + vh.half, lv.pos[:, 1] + vh.half,
                            lv.pos[:, 2] - vh.zmin[lv.batch]], -1).double().clamp(min=0)
        nijk = self._ijk(nrel, s) + 1
        qijk = self._ijk(rel, s) + 1
        R = (torch.maximum(nijk.amax(0), qijk.amax(0) if qijk.numel() else nijk.amax(0)) + 2).tolist()
        npos = ((lv.batch * R[0] + nijk[:, 0]) * R[1] + nijk[:, 1]) * R[2] + nijk[:, 2]
        qpos = ((qb * R[0] + qijk[:, 0]) * R[1] + qijk[:, 1]) * R[2] + qijk[:, 2]
        so = torch.argsort(npos, stable=True); sp = npos[so]
        M = qxyz.shape[0]
        out_n = torch.zeros(M, K, dtype=torch.long, device=dev)
        out_m = torch.ones(M, K, dtype=torch.bool, device=dev)
        out_o = torch.zeros(M, K, 3, device=dev)
        for a in range(0, M, chunk):
            e = min(a + chunk, M)
            qp = qpos[a:e]
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        t = qp + (dx * R[1] + dy) * R[2] + dz
                        lo = torch.searchsorted(sp, t); hi = torch.searchsorted(sp, t, right=True)
                        for j in range(self.per_pos):
                            k = lo + j
                            cand.append(torch.where(k < hi, so[k.clamp(max=lv.n - 1)], torch.full_like(k, -1)))
            cand = torch.stack(cand, 1)
            if own is not None:
                ow = own[a:e].unsqueeze(1)
                cand = torch.where(cand == ow, torch.full_like(cand, -1), cand)
                cand = torch.cat([ow, cand], 1)
            cv = cand >= 0; cc = cand.clamp(min=0)
            d = lv.pos[cc] - qxyz[a:e].unsqueeze(1)
            d2 = torch.where(cv, (d ** 2).sum(-1), torch.full(cv.shape, float("inf"), device=dev))
            if own is not None:
                d2[:, 0] = torch.where(cv[:, 0], torch.full_like(d2[:, 0], -1.0), d2[:, 0])
            k = min(K, cand.shape[1])
            dd, sel = d2.topk(k, dim=1, largest=False)
            out_n[a:e, :k] = cc.gather(1, sel)
            out_m[a:e, :k] = ~torch.isfinite(dd)
            out_o[a:e, :k] = d.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3)) / s
        out_m |= ~qvalid.unsqueeze(1)
        out_o = out_o.masked_fill(out_m.unsqueeze(-1), 0.0)
        return out_n, out_m, out_o
