"""
ezsp_hierarchy.py -- adaptive node hierarchy inside the columns (EZ-SP partition, GPU, per batch)
================================================================================================

Location: training/utils/datasets/ezsp_hierarchy.py

Same output structure as voxel_hierarchy.VoxelHierarchy, so HierStage is unchanged
(unet_vertical: hier, unet_hier_nodes: ezsp). Instead of a uniform voxel grid, the
nodes are graph components from the EZ-SP partition (torch-graph-components,
merge_components_by_contour_prior):

  1. 0.25 m voxels inside each column; graph edges between voxels of the same
     column within `radius` voxels (weight 1 / distance), so a component never
     crosses a column;
  2. voxel features: local shape from the voxel and its 26 touching neighbours
     (linearity, planarity, scattering, |normal_z|, |axis_z|) and the height above
     the local ground (compressed);
  3. level 0 = components of that graph (reg[0], min_size[0]);
  4. level l > 0 = components of the graph of level-(l-1) components (features =
     size-weighted means, edge weights summed), with reg[l]: merging components
     keeps the nesting (one parent per node, inside the column);
  5. attention neighbours: the K nearest nodes by centroid among the nodes whose
     centroid lies in the 27 grid cells (size `sizes[l]`) around the node, any
     column of the same sample, self first (as for the voxel hierarchy);
  6. level-0 read: up to m atoms per component (random subset).

Defaults come from the Blender study (1000-point columns): reg 0.02 -> ~2800 nodes per
50 m window (purity 0.979), reg 0.1 on top -> ~780 (0.968). A third level degenerates
into one component per connected structure of a column, i.e. the column itself.

torch_scatter: required by torch-graph-components. If it is missing, a pure-PyTorch
stand-in for the three functions the library uses is registered (slower, same results).
"""
from __future__ import annotations

import math
import sys
import types

import torch

from .voxel_hierarchy import VoxelHierarchy, VoxelHierarchyBuilder, VoxelLevel, _pad_groups


# =============================================================================
# torch_scatter: real package if installed, otherwise a pure-PyTorch stand-in
# =============================================================================
def _install_scatter_shim():
    try:
        import torch_scatter  # noqa: F401
        return False
    except ImportError:
        pass
    m = types.ModuleType("torch_scatter")
    m.__version__ = "pure-torch-shim"

    def _prep(src, index, dim, dim_size):
        dim = dim % src.dim()
        src = src.movedim(dim, 0)
        index = index.movedim(dim, 0) if index.dim() == src.dim() else index
        if index.dim() == 1 and src.dim() > 1:
            index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        size = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
        return src, index, dim, size

    def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
        src, index, dim, size = _prep(src, index, dim, dim_size)
        res = torch.zeros((size, *src.shape[1:]), dtype=src.dtype, device=src.device).scatter_add_(0, index, src)
        return res.movedim(0, dim)

    def _ext(src, index, dim, dim_size, reduce):
        src, index, dim, size = _prep(src, index, dim, dim_size)
        if src.is_floating_point():
            fill = float("inf") if reduce == "amin" else float("-inf")
        else:
            info = torch.iinfo(src.dtype); fill = info.max if reduce == "amin" else info.min
        val = torch.full((size, *src.shape[1:]), fill, dtype=src.dtype, device=src.device)
        val = val.scatter_reduce(0, index, src, reduce=reduce, include_self=True)
        cnt = torch.zeros((size, *src.shape[1:]), dtype=torch.long, device=src.device).scatter_add_(
            0, index, torch.ones_like(index))
        empty = cnt == 0
        val = torch.where(empty, torch.zeros_like(val), val)
        N = src.shape[0]
        pos = torch.arange(N, device=src.device).view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        cand = torch.where(src == val.gather(0, index), pos, torch.full_like(pos, N))
        arg = torch.full((size, *src.shape[1:]), N, dtype=torch.long, device=src.device).scatter_reduce(
            0, index, cand, reduce="amin", include_self=True)
        arg = torch.where(empty, torch.full_like(arg, N), arg)
        return val.movedim(0, dim), arg.movedim(0, dim)

    m.scatter_sum = scatter_sum
    m.scatter_add = scatter_sum
    m.scatter_min = lambda src, index, dim=-1, out=None, dim_size=None: _ext(src, index, dim, dim_size, "amin")
    m.scatter_max = lambda src, index, dim=-1, out=None, dim_size=None: _ext(src, index, dim, dim_size, "amax")
    sys.modules["torch_scatter"] = m
    print("[ezsp_hierarchy] torch_scatter not found: using the pure-PyTorch stand-in (slower)")
    return True


_install_scatter_shim()
import torch_graph_components as tgc  # noqa: E402


def eigh_safe(cov: torch.Tensor, chunk: int = 32768):
    """torch.linalg.eigh for a large batch of small symmetric matrices, robust on GPU:
    non-finite entries are zeroed, the matrices are symmetrised, and the batch is
    processed in chunks (cuSOLVER's batched solver rejects very large batches with
    CUSOLVER_STATUS_INVALID_VALUE)."""
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    if cov.shape[0] <= chunk:
        return torch.linalg.eigh(cov)
    ws, vs = zip(*(torch.linalg.eigh(cov[s:s + chunk]) for s in range(0, cov.shape[0], chunk)))
    return torch.cat(ws, 0), torch.cat(vs, 0)


class EzspHierarchyBuilder(VoxelHierarchyBuilder):
    """
    Args:
        regs:      EZ-SP regularisation per level (coarser with larger values).
        min_sizes: minimum component size per level (points at level 0, nodes above).
        sizes:     grid size per level for the neighbour search and the positional
                   units (RoPE, merge offsets, query offsets), e.g. (1.0, 2.0) m.
        vox:       voxel size of the partition graph (m); radius: edge radius in voxels.
        K, per_pos, m, half: as in VoxelHierarchyBuilder (per_pos: nodes considered
                   per grid cell of the neighbour search; components are smaller
                   than the cells, so use more than for voxels).
    """

    def __init__(self, regs=(0.02, 0.1), min_sizes=(10, 1), sizes=(1.0, 2.0), vox=0.25, radius=2,
                 half=25.0, K=16, per_pos=8, m=32, h_scale=5.0):
        assert len(regs) == len(min_sizes) == len(sizes), "one reg, min_size and size per level"
        self.regs, self.min_sizes = [float(r) for r in regs], [int(s) for s in min_sizes]
        self.sizes = [float(s) for s in sizes]
        self.vox, self.radius, self.h_scale = float(vox), int(radius), float(h_scale)
        self.half, self.K, self.per_pos, self.m = float(half), int(K), int(per_pos), int(m)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _merge(X, S, a, b, W, reg, min_size):
        E = torch.stack([a, b]).long()
        # the library draws random permutations (torch.randperm in its connected
        # components): fix the seed locally so a tile always gets the same partition,
        # without touching the training RNG streams
        devs = [X.device] if X.is_cuda else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(0)
            I, _, _ = tgc.merge_components_by_contour_prior(X.float(), S.float(), E, W.float(), float(reg),
                                                            int(min_size), verbose=False)
        return torch.unique(I.to(a.device), return_inverse=True)[1]       # compact labels 0..k-1

    @staticmethod
    def _coarsen(lab, X, S, a, b, W):
        K = int(lab.max()) + 1
        S2 = torch.zeros(K, dtype=S.dtype, device=S.device).index_add_(0, lab, S)
        X2 = torch.zeros(K, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, lab, X * S.unsqueeze(-1))
        X2 = X2 / S2.clamp(min=1e-9).unsqueeze(-1)
        la, lb = lab[a], lab[b]
        keep = la != lb
        lo, hi = torch.minimum(la[keep], lb[keep]), torch.maximum(la[keep], lb[keep])
        pair, inv = torch.unique(lo * K + hi, return_inverse=True)
        W2 = torch.zeros(pair.numel(), dtype=W.dtype, device=W.device).index_add_(0, inv, W[keep])
        return X2, S2, pair // K, pair % K, W2

    def _knn(self, pos, nb, zmin, size):
        """K nearest nodes (centroid) among the 27 grid cells around each node, same sample."""
        dev = pos.device
        n = pos.shape[0]
        rel = torch.stack([pos[:, 0] + self.half, pos[:, 1] + self.half, pos[:, 2] - zmin[nb]], -1).double().clamp(min=0)
        ijk = torch.floor(rel / size).long() + 1
        R = (ijk.amax(0) + 2).tolist()
        key = ((nb * R[0] + ijk[:, 0]) * R[1] + ijk[:, 1]) * R[2] + ijk[:, 2]
        so = torch.argsort(key, stable=True); sk = key[so]
        ar = torch.arange(n, device=dev).unsqueeze(1)
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    t = key + (dx * R[1] + dy) * R[2] + dz
                    lo = torch.searchsorted(sk, t); hi = torch.searchsorted(sk, t, right=True)
                    for j in range(self.per_pos):
                        k = lo + j
                        cand.append(torch.where(k < hi, so[k.clamp(max=n - 1)], torch.full_like(k, -1)))
        cand = torch.stack(cand, 1)
        cand = torch.where(cand == ar, torch.full_like(cand, -1), cand)
        cand = torch.cat([ar, cand], 1)
        cv = cand >= 0; cc = cand.clamp(min=0)
        d2 = ((pos[cc] - pos.unsqueeze(1)) ** 2).sum(-1)
        d2 = torch.where(cv, d2, torch.full_like(d2, float("inf")))
        d2[:, 0] = -1.0
        k = min(self.K, cand.shape[1])
        dd, sel = d2.topk(k, dim=1, largest=False)
        nbr = cc.gather(1, sel); nmask = ~torch.isfinite(dd)
        nbr = torch.where(nmask, ar.expand_as(nbr), nbr)
        return nbr, nmask

    # ------------------------------------------------------------------ partition
    @torch.no_grad()
    def partition_points(self, P, rel, gkey, ph, n_groups):
        """Nested EZ-SP partition of points. P [n, 3] (float64), rel [n, 3] grid frame,
        gkey [n] group id (graph edges only inside a group: column online, tile offline),
        ph [n] height above local ground. Returns [(lab_l, point_lab_l)] per level:
        point_lab_l = level-l component of each point (compact), lab_l maps the units of
        the previous level (voxels for l = 0) to level-l components."""
        dev = P.device
        r = self.radius
        # ---- 1. voxels inside columns, radius graph ----
        ijk = torch.floor(rel / self.vox).long() + r
        R = (ijk.amax(0) + r + 1).tolist() if ijk.numel() else [2 * r + 1] * 3
        gcol = gkey
        assert float(n_groups) * R[0] * R[1] * R[2] < 2 ** 62, "voxel key overflow"
        key = ((gcol * R[0] + ijk[:, 0]) * R[1] + ijk[:, 1]) * R[2] + ijk[:, 2]
        uk, vox = torch.unique(key, sorted=True, return_inverse=True)
        V = uk.numel()
        I, J, Wd = [], [], []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (dx, dy, dz) <= (0, 0, 0):
                        continue
                    nbk = uk + (dx * R[1] + dy) * R[2] + dz
                    p = torch.searchsorted(uk, nbk).clamp(max=V - 1)
                    ok = uk[p] == nbk
                    I.append(torch.nonzero(ok).squeeze(1)); J.append(p[ok])
                    Wd.append(torch.full((int(ok.sum()),), 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz),
                                         dtype=torch.float64, device=dev))
        I, J, Wd = torch.cat(I), torch.cat(J), torch.cat(Wd)
        # ---- 2. voxel features ----
        vc = torch.bincount(vox, minlength=V).double()
        S1 = torch.zeros(V, 3, dtype=torch.float64, device=dev).index_add_(0, vox, P)
        S2 = torch.zeros(V, 3, 3, dtype=torch.float64, device=dev).index_add_(0, vox, P.unsqueeze(-1) * P.unsqueeze(-2))
        touch = Wd >= 1.0 / math.sqrt(3) - 1e-9
        It, Jt = I[touch], J[touch]
        Nc = vc.clone().index_add_(0, It, vc[Jt]).index_add_(0, Jt, vc[It])
        N1 = S1.clone().index_add_(0, It, S1[Jt]).index_add_(0, Jt, S1[It])
        N2 = S2.clone().index_add_(0, It, S2[Jt]).index_add_(0, Jt, S2[It])
        mu = N1 / Nc.unsqueeze(-1)
        cov = N2 / Nc.view(-1, 1, 1) - mu.unsqueeze(-1) * mu.unsqueeze(-2)
        w, Vec = eigh_safe(cov)
        s = w.clamp(min=0).sqrt().flip(-1); s0 = s[:, 0].clamp(min=1e-6)
        vh = torch.zeros(V, dtype=torch.float64, device=dev).index_add_(0, vox, ph) / vc
        X = torch.stack([(s[:, 0] - s[:, 1]) / s0, (s[:, 1] - s[:, 2]) / s0, s[:, 2] / s0,
                         Vec[:, 2, 0].abs(), Vec[:, 2, 2].abs(), vh / (vh.abs() + self.h_scale)], -1)
        X[Nc < 5, :5] = 0.0
        # ---- 3-4. nested partitions ----
        Xn, Sn, a, bb, Wn = X, vc, I, J, Wd
        point_lab = vox                                                     # point -> current unit
        prev_lab, labs = None, []
        for reg, ms in zip(self.regs, self.min_sizes):
            lab = self._merge(Xn, Sn, a, bb, Wn, reg, ms)
            point_lab = lab[point_lab]
            labs.append((lab, point_lab))
            Xn, Sn, a, bb, Wn = self._coarsen(lab, Xn, Sn, a, bb, Wn)
        return labs

    # ------------------------------------------------------------------ build
    @torch.no_grad()
    def build(self, xyz, b, col, valid, B, L0, generator=None, point_h=None) -> VoxelHierarchy:
        dev = xyz.device
        BN = xyz.shape[0]
        vi = torch.nonzero(valid).squeeze(1)
        P, pb, pc = xyz[vi].double(), b[vi], col[vi]
        ph = (point_h[vi].double() if point_h is not None else torch.zeros(vi.numel(), dtype=torch.float64, device=dev))
        zmin = torch.full((B,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(0, pb, P[:, 2], reduce="amin")
        zmin = torch.where(torch.isfinite(zmin), zmin, torch.zeros_like(zmin))
        rel = torch.stack([P[:, 0] + self.half, P[:, 1] + self.half, P[:, 2] - zmin[pb]], -1).clamp(min=0)
        labs = self.partition_points(P, rel, pb * L0 + pc, ph, B * L0)
        return self._levels(labs, P, pb, pc, vi, zmin, B, L0, BN, generator)

    def _levels(self, labs, P, pb, pc, vi, zmin, B, L0, BN, generator):
        dev = P.device
        # ---- 5. levels ----
        levels = []
        for li, (lab, plab) in enumerate(labs):
            n = int(plab.max()) + 1
            cnt = torch.bincount(plab, minlength=n)
            cen = torch.zeros(n, 3, dtype=torch.float64, device=dev).index_add_(0, plab, P) / cnt.unsqueeze(-1)
            first = torch.full((n,), BN, dtype=torch.long, device=dev).scatter_reduce(
                0, plab, torch.arange(plab.numel(), device=dev), reduce="amin")
            nbatch, ncol = pb[first], pc[first]
            nbr, nmask = self._knn(cen.float(), nbatch, zmin, self.sizes[li])
            lv = VoxelLevel(size=self.sizes[li], n=n, batch=nbatch, col=ncol, pos=cen.float(), cnt=cnt,
                            nbr=nbr, nbr_mask=nmask, parent=torch.empty(0, dtype=torch.long, device=dev))
            if li > 0:
                par = lab                                                   # level-(l-1) node -> level-l node
                levels[-1].parent = par
                lv.children, lv.children_mask, _ = _pad_groups(par, n, torch.arange(levels[-1].n, device=dev))
            levels.append(lv)
        top = levels[-1]
        top.parent = top.batch * L0 + top.col
        col_children, col_mask, _ = _pad_groups(top.parent, B * L0, torch.arange(top.n, device=dev))
        ccnt = torch.zeros(B * L0, dtype=torch.float64, device=dev).index_add_(0, top.parent, top.cnt.double())
        cz = torch.zeros(B * L0, dtype=torch.float64, device=dev).index_add_(
            0, top.parent, top.pos[:, 2].double() * top.cnt.double()) / ccnt.clamp(min=1)
        col_pos = torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), cz], -1).float()
        # ---- 6. level-0 atoms ----
        plab0 = labs[0][1]
        point_node = torch.full((BN,), -1, dtype=torch.long, device=dev)
        point_node[vi] = plab0
        rnd = torch.rand(vi.numel(), device=dev, generator=generator)
        o = torch.argsort(rnd); o = o[torch.argsort(plab0[o], stable=True)]
        atoms, amask, dropped = _pad_groups(plab0[o], levels[0].n, vi[o], cap=self.m)
        stats = {f"nodes_L{i}": lv.n for i, lv in enumerate(levels)}
        stats["atoms_dropped"] = dropped
        stats["children_max"] = [int(lv.children.shape[1]) for lv in levels[1:]]
        stats["col_children_max"] = int(col_children.shape[1])
        return VoxelHierarchy(levels=levels, B=B, L0=L0, point_node=point_node, node_atoms=atoms,
                              node_atoms_mask=amask, col_children=col_children, col_children_mask=col_mask,
                              col_pos=col_pos, zmin=zmin.float(), half=self.half, stats=stats)


class PrecomputedEzspBuilder(EzspHierarchyBuilder):
    """Nodes from an EZ-SP partition precomputed once per tile WITHOUT the column
    constraint (scripts_atomizer/precompute_ezsp.py), split by column here.

    Valid under the training augmentation: rotations / flips are rigid (each point
    keeps its component), the square crop and the context subsampling only remove
    points, and splitting by column keeps the nesting ((L0, column) lies inside
    (L1, column) because each L0 component lies inside one L1 component).
    Only the cheap part runs per batch: unique labels, centroids, neighbour tables."""

    @torch.no_grad()
    def build(self, xyz, b, col, valid, B, L0, generator=None, point_h=None, part=None) -> VoxelHierarchy:
        assert part is not None, ("precomputed partition labels missing: run precompute_ezsp.py and "
                                  "load them with the dataset (batch['point_part'])")
        dev = xyz.device
        BN = xyz.shape[0]
        vi = torch.nonzero(valid).squeeze(1)
        P, pb, pc = xyz[vi].double(), b[vi], col[vi]
        zmin = torch.full((B,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(0, pb, P[:, 2], reduce="amin")
        zmin = torch.where(torch.isfinite(zmin), zmin, torch.zeros_like(zmin))
        pr = part.reshape(BN, -1)[vi].long()
        assert (pr >= 0).all(), "a real point has no precomputed partition label"
        labs, prev = [], None
        for l in range(pr.shape[1]):
            lab_l = pr[:, l]
            key = (pb * L0 + pc) * (int(lab_l.max()) + 1) + lab_l              # (sample, column, component)
            pl = torch.unique(key, return_inverse=True)[1]
            lab = None
            if prev is not None:                                                 # nested: one parent per node
                lab = torch.zeros(int(prev.max()) + 1, dtype=torch.long, device=dev).scatter_(0, prev, pl)
            labs.append((lab, pl)); prev = pl
        return self._levels(labs, P, pb, pc, vi, zmin, B, L0, BN, generator)
