"""
point_hierarchy.py
==================

Geometry for point clouds (DALES) in the Atomiser U-Net with the vertical stage.

Two parts
---------
STATIC (built once per configuration, shared by the whole batch)
    The usual HierarchyGeometry (hex anchors per level, 7-neighbour tables,
    merge / upsampling tables, overlap weights), built on a VIRTUAL regular
    site grid covering the fixed square patch. Consequences:
      * L per level is identical for every patch (no empty-anchor pruning),
        so batching needs no per-level padding; cells empty in a given patch
        are masked latents;
      * overlap weights count virtual sites, i.e. they are proportional to
        AREA, not to where the points happen to be;
      * a dense lookup table per level maps a virtual site to its cell, so
        assigning a point to its cell is O(1): floor onto the virtual grid
        and read the table (exact up to half a virtual site at boundaries).

DYNAMIC (per batch, on the GPU, vectorised)
    * point -> column (level-0 leaf), and occupancy per level (empty masks);
    * count-based vertical cells inside each column: points sorted by z,
      n_cells = clamp(ceil(count / kz), 1, vmax), balanced sizes;
    * per vertical cell: z_lo, z_hi, z_mean, count, and up to m atoms
      (random subset when a capped column makes a cell larger than m);
    * local ground per column: the column's low z quantile, replaced by the
      minimum over its 7-neighbour ring when it sits more than ground_jump_m
      above it (a roof with no ground below) or the column is empty; height
      above local ground per point / cell;
    * query lookup: each query's column and vertical cell, and its K nearest
      vertical cells in 3D among its own and neighbouring columns.

Layout used by the model (one CLS per column)
    Dense vertical tensor  [B, L0, vmax]   (vmask: True = padding)
    V pass:   attention over [CLS, cells] of one column
    H pass:   2D local block over the L0 CLS tokens (static local_idx[0])
    Collapse: the CLS is the column's 2D latent (level 0 of the U-Net)
    The leaf read fills the vertical cells from their atoms (packed form,
    `cell_atoms`), then scatters into the dense tensor with `cell_slot`.

Conventions
    * points: [B, N, C] with columns given by PointLayout (x, y, z in metres,
      x, y centred on the patch, z relative to a tile ground reference).
    * masks are True = padding / invalid (same as the rest of the codebase).
    * columns are indexed 0..L0-1; padded points get column L0 internally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Repository layout:
#   training/atomiser/GeographicPruningUNet.py        (HierarchyGeometry)
#   training/utils/datasets/dataset_dales_points.py   (PointLayout)
#   training/utils/datasets/point_hierarchy.py        (this file)
# The repo root (folder containing training/) must be importable; when this
# file is run or imported directly, it is added to sys.path below.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.atomiser.GeographicPruningUNet import HierarchyGeometry

try:  # imported as part of the training.utils.datasets package
    from .dataset_dales_points import PointLayout
except ImportError:  # imported as a top-level module (script in this folder)
    from training.utils.datasets.dataset_dales_points import PointLayout


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


# =============================================================================
# Per-batch geometry container
# =============================================================================
@dataclass
class PointBatchGeometry:
    B: int
    L0: int
    vmax: int
    m: int
    # per point [B, N]
    point_col: torch.Tensor        # column id, -1 for padding
    point_v: torch.Tensor          # vertical cell within the column, -1 for padding
    point_cell: torch.Tensor       # packed cell id, -1 for padding
    point_h: torch.Tensor          # height above local ground (m), 0 for padding
    # per column [B, L0]
    col_count: torch.Tensor        # points per column
    col_ncells: torch.Tensor       # vertical cells per column (0 if empty)
    col_empty: torch.Tensor        # True = no point
    col_ground: torch.Tensor       # local ground (m)
    # dense vertical cells [B, L0, vmax]
    vmask: torch.Tensor            # True = padding slot
    v_zlo: torch.Tensor
    v_zhi: torch.Tensor
    v_zmean: torch.Tensor
    v_count: torch.Tensor
    v_h: torch.Tensor              # z_mean - local ground
    # packed vertical cells [V] (V = number of occupied cells in the batch)
    cell_batch: torch.Tensor
    cell_col: torch.Tensor
    cell_v: torch.Tensor
    cell_slot: torch.Tensor        # flat index into [B * L0 * vmax]
    cell_atoms: torch.Tensor       # [V, m] flat index into [B * N]
    cell_atoms_mask: torch.Tensor  # [V, m] True = padding
    # occupancy of every 2D level
    level_count: list              # per level [B, L_l]
    level_empty: list              # per level [B, L_l], True = no point
    # cell centroids in x, y [B, L0, vmax] (column centre on padding slots)
    v_xmean: Optional[torch.Tensor] = None
    v_ymean: Optional[torch.Tensor] = None
    stats: Optional[dict] = None   # partition statistics (cluster mode)

    @property
    def num_cells(self) -> int:
        return int(self.cell_slot.shape[0])

    def scatter_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Packed [V, D] -> dense [B, L0, vmax, D] (zeros on padding slots)."""
        out = x.new_zeros(self.B * self.L0 * self.vmax, x.shape[-1])
        out[self.cell_slot] = x
        return out.view(self.B, self.L0, self.vmax, -1)

    def gather_cells(self, dense: torch.Tensor) -> torch.Tensor:
        """Dense [B, L0, vmax, D] -> packed [V, D]."""
        return dense.reshape(self.B * self.L0 * self.vmax, -1)[self.cell_slot]


# =============================================================================
# Module
# =============================================================================
class PointHierarchy(nn.Module):
    """
    Args:
        patch_size_m:     side of the fixed square (metres).
        points_per_leaf:  target points per level-0 column (k).
        density:          nominal points per m^2 (sets the leaf area k / density).
        levels, reduction, local_k: as in GeographicPruningUNet.
        site_spacing_m:   virtual site grid spacing (assignment resolution).
        kz:               target points per vertical cell.
        vmax:             max vertical cells per column (cap).
        m:                max atoms read per vertical cell (default kz).
        ground_quantile:  low z quantile per column used for the local ground.
        ground_jump_m:    a column whose low quantile is more than this above the
                          lowest one in its 7-ring (e.g. a roof with no ground
                          below) takes the ring minimum as its ground; other
                          columns keep their own (no downhill bias on slopes).
        partition:        how a column is cut into vertical cells:
                          "zslice"  count-based z slices of the whole column (original);
                          "cluster" connected clusters inside the column (voxels of
                                    vox_m, 26-neighbourhood; clusters < tiny points
                                    re-attached to a neighbouring cluster), count-based
                                    z slices inside each cluster, then consecutive slices
                                    of a cluster merged when their union is still one
                                    plane (RMS < plane_res, normals within plane_ang)
                                    or one line (linearity > line_lin, thickness <
                                    line_rad, axes within line_ang).
                          In "cluster" mode cells can sit side by side in a column, so
                          their anchor is the cell centroid (not the column centre) and
                          the descriptor gains the centroid offset (desc_dim 5).
    """

    def __init__(self, patch_size_m: float = 50.0, points_per_leaf: int = 250,
                 density: float = 46.1, levels: int = 3, reduction: int = 4,
                 local_k: int = 7, site_spacing_m: float = 0.1, kz: int = 50,
                 vmax: int = 24, m: Optional[int] = None,
                 ground_quantile: float = 0.02, ground_jump_m: float = 2.0,
                 partition: str = "zslice", vox_m: float = 0.35, tiny: int = 5,
                 plane_res: float = 0.08, plane_ang: float = 20.0,
                 line_lin: float = 0.70, line_rad: float = 0.20, line_ang: float = 20.0,
                 merge: bool = True, verbose: bool = True):
        super().__init__()
        self.P = float(patch_size_m)
        self.half = self.P / 2.0
        self.k = int(points_per_leaf)
        self.density = float(density)
        self.levels, self.reduction, self.local_k = levels, reduction, local_k
        self.s = float(site_spacing_m)
        self.n_side = int(round(self.P / self.s))
        assert abs(self.n_side * self.s - self.P) < 1e-6, "patch size must be a multiple of the site spacing"
        self.kz, self.vmax = int(kz), int(vmax)
        self.m = int(m) if m is not None else int(kz)
        self.ground_quantile = float(ground_quantile)
        self.ground_jump_m = float(ground_jump_m)
        assert partition in ("zslice", "cluster"), partition
        self.partition = partition
        self.vox, self.tiny, self.merge = float(vox_m), int(tiny), bool(merge)
        self.plane_res, self.line_lin, self.line_rad = float(plane_res), float(line_lin), float(line_rad)
        self.cos_plane = math.cos(math.radians(plane_ang))
        self.cos_line = math.cos(math.radians(line_ang))
        self.centroid_anchor = partition == "cluster"
        self.desc_dim = 5 if self.centroid_anchor else 3
        self.verbose = verbose
        self._static = {}          # device -> (HierarchyGeometry, tables); plain dict, DDP-safe

    # ------------------------------------------------------------------ static
    def static(self, device) -> HierarchyGeometry:
        return self._get_static(device)[0]

    def _get_static(self, device):
        key = str(device)
        if key not in self._static:
            n, s, h = self.n_side, self.s, self.half
            c = -h + (np.arange(n) + 0.5) * s
            gx, gy = np.meshgrid(c, c, indexing="ij")
            xy = np.stack([gx.ravel(), gy.ravel()], 1)                # site i*n + j
            sites_per_leaf = self.k / (self.density * s * s)
            g = HierarchyGeometry.build(xy, int(round(sites_per_leaf)), self.levels,
                                        self.reduction, self.local_k)
            # dense site -> cell tables (independent of the build's site order)
            sxy = g.site_xy.double().numpy()
            ij = np.rint((sxy + h) / s - 0.5).astype(np.int64)
            flat = ij[:, 0] * n + ij[:, 1]
            assert len(np.unique(flat)) == n * n, "virtual site grid not recovered"
            tables = []
            for lvl in range(self.levels):
                t = np.full(n * n, -1, dtype=np.int64)
                t[flat] = g.owner[lvl].numpy()
                assert (t >= 0).all()
                tables.append(torch.as_tensor(t))
            g = g.to(device)
            tables = [t.to(device) for t in tables]
            if self.verbose:
                print(f"[PointHierarchy] static lattice: {self.P:g} m patch, k={self.k} pts/leaf "
                      f"at {self.density:g} pts/m2 -> cells per level {g.sizes}; "
                      f"kz={self.kz}, vmax={self.vmax}, m={self.m}, partition={self.partition}")
            self._static[key] = (g, tables)
        return self._static[key]

    def _site_index(self, xy: torch.Tensor) -> torch.Tensor:
        ij = torch.floor((xy + self.half) / self.s).long().clamp_(0, self.n_side - 1)
        return ij[..., 0] * self.n_side + ij[..., 1]

    # ------------------------------------------------------------------ dynamic
    @torch.no_grad()
    def forward(self, points: torch.Tensor, mask: torch.Tensor,
                generator: Optional[torch.Generator] = None, cells: bool = True) -> PointBatchGeometry:
        """
        points: [B, N, C] (PointLayout columns), mask: [B, N] True = padding.
        generator: torch.Generator for the within-cell atom subset; pass a
                   seeded one at evaluation for a deterministic subset.
        """
        g, tables = self._get_static(points.device)
        dev = points.device
        B, N, _ = points.shape
        L0, vmax, kz, m = g.sizes[0], self.vmax, self.kz, self.m
        valid = ~mask.bool()
        xy = points[..., [PointLayout.X, PointLayout.Y]].float()
        z = points[..., PointLayout.Z].float()
        site = self._site_index(xy)                                  # [B, N]

        # ---- columns (padding -> dummy column L0) ----
        col = torch.where(valid, tables[0][site], torch.full_like(site, L0))
        G = B * (L0 + 1)
        bidx = torch.arange(B, device=dev).view(B, 1).expand(B, N)
        ckey = (bidx * (L0 + 1) + col).reshape(-1)                  # [BN]
        zf = torch.where(valid, z, torch.zeros_like(z)).reshape(-1)

        # sort by (column, z); stable so that ties in z (LAS z is quantised,
        # flat ground) are broken by point order -> deterministic, and a
        # sample gives the same cells alone or inside a batch
        order = torch.sort(zf, stable=True).indices
        order = order[torch.sort(ckey[order], stable=True).indices]
        ck = ckey[order]
        zs = zf[order]
        counts = torch.bincount(ckey, minlength=G)                   # [G]
        start = torch.cumsum(counts, 0) - counts
        BN = B * N
        rank = torch.arange(BN, device=dev) - start[ck]
        ncell = ((counts + kz - 1) // kz).clamp(1, vmax)             # [G]
        v_sorted = rank * ncell[ck] // counts[ck].clamp(min=1)

        # ---- column low quantile -> local ground over the 7-ring ----
        qpos = start + (self.ground_quantile * (counts - 1).clamp(min=0).float()).long()
        qpos = qpos.clamp(max=BN - 1)
        col_q = torch.where(counts > 0, zs[qpos], torch.full_like(zs[qpos], float("inf")))
        col_q = col_q.view(B, L0 + 1)[:, :L0]                        # [B, L0]
        ring = g.local_idx[0]                                        # [L0, K]
        ring_min = col_q[:, ring].amin(-1)                           # [B, L0]
        use_ring = ~torch.isfinite(col_q) | (col_q - ring_min > self.ground_jump_m)
        ground = torch.where(use_ring, ring_min, col_q)
        ground = torch.where(torch.isfinite(ground), ground, torch.zeros_like(ground))

        if not cells:
            # light path (hierarchical stage): columns, ground and heights only
            vflat = valid.reshape(-1)
            point_col = torch.where(valid, col, torch.full_like(col, -1))
            gcol = ground.gather(1, col.clamp(max=L0 - 1))
            point_h = torch.where(valid, z - gcol, torch.zeros_like(z))
            col_count = counts.view(B, L0 + 1)[:, :L0]
            none = None
            return PointBatchGeometry(
                B=B, L0=L0, vmax=vmax, m=0,
                point_col=point_col, point_v=none, point_cell=none, point_h=point_h,
                col_count=col_count, col_ncells=none, col_empty=col_count == 0, col_ground=ground,
                vmask=none, v_zlo=none, v_zhi=none, v_zmean=none, v_count=none, v_h=none,
                cell_batch=none, cell_col=none, cell_v=none, cell_slot=torch.zeros(0, dtype=torch.long, device=dev),
                cell_atoms=none, cell_atoms_mask=none, level_count=[], level_empty=[])

        # ---- partition: slot of every point within its column ----
        pstats = None
        if self.partition == "zslice":
            v_pt = torch.empty(BN, dtype=torch.long, device=dev)
            v_pt[order] = v_sorted
        else:
            xyz_f = torch.stack([xy[..., 0], xy[..., 1], z], -1).reshape(BN, 3)
            v_pt, pstats = self._cluster_partition(xyz_f, ckey, valid.reshape(-1), G, L0, vmax)

        # ---- vertical cells = segments of the (column, slot, z) order ----
        # (for "zslice" this is exactly the (column, z) order used above)
        vk_pt = ckey * vmax + v_pt
        order = torch.sort(zf, stable=True).indices
        order = order[torch.sort(vk_pt[order], stable=True).indices]
        vkey = vk_pt[order]
        zs = zf[order]
        xs = xy[..., 0].reshape(-1)[order]
        ys = xy[..., 1].reshape(-1)[order]
        new_seg = torch.ones(BN, dtype=torch.bool, device=dev)
        new_seg[1:] = vkey[1:] != vkey[:-1]
        seg = torch.cumsum(new_seg.long(), 0) - 1                    # [BN] segment id
        n_seg = int(seg[-1].item()) + 1 if BN > 0 else 0
        seg_key = vkey[new_seg]                                      # [n_seg]
        seg_cnt = torch.bincount(seg, minlength=n_seg)
        seg_first = torch.nonzero(new_seg).squeeze(1)
        seg_zlo = zs[seg_first]
        seg_zhi = zs[seg_first + seg_cnt - 1]
        seg_zsum = torch.zeros(n_seg, device=dev).index_add_(0, seg, zs)
        seg_xsum = torch.zeros(n_seg, device=dev).index_add_(0, seg, xs)
        seg_ysum = torch.zeros(n_seg, device=dev).index_add_(0, seg, ys)
        seg_col_g = seg_key // vmax                                  # global column key
        seg_b = seg_col_g // (L0 + 1)
        seg_c = seg_col_g % (L0 + 1)
        seg_v = seg_key % vmax
        keep = seg_c < L0                                            # drop dummy column
        cid = torch.full((n_seg,), -1, dtype=torch.long, device=dev)
        V = int(keep.sum().item())
        cid[keep] = torch.arange(V, device=dev)

        cell_b, cell_c, cell_v = seg_b[keep], seg_c[keep], seg_v[keep]
        cell_slot = (cell_b * L0 + cell_c) * vmax + cell_v
        c_cnt = seg_cnt[keep]
        c_zlo, c_zhi = seg_zlo[keep], seg_zhi[keep]
        c_zmean = seg_zsum[keep] / c_cnt.clamp(min=1).float()
        c_xmean = seg_xsum[keep] / c_cnt.clamp(min=1).float()
        c_ymean = seg_ysum[keep] / c_cnt.clamp(min=1).float()

        # ---- atoms per cell: random order within each cell, keep first m ----
        # (a generator must live on the same device as the points)
        rnd = torch.rand(BN, device=dev, generator=generator)
        o2 = torch.argsort(rnd)
        o2 = o2[torch.sort(seg[o2], stable=True).indices]            # positions in sorted array
        seg2 = seg[o2]
        pos2 = torch.arange(BN, device=dev) - seg_first[seg2]
        take = (pos2 < m) & (cid[seg2] >= 0)
        cell_atoms = torch.zeros(V, m, dtype=torch.long, device=dev)
        cell_atoms_mask = torch.ones(V, m, dtype=torch.bool, device=dev)
        cell_atoms[cid[seg2[take]], pos2[take]] = order[o2[take]]
        cell_atoms_mask[cid[seg2[take]], pos2[take]] = False

        # ---- per point ----
        point_v = torch.full((BN,), -1, dtype=torch.long, device=dev)
        point_cell = torch.full((BN,), -1, dtype=torch.long, device=dev)
        point_v = torch.where(valid.reshape(-1), v_pt, torch.full_like(v_pt, -1))
        point_cell[order] = cid[seg]
        point_v, point_cell = point_v.view(B, N), point_cell.view(B, N)
        point_col = torch.where(valid, col, torch.full_like(col, -1))
        gcol = ground.gather(1, col.clamp(max=L0 - 1))
        point_h = torch.where(valid, z - gcol, torch.zeros_like(z))

        # ---- dense tensors ----
        S = B * L0 * vmax
        def dense(vals, fill):
            out = torch.full((S,), fill, dtype=vals.dtype, device=dev)
            out[cell_slot] = vals
            return out.view(B, L0, vmax)
        vmask = dense(torch.zeros(V, dtype=torch.bool, device=dev), True)
        v_zlo, v_zhi, v_zmean = dense(c_zlo, 0.0), dense(c_zhi, 0.0), dense(c_zmean, 0.0)
        v_count = dense(c_cnt, 0)
        v_h = torch.where(vmask, torch.zeros_like(v_zmean), v_zmean - ground.unsqueeze(-1))
        cen = g.centers[0].float()
        v_xmean = torch.where(vmask, cen[:, 0].view(1, L0, 1).expand(B, L0, vmax), dense(c_xmean, 0.0))
        v_ymean = torch.where(vmask, cen[:, 1].view(1, L0, 1).expand(B, L0, vmax), dense(c_ymean, 0.0))

        col_count = counts.view(B, L0 + 1)[:, :L0]
        col_ncells = torch.bincount(cell_b * L0 + cell_c, minlength=B * L0).view(B, L0)

        # ---- occupancy of every 2D level ----
        level_count, level_empty = [], []
        for lvl in range(self.levels):
            Ll = g.sizes[lvl]
            cl = tables[lvl][site]
            key = torch.where(valid, bidx * Ll + cl, torch.full_like(cl, B * Ll))
            cnt = torch.bincount(key.reshape(-1), minlength=B * Ll + 1)[:B * Ll].view(B, Ll)
            level_count.append(cnt)
            level_empty.append(cnt == 0)

        return PointBatchGeometry(
            B=B, L0=L0, vmax=vmax, m=m,
            point_col=point_col, point_v=point_v, point_cell=point_cell, point_h=point_h,
            col_count=col_count, col_ncells=col_ncells, col_empty=col_count == 0,
            col_ground=ground,
            vmask=vmask, v_zlo=v_zlo, v_zhi=v_zhi, v_zmean=v_zmean, v_count=v_count, v_h=v_h,
            cell_batch=cell_b, cell_col=cell_c, cell_v=cell_v, cell_slot=cell_slot,
            cell_atoms=cell_atoms, cell_atoms_mask=cell_atoms_mask,
            level_count=level_count, level_empty=level_empty,
            v_xmean=v_xmean, v_ymean=v_ymean, stats=pstats,
        )

    # ------------------------------------------------------------------ cluster partition
    @staticmethod
    def _shape(cnt, s1, s2):
        """Moments (float64) -> sigmas s1 >= s2 >= s3, main axis, normal."""
        m = s1 / cnt.clamp(min=1).unsqueeze(-1)
        cov = s2 / cnt.clamp(min=1).view(-1, 1, 1) - m.unsqueeze(-1) * m.unsqueeze(-2)
        w, V = eigh_safe(cov)                                        # ascending
        s = w.clamp(min=0).sqrt().flip(-1)
        return s, V[..., 2], V[..., 0]

    def _kinds(self, cnt, s):
        lin = (s[:, 0] - s[:, 1]) / s[:, 0].clamp(min=1e-9)
        ok = (cnt >= 5) & (s[:, 0] > 0)
        line = ok & (lin > self.line_lin) & (s[:, 1] < self.line_rad)
        plane = ok & ~line & (s[:, 2] < self.plane_res)
        return torch.where(line, 1, torch.where(plane, 2, 0))        # 0 scatter, 1 line, 2 plane

    def _cluster_partition(self, xyz, ckey, valid, G, L0, vmax):
        """Per-point slot in its column (see the class docstring, partition="cluster").
        xyz [BN, 3] metres, ckey [BN] global column key (padding -> dummy column)."""
        dev = xyz.device
        BN = xyz.shape[0]
        # -- 1. voxels and 26-connectivity inside each column (pad 2 so +-2 offsets never wrap).
        #       The grid is anchored per sample (x, y at the patch corner, z at the sample's
        #       lowest point), so a sample gives the same clusters alone or in a batch.
        b_of = ckey // (L0 + 1)
        nb_ = int(b_of.max()) + 1 if BN else 1
        zmin = torch.full((nb_,), float("inf"), device=dev).scatter_reduce(
            0, b_of[valid], xyz[valid, 2], reduce="amin")
        zmin = torch.where(torch.isfinite(zmin), zmin, torch.zeros_like(zmin))
        rel = torch.stack([xyz[:, 0] + self.half, xyz[:, 1] + self.half, xyz[:, 2] - zmin[b_of]], -1)
        ijk = torch.floor(rel / self.vox).long().clamp(min=0)
        ijk = torch.where(valid.unsqueeze(-1), ijk, torch.zeros_like(ijk))
        hi = ijk.amax(0)
        ijk = ijk + 2
        R = (hi + 5).tolist()
        assert float(G) * R[0] * R[1] * R[2] < 2 ** 62, "voxel key overflow"
        key = ((ckey * R[0] + ijk[:, 0]) * R[1] + ijk[:, 1]) * R[2] + ijk[:, 2]
        uk, vox = torch.unique(key, sorted=True, return_inverse=True)
        Vn = uk.shape[0]

        def lookup(nb):
            pos = torch.searchsorted(uk, nb).clamp(max=Vn - 1)
            return pos, uk[pos] == nb

        I, J = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) <= (0, 0, 0):
                        continue                                     # 13 half-offsets
                    pos, ok = lookup(uk + (dx * R[1] + dy) * R[2] + dz)
                    I.append(torch.nonzero(ok).squeeze(1)); J.append(pos[ok])
        I, J = torch.cat(I), torch.cat(J)
        parent = torch.arange(Vn, device=dev)
        for _ in range(64):                                          # hooking + pointer jumping
            ri, rj = parent[I], parent[J]
            d = ri != rj
            if not bool(d.any()):
                break
            parent.scatter_reduce_(0, torch.maximum(ri[d], rj[d]), torch.minimum(ri[d], rj[d]), reduce="amin")
            for _ in range(32):
                p2 = parent[parent]
                if torch.equal(p2, parent):
                    break
                parent = p2
        # -- 2. re-attach tiny clusters to the nearest voxel of a larger cluster (radius 2 voxels)
        vcnt = torch.bincount(vox[valid], minlength=Vn)
        csize = torch.zeros(Vn, dtype=torch.long, device=dev).index_add_(0, parent, vcnt)
        comp_v = parent.clone()
        tiny_v = (csize[parent] < self.tiny) & (vcnt > 0)
        n_tiny = int(tiny_v.sum())
        if n_tiny:
            offs = [(dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3)
                    if (dx, dy, dz) != (0, 0, 0)]
            offs.sort(key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
            todo = torch.nonzero(tiny_v).squeeze(1)
            for dx, dy, dz in offs:
                if todo.numel() == 0:
                    break
                pos, ok = lookup(uk[todo] + (dx * R[1] + dy) * R[2] + dz)
                ok = ok & (csize[parent[pos]] >= self.tiny)
                comp_v[todo[ok]] = parent[pos[ok]]
                todo = todo[~ok]
        comp = comp_v[vox]                                           # [BN]
        # -- 3. count-based z slices inside each cluster
        o = torch.sort(xyz[:, 2], stable=True).indices
        o = o[torch.sort(comp[o], stable=True).indices]
        cc = comp[o]
        cnt = torch.bincount(comp, minlength=Vn)
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.arange(BN, device=dev) - start[cc]
        nsl = (cnt + self.kz - 1) // self.kz
        vin = torch.empty(BN, dtype=torch.long, device=dev)
        vin[o] = rank * nsl[cc] // cnt[cc].clamp(min=1)
        maxv = int(nsl.max()) + 1
        uc, cell = torch.unique(comp * maxv + vin, sorted=True, return_inverse=True)   # (cluster, slice)
        NC = uc.shape[0]
        # -- 4. merge consecutive slices of a cluster while the WHOLE union stays one plane
        #       or one line. Rounds of disjoint pairwise merges between neighbouring groups,
        #       each test on the union's full moments (no drift along a chain: a stack of
        #       thin crown slices is never accepted as one plane, as each pair would be).
        X = xyz.double()
        g_cnt = torch.bincount(cell, minlength=NC).double()
        S1 = torch.zeros(NC, 3, dtype=torch.float64, device=dev).index_add_(0, cell, X)
        S2 = torch.zeros(NC, 3, 3, dtype=torch.float64, device=dev).index_add_(
            0, cell, X.unsqueeze(-1) * X.unsqueeze(-2))
        s, ax, nr = self._shape(g_cnt, S1, S2)
        kind = self._kinds(g_cnt, s)
        g_comp = uc // maxv
        grp = torch.arange(NC, device=dev)                           # cell -> group
        merged_away, rounds = 0, 0
        if self.merge and NC > 1:
            for rnd in range(64):
                Ng = g_cnt.shape[0]
                if Ng < 2:
                    break
                a, b = torch.arange(Ng - 1, device=dev), torch.arange(1, Ng, device=dev)
                same = (g_comp[a] == g_comp[b]) & (kind[a] == kind[b]) & (kind[a] > 0)
                if not bool(same.any()):
                    break
                su, ua, un = self._shape(g_cnt[a] + g_cnt[b], S1[a] + S1[b], S2[a] + S2[b])
                ku = self._kinds(g_cnt[a] + g_cnt[b], su)
                isl = (kind[a] == 1).unsqueeze(-1)
                dir_a = torch.where(isl, ax[a], nr[a])
                dir_b = torch.where(isl, ax[b], nr[b])
                dir_u = torch.where(isl, ua, un)
                cmin = torch.where(kind[a] == 1, torch.full_like(g_cnt[a], self.cos_line),
                                   torch.full_like(g_cnt[a], self.cos_plane))
                cand = same & (ku == kind[a]) & ((dir_u * dir_a).sum(-1).abs() > cmin) & \
                    ((dir_u * dir_b).sum(-1).abs() > cmin)
                if not bool(cand.any()):
                    break
                # disjoint pairs, alternating parity of the group's rank INSIDE its cluster
                # (never a batch-global index, so a sample merges the same way alone or
                # in a batch)
                first = torch.ones(Ng, dtype=torch.bool, device=dev)
                first[1:] = g_comp[1:] != g_comp[:-1]
                fidx = torch.cummax(torch.where(first, torch.arange(Ng, device=dev),
                                                torch.zeros(Ng, dtype=torch.long, device=dev)), 0).values
                pos = torch.arange(Ng, device=dev) - fidx
                sel = cand & ((pos[a] % 2) == (rnd % 2))
                if not bool(sel.any()):
                    continue
                merge_prev = torch.zeros(Ng, dtype=torch.bool, device=dev)
                merge_prev[1:] = sel
                gid = torch.cumsum((~merge_prev).long(), 0) - 1
                Nn = int(gid[-1]) + 1
                g_cnt = torch.zeros(Nn, dtype=torch.float64, device=dev).index_add_(0, gid, g_cnt)
                S1 = torch.zeros(Nn, 3, dtype=torch.float64, device=dev).index_add_(0, gid, S1)
                S2 = torch.zeros(Nn, 3, 3, dtype=torch.float64, device=dev).index_add_(0, gid, S2)
                keep_first = ~merge_prev
                kind, g_comp = kind[keep_first], g_comp[keep_first]
                s, ax, nr = self._shape(g_cnt, S1, S2)
                grp = gid[grp]
                merged_away += int(sel.sum()); rounds += 1
        NG = g_cnt.shape[0]
        gpt = grp[cell]                                              # [BN]
        # -- 5. slots: groups of a column ordered by mean z, 0..n-1 (overflow -> last slot)
        g_col = torch.zeros(NG, dtype=torch.long, device=dev).scatter_(0, gpt, ckey)
        g_z = S1[:, 2] / g_cnt.clamp(min=1)
        og = torch.sort(g_z, stable=True).indices
        og = og[torch.sort(g_col[og], stable=True).indices]
        gc = g_col[og]
        per_col = torch.bincount(gc, minlength=G)
        gstart = torch.cumsum(per_col, 0) - per_col
        slot = torch.empty(NG, dtype=torch.long, device=dev)
        slot[og] = torch.arange(NG, device=dev) - gstart[gc]
        real = (g_col % (L0 + 1)) < L0
        overflow = int(((slot >= vmax) & real).sum())
        slot = slot.clamp(max=vmax - 1)
        g_kind = kind
        stats = {"clusters": int((csize > 0).sum()), "tiny_voxels": n_tiny,
                 "slices": NC, "latents": int(real.sum()), "merge_rounds": rounds,
                 "merged_away": merged_away, "overflow_to_last_slot": overflow,
                 "kinds": {k: int(((g_kind == i) & real).sum())
                           for i, k in ((0, "scatter"), (1, "line"), (2, "plane"))}}
        return slot[gpt], stats

    # ------------------------------------------------------------------ atoms
    @torch.no_grad()
    def atom_offsets(self, pg: PointBatchGeometry, points: torch.Tensor,
                     min_z_extent: float = 0.5) -> torch.Tensor:
        """
        Offsets of each cell's atoms to the cell anchor (column centre, z_mean),
        unit-free: dx, dy / column scale, dz / max(z_hi - z_lo, min_z_extent).
        Returns [V, m, 3] (zeros on padding atoms).
        """
        g = self.static(points.device)
        B, N, C = points.shape
        flat = points.reshape(B * N, C)
        a = flat[pg.cell_atoms]                                      # [V, m, C]
        slot = pg.cell_slot
        if self.centroid_anchor:                                     # cells can sit side by side
            cxy = torch.stack([pg.v_xmean.reshape(-1)[slot], pg.v_ymean.reshape(-1)[slot]], -1)
        else:
            cxy = g.centers[0][pg.cell_col]                          # [V, 2]
        csc = g.scales[0][pg.cell_col]                               # [V]
        zmean = pg.v_zmean.reshape(-1)[slot]
        zext = (pg.v_zhi.reshape(-1)[slot] - pg.v_zlo.reshape(-1)[slot]).clamp(min=min_z_extent)
        dxy = (a[..., [PointLayout.X, PointLayout.Y]] - cxy[:, None]) / csc[:, None, None]
        dz = (a[..., PointLayout.Z] - zmean[:, None]) / zext[:, None]
        out = torch.cat([dxy, dz.unsqueeze(-1)], -1)
        return out.masked_fill(pg.cell_atoms_mask.unsqueeze(-1), 0.0)

    @torch.no_grad()
    def cell_descriptors(self, pg: PointBatchGeometry, h_scale: float = 10.0) -> torch.Tensor:
        """Per dense slot [B, L0, vmax, desc_dim]: z extent (log), height above local
        ground (compressed), log count; in cluster mode also the centroid offset
        to the column centre (dx, dy / column scale). Zeros on padding."""
        ext = torch.log1p((pg.v_zhi - pg.v_zlo).clamp(min=0))
        h = pg.v_h / (h_scale + pg.v_h.abs())
        cnt = torch.log1p(pg.v_count.float())
        feats = [ext, h, cnt]
        if self.desc_dim == 5:
            g = self.static(pg.v_zmean.device)
            cen, sc = g.centers[0].float(), g.scales[0].float().view(1, -1, 1)
            feats += [(pg.v_xmean - cen[:, 0].view(1, -1, 1)) / sc, (pg.v_ymean - cen[:, 1].view(1, -1, 1)) / sc]
        out = torch.stack(feats, -1)
        return out.masked_fill(pg.vmask.unsqueeze(-1), 0.0)

    # ------------------------------------------------------------------ queries
    @torch.no_grad()
    def lookup_queries(self, pg: PointBatchGeometry, query_points: torch.Tensor,
                       query_mask: torch.Tensor, query_ctx_idx: Optional[torch.Tensor] = None,
                       K: int = 9, z_weight: float = 1.0, chunk: int = 16384) -> dict:
        """
        For every query: its column, its vertical cell, and K vertical cells:
        its own cell first, then the nearest ones in 3D among its own column
        and the 6 neighbouring columns (distance sqrt(dxy^2 + (z_weight dz)^2) to the cell anchors).

        query_ctx_idx: [B, Nq] row of the same point in the context (-1 if
            absent); when given, the query reuses that point's vertical cell,
            otherwise it is found from the column's z_lo boundaries.

        Returns dict:
            col   [B, Nq]     column (-1 padding)
            v     [B, Nq]     vertical cell in the column (-1 if column empty / padding)
            nbr   [B, Nq, K]  dense slot index into [L0 * vmax] (per sample)
            nbr_mask [B, Nq, K]  True = invalid
            nbr_off  [B, Nq, K, 3]  (dx, dy) / column scale, dz / z extent of the cell
        """
        g, tables = self._get_static(query_points.device)
        dev = query_points.device
        B, Nq, _ = query_points.shape
        L0, vmax = pg.L0, pg.vmax
        qvalid = ~query_mask.bool()
        qxy = query_points[..., [PointLayout.X, PointLayout.Y]].float()
        qz = query_points[..., PointLayout.Z].float()
        qcol = tables[0][self._site_index(qxy)]                      # [B, Nq]

        vm = pg.vmask.gather(1, qcol.unsqueeze(-1).expand(-1, -1, vmax))
        nc = pg.col_ncells.gather(1, qcol)
        if not self.centroid_anchor:
            # vertical cell from z boundaries (count of cells with z_lo <= z) ...
            zlo = pg.v_zlo.gather(1, qcol.unsqueeze(-1).expand(-1, -1, vmax))
            below = ((zlo <= qz.unsqueeze(-1)) & ~vm).sum(-1)
            qv = torch.where(nc > 0, (below - 1).clamp(min=0), torch.full_like(below, -1))
            qv = torch.minimum(qv, (nc - 1).clamp(min=-1))
        else:
            # cells can sit side by side: nearest cell centroid of the column in 3D
            ex = lambda t: t.gather(1, qcol.unsqueeze(-1).expand(-1, -1, vmax))
            d2 = (ex(pg.v_xmean) - qxy[..., :1]) ** 2 + (ex(pg.v_ymean) - qxy[..., 1:]) ** 2 + \
                 (z_weight * (ex(pg.v_zmean) - qz.unsqueeze(-1))) ** 2
            d2 = d2.masked_fill(vm, float("inf"))
            qv = torch.where(nc > 0, d2.argmin(-1), torch.full_like(nc, -1))
        # ... or from the context point itself
        if query_ctx_idx is not None:
            has = query_ctx_idx >= 0
            own = pg.point_v.gather(1, query_ctx_idx.clamp(min=0))
            qv = torch.where(has & (own >= 0), own, qv)
        qv = torch.where(qvalid, qv, torch.full_like(qv, -1))
        qcol_out = torch.where(qvalid, qcol, torch.full_like(qcol, -1))

        # ---- K cells among the 7 columns: own cell first, then nearest in 3D ----
        # Per-slot anchors are built once ([B, S, 4]: x, y, z_mean, z_extent),
        # and the 7 * vmax candidate slots of every column come from a static
        # table, so each query only gathers rows and runs one topk.
        S = L0 * vmax
        ring = g.local_idx[0]                                        # [L0, Kc]
        Kc = ring.shape[1]
        cand = (ring.unsqueeze(-1) * vmax + torch.arange(vmax, device=dev)).reshape(L0, Kc * vmax)
        if self.centroid_anchor:
            slot_xy = torch.stack([pg.v_xmean, pg.v_ymean], -1).reshape(B, S, 2)
        else:
            slot_xy = g.centers[0].repeat_interleave(vmax, 0).unsqueeze(0).expand(B, S, 2)
        slot_sc = g.scales[0].repeat_interleave(vmax, 0)             # [S]
        anchors = torch.cat([
            slot_xy,
            pg.v_zmean.reshape(B, S, 1),
            (pg.v_zhi - pg.v_zlo).clamp(min=0.5).reshape(B, S, 1),
            slot_sc.unsqueeze(0).expand(B, S).unsqueeze(-1),
        ], -1).reshape(B * S, 5)
        vm_flat = pg.vmask.reshape(B * S)
        k = min(K, Kc * vmax)
        nbr = torch.zeros(B, Nq, K, dtype=torch.long, device=dev)
        nbr_mask = torch.ones(B, Nq, K, dtype=torch.bool, device=dev)
        nbr_off = torch.zeros(B, Nq, K, 3, device=dev)
        boff = (torch.arange(B, device=dev) * S).view(B, 1, 1)
        for s0 in range(0, Nq, chunk):
            s1 = min(s0 + chunk, Nq)
            slots = cand[qcol[:, s0:s1]]                             # [B, n, Kc*vmax]
            fl = slots + boff
            a = anchors[fl]                                          # [B, n, Kc*vmax, 5]
            dxy = a[..., :2] - qxy[:, s0:s1].unsqueeze(2)
            dz = a[..., 2] - qz[:, s0:s1].unsqueeze(2)
            d2 = (dxy * dxy).sum(-1) + (z_weight * dz) ** 2
            d2 = d2.masked_fill(vm_flat[fl], float("inf"))
            own = (qcol[:, s0:s1] * vmax + qv[:, s0:s1]).unsqueeze(-1)
            d2 = d2.masked_fill((slots == own) & (qv[:, s0:s1] >= 0).unsqueeze(-1), -1.0)
            dist, sel = d2.topk(k, dim=-1, largest=False)
            nbr[:, s0:s1, :k] = slots.gather(2, sel)
            nbr_mask[:, s0:s1, :k] = ~torch.isfinite(dist)
            a_sel = a.gather(2, sel.unsqueeze(-1).expand(-1, -1, -1, 5))
            dxy_sel = a_sel[..., :2] - qxy[:, s0:s1].unsqueeze(2)
            dz_sel = a_sel[..., 2] - qz[:, s0:s1].unsqueeze(2)
            nbr_off[:, s0:s1, :k] = torch.cat([dxy_sel / a_sel[..., 4:5],
                                               (dz_sel / a_sel[..., 3]).unsqueeze(-1)], -1)
        nbr_mask |= ~qvalid.unsqueeze(-1)
        nbr_off = nbr_off.masked_fill(nbr_mask.unsqueeze(-1), 0.0)
        return {"col": qcol_out, "v": qv, "nbr": nbr, "nbr_mask": nbr_mask, "nbr_off": nbr_off}
