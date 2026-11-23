"""
GeographicPruningUNet
=====================

Drop-in replacement for the shared-batch (raster) path of GeographicPruning,
extended to build and cache the full U-Net hierarchy.

Same constructor and forward signature as GeographicPruning, same returned
triple (tokens_per_latent [B,L,k,D], masks_per_latent [B,L,k], bias [B,L,k]).

What it does differently
------------------------
* Geometry is built from the REAL token positions (geometry.get_token_centers)
  -> no assumption about atom order (pixel-major / band-major), and every
  anchor is expressed in your own coordinate frame and units.
* Atoms are grouped into SITES (unique positions: the C bands of one pixel).
  Voronoi assignment runs on sites, then expands to atoms (C x cheaper).
* One HierarchyGeometry per configuration, built once, holding for every level:
  site owners, occupied anchors (empty anchors pruned), kNN tables, padded
  merge / upsampling overlap tables, relative-geometry inputs, leaf atom cells.
* Valid atoms are ranked first when sampling, so masked atoms never consume the
  m budget while valid ones remain. Fully empty cells keep one zero-token slot
  (finite softmax) and are flagged in `self.last_empty`.

IMPORTANT: L comes from the geometry, not from ceil(N / k)
-----------------------------------------------------------
Empty border anchors are pruned, so L0 differs slightly from the model's
L_spatial (e.g. 686 vs 656 on PASTIS at k=250). The model must take its
latent count and anchor coordinates from `self.last_geometry`:

    tok, msk, bias = pruning(tokens, mask, latent_coords, geo_k, sigma, L_spatial)
    geom = pruning.last_geometry
    latent_coords = geom.centers[0]            # [L0, 2], your units/frame
    z0 = leaf_cross_attention(tok, msk, bias)  # [B, L0, D]
    d0 = unet(z0, geom)

`latent_coords` and `L_spatial` passed to forward() are only used to infer
atoms_per_latent (k ~= N / L_spatial) when it is not given explicitly.

Assumption (identical to the original shared-batch path): token index i is the
same physical position for every sample with the same N. Not valid for point
clouds; the per-sample DALES paths are not implemented here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


# =============================================================================
# Geometry helpers
# =============================================================================
def hex_anchors(lo: np.ndarray, hi: np.ndarray, area: float) -> np.ndarray:
    """Hexagonal lattice over the box [lo, hi] with Voronoi cell area ~= area."""
    a = math.sqrt(2.0 * area / math.sqrt(3.0))
    dy = a * math.sqrt(3.0) / 2.0
    pts = []
    for r, y in enumerate(np.arange(lo[0] + dy / 2.0, hi[0] + dy, dy)):
        for x in np.arange(lo[1] + (a / 2.0) * (r % 2), hi[1] + a, a):
            pts.append((y, x))
    return np.asarray(pts, dtype=np.float64)


def pad_groups(query_ids, key_ids, n_queries):
    """(query, key) pairs -> padded [n_queries, max_deg] index table + valid mask."""
    order = np.lexsort((key_ids, query_ids))
    q, k = query_ids[order], key_ids[order]
    deg = np.bincount(q, minlength=n_queries)
    assert deg.min() > 0, "every query must have at least one key"
    col = np.arange(len(q)) - (np.cumsum(deg) - deg)[q]
    idx = np.zeros((n_queries, deg.max()), dtype=np.int64)
    valid = np.zeros((n_queries, deg.max()), dtype=bool)
    idx[q, col] = k
    valid[q, col] = True
    return idx, valid


def rel_geo(cq, sq, ck, sk, idx):
    """[(c_key - c_query) / s_query, log(s_key / s_query)] -> [Lq, K, 3] (unit-free)."""
    d = (ck[idx] - cq[:, None, :]) / sq[:, None, None]
    r = np.log(sk[idx] / sq[:, None])
    return np.concatenate([d, r[..., None]], -1).astype(np.float32)


# =============================================================================
# Hierarchy geometry
# =============================================================================
@dataclass
class HierarchyGeometry:
    num_levels: int
    atoms_per_latent: int
    atoms_per_site: float
    site_spacing: float                               # your units (e.g. metres)
    origin: torch.Tensor = None                       # [2] bbox min
    extent: float = 1.0                               # max bbox side
    site_xy: torch.Tensor = None                      # [n_sites, 2]
    atom_to_site: torch.Tensor = None                 # [N]
    owner: list = field(default_factory=list)         # per level: [n_sites] cell id
    centers: list = field(default_factory=list)       # per level: [L, 2] your units
    scales: list = field(default_factory=list)        # per level: [L] sqrt(area)
    local_idx: list = field(default_factory=list)     # per level: [L, K], self first
    local_geo: list = field(default_factory=list)     # per level: [L, K, 3]
    merge_idx: list = field(default_factory=list)     # transition l: [L_{l+1}, Kc] fine ids
    merge_valid: list = field(default_factory=list)
    merge_geo: list = field(default_factory=list)
    up_idx: list = field(default_factory=list)        # transition l: [L_l, Kp] coarse ids
    up_valid: list = field(default_factory=list)
    up_geo: list = field(default_factory=list)
    overlap_pairs: list = field(default_factory=list)  # transition l: [3, E] fine, coarse, shared sites
    leaf_cells: torch.Tensor = None                   # [L0, max_cell] atom indices
    leaf_valid: torch.Tensor = None                   # [L0, max_cell]
    leaf_counts: torch.Tensor = None                  # [L0]

    @property
    def sizes(self):
        return [len(c) for c in self.centers]

    def normalized_centers(self, level: int) -> torch.Tensor:
        """Anchors mapped to ~[0, 1] (bbox min -> 0, longest side -> 1)."""
        return (self.centers[level] - self.origin) / self.extent

    @staticmethod
    def build(token_xy: np.ndarray, atoms_per_latent: int, levels: int = 3,
              reduction: int = 4, local_k: int = 7, seed: int = 0) -> "HierarchyGeometry":
        token_xy = np.asarray(token_xy, dtype=np.float64)
        N = len(token_xy)

        # ---- sites: unique physical positions -----------------------------
        span = float(np.ptp(token_xy, axis=0).max()) or 1.0
        key = np.round(token_xy / (span * 1e-7)).astype(np.int64)
        _, first, atom_to_site = np.unique(key, axis=0, return_index=True, return_inverse=True)
        atom_to_site = atom_to_site.reshape(-1)
        site_xy = token_xy[first]
        n_sites = len(site_xy)
        dists, _ = cKDTree(site_xy).query(site_xy, k=2)
        spacing = float(np.median(dists[:, 1]))
        site_area = spacing ** 2
        atoms_per_site = N / n_sites
        lo = site_xy.min(0) - spacing / 2.0
        hi = site_xy.max(0) + spacing / 2.0

        g = HierarchyGeometry(levels, atoms_per_latent, atoms_per_site, spacing)
        leaf_area = (atoms_per_latent / atoms_per_site) * site_area

        # ---- per-level partitions over the same sites ----------------------
        owners, centers, scales = [], [], []
        for lvl in range(levels):
            anchors = hex_anchors(lo, hi, leaf_area * reduction ** lvl)
            _, own = cKDTree(anchors).query(site_xy)
            occ, own = np.unique(own, return_inverse=True)       # prune empty anchors
            own = own.reshape(-1)
            counts = np.bincount(own)
            owners.append(own.astype(np.int64))
            centers.append(anchors[occ])
            scales.append(np.sqrt(counts * site_area))

        # ---- same-level kNN -------------------------------------------------
        local_idx, local_geo = [], []
        for lvl in range(levels):
            c, s = centers[lvl], scales[lvl]
            k = min(local_k, len(c))
            _, nn_idx = cKDTree(c).query(c, k=k)
            nn_idx = np.asarray(nn_idx).reshape(len(c), k)
            # put self first (ties at distance 0 are impossible: anchors are distinct)
            assert (nn_idx[:, 0] == np.arange(len(c))).all()
            local_idx.append(nn_idx)
            local_geo.append(rel_geo(c, s, c, s, nn_idx))

        # ---- overlap graph between adjacent levels -------------------------
        mi_l, mv_l, mg_l, ui_l, uv_l, ug_l, ov_l = [], [], [], [], [], [], []
        for lvl in range(levels - 1):
            f, co = owners[lvl], owners[lvl + 1]
            Lf, Lc = len(centers[lvl]), len(centers[lvl + 1])
            pair, cnt = np.unique(f * Lc + co, return_counts=True)
            fine, coarse = pair // Lc, pair % Lc
            assert np.array_equal(np.bincount(fine, cnt, Lf), np.bincount(f, minlength=Lf))
            assert np.array_equal(np.bincount(coarse, cnt, Lc), np.bincount(co, minlength=Lc))
            ov_l.append(np.stack([fine, coarse, cnt]))
            mi, mv = pad_groups(coarse, fine, Lc)
            ui, uv = pad_groups(fine, coarse, Lf)
            mi_l.append(mi); mv_l.append(mv); ui_l.append(ui); uv_l.append(uv)
            mg_l.append(rel_geo(centers[lvl + 1], scales[lvl + 1], centers[lvl], scales[lvl], mi))
            ug_l.append(rel_geo(centers[lvl], scales[lvl], centers[lvl + 1], scales[lvl + 1], ui))

        # ---- leaf atom cells (vectorised, one fixed shuffle for eval) -------
        atom_cell = owners[0][atom_to_site]
        L0 = len(centers[0])
        shuffle = np.random.default_rng(seed).permutation(N)
        order = shuffle[np.argsort(atom_cell[shuffle], kind="stable")]
        cell_of = atom_cell[order]
        counts = np.bincount(atom_cell, minlength=L0)
        col = np.arange(N) - (np.cumsum(counts) - counts)[cell_of]
        cells = np.zeros((L0, counts.max()), dtype=np.int64)
        cvalid = np.zeros((L0, counts.max()), dtype=bool)
        cells[cell_of, col] = order
        cvalid[cell_of, col] = True

        T = lambda xs, dt: [torch.as_tensor(x, dtype=dt) for x in xs]
        g.origin = torch.as_tensor(lo, dtype=torch.float32)
        g.extent = float((hi - lo).max())
        g.site_xy = torch.as_tensor(site_xy, dtype=torch.float32)
        g.atom_to_site = torch.as_tensor(atom_to_site, dtype=torch.long)
        g.owner, g.centers, g.scales = T(owners, torch.long), T(centers, torch.float32), T(scales, torch.float32)
        g.local_idx, g.local_geo = T(local_idx, torch.long), T(local_geo, torch.float32)
        g.merge_idx, g.merge_valid, g.merge_geo = T(mi_l, torch.long), T(mv_l, torch.bool), T(mg_l, torch.float32)
        g.up_idx, g.up_valid, g.up_geo = T(ui_l, torch.long), T(uv_l, torch.bool), T(ug_l, torch.float32)
        g.overlap_pairs = T(ov_l, torch.long)
        g.leaf_cells = torch.as_tensor(cells)
        g.leaf_valid = torch.as_tensor(cvalid)
        g.leaf_counts = torch.as_tensor(counts)
        return g

    def to(self, device):
        for name, val in vars(self).items():
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                setattr(self, name, [v.to(device) for v in val])
        return self

    def summary(self) -> str:
        out = [f"k={self.atoms_per_latent} | atoms/site={self.atoms_per_site:g} | "
               f"site spacing={self.site_spacing:g} | latents per level: {self.sizes}",
               f"leaf atoms/cell: min {int(self.leaf_counts.min())}, max {int(self.leaf_counts.max())}"]
        for l in range(self.num_levels - 1):
            out.append(f"L{l}<->L{l+1}: {self.overlap_pairs[l].shape[1]} overlap pairs | "
                       f"max children {self.merge_idx[l].shape[1]} | max parents {self.up_idx[l].shape[1]}")
        return "\n".join(out)


# =============================================================================
# The pruning module
# =============================================================================
class GeographicPruningUNet(nn.Module):
    """
    Args:
        geometry:          your existing geometry object (must provide
                           get_token_centers(tokens[0:1]) -> [1, N, 2]).
        levels:            hierarchy levels, bottleneck included.
        reduction:         area multiplier between levels.
        local_k:           same-level neighbours incl. self (7 for hexagons).
        atoms_per_latent_choices:
                           the k values used in training, e.g. (100, 250, 350).
                           Used to snap the inferred k = N / L_spatial.
        verbose:           print a summary when a geometry is built.
    """

    def __init__(self, geometry, levels: int = 3, reduction: int = 4, local_k: int = 7,
                 atoms_per_latent_choices: Optional[Sequence[int]] = None,
                 verbose: bool = True, **unused):
        super().__init__()
        self.geometry = geometry
        self.levels, self.reduction, self.local_k = levels, reduction, local_k
        self.k_choices = tuple(atoms_per_latent_choices) if atoms_per_latent_choices else None
        self.verbose = verbose
        self._cache: dict = {}                        # plain dict: not a buffer, DDP-safe
        self.last_geometry: Optional[HierarchyGeometry] = None
        self.last_empty: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ geometry
    def _infer_k(self, N, L_spatial):
        k_hat = N / max(L_spatial, 1)
        if self.k_choices is None:
            return int(round(k_hat))
        return min(self.k_choices, key=lambda k: abs(math.ceil(N / k) - L_spatial))

    def get_geometry(self, tokens, atoms_per_latent: int) -> HierarchyGeometry:
        B, N, _ = tokens.shape
        key = (N, atoms_per_latent, self.levels, self.reduction, self.local_k, str(tokens.device))
        if key not in self._cache:
            with torch.no_grad():
                xy = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)
            g = HierarchyGeometry.build(xy.detach().double().cpu().numpy(), atoms_per_latent,
                                        self.levels, self.reduction, self.local_k).to(tokens.device)
            if self.verbose:
                print(f"[GeographicPruningUNet] built geometry for N={N}\n" + g.summary())
            self._cache[key] = g
        return self._cache[key]

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                latent_coords: Optional[torch.Tensor] = None, geo_k: int = 1000,
                sigma: Optional[float] = None, L_spatial: Optional[int] = None,
                hexagonal: bool = True, patch_ids=None, token_latent_assignment=None,
                atoms_per_latent: Optional[int] = None):
        if patch_ids is not None or token_latent_assignment is not None:
            raise NotImplementedError("per-sample (point cloud) paths are not supported yet")
        B, N, D = tokens.shape
        if atoms_per_latent is None:
            if L_spatial is None:
                raise ValueError("pass atoms_per_latent or L_spatial")
            atoms_per_latent = self._infer_k(N, L_spatial)
        geom = self.get_geometry(tokens, atoms_per_latent)
        self.last_geometry = geom

        cells, cvalid = geom.leaf_cells, geom.leaf_valid            # [L, Mc]
        L, Mc = cells.shape
        k = min(geo_k, Mc)
        atom_ok = cvalid.unsqueeze(0) & ~mask.bool()[:, cells]      # [B, L, Mc]
        if self.training:
            score = torch.rand(B, L, Mc, device=tokens.device)
        else:   # deterministic subset: fixed build-time order
            score = torch.linspace(1, 0, Mc, device=tokens.device).expand(B, L, Mc)
        score = score + 2.0 * atom_ok + 2.0 * cvalid                # valid > masked > padding
        sel = score.topk(k, dim=-1).indices                         # [B, L, k]
        idx = torch.gather(cells.expand(B, L, Mc), 2, sel)
        ok = torch.gather(atom_ok, 2, sel)
        toks = torch.gather(tokens, 1, idx.reshape(B, L * k, 1).expand(-1, -1, D)).reshape(B, L, k, D)

        empty = ~ok.any(-1)                                         # [B, L]
        ok = ok.clone()
        ok[..., 0] |= empty                                         # keep one slot open
        slot0 = torch.zeros(k, dtype=torch.bool, device=tokens.device); slot0[0] = True
        toks = toks.masked_fill((empty[..., None] & slot0)[..., None], 0.0)
        masks = ~ok                                                 # True = invalid
        bias = torch.zeros(B, L, k, device=tokens.device, dtype=tokens.dtype).masked_fill(masks, float("-inf"))
        self.last_empty = empty
        return toks, masks, bias

    def clear_cache(self):
        self._cache.clear()

    def extra_repr(self):
        return (f"levels={self.levels}, reduction={self.reduction}, local_k={self.local_k}, "
                f"k_choices={self.k_choices}, cached={len(self._cache)}")
