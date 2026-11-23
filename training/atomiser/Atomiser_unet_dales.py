"""
Atomiser_unet_dales.py -- Atomiser U-Net for point clouds, with a vertical stage
==============================================================================

Location: training/atomiser/Atomiser_unet_dales.py

    from training.atomiser.Atomiser_unet_dales import Atomiser_UnetDales
    model = Atomiser_UnetDales(config)
    out = model(batch)                 # batch from collate_dales
    out["predictions"]                 # [B, Nq, num_classes], one row per query

Pipeline
    points [B, N, 7] (PointLayout: x, y, z, value, intensity, echo_a, echo_b)
      -> PointHierarchy: fixed hex lattice (shared), per-point column, count-based
         vertical cells, local ground, query lookup
      -> vertical leaf read: atoms of each vertical cell -> one cell latent
         (atom features + Fourier 3D offset to the cell anchor, cross-attention
         from a seed + MLP of the cell descriptor: z extent, height, count)
      -> vertical stage, one learned CLS per column (unet_vertical, e.g. "VHV"):
           V: attention over [CLS, cells] of one column, 1D RoPE on z (CLS unrotated)
           H: 2D local block over the CLS tokens (hex 7-neighbours, 2D RoPE)
         collapse: the CLS is the column's 2D latent (level 0 of the U-Net)
      -> AtomizerUNet (unchanged, from Atomiser_unet.py): local levels, merges,
         global bottleneck, overlap decoder -> D_0 [B, L0, D]
      -> uncollapse: D_0 into the CLS slot, unet_vertical_dec V blocks over
         [CLS, encoder cells] -> decoded cells [B, L0, vmax, D]
      -> query decoder: own point (atom embedding of the query) then K vertical
         cells (own cell first, then nearest in 3D) with Fourier 3D offsets,
         FFN, head.

unet_vertical: none  -> 2D baseline in the same code path: one cell per column
    (vmax = 1, up to unet_column_m atoms), the cell latent is the column latent,
    the query decoder reads the K nearest columns (with 3D offsets).

Config (config["Atomiser"]; U-Net keys as in Atomiser_unet.py):
    unet_vertical ("VHV")          pattern of V / H passes, or "none"
    unet_vertical_dec (1)          V blocks after the uncollapse (0 = add D_0 to cells)
    unet_vertical_kz (50), unet_vertical_vmax (24), unet_vertical_m (kz)
    unet_partition ("zslice" | "cluster")   vertical cells: z slices of the column, or
                                   connected clusters + z slices + plane / line merge
    unet_partition_vox_m (0.35), unet_partition_tiny (5), unet_partition_plane_res (0.08),
    unet_partition_plane_ang (20), unet_partition_line_lin (0.7), unet_partition_line_rad (0.2),
    unet_partition_line_ang (20), unet_partition_merge (true)
    unet_column_m (500)            atoms per column when unet_vertical: none
    unet_points_per_leaf (250), unet_point_density (46.1), unet_patch_size_m (50),
    unet_site_spacing_m (0.1), unet_ground_quantile (0.02), unet_ground_jump_m (2.0)
    unet_z_unit_m (1.0)            z unit of the vertical RoPE
    unet_vertical_rope ("z")       "z": 1D RoPE on cell height | "xyz": 3D RoPE on the cell
                                   centroid (half the pairs for z, a quarter each for x, y)
    unet_vertical_rope_xy_unit_m (0.25), unet_vertical_rope_base_xy (40): x, y periods 2..80 units
    unet_vertical_rope_base (100)  RoPE periods from 2 to 2 * base z units
    unet_decoder_neighbors (9)     K cells read per query
    unet_vertical: hier            hierarchical 3D stage instead of V / H passes (see HierStage):
      unet_hier_nodes ("voxel" | "ezsp" | "ezsp_pre")  node source: uniform voxel levels, nested EZ-SP
        components computed per batch, or EZ-SP precomputed per tile (precompute_ezsp.py), split by column
        components (ezsp_hierarchy.py: unet_ezsp_regs [0.02, 0.1], unet_ezsp_min_sizes [10, 1],
        unet_ezsp_vox_m 0.25, unet_ezsp_radius 2; unet_hier_sizes [1, 2] = neighbour grid / units)
      unet_hier_sizes ([0.5, 1, 2]) voxel size per level (doubling), inside the columns
      unet_hier_dims ([128, 256, dim]), unet_hier_depth ([1, 1, 1]) local-attention blocks per
      level, unet_hier_dec_depth ([1, 0, 0]) after each transport on the way down,
      unet_hier_k (16) neighbours per node (any column), unet_hier_per_pos (3),
      unet_hier_m (16) atoms per 0.5 m node, unet_hier_rope_base (16), unet_hier_chunk (32768)
    unet_lidar_use_intensity (false), unet_height_scale_m (10)
    unet_atom_dim (128), unet_query_dim (256), unet_leaf_chunk (4096 cells),
    unet_query_chunk (16384), unet_column_chunk (8192 columns per V-block chunk)
    unet_n_sensors (8)
    unet_checkpoint (true)         activation checkpointing (off: faster, more memory)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from training.atomiser.Atomiser_unet import AtomizerUNet, LocalBlock, FFN, fourier, maybe_checkpoint
from training.utils.datasets.point_hierarchy import PointHierarchy
from training.utils.datasets.voxel_hierarchy import VoxelHierarchyBuilder
from training.utils.datasets.dataset_dales_points import PointLayout as PL


def fourier_nd(d: torch.Tensor, n_freq: int = 16, max_freq: float = 16.0) -> torch.Tensor:
    """[..., n] -> [..., n (2F+1)]"""
    return torch.cat([fourier(d[..., i], n_freq, max_freq) for i in range(d.shape[-1])], -1)


def compress(x, s):
    return x / (s + x.abs())


PE3 = 3 * (2 * 16 + 1)

# activation checkpointing switch (config unet_checkpoint, default on): off = faster
# (no recomputation in the backward pass) but more memory
_CKPT = {"on": True}

# stage timing (profile_dales.py switches it on): synchronised wall time per labelled
# stage, accumulated in _TIMING["acc"]; always also a profiler record_function label
import time as _time
_TIMING = {"on": False, "acc": {}}


class _stage:
    def __init__(self, name):
        self.name, self.rf = name, record_function(name)

    def __enter__(self):
        if _TIMING["on"] and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t = _time.perf_counter()
        self.rf.__enter__()

    def __exit__(self, *exc):
        self.rf.__exit__(*exc)
        if _TIMING["on"]:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _TIMING["acc"][self.name] = _TIMING["acc"].get(self.name, 0.0) + _time.perf_counter() - self.t
        return False


# =============================================================================
# Atom embedding (LiDAR)
# =============================================================================
class LidarAtomEmbedding(nn.Module):
    """value (compressed elevation), height above local ground, echo pair (a, b),
    optional intensity, sensor embedding -> MLP -> atom_dim. No position here:
    offsets to the receiving anchor are added by the reader."""

    def __init__(self, atom_dim=128, n_sensors=8, use_intensity=False, h_scale=10.0):
        super().__init__()
        self.out_dim = atom_dim
        self.use_intensity, self.h_scale = use_intensity, float(h_scale)
        self.sensor = nn.Embedding(n_sensors, 8)
        in_dim = 17 + 17 + 9 + 9 + (9 if use_intensity else 0) + 8
        self.mlp = nn.Sequential(nn.Linear(in_dim, atom_dim), nn.GELU(), nn.Linear(atom_dim, atom_dim))

    def forward(self, p, h, s_emb):
        """p [..., C] points, h [...] height above local ground (m),
        s_emb: sensor embedding broadcastable to [..., 8]."""
        f = [fourier(p[..., PL.VALUE].float(), 8, 8.0),
             fourier(compress(h.float(), self.h_scale), 8, 8.0),
             fourier(p[..., PL.ECHO_A].float(), 4, 4.0),
             fourier(p[..., PL.ECHO_B].float(), 4, 4.0)]
        if self.use_intensity:
            f.append(fourier(p[..., PL.INTENSITY].float(), 4, 4.0))
        f.append(s_emb.expand(*h.shape, 8))
        return self.mlp(torch.cat(f, -1))


# =============================================================================
# Vertical leaf read: atoms of one vertical cell -> one cell latent
# =============================================================================
class VerticalLeafEncoder(nn.Module):
    def __init__(self, embed: LidarAtomEmbedding, dim, heads=8, chunk=4096, desc_dim=3):
        super().__init__()
        A = embed.out_dim
        self.embed, self.chunk = embed, chunk
        self.h, self.dh = heads, dim // heads
        self.mlp = nn.Sequential(nn.Linear(A + PE3, 2 * A), nn.GELU(), nn.Linear(2 * A, A))
        self.seed = nn.Parameter(torch.randn(dim) * 0.02)
        self.desc = nn.Sequential(nn.Linear(desc_dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.q = nn.Linear(dim, dim)
        self.ln_kv = nn.LayerNorm(A)
        self.kv = nn.Linear(A, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.ffn = FFN(dim)

    def _cells(self, pts, h, s_emb, off, amask, desc):
        # pts [v, m, C], h [v, m], s_emb [v, 1, 8], off [v, m, 3], amask [v, m] True = pad
        V, M = h.shape
        z = self.mlp(torch.cat([self.embed(pts, h, s_emb), fourier_nd(off.to(pts.dtype))], -1))
        k, v = self.kv(self.ln_kv(z)).view(V, M, 2, self.h, self.dh).unbind(2)
        q = self.q(self.seed).view(self.h, self.dh)
        s = torch.einsum("hd,vmhd->vmh", q, k) / math.sqrt(self.dh)
        s = s.masked_fill(amask.unsqueeze(-1), float("-inf"))         # every cell has >= 1 atom
        a = s.softmax(dim=1)
        out = torch.einsum("vmh,vmhd->vhd", a, v).reshape(V, -1)
        x = self.seed + self.desc(desc) + self.o(out)
        return x + self.ffn(x)

    def forward(self, pts, h, s_emb, off, amask, desc):
        outs = []
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        for s in range(0, h.shape[0], self.chunk):
            e = min(s + self.chunk, h.shape[0])
            outs.append(maybe_checkpoint(self._cells, pts[s:e], h[s:e], s_emb[s:e], off[s:e],
                                         amask[s:e], desc[s:e], enabled=ck))
        return torch.cat(outs, 0)                                       # [V, D]


# =============================================================================
# Vertical attention (within one column), 1D RoPE on z
# =============================================================================
class RoPE1D(nn.Module):
    def __init__(self, head_dim, base=100.0):
        super().__init__()
        assert head_dim % 2 == 0
        nf = head_dim // 2
        self.register_buffer("freqs", math.pi * base ** (-torch.arange(nf, dtype=torch.float32) / nf),
                             persistent=False)
        self.log_mult = nn.Parameter(torch.zeros(()))

    def forward(self, x, pos):
        # x [Bc, T, h, dh], pos [Bc, T] (z units)
        ang = (pos.float() * self.log_mult.exp()).unsqueeze(-1) * self.freqs      # [Bc, T, dh/2]
        cos, sin = ang.cos().unsqueeze(2), ang.sin().unsqueeze(2)
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


class RoPE3D(nn.Module):
    """Rotary embedding on (x, y, z): of the head_dim / 2 rotated pairs, half
    encode z and a quarter each x and y. Per axis, frequencies span pi (period 2
    units) down to pi / base (period 2 * base units), with a learned scale per
    axis (init 1). Scores depend only on relative offsets between cells."""

    def __init__(self, head_dim, base_xy=40.0, base_z=100.0):
        super().__init__()
        assert head_dim % 2 == 0
        nf = head_dim // 2
        nz = nf - 2 * (nf // 4)
        nxy = nf // 4
        f = lambda n, b: math.pi * b ** (-torch.arange(n, dtype=torch.float32) / max(n, 1))
        self.register_buffer("fx", f(nxy, base_xy), persistent=False)
        self.register_buffer("fy", f(nxy, base_xy), persistent=False)
        self.register_buffer("fz", f(nz, base_z), persistent=False)
        self.log_mult = nn.Parameter(torch.zeros(3))

    def forward(self, x, pos):
        # x [Bc, T, h, dh], pos [Bc, T, 3] (x, y, z in their units)
        p = pos.float() * self.log_mult.exp()
        ang = torch.cat([p[..., 0:1] * self.fx, p[..., 1:2] * self.fy, p[..., 2:3] * self.fz], -1)
        cos, sin = ang.cos().unsqueeze(2), ang.sin().unsqueeze(2)
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


class VerticalBlock(nn.Module):
    """Attention over [CLS, cells] of one column. Cells are rotated by their
    position: z only (rope="z"), or the cell centroid (x, y, z) (rope="xyz");
    relative offsets only. The CLS has no position and stays unrotated."""

    def __init__(self, dim, heads, dropout=0.0, rope_base=100.0, rope="z", rope_base_xy=40.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        assert rope in ("z", "xyz"), rope
        self.rope = RoPE1D(self.dh, rope_base) if rope == "z" else RoPE3D(self.dh, rope_base_xy, rope_base)
        self.ffn = FFN(dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, zpos, key_pad):
        # x [Bc, 1 + vmax, D], zpos [Bc, vmax] (rope z) or [Bc, vmax, 3] (rope xyz),
        # key_pad [Bc, 1 + vmax] True = padding
        Bc, T, D = x.shape
        q, k, v = self.qkv(self.ln(x)).view(Bc, T, 3, self.h, self.dh).unbind(2)
        q = torch.cat([q[:, :1], self.rope(q[:, 1:], zpos)], 1)
        k = torch.cat([k[:, :1], self.rope(k[:, 1:], zpos)], 1)
        allow = (~key_pad)[:, None, None, :]                              # CLS always allowed
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=allow,
                                             dropout_p=self.dropout if self.training else 0.0)
        x = x + self.o(out.transpose(1, 2).reshape(Bc, T, D))
        return x + self.ffn(x)


def _run_columns(blk, x, zpos, key_pad, chunk, ck):
    if x.shape[0] <= chunk:
        return maybe_checkpoint(blk, x, zpos, key_pad, enabled=ck)
    return torch.cat([maybe_checkpoint(blk, x[s:s + chunk], zpos[s:s + chunk], key_pad[s:s + chunk],
                                       enabled=ck) for s in range(0, x.shape[0], chunk)], 0)


class VerticalStage(nn.Module):
    """pattern of 'V' (within columns) and 'H' (across columns, CLS tokens only)."""

    def __init__(self, dim, heads, pattern="VHV", dropout=0.0, rope_base=100.0, chunk=8192,
                 rope="z", rope_base_xy=40.0):
        super().__init__()
        self.pattern = pattern.upper()
        assert self.pattern and set(self.pattern) <= {"V", "H"}, pattern
        self.blocks = nn.ModuleList(VerticalBlock(dim, heads, dropout, rope_base, rope, rope_base_xy) if c == "V"
                                    else LocalBlock(dim, heads, dropout) for c in self.pattern)
        self.cls_seed = nn.Parameter(torch.randn(dim) * 0.02)
        self.chunk = chunk

    def forward(self, cells, vmask, zpos, col_empty, empty_emb, g):
        B, L0, Vm, D = cells.shape
        cls = self.cls_seed + empty_emb * col_empty.unsqueeze(-1).to(cells.dtype)   # [B, L0, D]
        x = torch.cat([cls.unsqueeze(2).to(cells.dtype), cells], 2).reshape(B * L0, 1 + Vm, D)
        key_pad = torch.cat([torch.zeros(B * L0, 1, dtype=torch.bool, device=cells.device),
                             vmask.reshape(B * L0, Vm)], 1)
        zp = zpos.reshape(B * L0, Vm, *zpos.shape[3:])
        coords = g.centers[0] / g.scales[0].mean()
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        for c, blk in zip(self.pattern, self.blocks):
            if c == "V":
                x = _run_columns(blk, x, zp, key_pad, self.chunk, ck)
            else:
                t = blk(x[:, 0].reshape(B, L0, D), g.local_idx[0], coords)
                x = torch.cat([t.reshape(B * L0, 1, D), x[:, 1:]], 1)
        x = x.reshape(B, L0, 1 + Vm, D)
        return x[:, :, 0], x[:, :, 1:]                                    # CLS, cells


class VerticalDecoder(nn.Module):
    """Uncollapse: [D_0 as CLS, encoder cells] -> n V blocks -> decoded cells."""

    def __init__(self, dim, heads, n_blocks=1, dropout=0.0, rope_base=100.0, chunk=8192,
                 rope="z", rope_base_xy=40.0):
        super().__init__()
        self.blocks = nn.ModuleList(VerticalBlock(dim, heads, dropout, rope_base, rope, rope_base_xy)
                                    for _ in range(n_blocks))
        self.chunk = chunk

    def forward(self, d0, cells, vmask, zpos):
        B, L0, Vm, D = cells.shape
        x = torch.cat([d0.unsqueeze(2).to(cells.dtype), cells], 2).reshape(B * L0, 1 + Vm, D)
        key_pad = torch.cat([torch.zeros(B * L0, 1, dtype=torch.bool, device=cells.device),
                             vmask.reshape(B * L0, Vm)], 1)
        zp = zpos.reshape(B * L0, Vm, *zpos.shape[3:])
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        for blk in self.blocks:
            x = _run_columns(blk, x, zp, key_pad, self.chunk, ck)
        return x.reshape(B, L0, 1 + Vm, D)[:, :, 1:]


# =============================================================================
# Hierarchical 3D stage (unet_vertical: hier)
# =============================================================================
# Levels of voxels inside the columns (VoxelHierarchyBuilder): at each level,
# isotropic local attention over the K nearest nodes (any column, 3D RoPE on
# centroids), then merging into the next level (parents read their children,
# inside the column). The top level merges into the column latents (the 2D
# U-Net input). On the way down: gated transport from the parent, then local
# attention again. The query decoder reads the finest level.

class RoPE3DIso(nn.Module):
    """Rotary embedding on (x, y, z) with the rotated pairs split evenly between
    the three axes. Positions in units of the level's voxel size; periods from
    2 to 2 * base units; learned scale per axis (init 1)."""

    def __init__(self, head_dim, base=16.0):
        super().__init__()
        nf = head_dim // 2
        n = [nf // 3, nf // 3, nf - 2 * (nf // 3)]
        for name, k in zip("xyz", n):
            self.register_buffer(f"f{name}", math.pi * base ** (-torch.arange(k, dtype=torch.float32) / max(k, 1)),
                                 persistent=False)
        self.log_mult = nn.Parameter(torch.zeros(3))

    def forward(self, x, pos):
        # x [n, h, dh], pos [n, 3]
        p = pos.float() * self.log_mult.exp()
        ang = torch.cat([p[:, 0:1] * self.fx, p[:, 1:2] * self.fy, p[:, 2:3] * self.fz], -1).unsqueeze(1)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


class NodeBlock(nn.Module):
    """Local attention of every node over its neighbour table (self first), 3D RoPE."""

    def __init__(self, dim, head_dim=64, dropout=0.0, rope_base=16.0, chunk=32768):
        super().__init__()
        self.h = max(1, dim // head_dim); self.dh = dim // self.h
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.rope = RoPE3DIso(self.dh, rope_base)
        self.ffn = FFN(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.chunk = chunk

    def _attn(self, q, k, v, nbr, nmask):
        s = torch.einsum("chd,ckhd->ckh", q, k[nbr]) / math.sqrt(self.dh)
        s = s.masked_fill(nmask.unsqueeze(-1), float("-inf"))           # self is always valid
        a = self.drop(s.softmax(dim=1))
        return torch.einsum("ckh,ckhd->chd", a, v[nbr])

    def forward(self, x, pos, nbr, nmask):
        n, D = x.shape
        q, k, v = self.qkv(self.ln(x)).view(n, 3, self.h, self.dh).unbind(1)
        q, k = self.rope(q, pos), self.rope(k, pos)
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        out = torch.cat([maybe_checkpoint(self._attn, q[s:s + self.chunk], k, v, nbr[s:s + self.chunk],
                                          nmask[s:s + self.chunk], enabled=ck)
                         for s in range(0, n, self.chunk)], 0)
        x = x + self.o(out.reshape(n, D))
        return x + self.ffn(x)


class MergeBlock(nn.Module):
    """Parents read their children (padded table): query = learned seed + MLP of
    the parent descriptor; keys / values = children + Fourier offset (child
    centroid - parent centroid) / parent unit."""

    def __init__(self, dim_c, dim_p, head_dim=64, desc_dim=1, dropout=0.0):
        super().__init__()
        self.h = max(1, dim_p // head_dim); self.dh = dim_p // self.h
        self.seed = nn.Parameter(torch.randn(dim_p) * 0.02)
        self.desc = nn.Sequential(nn.Linear(desc_dim, 64), nn.GELU(), nn.Linear(64, dim_p))
        self.ln_k = nn.LayerNorm(dim_c)
        self.q = nn.Linear(dim_p, dim_p)
        self.kv = nn.Linear(dim_c, 2 * dim_p)
        self.pe = nn.Linear(PE3, 2 * dim_p)
        self.o = nn.Linear(dim_p, dim_p)
        self.ffn = FFN(dim_p, dropout=dropout)

    def forward(self, xc, pos_c, children, cmask, pos_p, unit, desc):
        n_p = children.shape[0]
        base = self.seed + self.desc(desc)                                  # [n_p, D]
        q = self.q(base).view(n_p, self.h, self.dh)
        kc, vc = self.kv(self.ln_k(xc)).chunk(2, -1)
        off = (pos_c[children] - pos_p.unsqueeze(1)) / unit                # [n_p, C, 3]
        pk, pv = self.pe(fourier_nd(off.to(xc.dtype))).chunk(2, -1)
        k = (kc[children] + pk).view(n_p, -1, self.h, self.dh)
        v = (vc[children] + pv).view(n_p, -1, self.h, self.dh)
        s = torch.einsum("phd,pchd->pch", q, k) / math.sqrt(self.dh)
        s = s.masked_fill(cmask.unsqueeze(-1), -1e4)
        a = s.softmax(dim=1) * (~cmask).any(1).view(-1, 1, 1).to(s.dtype)  # childless parent -> no message
        x = base + self.o(torch.einsum("pch,pchd->phd", a, v).reshape(n_p, -1))
        return x + self.ffn(x)


class TransportFuse(nn.Module):
    """Down path: x = skip + sigmoid(g(skip)) * W(parent state)[parent]; gate zero-init (0.5)."""

    def __init__(self, dim_p, dim_c):
        super().__init__()
        self.t = nn.Sequential(nn.LayerNorm(dim_p), nn.Linear(dim_p, dim_c))
        self.g_ln, self.g = nn.LayerNorm(dim_c), nn.Linear(dim_c, dim_c)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)

    def forward(self, skip, parent_state, parent):
        m = self.t(parent_state)[parent]
        return skip + torch.sigmoid(self.g(self.g_ln(skip))) * m


class PackedLeafEncoder(nn.Module):
    """Level-0 read without padding: the atoms of all nodes in one flat list, a
    softmax per node (segment softmax), the result summed per node. Same maths as
    VerticalLeafEncoder (seed query, atom MLP on [features ; Fourier offset],
    descriptor MLP, FFN), but cost and memory follow the real number of atoms."""

    def __init__(self, embed: LidarAtomEmbedding, dim, heads=8, desc_dim=2):
        super().__init__()
        A = embed.out_dim
        self.embed = embed
        self.h, self.dh = heads, dim // heads
        self.mlp = nn.Sequential(nn.Linear(A + PE3, 2 * A), nn.GELU(), nn.Linear(2 * A, A))
        self.seed = nn.Parameter(torch.randn(dim) * 0.02)
        self.desc = nn.Sequential(nn.Linear(desc_dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.q = nn.Linear(dim, dim)
        self.ln_kv = nn.LayerNorm(A)
        self.kv = nn.Linear(A, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.ffn = FFN(dim)

    def _read(self, pts, h, s_emb, off, node, desc):
        n = desc.shape[0]
        z = self.mlp(torch.cat([self.embed(pts, h, s_emb), fourier_nd(off.to(pts.dtype))], -1))   # [Na, A]
        k, v = self.kv(self.ln_kv(z)).view(-1, 2, self.h, self.dh).unbind(1)                    # [Na, h, dh]
        q = self.q(self.seed).view(self.h, self.dh)
        s = (k * q).sum(-1).float() / math.sqrt(self.dh)                                         # [Na, h]
        smax = torch.full((n, self.h), float("-inf"), device=s.device).scatter_reduce(
            0, node.unsqueeze(-1).expand_as(s), s, reduce="amax", include_self=True)
        e = torch.exp(s - smax[node])
        den = torch.zeros(n, self.h, device=s.device).index_add_(0, node, e)
        a = (e / den[node]).to(v.dtype)
        out = torch.zeros(n, self.h, self.dh, device=v.device, dtype=v.dtype).index_add_(0, node, a.unsqueeze(-1) * v)
        x = self.seed + self.desc(desc) + self.o(out.reshape(n, -1))
        return x + self.ffn(x)

    def forward(self, pts, h, s_emb, off, node, desc):
        """pts [Na, C], h [Na], s_emb [Na, 8], off [Na, 3], node [Na] (every node has >= 1 atom), desc [n, d]."""
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        return maybe_checkpoint(self._read, pts, h, s_emb, off, node, desc, enabled=ck)


class HierStage(nn.Module):
    def __init__(self, embed, dims, depth, dec_depth, col_dim, head_dim=64, dropout=0.0,
                 rope_base=16.0, leaf_chunk=8192, node_chunk=32768):
        super().__init__()
        S = len(dims)
        self.S, self.dims = S, list(dims)
        self.leaf = PackedLeafEncoder(embed, dims[0], max(1, dims[0] // head_dim), desc_dim=2)
        self.enc = nn.ModuleList(nn.ModuleList(NodeBlock(dims[l], head_dim, dropout, rope_base, node_chunk)
                                               for _ in range(depth[l])) for l in range(S))
        self.merge = nn.ModuleList(MergeBlock(dims[l], dims[l + 1], head_dim) for l in range(S - 1))
        self.col_merge = MergeBlock(dims[-1], col_dim, head_dim)
        self.down_col = TransportFuse(col_dim, dims[-1])
        self.down = nn.ModuleList(TransportFuse(dims[l + 1], dims[l]) for l in range(S - 1))
        self.dec = nn.ModuleList(nn.ModuleList(NodeBlock(dims[l], head_dim, dropout, rope_base, node_chunk)
                                               for _ in range(dec_depth[l])) for l in range(S))

    @staticmethod
    def _units(lv):
        return lv.pos / lv.size

    def encode(self, vh, pts_flat, h_flat, s_emb, col_xy, col_unit, empty_emb, col_empty):
        lv0 = vh.levels[0]
        # ---- level-0 read: atoms of each 0.5 m node ----
        keep = ~vh.node_atoms_mask                                                     # packed atoms (<= m per node)
        at = vh.node_atoms[keep]
        node = torch.nonzero(keep)[:, 0]
        a_pts, a_h = pts_flat[at], h_flat[at]
        off = (a_pts[:, [PL.X, PL.Y, PL.Z]].float() - lv0.pos[node]) / lv0.size
        node_h = torch.zeros(lv0.n, device=h_flat.device).index_add_(
            0, vh.point_node.clamp(min=0), h_flat * (vh.point_node >= 0).float()) / lv0.cnt.clamp(min=1).float()
        desc = torch.stack([torch.log1p(lv0.cnt.float()), node_h / (10.0 + node_h.abs())], -1).to(pts_flat.dtype)
        x = self.leaf(a_pts, a_h, s_emb[lv0.batch[node]], off, node, desc)               # [n0, d0]
        skips = []
        for l, lv in enumerate(vh.levels):
            u = self._units(lv)
            for blk in self.enc[l]:
                x = blk(x, u, lv.nbr, lv.nbr_mask)
            skips.append(x)
            if l < self.S - 1:
                nx = vh.levels[l + 1]
                x = self.merge[l](x, lv.pos, nx.children, nx.children_mask, nx.pos, nx.size,
                                  torch.log1p(nx.cnt.float()).unsqueeze(-1).to(x.dtype))
        # ---- top level -> columns ----
        top = vh.levels[-1]
        col_pos = torch.cat([col_xy, vh.col_pos[:, 2:3]], -1)                           # [B * L0, 3]
        ccnt = torch.zeros(col_pos.shape[0], device=x.device).index_add_(0, top.parent, top.cnt.float())
        z0 = self.col_merge(x, top.pos, vh.col_children, vh.col_children_mask, col_pos, col_unit,
                            torch.log1p(ccnt).unsqueeze(-1).to(x.dtype))
        z0 = z0 + empty_emb * col_empty.reshape(-1, 1).to(z0.dtype)
        return z0.view(vh.B, vh.L0, -1), skips

    def decode(self, d0, skips, vh):
        x = self.down_col(skips[-1], d0.reshape(vh.B * vh.L0, -1), vh.levels[-1].parent)
        for l in reversed(range(self.S)):
            lv = vh.levels[l]
            if l < self.S - 1:
                x = self.down[l](skips[l], x, lv.parent)
            u = self._units(lv)
            for blk in self.dec[l]:
                x = blk(x, u, lv.nbr, lv.nbr_mask)
        return x                                                                         # [n0, d0]


# =============================================================================
# Query decoder (3D)
# =============================================================================
class PointQueryDecoder(nn.Module):
    def __init__(self, embed: LidarAtomEmbedding, dim, qdim, num_classes, heads, chunk=16384):
        super().__init__()
        self.embed, self.chunk = embed, chunk
        self.h, self.dh = heads, qdim // heads
        self.q_init = nn.Parameter(torch.randn(qdim) * 0.02)
        self.obs = nn.Sequential(nn.LayerNorm(embed.out_dim), nn.Linear(embed.out_dim, qdim))
        self.lat_ln = nn.LayerNorm(dim)
        self.lat_kv = nn.Linear(dim, 2 * qdim)
        self.pe_kv = nn.Linear(PE3, 2 * qdim)
        self.q_ln = nn.LayerNorm(qdim)
        self.lat_q = nn.Linear(qdim, qdim)
        self.lat_o = nn.Linear(qdim, qdim)
        self.ffn = FFN(qdim)
        self.head = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, num_classes))

    def _chunk(self, qp, qh, s_emb, lat_k, lat_v, nbr, nmask, noff):
        B, M, _ = qp.shape
        x = self.q_init + self.obs(self.embed(qp, qh, s_emb))                  # own point
        bidx = torch.arange(B, device=qp.device)[:, None, None]
        pk, pv = self.pe_kv(fourier_nd(noff.to(qp.dtype))).chunk(2, -1)
        kk = (lat_k[bidx, nbr] + pk).view(B, M, -1, self.h, self.dh)
        vv = (lat_v[bidx, nbr] + pv).view(B, M, -1, self.h, self.dh)
        q = self.lat_q(self.q_ln(x)).view(B, M, self.h, self.dh)
        s = torch.einsum("bmhd,bmkhd->bmkh", q, kk) / math.sqrt(self.dh)
        s = s.masked_fill(nmask.unsqueeze(-1), -1e4)                        # padded queries stay finite
        a = s.softmax(dim=2)
        x = x + self.lat_o(torch.einsum("bmkh,bmkhd->bmhd", a, vv).reshape(B, M, -1))
        x = x + self.ffn(x)
        return self.head(x)

    def forward(self, qp, qh, s_emb, cells_flat, look):
        """qp [B, M, C], qh [B, M], s_emb [B, 1, 8], cells_flat [B, L0 * vmax, D] (per
        sample) or [n, D] (flat over the batch, look["nbr"] indexes it directly),
        look: PointHierarchy.lookup_queries / VoxelHierarchyBuilder.query_neighbors."""
        if cells_flat.dim() == 2:                                          # flat node table
            lat_k, lat_v = self.lat_kv(self.lat_ln(cells_flat)).unsqueeze(0).chunk(2, -1)
            lat_k, lat_v = lat_k.expand(qp.shape[0], -1, -1), lat_v.expand(qp.shape[0], -1, -1)
        else:
            lat_k, lat_v = self.lat_kv(self.lat_ln(cells_flat)).chunk(2, -1)     # once per cell
        M = qp.shape[1]
        ck = self.training and torch.is_grad_enabled() and _CKPT["on"]
        outs = []
        for s in range(0, M, self.chunk):
            e = min(s + self.chunk, M)
            outs.append(maybe_checkpoint(self._chunk, qp[:, s:e], qh[:, s:e], s_emb, lat_k, lat_v,
                                         look["nbr"][:, s:e], look["nbr_mask"][:, s:e],
                                         look["nbr_off"][:, s:e], enabled=ck))
        return torch.cat(outs, 1)


# =============================================================================
# The model
# =============================================================================
class Atomiser_UnetDales(nn.Module):
    def __init__(self, config, lookup_table=None):
        super().__init__()
        a = config.get("Atomiser", {})
        self.num_classes = int(config["trainer"]["num_classes"])
        _CKPT["on"] = bool(a.get("unet_checkpoint", True))
        levels = int(a.get("unet_levels", 3))
        dims = a.get("unet_dims", None)
        dims = [int(d) for d in dims] if dims is not None else [int(a.get("unet_dim", 384))] * levels
        head_dim = int(a.get("unet_head_dim", 64))
        dim = dims[0]
        heads = max(1, dim // head_dim)
        qdim = int(a.get("unet_query_dim", 256))

        # ---- geometry ----
        self.vertical = str(a.get("unet_vertical", "VHV"))
        self.is_hier = self.vertical.lower() == "hier"
        flat = self.vertical.lower() in ("none", "hier")       # hier: columns only from PointHierarchy"
        kz = int(a.get("unet_vertical_kz", 50))
        self.hier = PointHierarchy(
            patch_size_m=float(a.get("unet_patch_size_m", 50.0)),
            points_per_leaf=int(a.get("unet_points_per_leaf", 250)),
            density=float(a.get("unet_point_density", 46.1)),
            levels=levels, reduction=4, local_k=7,
            site_spacing_m=float(a.get("unet_site_spacing_m", 0.1)),
            kz=10 ** 9 if flat else kz,
            vmax=1 if flat else int(a.get("unet_vertical_vmax", 24)),
            m=int(a.get("unet_column_m", 500)) if flat else int(a.get("unet_vertical_m", kz)),
            ground_quantile=float(a.get("unet_ground_quantile", 0.02)),
            ground_jump_m=float(a.get("unet_ground_jump_m", 2.0)),
            partition="zslice" if flat else str(a.get("unet_partition", "zslice")),
            vox_m=float(a.get("unet_partition_vox_m", 0.35)),
            tiny=int(a.get("unet_partition_tiny", 5)),
            plane_res=float(a.get("unet_partition_plane_res", 0.08)),
            plane_ang=float(a.get("unet_partition_plane_ang", 20.0)),
            line_lin=float(a.get("unet_partition_line_lin", 0.70)),
            line_rad=float(a.get("unet_partition_line_rad", 0.20)),
            line_ang=float(a.get("unet_partition_line_ang", 20.0)),
            merge=bool(a.get("unet_partition_merge", True)))
        self.z_unit = float(a.get("unet_z_unit_m", 1.0))
        self.K = int(a.get("unet_decoder_neighbors", 9))
        rope_base = float(a.get("unet_vertical_rope_base", 100.0))
        self.vrope = str(a.get("unet_vertical_rope", "z"))
        self.xy_unit = float(a.get("unet_vertical_rope_xy_unit_m", 0.25))
        rope_base_xy = float(a.get("unet_vertical_rope_base_xy", 40.0))
        col_chunk = int(a.get("unet_column_chunk", 8192))
        dropout = float(a.get("unet_dropout", 0.0))

        # ---- atoms, leaf read, vertical stage ----
        self.embed = LidarAtomEmbedding(int(a.get("unet_atom_dim", 128)), int(a.get("unet_n_sensors", 8)),
                                        bool(a.get("unet_lidar_use_intensity", False)),
                                        float(a.get("unet_height_scale_m", 10.0)))
        self.leaf = (None if self.is_hier else       # hier: the level-0 read lives in HierStage
                     VerticalLeafEncoder(self.embed, dim, heads, int(a.get("unet_leaf_chunk", 4096)),
                                         desc_dim=self.hier.desc_dim))
        self.empty_emb = nn.Parameter(torch.zeros(dim))
        if self.is_hier:
            self.hier_nodes = str(a.get("unet_hier_nodes", "voxel"))
            half = float(a.get("unet_patch_size_m", 50.0)) / 2
            if self.hier_nodes in ("ezsp", "ezsp_pre"):
                from training.utils.datasets.ezsp_hierarchy import EzspHierarchyBuilder, PrecomputedEzspBuilder
                sizes = [float(x) for x in a.get("unet_hier_sizes", [1.0, 2.0])]
                cls_ = PrecomputedEzspBuilder if self.hier_nodes == "ezsp_pre" else EzspHierarchyBuilder
                self.vbuild = cls_(
                    regs=a.get("unet_ezsp_regs", [0.02, 0.1]), min_sizes=a.get("unet_ezsp_min_sizes", [10, 1]),
                    sizes=sizes, vox=float(a.get("unet_ezsp_vox_m", 0.25)), radius=int(a.get("unet_ezsp_radius", 2)),
                    half=half, K=int(a.get("unet_hier_k", 16)), per_pos=int(a.get("unet_hier_per_pos", 8)),
                    m=int(a.get("unet_hier_m", 32)))
            elif self.hier_nodes == "voxel":
                sizes = [float(x) for x in a.get("unet_hier_sizes", [0.5, 1.0, 2.0])]
                self.vbuild = VoxelHierarchyBuilder(sizes, half, int(a.get("unet_hier_k", 16)),
                                                    int(a.get("unet_hier_per_pos", 3)), int(a.get("unet_hier_m", 16)))
            else:
                raise ValueError(f"unet_hier_nodes must be 'voxel', 'ezsp' or 'ezsp_pre', got {self.hier_nodes!r}")
            hdims = [int(x) for x in a.get("unet_hier_dims", [128, 256, dim][-len(sizes):])]
            assert len(hdims) == len(sizes), "unet_hier_dims needs one width per level"
            dep = a.get("unet_hier_depth", [1] * len(sizes))
            ddep = a.get("unet_hier_dec_depth", [1] + [0] * (len(sizes) - 1))
            self.hstage = HierStage(self.embed, hdims, [int(x) for x in dep], [int(x) for x in ddep], dim,
                                    head_dim, dropout, float(a.get("unet_hier_rope_base", 16.0)),
                                    int(a.get("unet_leaf_chunk", 8192)), int(a.get("unet_hier_chunk", 32768)))
            self.query_dim_in = hdims[0]
        else:
            self.query_dim_in = dim
        self.stage = None if flat else VerticalStage(dim, heads, self.vertical, dropout, rope_base, col_chunk,
                                                     self.vrope, rope_base_xy)
        n_dec = int(a.get("unet_vertical_dec", 1))
        self.vdec = (VerticalDecoder(dim, heads, n_dec, dropout, rope_base, col_chunk, self.vrope, rope_base_xy)
                     if (not flat and n_dec > 0) else None)

        # ---- 2D U-Net (unchanged) ----
        self.unet = AtomizerUNet(dims, head_dim, int(a.get("unet_depth", 2)), dropout,
                                 a.get("unet_fusion", "gate"),
                                 decoder=a.get("unet_decoder", "overlap"),
                                 dec_dims=a.get("unet_dec_dims", None),
                                 dec_refine=a.get("unet_dec_refine", 0),
                                 merge=a.get("unet_merge", "cross"),
                                 group_final_read=bool(a.get("unet_group_final_read", True)),
                                 region_size_mlp=bool(a.get("unet_region_size_mlp", False)),
                                 dec_ffn=bool(a.get("unet_dec_ffn", False)),
                                 pull_k=int(a.get("unet_pull_k", 7)),
                                 pull_query=a.get("unet_pull_query", "skip"),
                                 pull_value_offset=bool(a.get("unet_pull_value_offset", True)),
                                 share_levels=bool(a.get("unet_share_levels", False)),
                                 share_decoder=bool(a.get("unet_share_decoder", False)),
                                 scale_embed=bool(a.get("unet_scale_embed", False)))
        if self.unet.out_dim != dim:
            raise ValueError(f"[Atomiser_UnetDales] decoder width {self.unet.out_dim} must equal the "
                             f"level-0 width {dim} (cells and D_0 share the vertical blocks)")
        if qdim % head_dim != 0:
            raise ValueError(f"unet_query_dim={qdim} must be divisible by unet_head_dim={head_dim}")
        self.decoder = PointQueryDecoder(self.embed, self.query_dim_in, qdim, self.num_classes, qdim // head_dim,
                                         int(a.get("unet_query_chunk", 16384)))
        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[Atomiser_UnetDales] {n:.1f}M params | vertical={self.vertical} "
              f"(kz={self.hier.kz if not flat else '-'}, vmax={self.hier.vmax}, m={self.hier.m}, "
              f"dec={0 if self.vdec is None else len(self.vdec.blocks)}, partition={self.hier.partition}, "
              f"rope={self.vrope}) "
              f"| levels={levels} dims={dims} "
              f"| K={self.K} | intensity={self.embed.use_intensity}")

    def gate_stats(self):
        od = getattr(self.unet, "odec", None)
        rec = getattr(od, "level_gate", None) if od is not None else None
        if rec:
            return {f"gate_L{l}": (v.item() if v is not None else None) for l, v in sorted(rec.items())}
        return {}

    def forward(self, batch, training=False, return_for_error=False, task=None):
        pts, pmask = batch["points"], batch["point_mask"].bool()
        qp, qm = batch["query_points"], batch["query_mask"].bool()
        qctx = batch["query_ctx_idx"]
        dev = pts.device
        B, N, C = pts.shape
        sensor = batch.get("sensor_idx", None)
        sensor = (torch.zeros(B, dtype=torch.long, device=dev) if sensor is None
                  else torch.as_tensor(sensor, device=dev).long().view(B))
        s_emb = self.embed.sensor(sensor)                                    # [B, 8]

        gen = None
        if not self.training:                                               # deterministic atom subset
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
        with _stage("geom/columns"):
            pg = self.hier(pts, pmask, generator=gen, cells=not self.is_hier)
        self.last_partition_stats = pg.stats                               # None for zslice
        g = self.hier.static(dev)
        L0, Vm = pg.L0, pg.vmax

        if self.is_hier:
            return self._forward_hier(batch, pts, pmask, qp, qm, qctx, s_emb, pg, g, gen)

        # ---- vertical leaf read ----
        flat_p = pts.reshape(B * N, C)
        flat_h = pg.point_h.reshape(B * N)
        off = self.hier.atom_offsets(pg, pts, min_z_extent=0.5)
        desc = self.hier.cell_descriptors(pg).reshape(-1, self.hier.desc_dim)[pg.cell_slot].to(pts.dtype)
        with _stage("enc/leaf_read"):
            cell_lat = self.leaf(flat_p[pg.cell_atoms], flat_h[pg.cell_atoms],
                                 s_emb[pg.cell_batch].unsqueeze(1), off, pg.cell_atoms_mask, desc)
        cells = pg.scatter_cells(cell_lat)                                   # [B, L0, vmax, D]
        if self.vrope == "xyz":                                            # cell centroid, column frame
            cen = g.centers[0].float()
            zpos = torch.stack([(pg.v_xmean - cen[:, 0].view(1, -1, 1)) / self.xy_unit,
                                (pg.v_ymean - cen[:, 1].view(1, -1, 1)) / self.xy_unit,
                                pg.v_zmean / self.z_unit], -1)                # [B, L0, vmax, 3]
        else:
            zpos = pg.v_zmean / self.z_unit                                # [B, L0, vmax]

        # ---- vertical stage + collapse ----
        if self.stage is not None:
            with _stage("enc/vertical_stage"):
                z0, cells_enc = self.stage(cells, pg.vmask, zpos, pg.col_empty, self.empty_emb, g)
        else:
            z0 = cells[:, :, 0] + self.empty_emb * pg.col_empty.unsqueeze(-1).to(cells.dtype)
            cells_enc = None

        # ---- 2D U-Net ----
        with _stage("unet2d"):
            d0 = self.unet(z0, g)                                           # [B, L0, D]

        # ---- uncollapse ----
        if self.vdec is not None:
            cells_dec = self.vdec(d0, cells_enc, pg.vmask, zpos)
        elif cells_enc is not None:
            cells_dec = cells_enc + d0.unsqueeze(2)
        else:
            cells_dec = d0.unsqueeze(2)                                      # vmax = 1

        # ---- queries ----
        with _stage("geom/query_lookup"):
            look = self.hier.lookup_queries(pg, qp, qm, qctx, K=self.K)
        col = look["col"].clamp(min=0)
        own_h = pg.point_h.gather(1, qctx.clamp(min=0))
        far_h = qp[..., PL.Z] - pg.col_ground.gather(1, col)
        qh = torch.where(qctx >= 0, own_h, far_h).masked_fill(qm, 0.0)
        with _stage("query_decoder"):
            preds = self.decoder(qp, qh, s_emb.unsqueeze(1), cells_dec.reshape(B, L0 * Vm, -1), look)
        return {"predictions": preds, "num_latents": L0, "num_cells": pg.num_cells}

    def _forward_hier(self, batch, pts, pmask, qp, qm, qctx, s_emb, pg, g, gen):
        B, N, C = pts.shape
        dev = pts.device
        valid = ~pmask.reshape(-1)
        bidx = torch.arange(B, device=dev).repeat_interleave(N)
        with _stage("geom/voxel_hierarchy"):
            vh = self.vbuild.build(pts[..., [PL.X, PL.Y, PL.Z]].reshape(-1, 3).float(), bidx,
                                   pg.point_col.reshape(-1), valid, B, pg.L0, generator=gen,
                                   point_h=pg.point_h.reshape(-1),
                                   **({"part": batch.get("point_part")} if self.hier_nodes == "ezsp_pre" else {}))
        self.last_partition_stats = vh.stats
        col_xy = g.centers[0].float().repeat(B, 1)                          # [B * L0, 2]
        col_unit = float(g.scales[0].mean())
        with _stage("enc/3d_levels"):
            z0, skips = self.hstage.encode(vh, pts.reshape(B * N, C), pg.point_h.reshape(-1), s_emb,
                                           col_xy, col_unit, self.empty_emb, pg.col_empty)
        with _stage("unet2d"):
            d0 = self.unet(z0, g)                                           # [B, L0, D]
        with _stage("dec/3d_levels"):
            x0 = self.hstage.decode(d0, skips, vh)                          # [n0, d0]
        # ---- queries: own point, then the K nearest 0.5 m nodes (own node first) ----
        Nq = qp.shape[1]
        qflat = qp[..., [PL.X, PL.Y, PL.Z]].reshape(-1, 3).float()
        qb = torch.arange(B, device=dev).repeat_interleave(Nq)
        ctx = qctx.reshape(-1)
        own = torch.where(ctx >= 0, vh.point_node[(qb * N + ctx.clamp(min=0))], torch.full_like(ctx, -1))
        with _stage("geom/query_lookup"):
            nbr, nmask, noff = self.vbuild.query_neighbors(vh, qflat, qb, ~qm.reshape(-1), K=self.K, own=own)
        look = {"nbr": nbr.view(B, Nq, -1), "nbr_mask": nmask.view(B, Nq, -1), "nbr_off": noff.view(B, Nq, -1, 3)}
        col = torch.where(qctx >= 0, pg.point_col.gather(1, qctx.clamp(min=0)), torch.zeros_like(qctx))
        own_h = pg.point_h.gather(1, qctx.clamp(min=0))
        far_h = qp[..., PL.Z] - pg.col_ground.gather(1, col.clamp(min=0))
        qh = torch.where(qctx >= 0, own_h, far_h).masked_fill(qm, 0.0)
        with _stage("query_decoder"):
            preds = self.decoder(qp, qh, s_emb.unsqueeze(1), x0, look)
        return {"predictions": preds, "num_latents": pg.L0, "num_cells": vh.levels[0].n}
