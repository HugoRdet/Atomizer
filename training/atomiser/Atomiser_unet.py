"""
Atomiser_Unet — single-timestep Atomizer with an overlap-based U-Net processor
==============================================================================

Drop-in encoder for Model_BurnScars_Skip (and the other *_Skip trainers):

    from training.atomiser.Atomiser_unet import Atomiser_Unet
    self.encoder = Atomiser_Unet(config=self.config, lookup_table=self.lookup_table)

    result = self.encoder(batch, training=..., return_for_error=...)
    result["predictions"]  -> [B, M, num_classes]   (one row per batch["queries"] row)

Pipeline
    atoms (batch["groups"][*]["tokens"], [B, N, 8])
      -> GeographicPruningUNet: cached hierarchy geometry + up to m atoms per leaf cell
      -> LeafEncoder: atom embedding + relative-offset PE + MLP + cross-attention
      -> AtomizerUNet: local levels -> global bottleneck (once) -> gated-skip decoder;
         every latent-to-latent attention uses 2D RoPE on anchor positions
      -> QueryDecoder: local raw-observation conditioning, then the 9 nearest leaf
         latents (with relative-offset PE), then the classification head.

Token / query columns (TokenBuilder convention)
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]

Every position inside the model is UNIT-FREE (offsets are divided by cell scales),
so x, y can be in pixels or metres.

Memory: the leaf read and the query decoder are processed in chunks, with
activation checkpointing during training (config: unet_leaf_chunk,
unet_query_chunk). This is what keeps 8 x 1.57M atoms and 8 x 100k queries
in memory on BurnScars.

ATOM FEATURES: the spectral (Gaussian integrals), reflectance, resolution and
time encoders are imported from the Atomizer-IO codebase (see AtomEmbedding),
so atoms are encoded exactly as in the baseline. Error predictor: not
implemented (keep use_error_predictor: false).

Config keys (config["Atomiser"], all optional):
    unet_levels (3), unet_dims (list, one width per level; default
    [unet_dim]*levels), unet_dim (512), unet_head_dim (64), unet_depth (2),
    unet_fusion ("gate" | "add"), unet_atom_dim (128), unet_query_dim (256),
    unet_task ("segmentation" | "classification"; classification = encoder only +
    mean pooling, returns predictions [B, num_classes]), unet_pool ("bottleneck" |
    "multiscale" | "cls" | "cls_bottleneck"),
    unet_merge ("cross": one cross-attention from seed queries per transition |
    "group_cls": one CLS token per coarse cell joins the fine level's blocks,
    reading its group one-way, then becomes the coarse latent),
    unet_group_final_read (true: after the level's last block, each group CLS
    reads its group's final features once more; group-CLS rows always use their
    own query projection in the shared blocks), unet_decoder ("attn" | "overlap"), unet_dec_dims (int or list per level; overlap
    decoder only; default = encoder widths), unet_dec_refine (int or list of
    S-1 entries, finest first; local RoPE blocks after each overlap fusion),
    unet_km_train ([[1000,1000],[1500,1500],[2000,2000]]),
    unet_km_eval ([1500,1500]), unet_decoder_neighbors (9),
    unet_leaf_heads / unet_leaf_head_dim (leaf cross-attention; default
    cross_heads (8) heads of size level-0 width / heads),
    unet_leaf_chunk (256 latents), unet_query_chunk (16384 queries),
    unet_encodings_package (package of spectral_encodings.py etc.;
    default training.utils.token_building),
    unet_temporal ("none": joint temporal encoding / single date | "per_level":
    per-date encoding + one small TemporalTransformer per level, decoder on
    the aggregated maps; expects tokens [B, T, N, 8]), unet_temporal_transformer
    (n_head 4, n_layers 1, dim_feedforward 2 x width, dropout 0.1, ...),
    unet_atom_features ("atomizer": Atomizer-IO TokenProcessor encoders |
    "legacy": learned embeddings + leaf offset MLP, raw x/y columns; uses
    unet_atom_dim (128), leaf heads atom_dim // 32),
    unet_region_size_mlp (false | true: legacy coarse-query size MLP),
    unet_fusion also accepts "gate_sm" (overlap decoder: gate sees [skip; message]),
    unet_dec_ffn (false | true: FFN after each overlap fusion),
    unet_decoder: pull — each fine cell pulls its unet_pull_k (7) nearest coarse
    decoder states (RoPE in coarse cell units); unet_pull_query: skip |
    skip_plus_transport (overlap message in query and residual, zero-init output);
    unet_pull_value_offset (true: Fourier offset added to the values),
    unet_dropout (0.0),
    unet_share_levels (false | true: ONE level processor = the local blocks shared by levels
    0..S-2, ONE downsampler = the merge shared by every transition, one region-size MLP and
    one coarse seed; the global bottleneck keeps its own weights; needs equal widths and
    unet_merge: cross), unet_share_decoder (false | true: ONE upsampler = transport, fusion
    gate, FFN and refinement blocks shared by every decoder stage; overlap decoder),
    unet_scale_embed (false | true: tells shared modules which level they process: a
    zero-initialised MLP of log2(mean cell size / level-0 cell size), added to the cells at
    the start of each local level, to the merge queries, and to the decoder state at each
    stage). Sharing reuses ONE module object at every level: per-level code paths are
    unchanged and each shared parameter is counted (and optimised) once.
"""
from __future__ import annotations

import importlib
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from training.atomiser.GeographicPruningUNet import GeographicPruningUNet, HierarchyGeometry
except ImportError:  # local testing
    from GeographicPruningUNet import GeographicPruningUNet, HierarchyGeometry

# U-Net blocks: identical to atomizer_unet.py, inlined so this file only depends
# on GeographicPruningUNet.py.


# =============================================================================
# Small helpers
# =============================================================================
COL_VALUE, COL_X, COL_Y, COL_SPEC, COL_LABEL, COL_QIDX, COL_RES, COL_TIME = range(8)


def fourier(x: torch.Tensor, n_freq: int, max_freq: float) -> torch.Tensor:
    """[..., ] -> [..., 2F+1]  : [x, sin(pi f x), cos(pi f x)] with f in linspace(1, max_freq)."""
    f = torch.linspace(1.0, max_freq, n_freq, device=x.device, dtype=x.dtype)
    a = math.pi * x.unsqueeze(-1) * f
    return torch.cat([x.unsqueeze(-1), a.sin(), a.cos()], -1)


def fourier_2d(d: torch.Tensor, n_freq=16, max_freq=16.0) -> torch.Tensor:
    """[..., 2] -> [..., 2(2F+1)]"""
    return torch.cat([fourier(d[..., 0], n_freq, max_freq), fourier(d[..., 1], n_freq, max_freq)], -1)


def maybe_checkpoint(fn, *args, enabled: bool):
    if enabled:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class TokenColumnGeometry:
    """Minimal geometry object for GeographicPruningUNet: positions = token columns x, y."""

    def get_token_centers(self, tokens):
        return tokens[..., COL_X:COL_Y + 1].float()


class FFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mult * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mult * dim, out_dim or dim))

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Atom embedding — Atomizer-IO's own token encoders
# =============================================================================
# Atom features are computed EXACTLY as TokenProcessor.process_data_for_encoder:
# same encoders (imported from the Atomizer-IO codebase), same order
# [pos, spectral, reflectance, resolution, time], same tokenizer MLP structure and
# output width (tokenizer_hidden_size / tokenizer_nb_layers / tokenizer_out_dim).
#   pos         : build_position_encoder on (atom centre - latent anchor), with
#                 compression_scale = RoPE.cross_compression_scale; token centres
#                 come from Atomizer-IO's SensorGeometry.get_token_centers
#   spectral    : closed-form Gaussian integrals over each band's support
#                 (config["wavelengths_encoding"]); learned vectors for abstract
#                 channels (VV, VH, ...)
#   reflectance : Fourier features of the value (bandvalue_* keys)
#   resolution  : resolution_* keys
#   time        : time_* keys (zeros when time_idx = -1)
# The only difference from Atomizer-IO is which anchor the offset is measured
# from: the U-Net's leaf cell anchor (a Voronoi cell of the hierarchy geometry).

# Atomizer-IO token building lives in training/utils/token_building/
# (geometry.py, positional_encodings.py, spectral_encodings.py, ...).
DEFAULT_ENCODING_PACKAGES = (
    "training.utils.token_building",
)


def _import_encoder_builders(package=None):
    """Find the package holding geometry.py, positional_encodings.py,
    spectral_encodings.py, reflectance_encodings.py, resolution_encodings.py and
    time_encodings.py (the modules TokenProcessor imports relatively)."""
    candidates = [package] if package else list(DEFAULT_ENCODING_PACKAGES)
    errors = []
    for pkg in candidates:
        try:
            mods = {name: importlib.import_module(f"{pkg}.{mod}")
                    for name, mod in (("position", "positional_encodings"),
                                      ("spectral", "spectral_encodings"),
                                      ("reflectance", "reflectance_encodings"),
                                      ("resolution", "resolution_encodings"),
                                      ("time", "time_encodings"))}
            geometry = importlib.import_module(f"{pkg}.geometry")
            print(f"[Atomiser_Unet] atom encoders imported from package '{pkg}'")
            return (mods["position"].build_position_encoder,
                    mods["spectral"].build_spectral_encoder,
                    mods["reflectance"].build_reflectance_encoder,
                    mods["resolution"].build_resolution_encoder,
                    mods["time"].build_time_encoder,
                    geometry.SensorGeometry)
        except (ImportError, AttributeError) as e:
            errors.append(f"  {pkg}: {type(e).__name__}: {e}")
    raise ImportError(
        "[Atomiser_Unet] Could not import the Atomizer-IO token encoders "
        "(spectral_encodings.py etc.). Set Atomiser.unet_encodings_package to the "
        "package that contains them (default: training.utils.token_building).\n" + "\n".join(errors))


def build_token_mlp(in_dim, hidden_dim, out_dim, num_layers):
    """Same structure as TokenProcessor's build_mlp(norm_output=True)."""
    if num_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
    layers += [nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)


def _flat_encode(encoder, x):
    """Call an Atomizer-IO sub-encoder on a FLAT index/value tensor and restore
    the leading shape: [...] -> [..., d]."""
    lead = x.shape
    out = encoder(x.reshape(-1))
    if out.dim() == 1:                       # encoder returned [N] for 1-dim features
        out = out.unsqueeze(-1)
    out = out.reshape(-1, out.shape[-1])
    if out.shape[0] != x.numel():
        raise RuntimeError(f"[Atomiser_Unet] {type(encoder).__name__} returned shape "
                           f"{tuple(out.shape)} for {x.numel()} inputs")
    return out.reshape(*lead, out.shape[-1])


class AtomEmbedding(nn.Module):
    """Raw [..., 8] token + anchor position -> [..., tokenizer_out_dim], computed as
    TokenProcessor.process_data_for_encoder does."""

    def __init__(self, config, lookup_table, package=None):
        super().__init__()
        if lookup_table is None:
            raise ValueError("[Atomiser_Unet] lookup_table is required for the Atomizer-IO "
                             "token encoders")
        b_pos, b_spec, b_refl, b_res, b_time, SensorGeometry = _import_encoder_builders(package)
        self.sensor_geometry = SensorGeometry(config, lookup_table)
        self.pos_encoder = b_pos(config)
        self.compression_scale = config["RoPE"].get("cross_compression_scale", 50.0)
        self.spectral_encoder = b_spec(config, lookup_table)
        self.reflectance_encoder = b_refl(config)
        self.resolution_encoder = b_res(config, lookup_table)
        self.time_encoder = b_time(config, lookup_table)
        a = config["Atomiser"]
        self.out_dim = int(a["tokenizer_out_dim"])
        in_dim = (self.pos_encoder.out_dim + self.spectral_encoder.out_dim
                  + self.reflectance_encoder.out_dim + self.resolution_encoder.out_dim
                  + self.time_encoder.out_dim)
        self.mlp = build_token_mlp(in_dim, int(a["tokenizer_hidden_size"]), self.out_dim,
                                   int(a["tokenizer_nb_layers"]))
        print(f"[Atomiser_Unet] atom features: pos={self.pos_encoder.out_dim} "
              f"(compression_scale={self.compression_scale}) + spectral={self.spectral_encoder.out_dim} + "
              f"reflectance={self.reflectance_encoder.out_dim} + "
              f"resolution={self.resolution_encoder.out_dim} + time={self.time_encoder.out_dim} "
              f"= {in_dim} -> MLP -> {self.out_dim}")

    def centers(self, t):
        """Token centres from Atomizer-IO's SensorGeometry: [..., 8] -> [..., 2]."""
        lead = t.shape[:-1]
        xy = self.sensor_geometry.get_token_centers(t.reshape(1, -1, t.shape[-1]))
        return xy.reshape(*lead, 2).float()

    def forward(self, t, anchor_xy):
        """t: [..., 8] tokens; anchor_xy: anchor positions broadcastable to [..., 2]
        (same units as SensorGeometry). Returns [..., tokenizer_out_dim]."""
        d = self.centers(t) - anchor_xy
        dx, dy = d[..., 0], d[..., 1]
        lead = dx.shape
        pos = self.pos_encoder(dx.reshape(-1), dy.reshape(-1),
                               compression_scale=self.compression_scale)
        pos = pos.reshape(dx.numel(), -1).reshape(*lead, -1)
        feats = [
            pos,
            _flat_encode(self.spectral_encoder, t[..., COL_SPEC].long()),
            _flat_encode(self.reflectance_encoder, t[..., COL_VALUE].float()),
            _flat_encode(self.resolution_encoder, t[..., COL_RES].long()),
            _flat_encode(self.time_encoder, t[..., COL_TIME].long()),
        ]
        return self.mlp(torch.cat([f.float() for f in feats], -1))


# =============================================================================
# LEGACY atom embedding (unet_atom_features: legacy) — the version used before
# the switch to Atomizer-IO's encoders (e.g. the BurnScars 90.29 run):
#   value       : Fourier features [v, sin, cos] (8 frequencies up to 8)
#   spectral    : learned embedding of spectral_idx (32-d)
#   resolution  : learned embedding of resolution_idx (8-d)
#   time        : learned embedding of time_idx + 1 (16-d, 0 = not available)
#   -> Linear -> unet_atom_dim (default 128). No position here: the legacy leaf
#   read adds Fourier features of (atom - anchor) / cell scale itself, and
#   positions are the raw token columns x, y (no SensorGeometry).
# =============================================================================
class LegacyAtomEmbedding(nn.Module):
    def __init__(self, atom_dim=128, n_spec=512, n_res=64, n_time=512):
        super().__init__()
        self.out_dim = atom_dim
        self.n_spec, self.n_res, self.n_time = n_spec, n_res, n_time
        self.spec = nn.Embedding(n_spec, 32)
        self.res = nn.Embedding(n_res, 8)
        self.time = nn.Embedding(n_time + 1, 16)
        self.proj = nn.Linear(17 + 32 + 8 + 16, atom_dim)
        self.sensor_geometry = TokenColumnGeometry()
        print(f"[Atomiser_Unet] atom features: LEGACY learned embeddings -> {atom_dim}")

    def centers(self, t):
        return t[..., COL_X:COL_Y + 1].float()

    def forward(self, t, anchor_xy=None):                 # anchor unused (legacy)
        v = fourier(t[..., COL_VALUE].float(), 8, 8.0)
        sp = self.spec(t[..., COL_SPEC].long().clamp(0, self.n_spec - 1))
        r = self.res(t[..., COL_RES].long().clamp(0, self.n_res - 1))
        tm = self.time((t[..., COL_TIME].long() + 1).clamp(0, self.n_time))
        return self.proj(torch.cat([v, sp, r, tm], -1))


# =============================================================================
# Leaf encoder: atoms of one Voronoi cell -> one latent
# =============================================================================
class LeafEncoder(nn.Module):
    """Atoms of one leaf Voronoi cell -> one latent. Atom features (including the
    positional encoding of the offset to the cell anchor) come from AtomEmbedding,
    exactly as Atomizer-IO's TokenProcessor computes them, and go straight into
    cross-attention from a shared learned seed.

    The attention works in an inner space of heads x head_dim, independent of the
    atom width: atom features [atom_dim] -> keys / values [heads * head_dim] ->
    output [dim]. Defaults: heads = cross_heads (Atomizer-IO: 8), head_dim =
    dim // heads, i.e. the inner width equals the level-0 latent width."""

    def __init__(self, embed: AtomEmbedding, atom_dim, dim, heads=8, head_dim=64, chunk=256,
                 legacy_offsets=False):
        super().__init__()
        self.embed = embed
        self.h, self.dh = heads, head_dim
        inner = heads * head_dim
        self.chunk = chunk
        # legacy: z = MLP([atom feature ; Fourier((atom - anchor) / cell scale)])
        self.legacy_offsets = legacy_offsets
        if legacy_offsets:
            pe_dim = 2 * (2 * 16 + 1)
            self.mlp = nn.Sequential(nn.Linear(atom_dim + pe_dim, 2 * atom_dim), nn.GELU(),
                                     nn.Linear(2 * atom_dim, atom_dim))
        self.seed = nn.Parameter(torch.randn(dim) * 0.02)
        self.q = nn.Linear(dim, inner)
        self.ln_kv = nn.LayerNorm(atom_dim)
        self.kv = nn.Linear(atom_dim, 2 * inner)
        self.o = nn.Linear(inner, dim)
        self.ffn = FFN(dim)

    def _cells(self, tok, bias, centers, scales):
        # tok [B, l, m, 8], bias [B, l, m], centers [l, 2], scales [l]
        B, l, m, _ = tok.shape
        if self.legacy_offsets:
            f = self.embed(tok)
            d = (tok[..., COL_X:COL_Y + 1].float() - centers[None, :, None]) / scales[None, :, None, None]
            z = self.mlp(torch.cat([f, fourier_2d(d)], -1))                      # [B, l, m, A]
        else:
            z = self.embed(tok, centers[None, :, None])                          # [B, l, m, A]
        k, v = self.kv(self.ln_kv(z)).view(B, l, m, 2, self.h, self.dh).unbind(3)
        q = self.q(self.seed).view(self.h, self.dh)
        s = torch.einsum("hd,blmhd->blmh", q, k) / math.sqrt(self.dh) + bias.unsqueeze(-1)
        a = s.softmax(dim=2)
        out = torch.einsum("blmh,blmhd->blhd", a, v).reshape(B, l, -1)
        x = self.seed + self.o(out)
        return x + self.ffn(x)

    def forward(self, tok, bias, geom: HierarchyGeometry):
        L = tok.shape[1]
        outs = []
        for s in range(0, L, self.chunk):
            e = min(s + self.chunk, L)
            outs.append(maybe_checkpoint(self._cells, tok[:, s:e], bias[:, s:e],
                                         geom.centers[0][s:e], geom.scales[0][s:e],
                                         enabled=self.training and torch.is_grad_enabled()))
        return torch.cat(outs, 1)                                                 # [B, L0, D]


# =============================================================================
# U-Net processor — positions enter ONLY through 2D RoPE
# =============================================================================
# Every latent-to-latent attention (local self-attention, merge, upsampling,
# bottleneck) rotates queries and keys by their anchor positions, so attention
# scores depend only on relative offsets. Coordinates are expressed in units of
# the mean cell size of the level whose frame is used:
#     self-attention at level l : both sides in level-l cell units
#     merge  l -> l+1           : coarse frame (level l+1 cell units)
#     upsample l+1 -> l         : fine frame   (level l cell units)
# so a one-cell offset has the same meaning at every level. Each attention has
# its own learned scale multiplier (init 1), like the baseline's learnable
# RoPE compression scale.

class RoPE2D(nn.Module):
    """Axial 2D rotary embedding. Half the rotated pairs encode the first
    coordinate, half the second. Frequencies span pi (period 2 cells) down to
    pi / base (period 2*base cells). Learned scale multiplier, init 1."""

    def __init__(self, head_dim, base=100.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
        nf = head_dim // 4
        self.register_buffer("freqs", math.pi * base ** (-torch.arange(nf, dtype=torch.float32) / nf),
                             persistent=False)
        self.log_mult = nn.Parameter(torch.zeros(()))

    def forward(self, x, coords):
        # x [B, L, h, dh], coords [L, 2] (cell units, float32)
        p = coords.float() * self.log_mult.exp()
        ang = torch.cat([p[:, :1] * self.freqs, p[:, 1:] * self.freqs], -1)    # [L, dh/2]
        cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
        return out.to(x.dtype)


class GatherAttention(nn.Module):
    """Each query attends over a padded per-query key table. Q and K are projected
    and RoPE-rotated ONCE per node, then gathered (no per-edge geometry work)."""

    def __init__(self, dim, heads=8, dropout=0.0, self_attn=False, kv_dim=None, cls_query=False):
        super().__init__()
        kv_dim = dim if (self_attn or kv_dim is None) else kv_dim
        self.h, self.dh = heads, dim // heads
        self.self_attn = self_attn
        self.ln_q = nn.LayerNorm(dim)
        self.ln_k = None if self_attn else nn.LayerNorm(kv_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)        # keys projected into the QUERY width
        self.o = nn.Linear(dim, dim)
        self.rope = RoPE2D(self.dh)
        self.drop = nn.Dropout(dropout)
        # group-CLS tokens have their own query projection (their role, summarising
        # a group, differs from the fine latents'); keys, values, norms, output
        # projection and FFN stay shared
        self.cls_query = cls_query
        if cls_query:
            assert self_attn, "cls_query needs a self-attention"
            self.q_cls = nn.Linear(dim, dim)

    def forward(self, xq, xk, idx, valid, cq, ck, n_main=None):
        # cq [Lq, 2], ck [Lk, 2]: anchor positions in the frame of this attention.
        # n_main: with cls_query, tokens [n_main:] are group CLS and use q_cls.
        B, Lq, D = xq.shape
        hq = self.ln_q(xq)
        hk = hq if self.self_attn else self.ln_k(xk)
        split = self.cls_query and n_main is not None and n_main < Lq
        if split:
            q = torch.cat([self.q(hq[:, :n_main]), self.q_cls(hq[:, n_main:])], 1)
        else:
            q = self.q(hq)
        q = self.rope(q.view(B, Lq, self.h, self.dh), cq)
        k, v = self.kv(hk).view(B, -1, 2, self.h, self.dh).unbind(2)
        k = self.rope(k, ck)
        k, v = k[:, idx], v[:, idx]                                   # [B, Lq, K, h, dh]
        s = torch.einsum("bqhd,bqkhd->bqkh", q, k) / math.sqrt(self.dh)
        if valid is not None:
            s = s.masked_fill(~valid[None, :, :, None], float("-inf"))
        a = self.drop(s.softmax(dim=2))
        return self.o(torch.einsum("bqkh,bqkhd->bqhd", a, v).reshape(B, Lq, D))


class FullAttentionRoPE(nn.Module):
    """Bottleneck: full self-attention, same RoPE as every other latent attention."""

    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.rope = RoPE2D(self.dh)
        self.dropout = dropout

    def forward(self, x, coords, n_extra=0):
        # x: [B, L + n_extra, D]; the last n_extra tokens (e.g. CLS) have no
        # position. Latent-latent scores use RoPE-rotated q/k (relative offsets);
        # every score involving an extra token uses UNROTATED q/k, so no absolute
        # position can leak in through the extra token.
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, L, 3, self.h, self.dh).unbind(2)
        if n_extra > 0:
            n = L - n_extra
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B, h, L, dh]
            s = (qh.float() @ kh.float().transpose(-1, -2)) / math.sqrt(self.dh)  # unrotated everywhere
            qr = self.rope(q[:, :n], coords).transpose(1, 2).float()
            kr = self.rope(k[:, :n], coords).transpose(1, 2).float()
            s[..., :n, :n] = (qr @ kr.transpose(-1, -2)) / math.sqrt(self.dh)    # rotated latent block
            a = s.softmax(-1)
            if self.training and self.dropout > 0:
                a = F.dropout(a, self.dropout)
            out = (a.to(vh.dtype) @ vh).transpose(1, 2).reshape(B, L, D)
            return self.o(out)
        q, k = self.rope(q, coords), self.rope(k, coords)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class LocalBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, cls_query=False):
        super().__init__()
        self.attn = GatherAttention(dim, heads, dropout, self_attn=True, cls_query=cls_query)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(self, x, idx, coords, valid=None, n_main=None):
        x = x + self.attn(x, x, idx, valid, coords, coords, n_main)
        return x + self.ffn(x)


class GroupFinalRead(nn.Module):
    """After the last block of a level, each group CLS reads its group once more,
    now from the FINAL fine features E_l: CLS <- CLS + Attn(CLS -> {self, group}),
    then CLS <- CLS + FFN(CLS). Only CLS rows are computed; fine latents are not
    updated (one-way)."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = GatherAttention(dim, heads, dropout)      # separate LN for queries / keys
        self.ffn = FFN(dim, dropout=dropout)

    def forward(self, xx, Lf, idx, valid, coords):
        q = xx[:, Lf:]
        q = q + self.attn(q, xx, idx[Lf:], valid[Lf:], coords[Lf:], coords)
        return q + self.ffn(q)


class GlobalBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = FullAttentionRoPE(dim, heads, dropout)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(self, x, coords, n_extra=0):
        x = x + self.attn(x, coords, n_extra)
        return x + self.ffn(x)


class GatedSkipFusion(nn.Module):
    def __init__(self, dim, mode="gate"):
        super().__init__()
        self.mode = mode
        if mode == "gate":
            self.ln = nn.LayerNorm(2 * dim)
            self.lin = nn.Linear(2 * dim, dim)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        self.last_gate_mean = None

    def forward(self, skip, msg):
        if self.mode == "add":
            return skip + msg
        g = torch.sigmoid(self.lin(self.ln(torch.cat([skip, msg], -1))))
        self.last_gate_mean = g.detach().float().mean()
        return skip + g * msg


def overlap_weights(geom, l):
    """Normalized overlap weights for the upsampling table of transition l:
    w[i, j] = shared sites(fine i, coarse parent j) / sites(fine i). Cached on geom."""
    cache = getattr(geom, "_up_w", None)
    if cache is None:
        cache = geom._up_w = {}
    if l not in cache:
        fine, coarse, cnt = geom.overlap_pairs[l]
        Lc = geom.sizes[l + 1]
        keys = fine * Lc + coarse                                   # sorted ascending (np.unique)
        tab = torch.arange(geom.sizes[l], device=keys.device)[:, None] * Lc + geom.up_idx[l]
        pos = torch.searchsorted(keys, tab).clamp(max=len(keys) - 1)
        w = cnt[pos].float() * geom.up_valid[l].float()
        assert torch.equal(keys[pos][geom.up_valid[l]], tab[geom.up_valid[l]]), "overlap lookup mismatch"
        cache[l] = w / w.sum(-1, keepdim=True)
    return cache[l]


class OverlapFusion(nn.Module):
    """mode="gate"   : D_i = s_i + sigmoid(g(s_i)) * m_i       (gate from the skip only)
    mode="gate_sm": D_i = s_i + sigmoid(g([s_i; m_i])) * m_i  (gate sees skip AND message)
    mode="add"    : D_i = s_i + m_i
    Gates are zero-initialised, so they open at 0.5."""

    def __init__(self, dim, mode="gate"):
        super().__init__()
        assert mode in ("gate", "gate_sm", "add"), mode
        self.mode = mode
        if mode in ("gate", "gate_sm"):
            w = 2 * dim if mode == "gate_sm" else dim
            self.ln = nn.LayerNorm(w)
            self.lin = nn.Linear(w, dim)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        self.last_gate_mean = None

    def forward(self, skip, msg):
        if self.mode == "add":
            return skip + msg
        inp = torch.cat([skip, msg], -1) if self.mode == "gate_sm" else skip
        g = torch.sigmoid(self.lin(self.ln(inp)))
        self.last_gate_mean = g.detach().float().mean()
        return skip + g * msg


class OverlapTransportDecoder(nn.Module):
    """Decoder without query-key scoring between levels.

        s_l     = P_l E_l                               (project saved skips once)
        D_{S-1} = s_{S-1}
        m_l,i   = sum_j  w_ij  W_l D_{l+1,j}            (w = normalized shared-site overlap)
        D_l,i   = s_l,i + sigmoid(g_l(s_l,i)) * m_l,i     (unet_fusion: gate)
                  s_l,i + m_l,i                         (unet_fusion: add)
        D_l     = refine_l(D_l)                         (0+ local RoPE blocks, per level)
    """

    def __init__(self, dims, dec_dims, refine, head_dim=64, dropout=0.0, fusion="gate", ffn=False):
        super().__init__()
        S = len(dims)
        self.S = S
        # optional FFN after each fusion (unet_dec_ffn), before refinement
        self.ffn = (nn.ModuleList(FFN(dec_dims[l], dropout=dropout) for l in range(S - 1))
                    if ffn else None)
        self.proj = nn.ModuleList(
            nn.Identity() if dims[l] == dec_dims[l] else
            nn.Sequential(nn.LayerNorm(dims[l]), nn.Linear(dims[l], dec_dims[l]))
            for l in range(S))
        self.transport = nn.ModuleList(nn.Sequential(nn.LayerNorm(dec_dims[l + 1]),
                                                     nn.Linear(dec_dims[l + 1], dec_dims[l]))
                                       for l in range(S - 1))
        self.fuse = nn.ModuleList(OverlapFusion(dec_dims[l], fusion) for l in range(S - 1))
        self.refine = nn.ModuleList(
            nn.ModuleList(LocalBlock(dec_dims[l], max(1, dec_dims[l] // head_dim), dropout)
                          for _ in range(refine[l]))
            for l in range(S - 1))
        self.scale_emb = None                    # set by AtomizerUNet with unet_scale_embed
        self.level_gate = {}                     # gate mean per level (also when shared)

    def forward(self, skips, geom, cell_coords, return_levels=False, start=None):
        """start=None: the full decoder. start=l (< S-1): decode from level l only -- D_l is
        level l's own features with NO message from above (its stage still runs: fusion with a
        zero message, FFN, refinement), then stages l-1 .. 0 as usual. Levels above l are not
        read (the 'student' path of intra-loop self-distillation; = ablation keep_L0-l)."""
        if start is None or start >= self.S - 1:
            d, top, record = self.proj[-1](skips[-1]), self.S - 1, True
        else:
            d, top, record = None, int(start) + 1, False
        levels = {} if d is None else {self.S - 1: d}
        for l in reversed(range(top)):
            s = self.proj[l](skips[l])
            if d is None:                                             # first stage of a student pass
                m = torch.zeros_like(s)
            else:
                t = self.transport[l](d)                              # once per coarse latent
                w = overlap_weights(geom, l).to(t.dtype)              # [L_l, Kp]
                m = (t[:, geom.up_idx[l]] * w[None, :, :, None]).sum(2)   # [B, L_l, d_l]
            x = self.fuse[l](s, m)
            if record:
                self.level_gate[l] = self.fuse[l].last_gate_mean
            if self.scale_emb is not None:
                x = x + self.scale_emb(scale_feature(geom, l)).to(x.dtype)
            if self.ffn is not None:
                x = x + self.ffn[l](x)
            c = cell_coords(geom, l, l)
            for blk in self.refine[l]:
                x = blk(x, geom.local_idx[l], c)
            d = x
            levels[l] = d
        if return_levels:
            return d, [levels[l] for l in sorted(levels)]
        return d


def pull_tables(geom, l, K):
    """For every fine cell of level l, its K nearest level-(l+1) anchors (by anchor
    distance; independent of site counts). Returns idx [L_l, K'], valid [L_l, K'],
    K' = min(K, L_{l+1}). Cached on the geometry."""
    cache = getattr(geom, "_pull_knn", None)
    if cache is None:
        cache = geom._pull_knn = {}
    if (l, K) not in cache:
        f, c = geom.centers[l].float(), geom.centers[l + 1].float()
        Kp = min(K, c.shape[0])
        idx = torch.cat([torch.cdist(f[s:s + 4096], c).topk(Kp, dim=-1, largest=False).indices
                         for s in range(0, f.shape[0], 4096)], 0)
        cache[(l, K)] = (idx, torch.ones_like(idx, dtype=torch.bool))
    return cache[(l, K)]


class PullAttention(nn.Module):
    """Fine queries read their K nearest coarse decoder states.
    RoPE on queries / keys in COARSE cell units; optional Fourier offset
    (coarse anchor - fine anchor, in coarse cell units) added to the values.
    zero_out: zero-initialised output projection (the attention starts silent)."""

    def __init__(self, dim, kv_dim, heads, dropout=0.0, value_offset=True, zero_out=False):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.ln_q, self.ln_k = nn.LayerNorm(dim), nn.LayerNorm(kv_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        if zero_out:
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)
        self.rope = RoPE2D(self.dh)
        self.pe_v = nn.Linear(2 * (2 * 16 + 1), dim) if value_offset else None
        self.drop = nn.Dropout(dropout)

    def forward(self, xq, xk, idx, valid, cq, ck):
        B, Lq, D = xq.shape
        q = self.rope(self.q(self.ln_q(xq)).view(B, Lq, self.h, self.dh), cq)
        k, v = self.kv(self.ln_k(xk)).view(B, -1, 2, self.h, self.dh).unbind(2)
        k = self.rope(k, ck)                                          # once per coarse latent
        k, v = k[:, idx], v[:, idx]                                   # [B, Lq, K, h, dh]
        if self.pe_v is not None:
            off = (ck[idx] - cq.unsqueeze(1)).to(xq.dtype)            # [Lq, K, 2] coarse units
            v = v + self.pe_v(fourier_2d(off)).view(1, Lq, -1, self.h, self.dh)
        s = torch.einsum("bqhd,bqkhd->bqkh", q, k) / math.sqrt(self.dh)
        s = s.masked_fill(~valid[None, :, :, None], float("-inf"))
        a = self.drop(s.softmax(dim=2))
        return self.o(torch.einsum("bqkh,bqkhd->bqhd", a, v).reshape(B, Lq, D))


class PullDecoder(nn.Module):
    """Decoder in which each fine cell pulls coarse context (unet_decoder: pull).

        s_l     = P_l E_l,   D_{S-1} = s_{S-1}
        query   q_i = s_{l,i}                       (unet_pull_query: skip)
                q_i = s_{l,i} + m_{l,i}             (skip_plus_transport; m = overlap message,
                                                     kept in the residual; W_o zero-init, so
                                                     the decoder starts as transport + FFN)
        x_i     = q_i + PullAttn(q_i, {D_{l+1,j} : j in kNN_K(anchor i)})
        D_l     = refine_l(x + FFN(x))
    """

    def __init__(self, dims, dec_dims, refine, head_dim=64, dropout=0.0, K=7,
                 query="skip", value_offset=True):
        super().__init__()
        assert query in ("skip", "skip_plus_transport"), query
        S = self.S = len(dims)
        self.K, self.query = K, query
        self.proj = nn.ModuleList(
            nn.Identity() if dims[l] == dec_dims[l] else
            nn.Sequential(nn.LayerNorm(dims[l]), nn.Linear(dims[l], dec_dims[l]))
            for l in range(S))
        prior = query == "skip_plus_transport"
        self.transport = (nn.ModuleList(nn.Sequential(nn.LayerNorm(dec_dims[l + 1]),
                                                      nn.Linear(dec_dims[l + 1], dec_dims[l]))
                                        for l in range(S - 1)) if prior else None)
        self.attn = nn.ModuleList(
            PullAttention(dec_dims[l], dec_dims[l + 1], max(1, dec_dims[l] // head_dim), dropout,
                          value_offset=value_offset, zero_out=prior)
            for l in range(S - 1))
        self.ffn = nn.ModuleList(FFN(dec_dims[l], dropout=dropout) for l in range(S - 1))
        self.refine = nn.ModuleList(
            nn.ModuleList(LocalBlock(dec_dims[l], max(1, dec_dims[l] // head_dim), dropout)
                          for _ in range(refine[l]))
            for l in range(S - 1))

    def forward(self, skips, geom, cell_coords, return_levels=False):
        d = self.proj[-1](skips[-1])
        levels = {self.S - 1: d}
        for l in reversed(range(self.S - 1)):
            q = self.proj[l](skips[l])
            if self.transport is not None:                          # geometric prior
                t = self.transport[l](d)
                w = overlap_weights(geom, l).to(t.dtype)
                q = q + (t[:, geom.up_idx[l]] * w[None, :, :, None]).sum(2)
            idx, valid = pull_tables(geom, l, self.K)
            x = q + self.attn[l](q, d, idx, valid,
                                 cell_coords(geom, l, l + 1), cell_coords(geom, l + 1, l + 1))
            x = x + self.ffn[l](x)
            c = cell_coords(geom, l, l)
            for blk in self.refine[l]:
                x = blk(x, geom.local_idx[l], c)
            d = x
            levels[l] = d
        return (d, [levels[l] for l in range(self.S)]) if return_levels else d


def group_cls_tables(geom, l):
    """Joint attention table for level l with one group-CLS token per coarse cell
    of level l+1. Tokens are [fine latents (L_l) ; group CLS (L_{l+1})].
        fine row i     : its 7 nearest fine neighbours (unchanged; never a CLS)
        CLS row L_l+j  : itself + the fine latents of group j (cells overlapping
                         coarse cell j, from the cached merge table)
    Returns idx [L_l + L_{l+1}, K], valid (same shape), coords [L_l + L_{l+1}, 2]
    in level-l cell units. Cached on the geometry."""
    cache = getattr(geom, "_group_cls", None)
    if cache is None:
        cache = geom._group_cls = {}
    if l not in cache:
        Lf, Lc = geom.sizes[l], geom.sizes[l + 1]
        loc, mi, mv = geom.local_idx[l], geom.merge_idx[l], geom.merge_valid[l]
        K = max(loc.shape[1], 1 + mi.shape[1])
        dev = loc.device
        idx = torch.zeros(Lf + Lc, K, dtype=torch.long, device=dev)
        val = torch.zeros(Lf + Lc, K, dtype=torch.bool, device=dev)
        idx[:Lf, :loc.shape[1]] = loc
        val[:Lf, :loc.shape[1]] = True
        idx[Lf:, 0] = torch.arange(Lf, Lf + Lc, device=dev)          # self
        val[Lf:, 0] = True
        idx[Lf:, 1:1 + mi.shape[1]] = mi
        val[Lf:, 1:1 + mi.shape[1]] = mv
        unit = geom.scales[l].mean()
        coords = torch.cat([geom.centers[l], geom.centers[l + 1]], 0) / unit
        cache[l] = (idx, val, coords)
    return cache[l]


def scale_feature(geom, lvl):
    """[1] tensor: log2(mean cell size of level lvl / mean cell size of level 0)."""
    return torch.log2(geom.scales[lvl].float().mean() / geom.scales[0].float().mean()).view(1)


class ScaleEmbedding(nn.Module):
    """MLP(log2 relative cell size) -> width, ZERO-initialised output: the model starts
    exactly as plain weight sharing and learns how much to tell each level its scale."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return self.net(feat)


class AtomizerUNet(nn.Module):
    """dims[l] = width at level l (level 0 finest). Heads per level = dims[l] // head_dim.
    Merges read fine keys (dims[l]) into coarse queries (dims[l+1]); upsampling reads
    coarse keys (dims[l+1]) into fine queries (dims[l]); skips stay at their own width.

    New coarse queries = the level's learned seed, identical for every cell. They
    carry no position: position enters only through RoPE."""

    def __init__(self, dims=(512, 512, 512), head_dim=64, depth=2, dropout=0.0, fusion="gate",
                 decoder="attn", dec_dims=None, dec_refine=0, merge="cross", group_final_read=True,
                 region_size_mlp=False, dec_ffn=False, pull_k=7, pull_query="skip",
                 pull_value_offset=True, share_levels=False, share_decoder=False, scale_embed=False):
        super().__init__()
        dims = list(dims)
        assert merge in ("cross", "group_cls"), merge
        self.merge_mode = merge
        S = self.S = len(dims)
        self.dims = dims
        self.decoder_mode = decoder
        H = [max(1, d // head_dim) for d in dims]
        for d, h in zip(dims, H):
            assert d % h == 0 and (d // h) % 4 == 0, f"width {d} / {h} heads must give head_dim % 4 == 0"
        cq = merge == "group_cls"                 # group CLS rows get their own query projection
        self.enc = nn.ModuleList(
            nn.ModuleList(LocalBlock(dims[l], H[l], dropout, cls_query=cq) for _ in range(depth)) if l < S - 1 else
            nn.ModuleList(GlobalBlock(dims[l], H[l], dropout) for _ in range(depth))
            for l in range(S))
        if merge == "cross":
            # one cross-attention per transition: coarse seed queries read their children
            self.merge = nn.ModuleList(GatherAttention(dims[l + 1], H[l + 1], dropout, kv_dim=dims[l])
                                       for l in range(S - 1))
        else:
            # group CLS: tokens live at the FINE width during level l, then are
            # projected to the next level's width
            self.group_seeds = nn.ParameterList(nn.Parameter(torch.randn(dims[l]) * 0.02)
                                                for l in range(S - 1))
            self.group_proj = nn.ModuleList(
                nn.Identity() if dims[l] == dims[l + 1] else
                nn.Sequential(nn.LayerNorm(dims[l]), nn.Linear(dims[l], dims[l + 1]))
                for l in range(S - 1))
            self.group_final = (nn.ModuleList(GroupFinalRead(dims[l], H[l], dropout)
                                              for l in range(S - 1))
                                if group_final_read else None)
        if decoder == "attn":
            self.dec = nn.ModuleList(nn.ModuleList(LocalBlock(dims[l], H[l], dropout) for _ in range(depth))
                                     for l in range(S - 1))
            self.up = nn.ModuleList(GatherAttention(dims[l], H[l], dropout, kv_dim=dims[l + 1])
                                    for l in range(S - 1))
            self.fuse = nn.ModuleList(GatedSkipFusion(dims[l], fusion) for l in range(S - 1))
            self.out_dim = dims[0]
        elif decoder in ("overlap", "pull"):
            dd = list(dims) if dec_dims is None else (
                [int(dec_dims)] * S if isinstance(dec_dims, (int, float)) else [int(x) for x in dec_dims])
            assert len(dd) == S, f"unet_dec_dims needs {S} entries"
            rf = [int(dec_refine)] * (S - 1) if isinstance(dec_refine, (int, float)) else [int(x) for x in dec_refine]
            assert len(rf) == S - 1, f"unet_dec_refine needs {S - 1} entries (finest first)"
            for d in dd:
                h = max(1, d // head_dim)
                assert d % h == 0 and (d // h) % 4 == 0, f"decoder width {d} incompatible with head_dim {head_dim}"
            if decoder == "overlap":
                self.odec = OverlapTransportDecoder(dims, dd, rf, head_dim, dropout, fusion, ffn=dec_ffn)
                self.fuse = self.odec.fuse                # same gate_stats interface
            else:
                self.odec = PullDecoder(dims, dd, rf, head_dim, dropout, K=pull_k, query=pull_query,
                                        value_offset=pull_value_offset)
                self.fuse = nn.ModuleList()               # no gates
            self.decoder_mode = "overlap"                 # decode() dispatches to self.odec
            self.out_dim = dd[0]
        elif decoder == "none":                         # classification: encoder only
            self.fuse = nn.ModuleList()
            self.out_dim = dims[-1]
        else:
            raise ValueError(f"unet_decoder must be 'attn', 'overlap', 'pull' or 'none', got {decoder!r}")
        # one shared learned seed per coarse level (index 0 unused: level 0 is
        # initialised by the leaf read's own seed)
        self.seeds = nn.ParameterList(nn.Parameter(torch.randn(d) * 0.02) for d in dims)
        # legacy (unet_region_size_mlp: true): seed + MLP(log relative cell size)
        self.region = (nn.ModuleList(nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, d))
                                     for d in dims) if region_size_mlp else None)

        # ---- weight sharing across levels (one module object reused at every level) ----
        self.share_levels, self.share_decoder = bool(share_levels), bool(share_decoder)
        if (share_levels or share_decoder or scale_embed) and len(set(dims)) != 1:
            raise ValueError(f"[AtomizerUNet] sharing / scale embedding need equal widths, got {dims}")
        if share_levels:
            if merge != "cross":
                raise ValueError("[AtomizerUNet] unet_share_levels needs unet_merge: cross")
            if S < 3:
                raise ValueError("[AtomizerUNet] unet_share_levels needs at least 3 levels")
            proc = self.enc[0]                                   # the level processor
            self.enc = nn.ModuleList([proc] * (S - 1) + [self.enc[S - 1]])
            self.merge = nn.ModuleList([self.merge[0]] * (S - 1))  # the downsampler
            self.seeds = nn.ParameterList([self.seeds[1]] * S)     # one coarse seed
            if self.region is not None:
                self.region = nn.ModuleList([self.region[1]] * S)
        if share_decoder:
            if decoder not in ("overlap",):
                raise ValueError("[AtomizerUNet] unet_share_decoder needs unet_decoder: overlap")
            od = self.odec
            od.transport = nn.ModuleList([od.transport[0]] * (S - 1))       # the upsampler
            od.fuse = nn.ModuleList([od.fuse[0]] * (S - 1))
            if od.ffn is not None:
                od.ffn = nn.ModuleList([od.ffn[0]] * (S - 1))
            n_ref = max(len(r) for r in od.refine)
            base = next(r for r in od.refine if len(r) == n_ref)            # shared refinement blocks
            od.refine = nn.ModuleList(nn.ModuleList(list(base)[:len(r)]) for r in od.refine)
            self.fuse = od.fuse
        self.scale_enc = ScaleEmbedding(dims[0]) if scale_embed else None   # cells, each local level
        self.scale_q = ScaleEmbedding(dims[0]) if scale_embed else None     # merge (coarse) queries
        if scale_embed and getattr(self, "odec", None) is not None:
            self.odec.scale_emb = ScaleEmbedding(dims[0])                   # decoder state, each stage

    @staticmethod
    def cell_coords(geom, lvl, frame):
        """Anchors of level `lvl` in units of level `frame`'s mean cell size."""
        return geom.centers[lvl] / geom.scales[frame].mean()

    def region_queries(self, geom, lvl):
        """Initial coarse tokens: the level's seed, identical for every cell (as in
        Atomizer-IO); with region_size_mlp (legacy), plus an MLP of the cell's size
        relative to the level's mean. Position enters only through RoPE."""
        if self.region is None:
            return self.seeds[lvl].expand(geom.sizes[lvl], -1)
        rel = torch.log(geom.scales[lvl] / geom.scales[lvl].mean()).unsqueeze(-1)
        return self.seeds[lvl] + self.region[lvl](rel)

    def encode(self, z0, geom, cls=None, cls_readers=None):
        """Returns the saved per-level features; with `cls` ([B, dims[-1]]), returns
        (skips, cls). cls_readers[l] (optional) lets the CLS read level l after its
        blocks (read-only: latents never attend to the CLS before the bottleneck).
        At the bottleneck the CLS joins full self-attention, unrotated."""
        S, B = self.S, z0.shape[0]
        assert z0.shape[1] == geom.sizes[0]
        x, skips = z0, []
        group = None
        for l in range(S):
            c = self.cell_coords(geom, l, l)
            if l < S - 1 and self.merge_mode == "group_cls":
                # one-way group CLS: every block runs on [fine ; group CLS]; fine
                # rows never attend to a CLS, so fine features are unchanged
                Lf = geom.sizes[l]
                idx, val, cc = group_cls_tables(geom, l)
                g0 = self.group_seeds[l].to(x.dtype).expand(B, geom.sizes[l + 1], -1)
                xx = torch.cat([x, g0], 1)
                for blk in self.enc[l]:
                    xx = blk(xx, idx, cc, val, n_main=Lf)
                x, group = xx[:, :Lf], xx[:, Lf:]
                if self.group_final is not None:              # read the final E_l once more
                    group = self.group_final[l](xx, Lf, idx, val, cc)
                if cls is not None and cls_readers is not None:
                    cls = cls_readers[l](cls, x)
            elif l < S - 1:
                if self.scale_enc is not None:
                    x = x + self.scale_enc(scale_feature(geom, l)).to(x.dtype)
                for blk in self.enc[l]:
                    x = blk(x, geom.local_idx[l], c)
                if cls is not None and cls_readers is not None:
                    cls = cls_readers[l](cls, x)
            elif cls is not None:
                x = torch.cat([x, cls.unsqueeze(1)], 1)
                for blk in self.enc[l]:
                    x = blk(x, c, n_extra=1)
                x, cls = x[:, :-1], x[:, -1]
            else:
                for blk in self.enc[l]:
                    x = blk(x, c)
            skips.append(x)
            if l < S - 1 and self.merge_mode == "group_cls":
                x = self.group_proj[l](group)              # group CLS -> next level's latents
            elif l < S - 1:
                q = self.region_queries(geom, l + 1).expand(B, -1, -1)
                if self.scale_q is not None:
                    q = q + self.scale_q(scale_feature(geom, l + 1)).to(q.dtype)
                x = q + self.merge[l](q, x, geom.merge_idx[l], geom.merge_valid[l],
                                      self.cell_coords(geom, l + 1, l + 1),
                                      self.cell_coords(geom, l, l + 1))
        return skips if cls is None else (skips, cls)

    def decode(self, skips, geom, return_levels=False, start=None):
        """D_0, or (D_0, [D_0, ..., D_{S-1}]) with return_levels (for visualisation).
        start=l: decode from level l only (overlap decoder; see OverlapTransportDecoder)."""
        if self.decoder_mode == "overlap":
            if start is not None and not isinstance(self.odec, OverlapTransportDecoder):
                raise ValueError("decoding from an intermediate level needs unet_decoder: overlap")
            if start is not None:
                return self.odec(skips, geom, self.cell_coords, return_levels, start=start)
            return self.odec(skips, geom, self.cell_coords, return_levels)
        if start is not None:
            raise ValueError("decoding from an intermediate level needs unet_decoder: overlap")
        d = skips[-1]
        levels = {self.S - 1: d}
        for l in reversed(range(self.S - 1)):
            msg = self.up[l](skips[l], d, geom.up_idx[l], geom.up_valid[l],
                             self.cell_coords(geom, l, l), self.cell_coords(geom, l + 1, l))
            x = self.fuse[l](skips[l], msg)
            c = self.cell_coords(geom, l, l)
            for blk in self.dec[l]:
                x = blk(x, geom.local_idx[l], c)
            d = x
            levels[l] = d
        return (d, [levels[l] for l in range(self.S)]) if return_levels else d

    def forward(self, z0, geom):
        return self.decode(self.encode(z0, geom), geom)
# =============================================================================
# Per-level temporal aggregation (unet_temporal: per_level)
# =============================================================================
# Each date is encoded separately through the U-Net encoder (shared weights,
# same cached geometry); at every level, a small temporal transformer
# aggregates the T latents of each cell into one (CLS readout, RoPE on the
# acquisition day) — a SIDE BRANCH: the per-date path continues upward
# unaggregated. The decoder runs once, on the aggregated maps.
# The transformer is Atomizer-IO's TemporalTransformer (Atomiser_temporal.py).

class PerLevelTemporalAggregation(nn.Module):
    def __init__(self, dims, cfg):
        super().__init__()
        from training.atomiser.Atomiser_temporal import TemporalTransformer
        self.levels = nn.ModuleList(
            TemporalTransformer(
                in_channels=d,
                n_head=int(cfg.get("n_head", 4)),
                n_layers=int(cfg.get("n_layers", 1)),
                dim_feedforward=int(cfg.get("dim_feedforward", 2 * d)),
                dropout=float(cfg.get("dropout", 0.1)),
                T=int(cfg.get("max_T", 1000)),
                positional_encoding=bool(cfg.get("positional_encoding", True)),
                rope_compression_scale=float(cfg.get("rope_compression_scale", 50.0)),
                rope_learnable_scale=bool(cfg.get("rope_learnable_scale", True)),
            ) for d in dims)

    def forward(self, skips_t, time_pos, pad):
        """skips_t[l]: [B, T, L_l, D_l]; time_pos: [B, T] (day of year) or None;
        pad: [B, T] True = padded date. Returns [B, L_l, D_l] per level."""
        out = []
        for l, x in enumerate(skips_t):
            B, T, L, D = x.shape
            seq = x.permute(0, 2, 1, 3).reshape(B * L, T, D)
            tp = None if time_pos is None else time_pos.unsqueeze(1).expand(B, L, T).reshape(B * L, T)
            pm = pad.unsqueeze(1).expand(B, L, T).reshape(B * L, T)
            out.append(self.levels[l](seq, time_indices=tp, pad_mask=pm).reshape(B, L, D))
        return out


# =============================================================================
# Query decoder
# =============================================================================
class QueryDecoder(nn.Module):
    """For each query: (1) cross-attend to the raw atoms at its own site (U-Net-like
    local skip, as in Atomizer-IO), (2) cross-attend to its k nearest leaf latents
    with relative-offset PE, (3) FFN + linear head."""

    def __init__(self, embed: AtomEmbedding, atom_dim, dim, qdim, num_classes,
                 heads=8, k_nn=9, chunk=16384):
        super().__init__()
        self.embed, self.k_nn, self.chunk = embed, k_nn, chunk
        assert qdim % heads == 0, f"query width {qdim} not divisible by {heads} heads"
        self.h, self.dh = heads, qdim // heads
        self.q_init = nn.Parameter(torch.randn(qdim) * 0.02)
        # (1) local observation conditioning
        self.obs_ln = nn.LayerNorm(atom_dim)
        self.obs_q = nn.Linear(qdim, qdim)
        self.obs_kv = nn.Linear(atom_dim, 2 * qdim)
        self.obs_o = nn.Linear(qdim, qdim)
        # (2) latent read: K/V projected once per latent, PE added per edge
        self.lat_ln = nn.LayerNorm(dim)
        self.lat_kv = nn.Linear(dim, 2 * qdim)
        pe_dim = 2 * (2 * 16 + 1)
        self.pe_kv = nn.Linear(pe_dim, 2 * qdim)
        self.q_ln = nn.LayerNorm(qdim)
        self.lat_q = nn.Linear(qdim, qdim)
        self.lat_o = nn.Linear(qdim, qdim)
        self.ffn = FFN(qdim)
        self.head = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, num_classes))

    # ---------------------------------------------------------------- site lookup
    @staticmethod
    def site_tables(geom: HierarchyGeometry):
        """Cached on the geometry: integer grid -> site id, and site -> atom indices."""
        if getattr(geom, "_site_grid", None) is not None:
            return geom._site_grid, geom._site_atoms, geom._site_atoms_valid
        dev = geom.site_xy.device
        ij = torch.floor((geom.site_xy - geom.origin) / geom.site_spacing).long()
        R, Cc = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
        grid = torch.full((R, Cc), -1, dtype=torch.long, device=dev)
        grid[ij[:, 0], ij[:, 1]] = torch.arange(len(ij), device=dev)
        a2s = geom.atom_to_site
        n_sites = len(geom.site_xy)
        order = torch.argsort(a2s, stable=True)
        counts = torch.bincount(a2s, minlength=n_sites)
        col = torch.arange(len(order), device=dev) - (torch.cumsum(counts, 0) - counts)[a2s[order]]
        atoms = torch.zeros(n_sites, int(counts.max()), dtype=torch.long, device=dev)
        valid = torch.zeros_like(atoms, dtype=torch.bool)
        atoms[a2s[order], col] = order
        valid[a2s[order], col] = True
        geom._site_grid, geom._site_atoms, geom._site_atoms_valid = grid, atoms, valid
        return grid, atoms, valid

    def query_sites(self, pq, geom):
        grid, _, _ = self.site_tables(geom)
        ij = torch.floor((pq - geom.origin) / geom.site_spacing).long()
        inb = (ij[..., 0] >= 0) & (ij[..., 0] < grid.shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < grid.shape[1])
        sid = grid[ij[..., 0].clamp(0, grid.shape[0] - 1), ij[..., 1].clamp(0, grid.shape[1] - 1)]
        sid = torch.where(inb, sid, torch.full_like(sid, -1))
        ok = sid >= 0
        near = (geom.site_xy[sid.clamp(min=0)] - pq).abs().amax(-1) <= 0.5 * geom.site_spacing + 1e-4
        return torch.where(ok & near, sid, torch.full_like(sid, -1))

    # ---------------------------------------------------------------- one chunk
    def _chunk(self, pq, sid, atoms_tok, atoms_ok, lat_k, lat_v, centers, scales):
        B, M, _ = pq.shape
        q0 = self.q_init.expand(B, M, -1)
        # (1) local observations at the query's site
        f = self.obs_ln(self.embed(atoms_tok, pq.unsqueeze(2)))                  # [B, M, c, A]
        k, v = self.obs_kv(f).view(B, M, -1, 2, self.h, self.dh).unbind(3)
        q = self.obs_q(q0).view(B, M, self.h, self.dh)
        s = torch.einsum("bmhd,bmchd->bmch", q, k) / math.sqrt(self.dh)
        s = s.masked_fill(~atoms_ok.unsqueeze(-1), -1e4)
        a = s.softmax(dim=2) * atoms_ok.any(-1)[..., None, None].to(s.dtype)     # no site -> zero message
        x = q0 + self.obs_o(torch.einsum("bmch,bmchd->bmhd", a, v).reshape(B, M, -1))
        # (2) k nearest leaf latents
        dist = torch.cdist(pq.float(), centers.float().unsqueeze(0).expand(B, -1, -1))
        knn = dist.topk(min(self.k_nn, centers.shape[0]), dim=-1, largest=False).indices   # [B, M, k]
        bidx = torch.arange(B, device=pq.device)[:, None, None]
        d = (centers[knn] - pq.unsqueeze(2)) / scales[knn].unsqueeze(-1)
        pk, pv = self.pe_kv(fourier_2d(d.to(pq.dtype))).chunk(2, -1)
        kk = (lat_k[bidx, knn] + pk).view(B, M, -1, self.h, self.dh)
        vv = (lat_v[bidx, knn] + pv).view(B, M, -1, self.h, self.dh)
        q = self.lat_q(self.q_ln(x)).view(B, M, self.h, self.dh)
        a = (torch.einsum("bmhd,bmkhd->bmkh", q, kk) / math.sqrt(self.dh)).softmax(dim=2)
        x = x + self.lat_o(torch.einsum("bmkh,bmkhd->bmhd", a, vv).reshape(B, M, -1))
        x = x + self.ffn(x)
        return self.head(x)

    @staticmethod
    def temporal_site_tables(geom: HierarchyGeometry, T: int, N: int):
        """Site -> atom indices in a time-flattened pool [T * N] (row = t * N + n):
        a pixel's atoms across all dates. Cached on the geometry per (T, N)."""
        cache = getattr(geom, "_site_atoms_T", None)
        if cache is None:
            cache = geom._site_atoms_T = {}
        if (T, N) not in cache:
            _, atoms, valid = QueryDecoder.site_tables(geom)
            offs = torch.arange(T, device=atoms.device).view(1, T, 1) * N
            cache[(T, N)] = ((atoms.unsqueeze(1) + offs).reshape(atoms.shape[0], -1),
                             valid.unsqueeze(1).expand(-1, T, -1).reshape(valid.shape[0], -1))
        return cache[(T, N)]

    def forward(self, queries, tokens, token_mask, d0, geom, n_dates=1):
        """tokens / token_mask: [B, N, 8] / [B, N], or the time-flattened pool
        [B, T * N, 8] / [B, T * N] with n_dates = T (row = t * N + n)."""
        B, M, _ = queries.shape
        pq = self.embed.centers(queries)                                          # SensorGeometry units
        if n_dates > 1:
            site_atoms, site_valid = self.temporal_site_tables(geom, n_dates,
                                                               tokens.shape[1] // n_dates)
        else:
            _, site_atoms, site_valid = self.site_tables(geom)
        sid = self.query_sites(pq, geom)                                          # [B, M]
        lat_k, lat_v = self.lat_kv(self.lat_ln(d0)).chunk(2, -1)                 # once per latent
        outs = []
        for s in range(0, M, self.chunk):
            e = min(s + self.chunk, M)
            sc = sid[:, s:e]
            aidx = site_atoms[sc.clamp(min=0)]                                    # [B, m, c]
            aok = site_valid[sc.clamp(min=0)] & (sc >= 0).unsqueeze(-1)
            flat = aidx.reshape(B, -1)
            atok = torch.gather(tokens, 1, flat.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
            atok = atok.view(B, e - s, -1, tokens.shape[-1])
            aok = aok & ~torch.gather(token_mask, 1, flat).view_as(aok)
            outs.append(maybe_checkpoint(self._chunk, pq[:, s:e], sc, atok, aok, lat_k, lat_v,
                                         geom.centers[0], geom.scales[0],
                                         enabled=self.training and torch.is_grad_enabled()))
        return torch.cat(outs, 1)                                                 # [B, M, classes]


# =============================================================================
# Classification head
# =============================================================================
class PoolingClassifier(nn.Module):
    """Mean-pool the encoder features and classify.

    pool="bottleneck": mean over the coarsest latents (the only ones that have
                       seen the whole scene).
    pool="multiscale": mean over every level, LayerNorm per level, concatenated.
    Means are taken over latents; cells clipped at the image border count the
    same as full cells.
    """

    def __init__(self, dims, num_classes, pool="bottleneck", dropout=0.0):
        super().__init__()
        assert pool in ("bottleneck", "multiscale"), pool
        self.pool = pool
        used = [dims[-1]] if pool == "bottleneck" else list(dims)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in used)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(sum(used), num_classes))

    def forward(self, skips):
        feats = [skips[-1]] if self.pool == "bottleneck" else skips
        pooled = [ln(f.mean(1)) for ln, f in zip(self.norms, feats)]
        return self.head(torch.cat(pooled, -1))                    # [B, num_classes]


class CLSReader(nn.Module):
    """CLS <- CLS + CrossAttn(CLS, level latents), then CLS <- CLS + FFN(CLS).
    Keys/values are projected from the level width into the CLS width.
    No positional encoding: the CLS has no position."""

    def __init__(self, cls_dim, kv_dim, head_dim=64, dropout=0.0):
        super().__init__()
        self.h = max(1, cls_dim // head_dim)
        self.dh = cls_dim // self.h
        self.ln_q, self.ln_k = nn.LayerNorm(cls_dim), nn.LayerNorm(kv_dim)
        self.q = nn.Linear(cls_dim, cls_dim)
        self.kv = nn.Linear(kv_dim, 2 * cls_dim)
        self.o = nn.Linear(cls_dim, cls_dim)
        self.ffn = FFN(cls_dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, cls, x):
        B, L, _ = x.shape
        q = self.q(self.ln_q(cls)).view(B, 1, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.ln_k(x)).view(B, L, 2, self.h, self.dh).unbind(2)
        out = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2),
                                             dropout_p=self.dropout if self.training else 0.0)
        cls = cls + self.o(out.transpose(1, 2).reshape(B, -1))
        return cls + self.ffn(cls)


class CLSClassifier(nn.Module):
    """pool="cls_bottleneck": a learned CLS token joins the bottleneck's full
    self-attention only.
    pool="cls": the CLS also reads every finer level after its encoder blocks
    (CLSReader), then joins the bottleneck. Classify from the final CLS."""

    def __init__(self, dims, num_classes, pool="cls", head_dim=64, dropout=0.0):
        super().__init__()
        self.pool = pool
        self.token = nn.Parameter(torch.randn(dims[-1]) * 0.02)
        self.readers = (nn.ModuleList(CLSReader(dims[-1], dims[l], head_dim, dropout)
                                      for l in range(len(dims) - 1))
                        if pool == "cls" else None)
        self.head = nn.Sequential(nn.LayerNorm(dims[-1]), nn.Dropout(dropout),
                                  nn.Linear(dims[-1], num_classes))

    def forward(self, unet, z0, geom):
        cls = self.token.expand(z0.shape[0], -1).to(z0.dtype)
        _, cls = unet.encode(z0, geom, cls=cls, cls_readers=self.readers)
        return self.head(cls)


# =============================================================================
# The encoder the trainer instantiates
# =============================================================================
class Atomiser_Unet(nn.Module):
    def __init__(self, config, lookup_table=None):
        super().__init__()
        a = config.get("Atomiser", {})
        self.num_classes = int(config["trainer"]["num_classes"])
        levels = int(a.get("unet_levels", 3))
        dims = a.get("unet_dims", None)
        dims = [int(d) for d in dims] if dims is not None else [int(a.get("unet_dim", 512))] * levels
        if len(dims) != levels:
            raise ValueError(f"unet_dims has {len(dims)} entries but unet_levels={levels}")
        head_dim = int(a.get("unet_head_dim", 64))
        dim = dims[0]                                     # leaf / query-decoder side width
        heads = max(1, dim // head_dim)
        self.atom_features = a.get("unet_atom_features", "atomizer")
        if self.atom_features not in ("atomizer", "legacy"):
            raise ValueError(f"unet_atom_features must be 'atomizer' or 'legacy', got {self.atom_features!r}")
        legacy = self.atom_features == "legacy"
        if not legacy and "unet_atom_dim" in a and int(a["unet_atom_dim"]) != int(a["tokenizer_out_dim"]):
            print(f"[Atomiser_Unet] NOTE: unet_atom_dim={a['unet_atom_dim']} is ignored; atom "
                  f"features use tokenizer_out_dim={a['tokenizer_out_dim']} (as Atomizer-IO)")
        qdim = int(a.get("unet_query_dim", 256))
        self.task = a.get("unet_task", "segmentation")
        if self.task not in ("segmentation", "classification"):
            raise ValueError(f"unet_task must be 'segmentation' or 'classification', got {self.task!r}")
        self.km_train = [tuple(x) for x in a.get("unet_km_train", [[1000, 1000], [1500, 1500], [2000, 2000]])]
        self.km_eval = tuple(a.get("unet_km_eval", [1500, 1500]))

        if legacy:
            # The legacy learned time embedding consumes raw time_idx values. With
            # Atomizer-IO features, TimeEncoder pre-registers every day of year in
            # the lookup table at model construction, so all DataLoader workers
            # share one day -> index mapping. Without it, each worker would register
            # days lazily in its own order (inconsistent indices). Do the same here.
            if lookup_table is not None and hasattr(lookup_table, "get_or_register_time_idx"):
                for doy in range(1, 367):
                    lookup_table.get_or_register_time_idx(doy)
            self.embed = LegacyAtomEmbedding(int(a.get("unet_atom_dim", 128)),
                                             int(a.get("unet_num_spectral", 512)),
                                             int(a.get("unet_num_resolution", 64)),
                                             int(a.get("unet_num_time", 512)))
        else:
            self.embed = AtomEmbedding(config, lookup_table, a.get("unet_encodings_package"))
        atom_dim = self.embed.out_dim
        self.pruning = GeographicPruningUNet(
            self.embed.sensor_geometry, levels=levels,
            atoms_per_latent_choices=sorted({k for k, _ in self.km_train} | {self.km_eval[0]}))
        # leaf read: same shape as Atomizer-IO's cross-attention by default
        # heads: Atomizer-IO's cross_heads; head size: level-0 width / heads by default
        # (an inner width far above the latent width trained poorly in tests).
        # legacy: atom_dim // 32 heads, inner width = atom width
        if legacy:
            leaf_heads = int(a.get("unet_leaf_heads", max(1, atom_dim // 32)))
            leaf_head_dim = int(a.get("unet_leaf_head_dim", atom_dim // leaf_heads))
        else:
            leaf_heads = int(a.get("unet_leaf_heads", a.get("cross_heads", 8)))
            leaf_head_dim = int(a.get("unet_leaf_head_dim", max(1, dim // leaf_heads)))
        self.leaf = LeafEncoder(self.embed, atom_dim, dim, heads=leaf_heads, head_dim=leaf_head_dim,
                                chunk=int(a.get("unet_leaf_chunk", 256)), legacy_offsets=legacy)
        print(f"[Atomiser_Unet] leaf read: {leaf_heads} heads x {leaf_head_dim} "
              f"(atoms {atom_dim}-d -> {leaf_heads * leaf_head_dim}-d attention -> {dim}-d latent)")
        self.empty_emb = nn.Parameter(torch.zeros(dim))
        self.unet = AtomizerUNet(dims, head_dim, int(a.get("unet_depth", 2)),
                                 float(a.get("unet_dropout", 0.0)), a.get("unet_fusion", "gate"),
                                 decoder="none" if self.task == "classification" else a.get("unet_decoder", "attn"),
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
        if a.get("unet_share_levels") or a.get("unet_share_decoder") or a.get("unet_scale_embed"):
            print(f"[Atomiser_Unet] SHARED: level processor + downsampler="
                  f"{bool(a.get('unet_share_levels', False))} | upsampler="
                  f"{bool(a.get('unet_share_decoder', False))} | scale embedding="
                  f"{bool(a.get('unet_scale_embed', False))} | global bottleneck: own weights")
        if self.task == "classification":
            pool = a.get("unet_pool", "bottleneck")
            if pool in ("bottleneck", "multiscale"):
                self.classifier = PoolingClassifier(dims, self.num_classes, pool,
                                                    float(a.get("unet_dropout", 0.0)))
            elif pool in ("cls", "cls_bottleneck"):
                self.classifier = CLSClassifier(dims, self.num_classes, pool, head_dim,
                                                float(a.get("unet_dropout", 0.0)))
            else:
                raise ValueError(f"unet_pool must be bottleneck | multiscale | cls | cls_bottleneck, "
                                 f"got {pool!r}")
            self.decoder = None
        else:
            # decoder heads follow the QUERY width, not the latent width
            if qdim % head_dim != 0:
                raise ValueError(f"unet_query_dim={qdim} must be divisible by unet_head_dim={head_dim}")
            self.decoder = QueryDecoder(self.embed, atom_dim, self.unet.out_dim, qdim, self.num_classes,
                                        qdim // head_dim,
                                        int(a.get("unet_decoder_neighbors", 9)),
                                        int(a.get("unet_query_chunk", 16384)))
        # ---- temporal mode ------------------------------------------------
        self.temporal_mode = a.get("unet_temporal", "none")
        if self.temporal_mode not in ("none", "per_level"):
            raise ValueError(f"unet_temporal must be 'none' or 'per_level', got {self.temporal_mode!r}")
        if self.temporal_mode == "per_level":
            if self.task != "segmentation":
                raise ValueError("unet_temporal: per_level is implemented for segmentation")
            self.temporal = PerLevelTemporalAggregation(
                dims, a.get("unet_temporal_transformer", a.get("transformer", {})))
            print(f"[Atomiser_Unet] temporal: per-level TemporalTransformer "
                  f"({sum(p.numel() for p in self.temporal.parameters()) / 1e6:.2f}M params)")
        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[Atomiser_Unet] {self.task} | {n:.1f}M params | levels={levels} dims={dims} "
              f"km_train={self.km_train} km_eval={self.km_eval} merge={a.get('unet_merge', 'cross')} "
              f"atoms={self.atom_features} region_mlp={bool(a.get('unet_region_size_mlp', False))} " + (
                  f"pool={a.get('unet_pool', 'bottleneck')}" if self.task == "classification" else
                  f"decoder={a.get('unet_decoder', 'attn')}"
                  + ("" if a.get('unet_decoder', 'attn') == 'attn' else
                     f" dec_dims={a.get('unet_dec_dims')} refine={a.get('unet_dec_refine', 0)}")))

    @staticmethod
    def _gather_inputs(batch):
        """[B, N, 8] tokens, or [B, T, N, 8] per-date tokens; groups are
        concatenated along the atom axis."""
        toks, masks = [], []
        for res in sorted(batch["groups"].keys(), key=float):
            g = batch["groups"][res]
            toks.append(g["tokens"])
            masks.append(g["mask"].bool())
        dim = toks[0].dim() - 2
        return torch.cat(toks, dim), torch.cat(masks, dim)

    def gate_stats(self):
        od = getattr(self.unet, "odec", None)
        rec = getattr(od, "level_gate", None) if od is not None else None
        if rec:                                    # per level, also when the gate is shared
            return {f"gate_L{l}": (v.item() if v is not None else None) for l, v in sorted(rec.items())}
        return {f"gate_L{l}": (f.last_gate_mean.item() if f.last_gate_mean is not None else None)
                for l, f in enumerate(self.unet.fuse)}

    def forward(self, batch, training=False, return_for_error=False, task=None):
        # `task` is accepted for trainers that pass it (e.g. Model_ForestNet); it
        # must match the task the model was built for (config unet_task).
        if task is not None and task != self.task:
            raise ValueError(f"[Atomiser_Unet] trainer asked for task={task!r} but the model was "
                             f"built with unet_task={self.task!r}")
        tokens, mask = self._gather_inputs(batch)
        k, m = random.choice(self.km_train) if self.training else self.km_eval

        if tokens.dim() == 4:                                    # [B, T, N, 8]: one set per date
            if self.temporal_mode != "per_level":
                raise ValueError("[Atomiser_Unet] got per-date tokens [B, T, N, 8]; set "
                                 "Atomiser.unet_temporal: per_level (or use joint encoding)")
            return self._forward_temporal(batch, tokens, mask, k, m)
        if self.temporal_mode == "per_level":
            raise ValueError("[Atomiser_Unet] unet_temporal: per_level expects per-date tokens "
                             "[B, T, N, 8] (PastisHDTemporalDataset)")

        tok_sel, _, bias = self.pruning(tokens, mask, None, geo_k=m, atoms_per_latent=k)
        geom = self.pruning.last_geometry
        z0 = self.leaf(tok_sel, bias, geom)
        z0 = z0 + self.empty_emb * self.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
        if self.task == "classification":
            if isinstance(self.classifier, CLSClassifier):
                logits = self.classifier(self.unet, z0, geom)
            else:
                logits = self.classifier(self.unet.encode(z0, geom))
            return {"predictions": logits, "num_latents": geom.sizes[0]}
        d0 = self.unet(z0, geom)
        preds = self.decoder(batch["queries"], tokens, mask, d0, geom)
        return {"predictions": preds, "num_latents": geom.sizes[0]}

    def forward_ilsd(self, batch, student_level, query_idx=None):
        """Intra-loop self-distillation (ELT): ONE encoder pass, two decodes.
        teacher: the full model; student: decoded from level `student_level` only (levels
        above it unused), on the queries `query_idx` ([M_s] indices; None = all).
        Returns {"predictions": teacher [B, M, C], "student_predictions": [B, M_s, C], ...}."""
        tokens, mask = self._gather_inputs(batch)
        k, m = random.choice(self.km_train) if self.training else self.km_eval
        if tokens.dim() == 4:                                    # per-date (per-level temporal)
            if self.temporal_mode != "per_level":
                raise ValueError("[Atomiser_Unet] per-date tokens need unet_temporal: per_level")
            B, T, N, C = tokens.shape
            tok_sel, _, bias = self.pruning(tokens.reshape(B * T, N, C), mask.reshape(B * T, N),
                                            None, geo_k=m, atoms_per_latent=k)
            geom = self.pruning.last_geometry
            z0 = self.leaf(tok_sel, bias, geom)
            z0 = z0 + self.empty_emb * self.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
            skips = self.unet.encode(z0, geom)
            skips_t = [s.view(B, T, s.shape[1], s.shape[2]) for s in skips]
            tp = batch.get("time_positions", batch.get("time_indices", None))
            tp = None if tp is None else tp.to(z0.device).float()
            maps = self.temporal(skips_t, tp, mask.all(-1))
            dtok, dmsk, nd = tokens.reshape(B, T * N, C), mask.reshape(B, T * N), T
        else:
            if self.temporal_mode == "per_level":
                raise ValueError("[Atomiser_Unet] unet_temporal: per_level expects per-date tokens")
            tok_sel, _, bias = self.pruning(tokens, mask, None, geo_k=m, atoms_per_latent=k)
            geom = self.pruning.last_geometry
            z0 = self.leaf(tok_sel, bias, geom)
            z0 = z0 + self.empty_emb * self.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
            maps = self.unet.encode(z0, geom)
            dtok, dmsk, nd = tokens, mask, 1
        q = batch["queries"]
        d0 = self.unet.decode(maps, geom)
        preds = self.decoder(q, dtok, dmsk, d0, geom, n_dates=nd)
        d0_s = self.unet.decode(maps, geom, start=student_level)
        q_s = q if query_idx is None else q[:, query_idx]
        preds_s = self.decoder(q_s, dtok, dmsk, d0_s, geom, n_dates=nd)
        return {"predictions": preds, "student_predictions": preds_s,
                "student_level": int(student_level), "num_latents": geom.sizes[0]}

    @torch.no_grad()
    def predict_at_depth(self, batch, level):
        """Predictions decoded from level `level` only (level = S-1: the full model)."""
        out = self.forward_ilsd(batch, min(int(level), self.unet.S - 1))
        return out["predictions"] if level >= self.unet.S - 1 else out["student_predictions"]

    def _forward_temporal(self, batch, tokens, mask, k, m):
        """Per-date encoding + per-level temporal aggregation + one decode.
        tokens [B, T, N, 8], mask [B, T, N] (True = invalid). Every date shares the
        token layout, so one cached geometry serves all dates: dates are folded
        into the batch for the leaf read and the encoder."""
        B, T, N, C = tokens.shape
        flat_tok, flat_mask = tokens.reshape(B * T, N, C), mask.reshape(B * T, N)
        tok_sel, _, bias = self.pruning(flat_tok, flat_mask, None, geo_k=m, atoms_per_latent=k)
        geom = self.pruning.last_geometry
        z0 = self.leaf(tok_sel, bias, geom)
        z0 = z0 + self.empty_emb * self.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
        skips = self.unet.encode(z0, geom)                       # per date, shared weights
        skips_t = [s.view(B, T, s.shape[1], s.shape[2]) for s in skips]
        pad = mask.all(-1)                                       # [B, T] date fully padded
        time_pos = batch.get("time_positions", batch.get("time_indices", None))
        if time_pos is not None:
            time_pos = time_pos.to(z0.device).float()
        agg = self.temporal(skips_t, time_pos, pad)               # [B, L_l, D_l] per level
        d0 = self.unet.decode(agg, geom)
        preds = self.decoder(batch["queries"], tokens.reshape(B, T * N, C),
                             mask.reshape(B, T * N), d0, geom, n_dates=T)
        return {"predictions": preds, "num_latents": geom.sizes[0]}
