"""
AtomizerPTv3: Atomizer per-pixel tokenizer + PTv3 backbone
==========================================================

Step 1 of the "replace PTv3 parts by Atomizer parts" ladder: PTv3's input
(the raw band vector of each pixel) is replaced by Atomizer's per-pixel
embedding; everything else is the PTv3 used on Cashew.

    pixel p's C atoms (band value + band / resolution / time metadata)
      -> TokenProcessor.process_data_for_encoder          (Atomizer, as is)
      -> learned pixel query cross-attends to the C atoms (as _pixel_skip)
      -> LayerNorm -> D-dim pixel embedding
      -> PTv3 (stem, encoder, decoder; pixels as points)  -> linear head
      -> logits gathered at the batch's query pixels

Drop-in replacement for Atomiser_Senflood_Skip:
    model = AtomizerPTv3(config=config, lookup_table=lookup_table)
    logits = model(batch, training=...)        # [B, M, num_classes]
It reads the same batch as CashewSkipDataset produces (groups[10.0]["tokens"]
[B, N, 8] channel-major, groups[..]["shape"] = (C, H, W), queries [B, M, 8],
queries_mask [B, M]), so the Atomizer dataset, collate, trainer, loss and
metrics are unchanged. query_token_idx is not needed (every pixel is
embedded) and is ignored.

Pixel layout: TokenBuilder flattens (c h w) -> row, so pixel p = h*W + w owns
rows {p + c*H*W}. tokens.view(B, C, HW, 8) therefore gives each pixel's atoms,
in the same row-major pixel order as the PTv3 grid. Queries are matched to
pixels through the token position fields (1 = x index, 2 = y index, as in
Atomiser._compute_deltas); the match is checked, not assumed.

Config (new optional section; defaults in brackets):
    PTv3:
      pixel_dim:      [Atomiser.latent_dim]  embedding width fed to PTv3
      pixel_heads:    [Atomiser.cross_heads]
      atom_drop_p:    [0.0]   train-only fraction of a pixel's atoms dropped
                               (Atomizer's decoder_skip_drop_p is 0.5; 0 keeps
                               the comparison with PTv3 to one change)
      pixel_chunk:    [16384] pixels per tokenizer chunk (memory)
      checkpoint_tokenizer: [true]  recompute tokenizer chunks in backward
      stem:           [conv]  "conv" = PTv3's 5x5 stem; "linear" = no spatial
                               mixing before the first block (ladder step 2)
      enc_channels:   [32, 64, 128, 256, 512]
      enc_depths:     [2, 2, 2, 6, 2]
      dec_channels:   [64, 64, 128, 256]
      dec_depths:     [2, 2, 2, 2]
      head_dim:       [16]
      patch_size:     [1024]
      drop_path:      [0.3]
      orders:         [z, z-trans, hilbert, hilbert-trans]
      shuffle_orders: [true]
      gsd:            [10.0]
      precision:      [auto]
      norm:           [bn]    "ln" = LayerNorm instead of BatchNorm in the stem,
                               poolings and unpoolings (Sonata); blocks always
                               use LayerNorm. Use "ln" when batches alternate
                               between datasets (distillation) and to fine-tune
                               weights distilled with "ln"
      cpe:            [true]  false = remove the convolutional positional
                               encoding (3x3 sparse conv + Linear + LayerNorm)
                               from EVERY block, encoder and decoder
      rope:           [none]  "2d" = 2D rotary position embedding of q and k
                               in every block's attention (see Rope2d)
      rope_base:      [100.0] RoPE frequency base
      rope_unit:      [gsd]   coordinates are divided by this before RoPE, so
                               positions are in input pixels at every level

Ladder step "RoPE instead of CPE":  cpe: false, rope: 2d.  The stem conv is
kept (stem: linear removes it too, i.e. no convolution left anywhere).

Spectral streams (mode: streams)
--------------------------------
Instead of one embedding per pixel from all its atoms, each band GROUP gets
its own embedding (same pixel cross-attention, shared weights, on that
group's atoms only) and its own copy of the spatial hierarchy:

    group g's atoms -> pixel CA -> stream g          (S streams per tile)
    levels 0..fuse_level : PTv3 encoder stages on the S streams, shared
                           weights, no exchange between streams
                           (each stream is its own point cloud)
    at each of these levels: a transformer over the S streams per location
                           (CLS readout, as the temporal transformer) gives
                           the fused skip for the decoder
    levels > fuse_level  : the fused features of level fuse_level continue as
                           ONE stream through the remaining encoder stages
    decoder              : once per tile, on the fused features and skips

    PTv3:
      mode:          [pixel]  "streams" enables the above
      band_groups:   [[1,2,3,7], [4,5,6,8,10,11], [0,9]]
                     channel indices per stream, in the dataset's band order.
                     Default = Sentinel-2 native resolution on Cashew
                     (0 B01, 1 B02, 2 B03, 3 B04, 4 B05, 5 B06, 6 B07, 7 B08,
                      8 B8A, 9 B09, 10 B11, 11 B12):
                     10 m: B02 B03 B04 B08 | 20 m: B05 B06 B07 B8A B11 B12 |
                     60 m: B01 B09
      fuse_level:    [2]      last level with separate streams (0..4)
      fusion_layers: [1]      transformer layers per fusion
      fusion_head_dim: [32]
      fusion_group_embedding: [false]  add a learned per-stream embedding to
                     the fusion tokens (stream identity then also comes from
                     the stream's index, not only from its content)

Needs the standalone point_transformer_v3 package on PYTHONPATH.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from point_transformer_v3 import Config, Point, PointTransformerV3
from point_transformer_v3.model import _LevelOrders
from point_transformer_v3.modules import Block, check_input, check_orders
from point_transformer_v3.structure import Level, auto_depth
from training.utils.token_building.processor import TokenProcessor


ALL_ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


class NoCpe(nn.Module):
    """Stands in for a block's CPE: Block.forward does x = x + cpe(x, grid),
    so returning a scalar zero removes the convolutional positional encoding
    while leaving the rest of the block unchanged."""

    def forward(self, x, grid):
        return x.new_zeros(())


class Rope2d(nn.Module):
    """2D rotary position embedding of queries and keys, for pixels as points.

    Each head of size hd splits into an x half and a y half; each half is
    rotated by the point's x (resp. y) coordinate with the usual RoPE
    frequencies inv_freq = 1 / base ** (arange(0, hd/2, 2) / (hd/2)), using
    the GPT-NeoX rotate_half pairing. Values are not rotated. The attention
    score of a query and a key then depends on their relative position only.

    Drop-in for the package's SerializedAttention.rope (same call: qkv
    (N, 3*H*hd) laid out [q | k | v], each head-major; coord (N, 3) float,
    the level's coordinates in metres). Coordinates are divided by `unit`
    (the GSD), so positions are in input pixels at every level (coarse
    levels carry the mean coordinate of their cells). z is ignored.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim, base=100.0, unit=1.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"2D RoPE needs head_dim divisible by 4, got {head_dim}")
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.unit = float(unit)
        half = self.head_dim // 2
        self.register_buffer(
            "inv_freq", 1.0 / (self.base ** (torch.arange(0, half, 2).float() / half)))

    @staticmethod
    def _rot_half(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    def forward(self, qkv, coord):
        hd, half = self.head_dim, self.head_dim // 2
        xy = coord[:, :2].float() / self.unit                       # (N, 2) pixels
        ax = xy[:, 0:1] * self.inv_freq                             # (N, half/2)
        ay = xy[:, 1:2] * self.inv_freq
        ang = torch.cat([ax, ax, ay, ay], dim=-1)[:, None, None, :] # (N, 1, 1, hd)
        cos, sin = ang.cos(), ang.sin()

        t = qkv.unflatten(1, (3, -1, hd))                           # (N, 3, H, hd)
        qk, v = t[:, :2], t[:, 2:]
        qx, qy = qk.float().split(half, dim=-1)
        rotated = torch.cat([self._rot_half(qx), self._rot_half(qy)], dim=-1)
        qk = qk.float() * cos + rotated * sin
        return torch.cat([qk.to(qkv.dtype), v], dim=1).flatten(1)

    def extra_repr(self):
        return f"head_dim={self.head_dim}, base={self.base:g}, unit={self.unit:g}"


class StreamFusion(nn.Module):
    """Transformer over the S streams at one location, CLS readout:
    x [N, S, D], pad [N, S] (True = missing stream) -> [N, D].
    Same structure as the temporal aggregator of PTv3-Temporal, without dates."""

    # Locations per attention call. Every location is one tiny sequence, and the
    # fused SDPA kernels cannot launch on more than 65,535 sequences at once
    # ("invalid configuration argument"), so large levels are processed in chunks.
    CHUNK = 32768

    def __init__(self, dim, n_streams, n_layers=1, head_dim=32, ff_mult=2,
                 group_embedding=False):
        super().__init__()
        heads = max(1, dim // head_dim)
        while dim % heads:
            heads -= 1
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.group_emb = None
        if group_embedding:
            self.group_emb = nn.Parameter(torch.zeros(1, n_streams, dim))
            nn.init.trunc_normal_(self.group_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def _run(self, x, pad):
        N = x.shape[0]
        x = x.float()
        if self.group_emb is not None:
            x = x + self.group_emb
        x = torch.cat([self.cls.expand(N, 1, -1), x], dim=1)
        kpm = None
        if pad is not None:
            kpm = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), pad], 1)
        x = self.encoder(x, src_key_padding_mask=kpm)
        return self.norm(x[:, 0])

    def forward(self, x, pad=None):
        if x.shape[0] <= self.CHUNK:
            return self._run(x, pad)
        return torch.cat([
            self._run(x[i:i + self.CHUNK], None if pad is None else pad[i:i + self.CHUNK])
            for i in range(0, x.shape[0], self.CHUNK)], dim=0)


class AtomizerPTv3(pl.LightningModule):

    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=["lookup_table"])
        self.config = config
        a = config["Atomiser"]
        p = config.get("PTv3", {}) or {}

        # ── Atomizer tokenizer (unchanged module) ────────────────────────
        self.input_processor = TokenProcessor(config, lookup_table)
        self.input_dim = self.input_processor.get_encoder_output_dim()
        self.num_classes = config["trainer"]["num_classes"]
        self.skip_res = float(a.get("skip_resolution", 10.0))

        # ── per-pixel cross-attention (same structure as _pixel_skip) ────
        self.pixel_dim = int(p.get("pixel_dim", a.get("latent_dim", self.input_dim)))
        heads = int(p.get("pixel_heads", a["cross_heads"]))
        assert self.pixel_dim % heads == 0, (self.pixel_dim, heads)
        self.pixel_query = nn.Parameter(torch.zeros(1, 1, self.pixel_dim))
        nn.init.trunc_normal_(self.pixel_query, std=0.02, a=-2., b=2.)
        self.pixel_q_norm = nn.LayerNorm(self.pixel_dim)
        self.pixel_cross_attn = nn.MultiheadAttention(
            embed_dim=self.pixel_dim, num_heads=heads,
            kdim=self.input_dim, vdim=self.input_dim,
            dropout=float(a.get("attn_dropout", 0.0)), batch_first=True,
        )
        self.pixel_out_norm = nn.LayerNorm(self.pixel_dim)
        self.atom_drop_p = float(p.get("atom_drop_p", 0.0))
        self.pixel_chunk = int(p.get("pixel_chunk", 16384))
        self.checkpoint_tokenizer = bool(p.get("checkpoint_tokenizer", True))

        # ── PTv3 backbone (same recipe as the Cashew PTv3 baseline) ──────
        enc_channels = tuple(p.get("enc_channels", (32, 64, 128, 256, 512)))
        enc_depths   = tuple(p.get("enc_depths",   (2, 2, 2, 6, 2)))
        dec_channels = tuple(p.get("dec_channels", (64, 64, 128, 256)))
        dec_depths   = tuple(p.get("dec_depths",   (2, 2, 2, 2)))
        head_dim     = int(p.get("head_dim", 16))
        patch_size   = int(p.get("patch_size", 1024))
        for c in enc_channels + dec_channels:
            assert c % head_dim == 0, f"PTv3 width {c} not divisible by head_dim={head_dim}"

        norm = str(p.get("norm", "bn")).lower()
        if norm not in ("bn", "ln"):
            raise ValueError(f"PTv3.norm must be 'bn' or 'ln', got {norm!r}")
        cfg_kw = {} if norm == "bn" else {"norm": norm}       # "bn" = package default
        cfg = Config(
            in_channels=self.pixel_dim,
            stem=p.get("stem", "conv"),
            **cfg_kw,
            orders=tuple(p.get("orders", ALL_ORDERS)),
            shuffle_orders=bool(p.get("shuffle_orders", True)),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_heads=tuple(c // head_dim for c in enc_channels),
            enc_patch_sizes=(patch_size,) * len(enc_channels),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_heads=tuple(c // head_dim for c in dec_channels),
            dec_patch_sizes=(patch_size,) * len(dec_channels),
            drop_path=float(p.get("drop_path", 0.3)),
        )
        self.backbone = PointTransformerV3(config=cfg, precision=p.get("precision", "auto"))
        n_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.backbone.modules())
        if norm == "ln" and n_bn:
            raise RuntimeError(f"PTv3.norm: ln but the backbone still has {n_bn} BatchNorm layers "
                               "(this PTv3 package version ignores Config.norm?)")
        self.norm_type = norm
        self.head = nn.Linear(dec_channels[0], self.num_classes)
        self.gsd = float(p.get("gsd", 10.0))
        self._grid_cache = {}

        # ── block-level positional encoding: CPE and/or RoPE ─────────────
        n_bb_before = sum(q.numel() for q in self.backbone.parameters())
        self.use_cpe = bool(p.get("cpe", True))
        self.rope = p.get("rope", None)
        self.rope = None if self.rope in (None, "none", False) else str(self.rope)
        if self.rope not in (None, "2d"):
            raise ValueError(f"PTv3.rope must be none or '2d', got {self.rope!r}")
        rope_base = float(p.get("rope_base", 100.0))
        rope_unit = float(p.get("rope_unit", self.gsd))
        n_blocks = 0
        for m in self.backbone.modules():
            if isinstance(m, Block):
                n_blocks += 1
                if not self.use_cpe:
                    m.cpe = NoCpe()
                if self.rope == "2d":
                    if m.attn.rope is not None:
                        raise RuntimeError("block already has a RoPE (Config.rope_base set?)")
                    m.attn.rope = Rope2d(m.attn.head_dim, base=rope_base, unit=rope_unit)
        n_bb_after = sum(q.numel() for q in self.backbone.parameters())

        # ── spectral streams ─────────────────────────────────────────────
        self.mode = str(p.get("mode", "pixel"))
        if self.mode not in ("pixel", "streams"):
            raise ValueError(f"PTv3.mode must be 'pixel' or 'streams', got {self.mode!r}")
        self.band_groups = None
        if self.mode == "streams":
            groups = p.get("band_groups", [[1, 2, 3, 7], [4, 5, 6, 8, 10, 11], [0, 9]])
            self.band_groups = [list(map(int, g)) for g in groups]
            flat = [c for g in self.band_groups for c in g]
            if len(flat) != len(set(flat)) or any(len(g) == 0 for g in self.band_groups):
                raise ValueError(f"PTv3.band_groups must be non-empty, disjoint: {groups}")
            for i, g in enumerate(self.band_groups):
                self.register_buffer(f"_group_idx_{i}", torch.tensor(g, dtype=torch.long),
                                     persistent=False)
            self.fuse_level = int(p.get("fuse_level", 2))
            if not 0 <= self.fuse_level < len(enc_channels):
                raise ValueError(f"PTv3.fuse_level must be in 0..{len(enc_channels) - 1}")
            S = len(self.band_groups)
            self.fusions = nn.ModuleList([
                StreamFusion(enc_channels[l], S,
                             n_layers=int(p.get("fusion_layers", 1)),
                             head_dim=int(p.get("fusion_head_dim", 32)),
                             group_embedding=bool(p.get("fusion_group_embedding", False)))
                for l in range(self.fuse_level + 1)
            ])
            n_fus = sum(q.numel() for q in self.fusions.parameters())
            print(f"[AtomizerPTv3] mode=streams: {S} streams {self.band_groups}, separate "
                  f"through level {self.fuse_level}, fused after; fusion params "
                  f"{n_fus/1e6:.3f}M")
        print(f"[AtomizerPTv3] {n_blocks} blocks: norm {self.norm_type} (stem/pooling/unpooling; blocks LayerNorm), "
              f"CPE {'on' if self.use_cpe else 'REMOVED'}, "
              f"RoPE {self.rope or 'off'}"
              + (f" (base={rope_base:g}, unit={rope_unit:g} m = 1 px)" if self.rope else "")
              + f"; backbone params {n_bb_before/1e6:.2f}M -> {n_bb_after/1e6:.2f}M")

        n_tok = sum(q.numel() for q in self.input_processor.parameters())
        n_pix = sum(q.numel() for m in (self.pixel_cross_attn, self.pixel_q_norm,
                                        self.pixel_out_norm) for q in m.parameters())
        n_bb = sum(q.numel() for q in self.backbone.parameters())
        print(f"[AtomizerPTv3] tokenizer {n_tok/1e6:.2f}M + pixel-CA {n_pix/1e6:.2f}M "
              f"(input_dim={self.input_dim} -> pixel_dim={self.pixel_dim}, heads={heads}, "
              f"atom_drop_p={self.atom_drop_p}) + PTv3 {n_bb/1e6:.2f}M "
              f"(stem={cfg.stem}, patch={patch_size}, enc={enc_channels})")

    # =========================================================================
    # helpers
    # =========================================================================

    @staticmethod
    def _chw(shape):
        """groups[res]["shape"] after collate: (C, H, W) of ints, or per-dim
        tensors/lists over the batch, or a list of per-sample tuples."""
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        if isinstance(shape, (list, tuple)) and len(shape) and isinstance(
                shape[0], (list, tuple)) and len(shape[0]) == 3:
            shape = shape[0]                                   # list of per-sample tuples
        out = []
        for s in shape:
            if isinstance(s, torch.Tensor):
                s = s.reshape(-1)[0].item()
            elif isinstance(s, (list, tuple)):
                s = s[0]
            out.append(int(s))
        assert len(out) == 3, f"unexpected group shape {shape}"
        return out

    def _grid(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(torch.arange(H, device=device),
                                    torch.arange(W, device=device), indexing="ij")
            g = torch.stack([xs, ys, torch.zeros_like(xs)], -1).reshape(-1, 3)
            self._grid_cache[key] = g.int()                   # row-major p = h*W + w
        return self._grid_cache[key]

    def _encode_chunk(self, tok, msk, coords, training):
        """tok [B, P, C, 8], msk [B, P, C] (True = masked), coords [B, P, 2]
        -> [B, P, pixel_dim]."""
        B, P, C, _ = tok.shape
        if training and self.atom_drop_p > 0.0:
            c_keep = max(1, math.ceil((1.0 - self.atom_drop_p) * C))   # as _pixel_skip
            sel = torch.rand(B, P, C, device=tok.device).argsort(-1)[..., :c_keep]
            tok = torch.gather(tok, 2, sel.unsqueeze(-1).expand(-1, -1, -1, tok.shape[-1]))
            msk = torch.gather(msk, 2, sel)
            C = c_keep

        enc = self.input_processor.process_data_for_encoder(
            tok, msk, latent_positions=coords)                   # [B, P, C, input_dim]

        kpm = msk.clone()
        none = kpm.all(-1)                                       # pixel with no real atom
        if none.any():
            kpm[..., 0] &= ~none                                 # keep one (output unused)

        BP = B * P
        q = self.pixel_q_norm(self.pixel_query).reshape(1, -1)  # same query for every pixel
        kv = enc.reshape(BP, C, -1)
        out = self._pixel_attention(q, kv, kpm.reshape(BP, C), training)
        return self.pixel_out_norm(out).reshape(B, P, -1)

    def _pixel_attention(self, q, kv, kpm, training):
        """Single-query multi-head attention with pixel_cross_attn's own weights,
        computed explicitly: q [1, D], kv [BP, C, Din], kpm [BP, C] (True = ignore)
        -> [BP, D].

        Same maths as nn.MultiheadAttention(q, kv, kv, key_padding_mask=kpm), but
        without the fused SDPA kernel, which rejects more than 65,535 sequences
        per call when attention dropout is on (here: one sequence per pixel)."""
        mha = self.pixel_cross_attn
        D, h = mha.embed_dim, mha.num_heads
        hd = D // h
        if mha._qkv_same_embed_dim:
            w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
        else:
            w_q, w_k, w_v = mha.q_proj_weight, mha.k_proj_weight, mha.v_proj_weight
        b_q = b_k = b_v = None
        if mha.in_proj_bias is not None:
            b_q, b_k, b_v = mha.in_proj_bias.chunk(3)

        qh = F.linear(q, w_q, b_q).reshape(h, hd)                         # [h, hd]
        k = F.linear(kv, w_k, b_k).reshape(kv.shape[0], kv.shape[1], h, hd)
        v = F.linear(kv, w_v, b_v).reshape(kv.shape[0], kv.shape[1], h, hd)

        scores = torch.einsum("nchd,hd->nhc", k, qh) / math.sqrt(hd)       # [BP, h, C]
        scores = scores.float().masked_fill(kpm[:, None, :], float("-inf"))
        attn = scores.softmax(dim=-1).to(v.dtype)
        if training and mha.dropout > 0.0:
            attn = F.dropout(attn, p=mha.dropout)
        out = torch.einsum("nhc,nchd->nhd", attn, v).reshape(kv.shape[0], D)
        return mha.out_proj(out)

    def _embed_pixels(self, tok, msk, training):
        """tok [B, HW, C, 8] -> [B, HW, pixel_dim], in chunks of pixels."""
        coords = self.input_processor.geometry.get_token_centers(tok[:, :, 0])
        outs = []
        for s in range(0, tok.shape[1], self.pixel_chunk):
            sl = slice(s, s + self.pixel_chunk)
            args = (tok[:, sl], msk[:, sl], coords[:, sl], training)
            if self.checkpoint_tokenizer and training and torch.is_grad_enabled():
                outs.append(torch_checkpoint(self._encode_chunk, *args, use_reentrant=False))
            else:
                outs.append(self._encode_chunk(*args))
        return torch.cat(outs, 1)

    # =========================================================================
    # spectral streams
    # =========================================================================

    def _point(self, g, n_clouds, HW, feat):
        return Point(
            coord=(g.float() * self.gsd).repeat(n_clouds, 1),
            grid_coord=g.repeat(n_clouds, 1),
            feat=feat,
            offset=torch.arange(1, n_clouds + 1, device=g.device) * HW,
            grid_size=self.gsd,
        )

    def _grid_chain(self, feat0, grid0, n_levels):
        """Level grids 0..n_levels-1, as PointTransformerV3._forward_levels builds them."""
        cfg = self.backbone.cfg
        depth = cfg.serialization_depth
        auto = not isinstance(depth, int)
        grid = check_input(Level(feat0, grid0), cfg.in_channels, None if auto else depth)
        grids = [grid]
        max_coord = getattr(grid, "_max_coord", None)
        for i, stride in enumerate(cfg.strides[:n_levels - 1]):
            d = None
            if auto and cfg.pool_policy == "reencode":
                bits = stride.bit_length() - 1
                max_coord = None if max_coord is None else max_coord >> bits
                nxt = cfg.strides[i + 1].bit_length() if i + 1 < len(cfg.strides) else 1
                d = auto_depth(max_coord or 0, rule=str(cfg.serialization_depth), minimum=nxt)
            grids.append(grids[-1].downsample(stride, cfg.pool_policy,
                                              order=cfg.orders[0], depth=d))
        return grids

    @staticmethod
    def _release_grids(grids):
        seen = set()
        for g in grids:
            while g is not None and id(g) not in seen:
                seen.add(id(g))
                parent = getattr(g, "parent", None)
                if hasattr(g, "cache"):
                    g.cache.clear()
                g.parent = None
                g = parent

    def _fuse(self, feat_s, grid_s, grid_b, fusion, B, S, level):
        """feat_s: rows of the S-stream level (clouds ordered (b, s)) -> rows of the
        single-stream level, every location fused over its S streams.

        Rows are matched by cell position (grid_coord), not by row index, so the
        result is correct whatever row order the package gives each pack; a
        location missing from a stream is an error."""
        seq_s, seq_b = tuple(grid_s.seq_lens), tuple(grid_b.seq_lens)
        if len(seq_s) != B * S or len(set(seq_s)) != 1 or len(seq_b) != B or seq_b[0] != seq_s[0]:
            raise RuntimeError(f"level {level}: stream / fused point counts differ "
                               f"({seq_s[:4]}..., {seq_b[:4]}...)")
        N, Cl = seq_s[0], feat_s.shape[1]
        gs = grid_s.grid_coord.view(B, S, N, 3).long()
        gb = grid_b.grid_coord.view(B, N, 3).long()
        x = feat_s.view(B, S, N, Cl)

        if (gs == gb[:, None]).all():                            # same row order: fast path
            fused = fusion(x.permute(0, 2, 1, 3).reshape(B * N, S, Cl))
            return fused.to(feat_s.dtype)

        K = int(torch.maximum(gs[..., 0].max(), gb[..., 0].max()).item()) + 1
        ks = gs[..., 1] * K + gs[..., 0]                         # [B, S, N]
        kb = gb[..., 1] * K + gb[..., 0]                         # [B, N]
        ks_sorted, os_ = ks.sort(dim=-1)
        kb_sorted, ob = kb.sort(dim=-1)
        if not (ks_sorted == kb_sorted[:, None]).all():
            raise RuntimeError(f"level {level}: streams and fused level hold different cells")
        x = torch.gather(x, 2, os_.unsqueeze(-1).expand(-1, -1, -1, Cl))   # sorted by cell
        fused = fusion(x.permute(0, 2, 1, 3).reshape(B * N, S, Cl)).view(B, N, Cl)
        out = torch.empty_like(fused)
        out.scatter_(1, ob.unsqueeze(-1).expand(-1, -1, Cl), fused)       # back to grid_b rows
        return out.reshape(B * N, Cl).to(feat_s.dtype)

    def _forward_streams(self, tok, msk, g, B, HW, C, training):
        """-> level-0 decoder features [B*HW, dec_channels[0]], rows in input order."""
        bb, cfg = self.backbone, self.backbone.cfg
        S = len(self.band_groups)
        n_levels = len(bb.enc)
        dev = tok.device
        if max(c for grp in self.band_groups for c in grp) >= C:
            raise RuntimeError(f"band_groups {self.band_groups} index beyond C={C} bands")

        # 1. one embedding per (pixel, stream), shared tokenizer weights
        embs = []
        for i in range(S):
            idx = getattr(self, f"_group_idx_{i}")
            embs.append(self._embed_pixels(tok[:, :, idx], msk[:, :, idx], training))
        feat = torch.stack(embs, 1).reshape(B * S * HW, -1).float()  # clouds (b, s)

        with bb._autocast(dev):
            grid0_s, feat0_s = bb._level0(self._point(g, B * S, HW, feat))
            grid0_b, feat0_b = bb._level0(
                self._point(g, B, HW, feat.new_zeros(B * HW, feat.shape[1])))
            grids_s = self._grid_chain(feat0_s, grid0_s, self.fuse_level + 1)
            grids_b = self._grid_chain(feat0_b, grid0_b, n_levels)

            orders = bb.draw_orders()
            check_orders(orders, bb.num_levels, len(cfg.orders))
            lo_s = _LevelOrders(cfg.orders, orders, dev)
            lo_b = _LevelOrders(cfg.orders, orders, dev)

            # 2. stem + encoder stages 0..fuse_level on the S streams
            x = (bb.embedding(feat0_s, grids_s[0]) if cfg.stem == "conv"
                 else bb.embedding(feat0_s))
            skips = []
            for l in range(self.fuse_level + 1):
                x = bb.enc[l](x, grids_s[l], lo_s(l, grids_s[l]))
                # 3. fused skip at this level
                skips.append(self._fuse(x, grids_s[l], grids_b[l], self.fusions[l],
                                        B, S, l))

            # 4. remaining encoder stages on the fused stream
            x = skips[-1]
            feats = list(skips)
            for l in range(self.fuse_level + 1, n_levels):
                x = bb.enc[l](x, grids_b[l], lo_b(l, grids_b[l]))
                feats.append(x)

            # 5. decoder once per tile, as in _forward_levels
            for l in reversed(range(len(bb.dec))):
                x = bb.dec[l](x, feats[l], grids_b[l + 1], lo_b(l, grids_b[l]))

        if not torch.equal(grids_b[0].grid_coord.int(), g.repeat(B, 1)):
            raise RuntimeError("fused level-0 rows are not in input order")
        self._release_grids(grids_s + grids_b)
        return x

    @staticmethod
    def _query_pixel_index(pix_tok, queries, queries_mask):
        """Row p of each query pixel, matched on the token (x, y) index fields.
        pix_tok [B, HW, 8] (one atom per pixel), queries [B, M, 8] -> [B, M]."""
        px, py = pix_tok[..., 1].long(), pix_tok[..., 2].long()
        qx, qy = queries[..., 1].long(), queries[..., 2].long()
        S = int(torch.maximum(px.max(), qx.max()).item()) + 1
        pkey = py * S + px
        qkey = qy * S + qx
        skey, order = pkey.sort(dim=1)
        pos = torch.searchsorted(skey, qkey).clamp(max=skey.shape[1] - 1)
        matched = torch.gather(skey, 1, pos) == qkey
        valid = queries_mask <= 0.5 if queries_mask is not None else torch.ones_like(matched)
        if not bool(matched[valid].all()):
            n = int((~matched & valid).sum())
            raise RuntimeError(f"[AtomizerPTv3] {n} valid queries match no pixel token "
                               f"(position fields 1/2 differ between queries and tokens)")
        return torch.gather(order, 1, pos)

    # =========================================================================
    # forward
    # =========================================================================

    def forward(self, batch, training=True, task="reconstruction", **kwargs):
        if task not in ("reconstruction", "segmentation"):
            raise NotImplementedError(f"[AtomizerPTv3] task={task!r} not supported")

        groups = batch["groups"]
        res = self.skip_res if self.skip_res in groups else min(groups.keys())
        tokens = groups[res]["tokens"]                           # [B, N, 8]
        mask = groups[res]["mask"]
        C, H, W = self._chw(groups[res]["shape"])
        B, N, Dt = tokens.shape
        HW = H * W
        if N != C * HW:
            raise RuntimeError(f"[AtomizerPTv3] {N} tokens != C*H*W = {C}*{H}*{W}: the pool "
                               f"must be the full, unsubsampled (c h w) grid")

        tok = tokens.view(B, C, HW, Dt).transpose(1, 2)          # [B, HW, C, 8]
        msk = (mask.view(B, C, HW) > 0.5 if not mask.dtype == torch.bool
               else mask.view(B, C, HW)).transpose(1, 2)         # [B, HW, C] True = masked

        g = self._grid(H, W, tokens.device)
        if self.mode == "streams":
            out_feat = self._forward_streams(tok, msk, g, B, HW, C, training)
        else:
            feat = self._embed_pixels(tok, msk, training)        # [B, HW, pixel_dim]
            point = Point(
                coord=(g.float() * self.gsd).repeat(B, 1),
                grid_coord=g.repeat(B, 1),
                feat=feat.reshape(B * HW, -1).float(),
                offset=torch.arange(1, B + 1, device=tokens.device) * HW,
                grid_size=self.gsd,
            )
            out_feat = self.backbone(point)["feat"]              # level-0 rows, input order
        logits = self.head(out_feat.float()).view(B, HW, -1)     # [B, HW, K]

        idx = self._query_pixel_index(tok[:, :, 0], batch["queries"],
                                      batch.get("queries_mask", None))
        return torch.gather(logits, 1, idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))

    # =========================================================================
    # Atomiser-compatible no-ops / freeze helpers
    # =========================================================================

    def _set_requires_grad(self, modules, flag):
        for m in modules:
            if isinstance(m, nn.Parameter):
                m.requires_grad = flag
            else:
                for q in m.parameters():
                    q.requires_grad = flag

    def _encoder_parts(self):
        parts = [self.input_processor, self.pixel_cross_attn, self.pixel_q_norm,
                 self.pixel_out_norm, self.pixel_query, self.backbone]
        if getattr(self, "fusions", None) is not None:
            parts.append(self.fusions)
        return parts

    def freeze_encoder(self):
        self._set_requires_grad(self._encoder_parts(), False)

    def unfreeze_encoder(self):
        self._set_requires_grad(self._encoder_parts(), True)

    def freeze_decoder(self):
        self._set_requires_grad([self.head], False)

    def unfreeze_decoder(self):
        self._set_requires_grad([self.head], True)

    def freeze_all(self):
        for q in self.parameters():
            q.requires_grad = False

    def unfreeze_all(self):
        for q in self.parameters():
            q.requires_grad = True
