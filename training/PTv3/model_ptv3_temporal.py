"""
PTv3-Temporal: Point Transformer V3 with level-aligned temporal attention
==========================================================================

For satellite image time series (PASTIS): every frame is the same H x W
pixel grid, so every frame becomes an identical point cloud (one point per
pixel). PTv3's serialization and grid pooling depend only on the
coordinates, so the points of every encoder level line up across time:
row n of frame t and row n of frame t' are the same location.

That lets us do what U-TAE does with a CNN:

    frames [B, T, C, H, W]
      │  one point cloud per frame, all B*T clouds packed with `offset`
      ▼
    PTv3 embedding + encoder  (shared weights, frames processed independently)
      │  5 levels: 1, 1/2, 1/4, 1/8, 1/16 resolution (per side)
      ▼
    at each chosen level: temporal transformer over T, per location
      [B*N_l, T, C_l] -> [B*N_l, C_l]   (CLS readout, date encoding,
                                          padded frames masked)
      the aggregated feature replaces the T per-frame features
      ▼
    PTv3 decoder (unpooling + skip connections) on the aggregated levels
      ▼
    final temporal readout per pixel + linear head -> [B, K, H, W]

With all levels aggregated, the decoder sees identical inputs for every
frame; the final readout then just merges T (near-)identical copies. With
temporal_levels=() it becomes pure late fusion: per-frame PTv3, temporal
attention per pixel at the output.

Implementation notes
--------------------
* Uses the standalone `point_transformer_v3` package, random init:
      export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH
* Written for the vendored Pointcept2 package (point_transformer_v3/model.py
  with `_level0`, `_forward_levels`, `enc` / `dec` ModuleLists). The
  backbone is NOT modified: `_run_levels` below is a copy of the package's
  `PointTransformerV3._forward_levels`, with the temporal aggregation
  inserted between the encoder loop and the decoder loop. If the package
  is updated, compare `_run_levels` with its `_forward_levels` again.
* The alignment of every level across frames is ASSERTED at runtime
  (identical grid_coord and sample sizes for all frames).
* Cost: the decoder runs on all B*T clouds (it cannot be shrunk to B
  without rebuilding PTv3's internal pooling chain). It is the smaller half
  of PTv3, but this is still T x the decoder cost of a single frame.

Interface (BaselineTrainer, temporal=True):
    forward(x [B, T, C, H, W], doy=[B, T]) -> logits [B, K, H, W]
    doy = days since a reference date; padded frames have doy < 0.
"""

import torch
import torch.nn as nn

from point_transformer_v3 import Config, Point, PointTransformerV3
from point_transformer_v3.model import _LevelOrders
from point_transformer_v3.modules import check_input, check_orders
from point_transformer_v3.structure import Level, auto_depth


# =============================================================================
# Temporal aggregator: transformer over T + CLS readout
# =============================================================================

class TemporalAggregator(nn.Module):
    """
    x [N, T, D], days [N, T], pad [N, T] (True = padded frame) -> [N, D]

    Acquisition dates enter as a fixed sinusoidal encoding of the number of
    days (period base 1000, as in L-TAE / U-TAE), added to the features.
    Same call signature as the Atomiser TemporalTransformer, so either can
    be swapped in.
    """

    def __init__(self, dim, n_layers=1, head_dim=32, ff_mult=2, dropout=0.0):
        super().__init__()
        heads = max(1, dim // head_dim)
        assert dim % heads == 0 and dim % 2 == 0, dim
        self.dim = dim

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

        half = dim // 2
        freqs = 1.0 / (1000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("freqs", freqs, persistent=False)

    def date_encoding(self, days):
        ang = days.float().unsqueeze(-1) * self.freqs              # [N, T, D/2]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)          # [N, T, D]

    def forward(self, x, days, pad):
        N = x.shape[0]
        x = x.float() + self.date_encoding(days)
        x = torch.cat([self.cls.expand(N, 1, -1), x], dim=1)        # [N, 1+T, D]
        key_pad = torch.cat(
            [torch.zeros(N, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        x = self.encoder(x, src_key_padding_mask=key_pad)
        return self.norm(x[:, 0])                                   # [N, D]


# =============================================================================
# PTv3-Temporal
# =============================================================================

class PTv3Temporal(nn.Module):

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 20,
        gsd: float = 10.0,
        patch_size: int = 1024,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        dec_channels=(64, 64, 128, 256),
        dec_depths=(2, 2, 2, 2),
        head_dim: int = 16,
        drop_path: float = 0.3,
        temporal_levels=(0, 1, 2, 3, 4),
        temporal_layers: int = 1,
        temporal_head_dim: int = 32,
        temporal_dropout: float = 0.0,
        precision: str = "auto",
    ):
        super().__init__()
        enc_channels = tuple(enc_channels)
        dec_channels = tuple(dec_channels)
        enc_depths   = tuple(enc_depths)
        dec_depths   = tuple(dec_depths)
        assert len(enc_channels) == len(enc_depths) == 5
        assert len(dec_channels) == len(dec_depths) == 4
        for c in enc_channels + dec_channels:
            assert c % head_dim == 0, f"width {c} not divisible by head_dim={head_dim}"

        cfg = Config(
            in_channels=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_heads=tuple(c // head_dim for c in enc_channels),
            enc_patch_sizes=(patch_size,) * len(enc_channels),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_heads=tuple(c // head_dim for c in dec_channels),
            dec_patch_sizes=(patch_size,) * len(dec_channels),
            drop_path=drop_path,
        )
        self.backbone = PointTransformerV3(config=cfg, precision=precision)
        n_levels = self.backbone.num_levels
        assert len(self.backbone.enc) == n_levels == 5 and len(self.backbone.dec) == 4

        self.enc_channels    = enc_channels
        self.temporal_levels = tuple(sorted(set(int(l) for l in temporal_levels)))
        for l in self.temporal_levels:
            assert 0 <= l < 5, f"temporal level {l} out of range 0..4"

        agg_kw = dict(n_layers=temporal_layers, head_dim=temporal_head_dim,
                      dropout=temporal_dropout)
        self.level_aggs = nn.ModuleDict({
            str(l): TemporalAggregator(enc_channels[l], **agg_kw)
            for l in self.temporal_levels
        })
        self.out_agg = TemporalAggregator(dec_channels[0], **agg_kw)
        self.head    = nn.Linear(dec_channels[0], num_classes)

        self.gsd = float(gsd)
        self.num_classes = num_classes
        self._grid_cache = {}

    # -------------------------------------------------------------------------

    def _grid(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            g = torch.stack([xs, ys, torch.zeros_like(xs)], -1).reshape(-1, 3)
            self._grid_cache[key] = g.int()
        return self._grid_cache[key]

    def _aggregate(self, feat, grid, agg, B, T, days, pad, level):
        """feat [B*T*N, C] on an aligned level -> same shape, every frame
        replaced by the temporal aggregate of its location."""
        BT = B * T
        seq = tuple(grid.seq_lens)
        if len(seq) != BT or len(set(seq)) != 1:
            raise RuntimeError(f"level {level}: per-frame point counts differ: {seq[:8]}...")
        N, C = seq[0], feat.shape[1]
        gc = grid.grid_coord.view(BT, N, 3)
        if not (gc == gc[:1]).all():
            raise RuntimeError(f"level {level}: points are not aligned across frames")

        x  = feat.view(B, T, N, C).permute(0, 2, 1, 3).reshape(B * N, T, C)
        dd = days[:, None, :].expand(B, N, T).reshape(B * N, T)
        pp = pad[:, None, :].expand(B, N, T).reshape(B * N, T)
        a  = agg(x, dd, pp)                                          # [B*N, C]
        out = (a.view(B, N, 1, C).expand(B, N, T, C)
                .permute(0, 2, 1, 3).reshape(BT * N, C))
        return out.to(feat.dtype)

    def _run_levels(self, feat0, grid0, B, T, days, pad):
        """Copy of PointTransformerV3._forward_levels (vendored Pointcept2),
        with the temporal aggregation between encoder and decoder.
        Returns the level-0 decoder features [B*T*HW, dec_channels[0]]."""
        bb, cfg = self.backbone, self.backbone.cfg
        depth = cfg.serialization_depth
        auto = not isinstance(depth, int)
        point = Level(feat0, grid0)
        grid = check_input(point, cfg.in_channels, None if auto else depth)
        orders = bb.draw_orders()
        check_orders(orders, bb.num_levels, len(cfg.orders))
        level_orders = _LevelOrders(cfg.orders, orders, grid.device)

        grids = [grid]
        max_coord = getattr(grid, "_max_coord", None)
        for i, stride in enumerate(cfg.strides):
            d = None
            if auto and cfg.pool_policy == "reencode":
                bits = stride.bit_length() - 1
                max_coord = None if max_coord is None else max_coord >> bits
                nxt = cfg.strides[i + 1].bit_length() if i + 1 < len(cfg.strides) else 1
                d = auto_depth(max_coord or 0, rule=str(cfg.serialization_depth), minimum=nxt)
            grids.append(grids[-1].downsample(stride, cfg.pool_policy,
                                              order=cfg.orders[0], depth=d))

        # ---- encoder (all B*T frames, shared weights)
        x = bb.embedding(point.feat, grid) if cfg.stem == "conv" else bb.embedding(point.feat)
        feats = []
        for s_, stage in enumerate(bb.enc):
            x = stage(x, grids[s_], level_orders(s_, grids[s_]))
            feats.append(x)

        # ---- temporal aggregation per level
        for l in self.temporal_levels:
            feats[l] = self._aggregate(feats[l], grids[l], self.level_aggs[str(l)],
                                       B, T, days, pad, l)

        # ---- decoder (unpooling + skips), as in _forward_levels
        x = feats[-1]
        for s_ in reversed(range(len(bb.dec))):
            x = bb.dec[s_](x, feats[s_], grids[s_ + 1], level_orders(s_, grids[s_]))

        # free the grid chain's caches (as the package's _release)
        seen = set()
        for g_ in grids:
            while g_ is not None and id(g_) not in seen:
                seen.add(id(g_))
                parent = getattr(g_, "parent", None)
                if hasattr(g_, "cache"):
                    g_.cache.clear()
                g_.parent = None
                g_ = parent
        return x

    # -------------------------------------------------------------------------

    def forward(self, x, doy=None):
        B, T, C, H, W = x.shape
        HW, BT = H * W, B * T
        dev = x.device

        if doy is None:
            doy = torch.arange(T, device=dev, dtype=torch.float32).expand(B, T)
        pad  = doy < 0                                 # padded frames
        days = doy.clamp(min=0).float()

        g = self._grid(H, W, dev)
        point = Point(dict(
            coord=(g.float() * self.gsd).repeat(BT, 1),
            grid_coord=g.repeat(BT, 1),
            feat=x.reshape(BT, C, H, W).permute(0, 2, 3, 1).reshape(-1, C).float(),
            offset=torch.arange(1, BT + 1, device=dev) * HW,
            grid_size=self.gsd,
        ))

        bb = self.backbone
        with bb._autocast(dev):
            grid0, feat0 = bb._level0(point)
            f = self._run_levels(feat0, grid0, B, T, days, pad)   # [BT*HW, Cd], input rows

        f   = f.float()
        Cd  = f.shape[1]
        f   = f.view(B, T, HW, Cd).permute(0, 2, 1, 3).reshape(B * HW, T, Cd)
        dd  = days[:, None, :].expand(B, HW, T).reshape(B * HW, T)
        pp  = pad[:, None, :].expand(B, HW, T).reshape(B * HW, T)
        f   = self.out_agg(f, dd, pp)                  # [B*HW, Cd]

        logits = self.head(f)                          # [B*HW, K]
        return logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
