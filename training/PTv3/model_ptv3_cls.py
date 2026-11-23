"""
PTv3 classifier (pixels-as-points), random init, no pretrained weights.

For scene classification (EuroSAT): each pixel becomes one point (x, y, 0)
with its band values as features; PTv3 runs ENCODER ONLY (dec_depths=(),
supported by the vendored Pointcept2 package), and the coarsest level is
average-pooled per sample, followed by LayerNorm + Linear.

    forward: [B, C, H, W] -> logits [B, num_classes]

On a 64x64 tile the 5 encoder levels are 64^2, 32^2, 16^2, 8^2 and 4^2
points; the pooled level holds 16 points per sample.

Uses the standalone `point_transformer_v3` package:
    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH
"""

import torch
import torch.nn as nn

from point_transformer_v3 import Config, Point, PointTransformerV3


class PTv3Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        num_classes: int = 10,
        gsd: float = 10.0,
        patch_size: int = 256,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        head_dim: int = 16,
        drop_path: float = 0.3,
        head_dropout: float = 0.0,
        precision: str = "auto",
    ):
        super().__init__()
        enc_channels = tuple(enc_channels)
        enc_depths = tuple(enc_depths)
        assert len(enc_channels) == len(enc_depths) == 5
        for c in enc_channels:
            assert c % head_dim == 0, f"width {c} not divisible by head_dim={head_dim}"

        cfg = Config(
            in_channels=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_heads=tuple(c // head_dim for c in enc_channels),
            enc_patch_sizes=(patch_size,) * len(enc_channels),
            dec_depths=(),                       # encoder only
            drop_path=drop_path,
        )
        self.backbone = PointTransformerV3(config=cfg, precision=precision)
        self.norm = nn.LayerNorm(enc_channels[-1])
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(enc_channels[-1], num_classes)
        self.gsd = float(gsd)
        self._grid_cache = {}

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

    def forward(self, x):
        B, C, H, W = x.shape
        g = self._grid(H, W, x.device)
        point = Point(dict(
            coord=(g.float() * self.gsd).repeat(B, 1),
            grid_coord=g.repeat(B, 1),
            feat=x.permute(0, 2, 3, 1).reshape(-1, C).float(),
            offset=torch.arange(1, B + 1, device=x.device) * (H * W),
            grid_size=self.gsd,
        ))
        out = self.backbone(point)               # coarsest level (encoder only)

        feat = out["feat"].float()               # [N_coarse, C_last]
        batch = out["batch"].long()              # [N_coarse] sample index
        sums = feat.new_zeros(B, feat.shape[1]).index_add_(0, batch, feat)
        cnt = torch.bincount(batch, minlength=B).clamp(min=1).unsqueeze(1)
        pooled = sums / cnt                      # per-sample mean, [B, C_last]

        return self.head(self.drop(self.norm(pooled)))
