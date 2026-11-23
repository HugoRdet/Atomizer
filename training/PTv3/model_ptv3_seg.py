"""
PTv3 segmenter (pixels-as-points), random init, no pretrained weights.

Uses the standalone `point_transformer_v3` package
(github.com/Pointcept/PointTransformerV3): pure PyTorch + Triton, no
spconv / torch_scatter / flash-attn needed. Make it importable with

    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH

Each pixel becomes one point (x, y, 0) with its C band values as the
feature. The whole batch is packed into one point set and separated with
`offset`. Level 0 keeps the rows it is given, so the decoder output is in
the same order as the input and can be reshaped back to [B, K, H, W].

    forward: [B, C, H, W] -> [B, num_classes, H, W]
"""

import torch
import torch.nn as nn

from point_transformer_v3 import Config, Point, PointTransformerV3


class PTv3Segmenter(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 7,
        gsd: float = 10.0,
        patch_size: int = 1024,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        dec_channels=(64, 64, 128, 256),
        dec_depths=(2, 2, 2, 2),
        head_dim: int = 16,
        drop_path: float = 0.3,
        precision: str = "auto",
    ):
        super().__init__()
        enc_channels = tuple(enc_channels)
        dec_channels = tuple(dec_channels)
        enc_depths = tuple(enc_depths)
        dec_depths = tuple(dec_depths)
        assert len(enc_channels) == len(enc_depths) == 5
        assert len(dec_channels) == len(dec_depths) == 4
        for c in enc_channels + dec_channels:
            assert c % head_dim == 0, (
                f"channel width {c} must be divisible by head_dim={head_dim}"
            )

        cfg = Config(
            in_channels=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            # constant head_dim (the default recipe also uses 16)
            enc_num_heads=tuple(c // head_dim for c in enc_channels),
            enc_patch_sizes=(patch_size,) * len(enc_channels),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_heads=tuple(c // head_dim for c in dec_channels),
            dec_patch_sizes=(patch_size,) * len(dec_channels),
            drop_path=drop_path,
        )
        self.backbone = PointTransformerV3(config=cfg, precision=precision)
        self.head = nn.Linear(dec_channels[0], num_classes)
        self.gsd = gsd
        self.num_classes = num_classes
        self._grid_cache = {}

    def _grid(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            # row-major (y, x) == feat flatten order of x.permute(0,2,3,1)
            g = torch.stack([xs, ys, torch.zeros_like(xs)], -1).reshape(-1, 3)
            self._grid_cache[key] = g.int()
        return self._grid_cache[key]

    def forward(self, x):
        B, C, H, W = x.shape
        g = self._grid(H, W, x.device)
        point = Point(
            coord=(g.float() * self.gsd).repeat(B, 1),
            grid_coord=g.repeat(B, 1),
            feat=x.permute(0, 2, 3, 1).reshape(-1, C).float(),
            offset=torch.arange(1, B + 1, device=x.device) * (H * W),
            grid_size=float(self.gsd),
        )
        out = self.backbone(point)                     # level-0 rows, input order
        logits = self.head(out["feat"].float())        # [B*H*W, K]
        return logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()