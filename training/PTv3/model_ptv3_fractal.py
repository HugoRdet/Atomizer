"""
PTv3 for FRACTAL (LiDAR + VHR), random init, no pretrained weights.

Uses the standalone `point_transformer_v3` package:
    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH

    forward(batch) -> logits [Nv, num_classes]
        one row per voxel representative, in input order
        (batch["coord"] / ["grid_coord"] / ["feat"] / ["offset"] from
        fractal_ptv3_collate). The trainer gathers them onto the
        evaluated LiDAR points with batch["eval_index"].
"""

import torch
import torch.nn as nn

from point_transformer_v3 import Config, Point, PointTransformerV3


class PTv3Fractal(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 7,
        grid_size: float = 0.1,
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
        enc_depths   = tuple(enc_depths)
        dec_depths   = tuple(dec_depths)
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
            enc_num_heads=tuple(c // head_dim for c in enc_channels),
            enc_patch_sizes=(patch_size,) * len(enc_channels),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_heads=tuple(c // head_dim for c in dec_channels),
            dec_patch_sizes=(patch_size,) * len(dec_channels),
            drop_path=drop_path,
        )
        self.backbone    = PointTransformerV3(config=cfg, precision=precision)
        self.head        = nn.Linear(dec_channels[0], num_classes)
        self.grid_size   = float(grid_size)
        self.num_classes = num_classes

    def forward(self, batch):
        grid = batch["grid_coord"].int()
        point = Point(
            coord=batch["coord"].float(),
            grid_coord=grid,
            feat=batch["feat"].float(),
            offset=batch["offset"].long(),
            grid_size=self.grid_size,
        )
        out = self.backbone(point)
        # Cheap guard: output rows must be the input rows, in input order
        # (verified on Cashew; kept as a runtime check since eval_index
        # relies on it).
        if not torch.equal(out["grid_coord"].int(), grid):
            raise RuntimeError("PTv3 output rows are not in input order.")
        return self.head(out["feat"].float())          # [Nv, K]
