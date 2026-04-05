"""
UNet — PANGAEA Architecture, Standalone Interface
===================================================

Same architecture as PANGAEA benchmark UNet (encoder + decoder),
but with a simple tensor interface:

    model = UNet(in_channels=36, num_classes=2)
    logits = model(x)  # [B, 36, 256, 256] → [B, 2, 256, 256]

No PANGAEA base class dependency. No dict input.
Topology default: [64, 128, 256, 512, 1024]
"""

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS (from PANGAEA)
# ═══════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.mpconv(x)


class UpBlock(nn.Module):
    """ConvTranspose → cat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if sizes don't match
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════
# ENCODER
# ═══════════════════════════════════════════════════════════════════════

class UNetEncoder(nn.Module):
    """
    UNet encoder: InConv → Down × (n_levels - 1)
    Returns list of features [deepest, ..., shallowest]
    """

    def __init__(self, in_channels: int, topology: Sequence[int]):
        super().__init__()

        self.in_conv = InConv(in_channels, topology[0])

        down_dict = OrderedDict()
        n_layers = len(topology)
        for idx in range(n_layers):
            is_not_last = idx != n_layers - 1
            in_dim = topology[idx]
            out_dim = topology[idx + 1] if is_not_last else topology[idx]
            down_dict[f"down{idx + 1}"] = DownBlock(in_dim, out_dim)

        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x: torch.Tensor) -> list:
        x = self.in_conv(x)
        features = [x]
        for layer in self.down_seq.values():
            features.append(layer(features[-1]))
        features.reverse()  # deepest first
        return features


# ═══════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════

class UNetDecoder(nn.Module):
    """
    UNet decoder: Up × (n_levels - 1)
    Takes feature list [deepest, ..., shallowest]
    """

    def __init__(self, topology: Sequence[int]):
        super().__init__()

        n_layers = len(topology)

        # Build upward topology
        up_topo = [topology[0]]
        for idx in range(n_layers):
            is_not_last = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last else topology[idx]
            up_topo.append(out_dim)

        up_dict = OrderedDict()
        for idx in reversed(range(n_layers)):
            is_not_last = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            up_dict[f"up{idx + 1}"] = UpBlock(in_dim, out_dim)

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:
        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)
        return x1


# ═══════════════════════════════════════════════════════════════════════
# FULL UNET
# ═══════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """
    Full UNet: encoder + decoder + classification head.

    Same architecture as PANGAEA benchmark, standalone interface.

    Args:
        in_channels: input channels (e.g. 36 = 12 bands × 3 timesteps)
        num_classes: output classes
        topology: feature dims at each level [64, 128, 256, 512, 1024]

    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        topology: Sequence[int] = (64, 128, 256, 512, 1024),
    ):
        super().__init__()

        self.encoder = UNetEncoder(in_channels, topology)
        self.decoder = UNetDecoder(topology)
        self.out_conv = OutConv(topology[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)
        return self.out_conv(x)