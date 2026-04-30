"""
ResNet + UPerNet — Multi-Temporal Segmentation via Channel-Concat
====================================================================

Early temporal fusion: stack T frames along the channel dimension and
project back to C channels via a small conv block (PANGAEA-style
DoubleConv) before the encoder.

This matches PANGAEA's `UNetMT` pattern so numbers are directly comparable
to their published baselines on multi-temporal datasets like PASTIS.

Pipeline (multi-temporal, num_frames > 1):
    [B, T, C, H, W]
        → reshape: [B, T*C, H, W]
        → TimeMerge (DoubleConv): Conv(T*C → C) + BN + ReLU + Conv(C → C) + BN + ReLU
        → ResNet encoder → 4 multi-scale features
        → UPerNet decoder → [B, num_classes, H', W']
        → bilinear upsample → [B, num_classes, H, W]

Pipeline (single-frame, num_frames = 1):
    [B, C, H, W] (or [B, 1, C, H, W] which is squeezed)
        → standard ResNet+UPerNet (no TimeMerge — saves params, identical
          to a plain single-frame ResNetUPerNet)

ResNet variants exposed:
    - resnet_super_small  : Bottleneck [1,1,1,1]
    - resnet_small        : Bottleneck [1,2,3,1]
    - resnet50            : Bottleneck [3,4,6,3]
    - resnet101           : Bottleneck [3,4,23,3]
    - resnet152           : Bottleneck [3,8,36,3]

This file replaces the old ResNetUPerNetLTAE (late-fusion) variant.
ResNetUPerNet (single-frame) and ResNetClassifier are kept unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the canonical UPerNet decoder
from training.VIT.model_vit_upernet import UPerNetDecoder


# ═══════════════════════════════════════════════════════════════════════
# RESNET BUILDING BLOCKS (Bottleneck)
# ═══════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block (1×1 → 3×3 → 1×1, expansion=4)."""
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))

        if self.i_downsample is not None:
            identity = self.i_downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ═══════════════════════════════════════════════════════════════════════
# RESNET FEATURE EXTRACTOR (no classification head, exposes 4 scales)
# ═══════════════════════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    """
    ResNet backbone returning multi-scale feature maps for FPN/UPerNet.

    Drops the classification head (avgpool + fc). Exposes outputs of
    layer1..layer4 as a 4-element feature pyramid.

    Spatial reductions (relative to input size H):
        after stem (conv1+maxpool): H/4
        after layer1: H/4   (stride=1 in the first layer)
        after layer2: H/8
        after layer3: H/16
        after layer4: H/32

    Channel counts (Bottleneck, expansion=4):
        layer1: 256, layer2: 512, layer3: 1024, layer4: 2048
    """

    def __init__(self, ResBlock, layer_list, num_channels=3):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(num_channels, 64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 stages
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        e = ResBlock.expansion
        self.output_dim = [64 * e, 128 * e, 256 * e, 512 * e]

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers = [ResBlock(self.in_channels, planes,
                           i_downsample=ii_downsample, stride=stride)]
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            list of 4 multi-scale feature maps:
                [B, 256,  H/4,  W/4 ],
                [B, 512,  H/8,  W/8 ],
                [B, 1024, H/16, W/16],
                [B, 2048, H/32, W/32]
        """
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]


# ═══════════════════════════════════════════════════════════════════════
# TIME-MERGE BLOCK (PANGAEA-style DoubleConv)
# ═══════════════════════════════════════════════════════════════════════

class TimeMerge(nn.Module):
    """
    PANGAEA-style temporal aggregation block.

    Takes [B, T*C, H, W] and projects to [B, C, H, W] via two 3×3 conv
    layers with BN+ReLU. This mirrors the `DoubleConv(C*T, C)` pattern
    used in PANGAEA's UNetMT for multi-temporal datasets.

    The 3×3 kernel gives the merge slight spatial awareness; the two-layer
    depth provides non-linearity. This is what early-fusion baselines use
    in PANGAEA so our numbers are directly comparable to their PASTIS
    reference points.

    Args:
        in_channels:  channel count per frame (C)
        num_frames:   number of frames T (input has T*C channels)
    """

    def __init__(self, in_channels: int, num_frames: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        merged_in = in_channels * num_frames

        self.block = nn.Sequential(
            nn.Conv2d(merged_in, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T*C, H, W] (already channel-concatenated)
        Returns:
            [B, C, H, W]
        """
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════
# RESNET + UPERNET + CHANNEL-CONCAT (multi-temporal early fusion)
# ═══════════════════════════════════════════════════════════════════════

class ResNetUPerNetMT(nn.Module):
    """
    ResNet + UPerNet with channel-concat temporal aggregation (early fusion).

    Replaces the older LTAE-based late-fusion variant. Designed to match
    PANGAEA's UNetMT temporal handling for direct comparability on PASTIS.

    Pipeline (T > 1):
        [B, T, C, H, W]
            → reshape: [B, T*C, H, W]
            → TimeMerge (DoubleConv): [B, C, H, W]
            → ResNet encoder → 4 multi-scale features
            → UPerNet decoder → [B, num_classes, H', W']
            → bilinear upsample → [B, num_classes, H, W]

    Pipeline (T = 1): identical to ResNetUPerNet — TimeMerge is skipped
    entirely (saves ~C² × 18 params and one extra forward layer).

    Forward accepts inputs in either of these shapes:
        [B, C, H, W]      — single frame, T inferred as 1
        [B, T, C, H, W]   — multi-temporal, T inferred from shape

    Args:
        ResBlock:         Bottleneck class
        layer_list:       e.g. [3, 4, 6, 3] for ResNet50
        in_channels:      Bands per frame (e.g. 10 for S2)
        num_classes:      Output segmentation classes
        num_frames:       Number of temporal frames T. If 1, TimeMerge
                          is not constructed (identical to single-frame
                          ResNetUPerNet for cashew/sen1floods/burnscars/
                          forestnet/pv4ger).
        decoder_channels: UPerNet hidden channels
    """

    def __init__(
        self,
        ResBlock,
        layer_list,
        in_channels: int,
        num_classes: int,
        num_frames: int = 1,
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels

        # Time-merge only constructed when needed.
        if num_frames > 1:
            self.time_merge = TimeMerge(in_channels, num_frames)
        else:
            self.time_merge = None

        self.encoder = ResNetEncoder(
            ResBlock=ResBlock,
            layer_list=layer_list,
            num_channels=in_channels,
        )

        self.decoder = UPerNetDecoder(
            in_channels=self.encoder.output_dim,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   [B, C, H, W] (single frame) OR [B, T, C, H, W] (multi-temporal)
            doy: [B, T] day-of-year (accepted for trainer compatibility but
                 IGNORED — channel-concat early fusion has no notion of
                 temporal ordering / position. Matches PANGAEA's UNetMT
                 behavior on PASTIS.)

        Returns:
            logits: [B, num_classes, H, W]
        """
        if x.dim() == 5:
            # [B, T, C, H, W] → reshape to [B, T*C, H, W]
            B, T, C, H, W = x.shape
            if self.time_merge is None:
                # Model built for T=1 but received T>1.
                if T == 1:
                    x = x.squeeze(1)              # [B, C, H, W]
                else:
                    raise RuntimeError(
                        f"Model built with num_frames=1 but received T={T}. "
                        f"Construct with num_frames={T} for multi-temporal input."
                    )
            else:
                if T != self.num_frames:
                    raise RuntimeError(
                        f"Model built with num_frames={self.num_frames} but "
                        f"received T={T}. Mismatch — rebuild model or pad/trim "
                        f"input to T={self.num_frames}."
                    )
                x = x.reshape(B, T * C, H, W)
                x = self.time_merge(x)             # [B, C, H, W]
        else:
            # Already [B, C, H, W]
            if self.time_merge is not None:
                raise RuntimeError(
                    f"Model built for T={self.num_frames} (multi-temporal) but "
                    f"received 4D input [B, C, H, W]. Use 5D [B, T, C, H, W]."
                )

        H, W = x.shape[-2], x.shape[-1]
        features = self.encoder(x)
        logits = self.decoder(features, output_shape=(H, W))
        return logits


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE BUILDERS
# ═══════════════════════════════════════════════════════════════════════

_LAYER_CONFIGS = {
    "resnet_super_small": [1, 1, 1, 1],
    "resnet_small":       [1, 2, 3, 1],
    "resnet50":           [3, 4, 6, 3],
    "resnet101":          [3, 4, 23, 3],
    "resnet152":          [3, 8, 36, 3],
}


def build_resnet_upernet_mt(
    variant: str,
    in_channels: int,
    num_classes: int,
    num_frames: int = 1,
    decoder_channels: int = 256,
) -> ResNetUPerNetMT:
    """
    Build a ResNet+UPerNet with channel-concat temporal aggregation.

    For T=1 (single-frame), this is functionally identical to a plain
    ResNetUPerNet — TimeMerge is not constructed.
    For T>1, a TimeMerge block is added before the encoder.

    Args:
        variant:          one of _LAYER_CONFIGS keys (e.g. 'resnet50')
        in_channels:      bands per frame
        num_classes:      output segmentation classes
        num_frames:       T — number of temporal frames (default 1)
        decoder_channels: UPerNet hidden channels
    """
    if variant not in _LAYER_CONFIGS:
        raise ValueError(
            f"Unknown ResNet variant: {variant}. "
            f"Available: {list(_LAYER_CONFIGS.keys())}"
        )
    return ResNetUPerNetMT(
        ResBlock=Bottleneck,
        layer_list=_LAYER_CONFIGS[variant],
        in_channels=in_channels,
        num_classes=num_classes,
        num_frames=num_frames,
        decoder_channels=decoder_channels,
    )


# Back-compat alias: callers expecting the old `build_resnet_upernet`
# (single-frame) get an MT model with num_frames=1, which behaves
# identically to a non-temporal ResNet+UPerNet (no TimeMerge constructed).
def build_resnet_upernet(
    variant: str,
    in_channels: int,
    num_classes: int,
    decoder_channels: int = 256,
) -> ResNetUPerNetMT:
    """Build a single-frame ResNet+UPerNet (kept for back-compat)."""
    return build_resnet_upernet_mt(
        variant=variant,
        in_channels=in_channels,
        num_classes=num_classes,
        num_frames=1,
        decoder_channels=decoder_channels,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION: ResNet with avgpool + fc head (unchanged)
# ═══════════════════════════════════════════════════════════════════════

class ResNetClassifier(nn.Module):
    """
    ResNet for single-label image classification.

    Pipeline:
        [B, C, H, W]
            → ResNet stem + 4 stages → [B, 2048, H/32, W/32] (ResNet50)
            → AdaptiveAvgPool2d → [B, 2048, 1, 1]
            → flatten → [B, 2048]
            → Dropout → Linear → [B, num_classes]
    """

    def __init__(
        self,
        ResBlock,
        layer_list,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = ResNetEncoder(
            ResBlock=ResBlock,
            layer_list=layer_list,
            num_channels=in_channels,
        )

        feat_dim = self.encoder.output_dim[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   [B, C, H, W]
            doy: optional, accepted for trainer compatibility (ignored —
                 classification on a single frame doesn't use temporal info).
        Returns:
            logits: [B, num_classes]
        """
        feats = self.encoder(x)
        last = feats[-1]
        pooled = self.avgpool(last).flatten(1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def build_resnet_classifier(
    variant: str,
    in_channels: int,
    num_classes: int,
    dropout: float = 0.0,
) -> ResNetClassifier:
    """Build a ResNetClassifier by named variant (e.g. 'resnet50')."""
    if variant not in _LAYER_CONFIGS:
        raise ValueError(
            f"Unknown ResNet variant: {variant}. "
            f"Available: {list(_LAYER_CONFIGS.keys())}"
        )
    return ResNetClassifier(
        ResBlock=Bottleneck,
        layer_list=_LAYER_CONFIGS[variant],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )