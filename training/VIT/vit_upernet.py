"""
ViT-Small + UPerNet Baseline for MDAS Segmentation
====================================================

Standalone module: ViT-Small encoder with UPerNet decoder.
No dependency on PANGAEA base classes.

ViT-Small config:
    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, patch_size=4
    On 64×64 input → 16×16 = 256 patches (comparable to ViT on 224×224 with patch_size=16)

UPerNet decoder:
    Feature Pyramid from 4 intermediate ViT layers → PSP module → FPN → segmentation head.

Input:  [B, C, 64, 64]  (C adapts to any channel count)
Output: [B, num_classes, 64, 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# =============================================================================
# ViT BUILDING BLOCKS
# =============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with flexible input channels."""

    def __init__(self, img_size: int = 64, patch_size: int = 4,
                 in_channels: int = 368, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, num_patches, embed_dim]"""
        x = self.proj(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# ViT-Small ENCODER
# =============================================================================

class ViTSmallEncoder(nn.Module):
    """
    ViT-Small encoder that outputs multi-scale features from 4 intermediate layers.

    Args:
        in_channels: number of input channels
        img_size: spatial input size (default: 64)
        patch_size: patch size (default: 4 → 16×16 grid on 64×64)
        embed_dim: transformer hidden dimension (default: 384 for ViT-S)
        depth: number of transformer blocks (default: 12)
        num_heads: attention heads (default: 6)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        output_layers: which block indices to tap for multi-scale features
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int = 64,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        output_layers: tuple = (2, 5, 8, 11),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_layers = output_layers
        self.grid_size = img_size // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Positional embedding (learnable)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            list of 4 feature maps, each [B, embed_dim, grid_h, grid_w]
        """
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x)  # [B, N, D]

        # Prepend cls token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed

        # Forward through blocks, collect features
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in self.output_layers:
                # Apply norm at last layer
                out = self.norm(x) if i == self.blocks.__len__() - 1 else x

                # Remove cls token, reshape to spatial
                out = out[:, 1:]  # [B, N, D]
                out = out.transpose(1, 2).reshape(
                    B, self.embed_dim, self.grid_size, self.grid_size
                )
                outputs.append(out)

        return outputs


# =============================================================================
# UPerNet DECODER (standalone, no PANGAEA dependencies)
# =============================================================================

class PPM(nn.ModuleList):
    """Pooling Pyramid Module from PSPNet."""

    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        for scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        outs = []
        for ppm in self:
            out = ppm(x)
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
            outs.append(out)
        return outs


class Feature2Pyramid(nn.Module):
    """Convert same-resolution ViT features to a multi-scale pyramid."""

    def __init__(self, embed_dim: int, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()

        for k in rescales:
            if k == 4:
                self.ops.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ))
            elif k == 2:
                self.ops.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unsupported rescale factor: {k}")

    def forward(self, inputs: list) -> list:
        assert len(inputs) == len(self.rescales)
        return [op(x) for op, x in zip(self.ops, inputs)]


class UPerNetDecoder(nn.Module):
    """
    UPerNet decoder: PSP module + FPN on multi-scale features.

    Args:
        in_channels: list of input channel dims per scale (all same for ViT)
        channels: internal feature dimension
        num_classes: output classes
        pool_scales: PSP pooling scales
    """

    def __init__(
        self,
        in_channels: list,
        channels: int = 256,
        num_classes: int = 6,
        pool_scales: tuple = (1, 2, 3, 6),
    ):
        super().__init__()
        self.channels = channels

        # PSP on the deepest feature
        self.psp = PPM(pool_scales, in_channels[-1], channels)
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # FPN lateral + output convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels[:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

        # Final bottleneck
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: list of 4 feature maps [B, C, H_i, W_i] (multi-scale)

        Returns:
            logits: [B, num_classes, H_0, W_0] (at finest feature scale)
        """
        # PSP on deepest feature
        psp_out = [features[-1]]
        psp_out.extend(self.psp(features[-1]))
        psp_out = torch.cat(psp_out, dim=1)
        psp_out = self.psp_bottleneck(psp_out)

        # FPN laterals
        laterals = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )

        # FPN outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])

        # Upsample all to finest resolution
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size,
                mode="bilinear", align_corners=False,
            )

        out = torch.cat(fpn_outs, dim=1)
        out = self.fpn_bottleneck(out)
        out = self.dropout(out)
        out = self.conv_seg(out)
        return out


# =============================================================================
# COMBINED MODEL: ViT-Small + UPerNet
# =============================================================================

class ViTUPerNet(nn.Module):
    """
    ViT-Small encoder + UPerNet decoder for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 368 for HySpex, 12 for S2).
    num_classes : int
        Number of output classes (default: 6 for MDAS).
    img_size : int
        Spatial input size (default: 64).
    patch_size : int
        ViT patch size (default: 4 → 16×16 grid on 64×64).
    embed_dim : int
        ViT hidden dim (default: 384 for ViT-Small).
    depth : int
        ViT depth (default: 12).
    num_heads : int
        Attention heads (default: 6).
    decoder_channels : int
        UPerNet internal feature dim (default: 256).
    output_layers : tuple
        Which ViT blocks to tap for multi-scale features.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        img_size: int = 64,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        decoder_channels: int = 256,
        output_layers: tuple = (2, 5, 8, 11),
    ):
        super().__init__()

        self.encoder = ViTSmallEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            output_layers=output_layers,
        )

        # ViT outputs same dim at all layers; Feature2Pyramid creates multi-scale
        # rescales: [4x, 2x, 1x, 0.5x] relative to patch grid
        self.neck = Feature2Pyramid(embed_dim=embed_dim, rescales=[4, 2, 1, 0.5])

        # After neck, all features have embed_dim channels but different spatial sizes
        in_channels_decoder = [embed_dim] * len(output_layers)

        self.decoder = UPerNetDecoder(
            in_channels=in_channels_decoder,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] (H=W=img_size expected)

        Returns:
            logits: [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        # Encoder: multi-scale features
        features = self.encoder(x)  # list of 4 × [B, D, grid, grid]

        # Neck: create pyramid
        features = self.neck(features)  # [B,D,64,64], [B,D,32,32], [B,D,16,16], [B,D,8,8]

        # Decoder
        logits = self.decoder(features)  # [B, num_classes, 64, 64]

        # Ensure output matches input spatial size
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)