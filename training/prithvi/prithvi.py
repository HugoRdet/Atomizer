"""
Prithvi + UPerNet — Standalone, From-Scratch Training
=======================================================

Prithvi 3D ViT encoder + UPerNet decoder for multi-temporal segmentation.
No PANGAEA base class dependency. No pretrained weights.

Adapted from: https://github.com/NASA-IMPACT/hls-foundation-os

Architecture comparison vs Atomizer:
    Prithvi:  metadata as explicit 3D positional embedding + tubelet conv
    Atomizer: metadata natively encoded in tokens

Both trained from scratch — isolates architectural contribution.

Input format: [B, C, T, H, W]

For PASTIS (S2+S1, 6 frames, 128×128):
    PrithviUPerNet(in_chans=12, num_frames=6, img_size=128, num_classes=20)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block


# =============================================================================
# 3D SIN-COS POSITIONAL EMBEDDING
# Reimplemented locally — no PANGAEA dependency
# =============================================================================

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: (T, H, W) tuple
    Returns: [T*H*W, embed_dim] or [1+T*H*W, embed_dim] with cls token
    """
    assert embed_dim % 3 == 0, "embed_dim must be divisible by 3 for 3D sincos"
    T, H, W = grid_size

    grid_t = np.arange(T, dtype=np.float32)
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)

    grid   = np.meshgrid(grid_w, grid_h, grid_t, indexing="xy")
    grid   = np.stack(grid, axis=0).reshape(3, 1, W, H, T)

    d          = embed_dim // 3
    pos_embed  = np.concatenate([
        _get_1d_sincos(d, grid[2].reshape(-1)),  # temporal
        _get_1d_sincos(d, grid[1].reshape(-1)),  # height
        _get_1d_sincos(d, grid[0].reshape(-1)),  # width
    ], axis=1)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)
    out    = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# =============================================================================
# PATCH EMBED (3D — tubelet)
# Matches PANGAEA's PatchEmbed exactly
# =============================================================================

class PatchEmbed(nn.Module):
    """
    3D patch embedding: [B, C, T, H, W] → [B, N, D]
    Matches PANGAEA's PatchEmbed.
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=3,
                 tubelet_size=1, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size     = img_size
        self.patch_size   = patch_size
        self.num_frames   = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size    = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = (self.grid_size[0]
                            * self.grid_size[1]
                            * self.grid_size[2])
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size,     patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return self.norm(x)


# =============================================================================
# PRITHVI ENCODER — standalone
# =============================================================================

class PrithviEncoder(nn.Module):
    """
    Prithvi 3D ViT encoder — from scratch, no pretrained weights.
    Matches PANGAEA's Prithvi_Encoder architecture exactly.

    Key difference from PANGAEA's forward:
        PANGAEA uses .squeeze(2) designed for single-temporal output (T_patches=1).
        Here we use .mean(dim=2) to collapse temporal → [B, D, H', W'] for UPerNet.
        This is equivalent when tubelet_size=num_frames.

    Args:
        img_size:      spatial size
        patch_size:    spatial patch size
        num_frames:    temporal frames
        tubelet_size:  temporal patch size (1 = frame-by-frame)
        in_chans:      channels per frame
        embed_dim:     transformer hidden dim
        depth:         transformer depth
        num_heads:     attention heads
        output_layers: block indices to tap for multi-scale features
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        output_layers: tuple = (5, 11, 17, 23),
    ):
        super().__init__()

        self.img_size      = img_size
        self.patch_size    = patch_size
        self.num_frames    = num_frames
        self.tubelet_size  = tubelet_size
        self.in_chans      = in_chans
        self.embed_dim     = embed_dim
        self.output_layers = set(output_layers)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            num_frames=num_frames, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )

        num_patches    = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,   # fixed sin-cos
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            list of [B, D, H', W'] feature maps from output_layers
        """
        B = x.shape[0]
        H_patches = self.img_size // self.patch_size
        T_patches = self.num_frames // self.tubelet_size

        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed

        last_idx = len(self.blocks) - 1
        outputs  = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = self.norm(x) if i == last_idx else x
                # [B, N, D] → [B, D, T', H', W']
                out = (out[:, 1:, :]
                       .permute(0, 2, 1)
                       .view(B, self.embed_dim, T_patches,
                             H_patches, H_patches)
                       .contiguous())
                # Collapse temporal → [B, D, H', W']
                # PANGAEA uses .squeeze(2) which assumes T_patches=1
                # We use .mean(dim=2) which handles any T_patches
                out = out.mean(dim=2)
                outputs.append(out)

        return outputs  # 4 × [B, D, H', W']


# =============================================================================
# UPerNet DECODER
# =============================================================================

class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        for scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        return [F.interpolate(ppm(x), size=x.shape[2:],
                              mode="bilinear", align_corners=False)
                for ppm in self]


class Feature2Pyramid(nn.Module):
    def __init__(self, embed_dim, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.ops = nn.ModuleList()
        for k in rescales:
            if k == 4:
                self.ops.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
                    nn.BatchNorm2d(embed_dim), nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
                ))
            elif k == 2:
                self.ops.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2))
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(2, stride=2))
            else:
                raise ValueError(f"Unsupported rescale: {k}")

    def forward(self, inputs):
        return [op(x) for op, x in zip(self.ops, inputs)]


class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels, channels=256,
                 num_classes=20, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.psp = PPM(pool_scales, in_channels[-1], channels)
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels,
                      channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for c in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for _ in in_channels[:-1]
        ])
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels,
                      3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.dropout  = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features):
        psp_out = torch.cat([features[-1]] + self.psp(features[-1]), dim=1)
        psp_out = self.psp_bottleneck(psp_out)

        laterals = [conv(features[i])
                    for i, conv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False)

        fpn_outs    = [self.fpn_convs[i](laterals[i])
                       for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size,
                                        mode="bilinear", align_corners=False)

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.conv_seg(self.dropout(out))


# =============================================================================
# FULL MODEL
# =============================================================================

class PrithviUPerNet(nn.Module):
    """
    Prithvi 3D ViT + UPerNet — from scratch, no pretrained weights.

    Uses Prithvi's native 3D tubelet architecture:
        [B, C, T, H, W] → 3D Conv tubelet embedding → transformer → UPerNet

    This preserves Prithvi's original design for a fair architectural
    comparison: Prithvi handles temporal via explicit 3D positional
    embedding, Atomizer handles it natively in tokens.

    For PASTIS S2+S1, 6 frames, 128×128:
        PrithviUPerNet(in_chans=12, num_frames=6,
                       img_size=128, num_classes=20)
    """

    def __init__(
        self,
        in_chans: int = 6,
        num_frames: int = 6,
        img_size: int = 128,
        patch_size: int = 16,
        tubelet_size: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_classes: int = 20,
        decoder_channels: int = 256,
        output_layers: tuple = (2, 5, 8, 11),
    ):
        super().__init__()

        self.encoder = PrithviEncoder(
            img_size=img_size, patch_size=patch_size,
            num_frames=num_frames, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads,
            output_layers=output_layers,
        )
        self.neck    = Feature2Pyramid(embed_dim=embed_dim,
                                       rescales=[4, 2, 1, 0.5])
        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * len(output_layers),
            channels=decoder_channels,
            num_classes=num_classes,
        )

        params = self.count_parameters()
        print(f"[PrithviUPerNet] {params:,} parameters "
              f"({in_chans} ch/frame × {num_frames} frames, "
              f"{img_size}×{img_size}, {num_classes} classes, "
              f"3D tubelet architecture)")

    def forward(self, x, doy=None):
        """
        Args:
            x:   [B, C, T, H, W] — Prithvi native format
            doy: ignored (temporal via 3D sin-cos pos embed)
        Returns:
            logits: [B, num_classes, H, W]
        """
        input_size = x.shape[-2:]
        features   = self.neck(self.encoder(x))
        logits     = self.decoder(features)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# UTILITY: interpolate pos embed for variable T at inference
# =============================================================================

def interpolate_prithvi_pos_embed(model, train_T, test_T,
                                   patch_size=16, img_size=128):
    """
    Interpolate Prithvi's 3D sin-cos positional embedding from train_T to test_T.

    Called at inference time when evaluating on a different number of frames
    than the model was trained on. Uses trilinear interpolation.

    Args:
        model:      PrithviUPerNet instance
        train_T:    number of frames used during training
        test_T:     number of frames to use at inference
        patch_size: spatial patch size (default 16)
        img_size:   spatial image size (default 128)
    """
    if train_T == test_T:
        return

    H_patches = img_size // patch_size
    D         = model.encoder.pos_embed.shape[-1]

    pos_embed = model.encoder.pos_embed.data   # [1, 1+T*H'*W', D]
    cls_tok   = pos_embed[:, :1, :]             # [1, 1, D]
    spatial   = pos_embed[:, 1:, :]             # [1, T*H'*W', D]

    # [1, T*H'*W', D] → [1, D, T, H', W']
    spatial = spatial.reshape(1, train_T, H_patches, H_patches, D)
    spatial = spatial.permute(0, 4, 1, 2, 3)

    # Trilinear interpolation to new T
    spatial_interp = F.interpolate(
        spatial,
        size=(test_T, H_patches, H_patches),
        mode="trilinear",
        align_corners=False,
    )

    # [1, D, T', H', W'] → [1, T'*H'*W', D]
    spatial_interp = (spatial_interp
                      .permute(0, 2, 3, 4, 1)
                      .reshape(1, test_T * H_patches * H_patches, D))

    new_pos_embed = torch.cat([cls_tok, spatial_interp], dim=1)
    model.encoder.pos_embed = torch.nn.Parameter(
        new_pos_embed, requires_grad=False)

    print(f"[PosEmbed] Interpolated T={train_T} → T={test_T}, "
          f"{pos_embed.shape} → {new_pos_embed.shape}")