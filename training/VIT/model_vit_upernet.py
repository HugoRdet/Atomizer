"""
ViT + LTAE + UPerNet — Standalone Temporal Segmentation
=========================================================

Three model variants:

  ViTUPerNet:        Non-temporal. Channel-stacked frames → ViT → UPerNet.
                     Input: [B, C, H, W]

  ViTLTAEUPerNet:    LTAE between encoder and decoder (per FPN layer).
                     Per-frame ViT → per-layer LTAE → UPerNet.
                     Input: [B, T, C, H, W], doy=[B, T]

  ViTUPerNetLTAE:    LTAE AFTER full UPerNet decode (at output resolution).
                     Per-frame ViT → per-frame UPerNet (features only)
                     → SpatioTemporalLTAE → 1×1 conv → upsample.
                     Input: [B, T, C, H, W], doy=[B, T]

The LTAE-AFTER variant matches the same architectural template as UNetLTAE
and AtomiserLTAE: temporal aggregation happens at output resolution after
the full encode-decode pipeline, before the final classifier head.
"""

import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

# Reuse the canonical SpatioTemporalLTAE implementation.
from training.ltae.ltae import SpatioTemporalLTAE


# ═══════════════════════════════════════════════════════════════════════
# VIT ENCODER
# ═══════════════════════════════════════════════════════════════════════

class ViTEncoder(nn.Module):
    """
    ViT encoder returning multi-scale feature maps.

    Extracts features at specified block indices, reshapes to spatial maps.

    Input:  [B, C, H, W]
    Output: list of [B, embed_dim, H/P, W/P] at output_layers
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int = 256,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        output_layers: tuple = (2, 5, 8, 11),
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_layers = list(output_layers)
        self.output_dim = [embed_dim] * len(output_layers)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.spatial_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,
        )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_pos_embed()
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_pos_embed(self):
        """Sin-cos positional embeddings."""
        num_patches = self.patch_embed.num_patches
        grid_size = int(num_patches ** 0.5)
        d = self.embed_dim

        pos = np.arange(grid_size, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(pos, pos, indexing='ij')
        grid = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)

        pe = np.zeros((num_patches, d))
        for i in range(d // 4):
            freq = 1.0 / (10000 ** (4 * i / d))
            pe[:, 4*i]   = np.sin(grid[:, 0] * freq)
            pe[:, 4*i+1] = np.cos(grid[:, 0] * freq)
            pe[:, 4*i+2] = np.sin(grid[:, 1] * freq)
            pe[:, 4*i+3] = np.cos(grid[:, 1] * freq)

        cls_pe = np.zeros((1, d))
        full_pe = np.concatenate([cls_pe, pe], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(full_pe).float().unsqueeze(0))

    def forward(self, x: torch.Tensor) -> list:
        B = x.shape[0]
        S = self.spatial_size

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)
            if i in self.output_layers:
                feat = x[:, 1:]
                feat = feat.transpose(1, 2).reshape(B, -1, S, S)
                output.append(feat.contiguous())
        return output


# ═══════════════════════════════════════════════════════════════════════
# LTAE (per-FPN-layer variant — used by ViTLTAEUPerNet)
# ═══════════════════════════════════════════════════════════════════════

class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""

    def __init__(self, d_model: int, T: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.T = T
        self.denom = torch.pow(
            T, 2 * (torch.arange(d_model).float() // 2) / d_model
        )
        self._device_set = False

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if not self._device_set:
            self.denom = self.denom.to(positions.device)
            self._device_set = True
        table = positions.unsqueeze(-1) / self.denom.unsqueeze(0).unsqueeze(0)
        table[..., 0::2] = torch.sin(table[..., 0::2])
        table[..., 1::2] = torch.cos(table[..., 1::2])
        return table


class LTAE(nn.Module):
    """
    Per-FPN-layer LTAE used inside ViTLTAEUPerNet.

    Takes [B, D, T, H, W] → [B, D_out, H, W] via temporal attention.
    """

    def __init__(
        self,
        in_channels: int = 384,
        n_head: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        mlp_dims: list = None,
        positional_encoding: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        if mlp_dims is None:
            mlp_dims = [self.d_model, in_channels]

        self.in_norm = nn.GroupNorm(
            num_groups=min(n_head, in_channels),
            num_channels=in_channels,
        )

        if positional_encoding:
            self.pe = TemporalPositionalEncoder(self.d_model // n_head)
        else:
            self.pe = None

        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc_k = nn.Linear(self.d_model, n_head * d_k)
        nn.init.normal_(self.fc_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.BatchNorm1d(mlp_dims[i + 1]),
                nn.ReLU(),
            ])
        self.mlp = nn.Sequential(*layers)

        self.out_norm = nn.GroupNorm(
            num_groups=min(n_head, mlp_dims[-1]),
            num_channels=mlp_dims[-1],
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None):
        B, D, T, H, W = x.shape
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous()
        x_flat = x_perm.view(B * H * W, T, D)

        x_normed = self.in_norm(x_flat.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            x_proj = self.inconv(x_normed.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x_proj = x_normed

        if self.pe is not None and batch_positions is not None:
            bp = batch_positions.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
            bp = bp.reshape(B * H * W, T)
            pe = self.pe(bp)
            pe = pe.repeat(1, 1, self.n_head)
            x_proj = x_proj + pe

        K = self.fc_k(x_proj)
        K = K.view(B * H * W, T, self.n_head, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, T, self.d_k)

        Q = self.Q.unsqueeze(1).repeat(1, B * H * W, 1).view(-1, 1, self.d_k)

        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        d_v = self.d_model // self.n_head
        V = x_proj.view(B * H * W, T, self.n_head, d_v)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, T, d_v)

        out = torch.bmm(attn, V)
        out = out.view(self.n_head, B * H * W, d_v)
        out = out.permute(1, 0, 2).contiguous().view(B * H * W, -1)

        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)

        D_out = out.shape[-1]
        out = out.view(B, H, W, D_out).permute(0, 3, 1, 2)
        return out


# ═══════════════════════════════════════════════════════════════════════
# PPM (Pyramid Pooling Module)
# ═══════════════════════════════════════════════════════════════════════

class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        for ps in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        outs = []
        for ppm in self:
            out = ppm(x)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear',
                                align_corners=False)
            outs.append(out)
        return outs


# ═══════════════════════════════════════════════════════════════════════
# UPERNET DECODER
# ═══════════════════════════════════════════════════════════════════════

class UPerNetDecoder(nn.Module):
    """
    UPerNet FPN + PPM decoder.

    Takes multi-scale feature list → segmentation logits OR pre-classifier
    features (when return_features=True).

    The optional `return_features=True` mode skips the final 1×1 conv_seg
    and skips the upsample-to-output_shape step. This is used by
    ViTUPerNetLTAE which performs temporal aggregation on features then
    classifies once.
    """

    def __init__(
        self,
        in_channels: list,
        channels: int = 256,
        num_classes: int = 2,
        pool_scales: tuple = (1, 2, 3, 6),
    ):
        super().__init__()

        self.channels = channels

        self.rescale = nn.ModuleList([nn.Identity() for _ in in_channels])

        self.psp = PPM(pool_scales, in_channels[-1], channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ic in in_channels[:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(ic, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: list, output_shape=None,
                return_features: bool = False):
        """
        Args:
            features:        list of [B, D, H', W'] from encoder
            output_shape:    (H, W) target spatial size for final upsample
            return_features: if True, return pre-classifier features at FPN's
                             native resolution (skips conv_seg and upsample).
        Returns:
            logits:   [B, num_classes, H, W]   (default)
            features: [B, channels,    H', W'] (if return_features=True)
        """
        features = [self.rescale[i](f) for i, f in enumerate(features)]

        psp_outs = [features[-1]]
        psp_outs.extend(self.psp(features[-1]))
        psp_out = self.bottleneck(torch.cat(psp_outs, dim=1))

        laterals = [lconv(features[i]) for i, lconv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False,
            )

        fpn_outs = [self.fpn_convs[i](laterals[i])
                    for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])

        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size,
                mode='bilinear', align_corners=False,
            )

        feat = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        feat = self.dropout(feat)

        if return_features:
            return feat  # [B, channels, H_fpn, W_fpn]

        logits = self.conv_seg(feat)

        if output_shape is not None:
            logits = F.interpolate(logits, size=output_shape,
                                   mode='bilinear', align_corners=False)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# NON-TEMPORAL: ViT + UPerNet (channel stacking)
# ═══════════════════════════════════════════════════════════════════════

class ViTUPerNet(nn.Module):
    """
    ViT + UPerNet for non-temporal segmentation (channel stacking).

    Input:  [B, C, H, W]
    Output: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 256,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.img_size = img_size

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )

        self.decoder = UPerNetDecoder(
            in_channels=self.encoder.output_dim,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        features = self.encoder(x)
        return self.decoder(features, output_shape=(H, W))


# ═══════════════════════════════════════════════════════════════════════
# TIME-MERGE BLOCK (PANGAEA-style DoubleConv)
# ═══════════════════════════════════════════════════════════════════════

class TimeMerge(nn.Module):
    """
    PANGAEA-style temporal aggregation block.

    Takes [B, T*C, H, W] and projects to [B, C, H, W] via two 3×3 conv
    layers with BN+ReLU. Mirrors the `DoubleConv(C*T, C)` pattern in
    PANGAEA's UNetMT / shared with ResNetUPerNetMT.

    For multi-task training, this serves as a per-task input adapter that
    maps variable temporal-channel dimensions (T*C per task) to a uniform
    C dimension, allowing a single shared backbone across tasks.

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
# VIT + UPERNET + CHANNEL-CONCAT TEMPORAL (early fusion via TimeMerge)
# ═══════════════════════════════════════════════════════════════════════

class ViTUPerNetMT(nn.Module):
    """
    ViT + UPerNet with channel-concat temporal aggregation (early fusion).

    Mirror of ResNetUPerNetMT but with a ViT spatial encoder. Uses a
    PANGAEA-style TimeMerge DoubleConv to project T*C channels down to C
    before the patch embedding.

    Pipeline (T > 1):
        [B, T, C, H, W]
            → reshape: [B, T*C, H, W]
            → TimeMerge (DoubleConv): [B, C, H, W]
            → ViT encoder → 4 multi-scale features
            → UPerNet decoder → [B, num_classes, H', W']
            → bilinear upsample → [B, num_classes, H, W]

    Pipeline (T = 1): identical to ViTUPerNet — TimeMerge is skipped
    entirely (saves the extra DoubleConv layer).

    Forward accepts inputs in either of these shapes:
        [B, C, H, W]      — single frame, T inferred as 1
        [B, T, C, H, W]   — multi-temporal, T inferred from shape

    Args:
        in_channels:      Bands per frame (e.g. 10 for S2)
        num_classes:      Output segmentation classes
        num_frames:       Number of temporal frames T. If 1, TimeMerge
                          is not constructed.
        img_size:         Input spatial size (must equal --crop_size).
        embed_dim:        ViT hidden dim
        depth:            ViT depth
        num_heads:        ViT attention heads
        patch_size:       ViT patch size
        output_layers:    Block indices to extract features from
        decoder_channels: UPerNet hidden channels
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_frames: int = 1,
        img_size: int = 256,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.img_size = img_size

        # TimeMerge only when T > 1.
        if num_frames > 1:
            self.time_merge = TimeMerge(in_channels, num_frames)
        else:
            self.time_merge = None

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
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
                 temporal ordering / position).

        Returns:
            logits: [B, num_classes, H, W]
        """
        if x.dim() == 5:
            # [B, T, C, H, W] → reshape to [B, T*C, H, W]
            B, T, C, H, W = x.shape
            if self.time_merge is None:
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
        return self.decoder(features, output_shape=(H, W))


def build_vit_upernet_mt(
    in_channels: int,
    num_classes: int,
    num_frames: int = 1,
    img_size: int = 256,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    patch_size: int = 16,
    output_layers: tuple = (2, 5, 8, 11),
    decoder_channels: int = 256,
) -> ViTUPerNetMT:
    """
    Build a ViTUPerNetMT (channel-concat early fusion via TimeMerge).

    For T=1 (single-frame), this is functionally identical to a plain
    ViTUPerNet — TimeMerge is not constructed.
    For T>1, a TimeMerge block is added before the ViT encoder.
    """
    return ViTUPerNetMT(
        in_channels=in_channels,
        num_classes=num_classes,
        num_frames=num_frames,
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        output_layers=output_layers,
        decoder_channels=decoder_channels,
    )


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL: ViT + per-FPN-LTAE + UPerNet (LTAE BETWEEN encoder and decoder)
# ═══════════════════════════════════════════════════════════════════════

class ViTLTAEUPerNet(nn.Module):
    """
    ViT + LTAE + UPerNet for multi-temporal segmentation.

    LTAE placement: BETWEEN encoder and decoder (one per FPN feature layer).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 256,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
        ltae_n_head: int = 16,
        ltae_d_k: int = 4,
        ltae_d_model: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_layers = len(output_layers)

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )

        self.ltaes = nn.ModuleList([
            LTAE(
                in_channels=embed_dim,
                n_head=ltae_n_head,
                d_k=ltae_d_k,
                d_model=ltae_d_model,
                mlp_dims=[ltae_d_model, embed_dim],
                positional_encoding=True,
            )
            for _ in range(self.n_layers)
        ])

        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * self.n_layers,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x, doy=None):
        B, T, C, H, W = x.shape

        all_feats = [[] for _ in range(self.n_layers)]
        for t in range(T):
            frame_feats = self.encoder(x[:, t])
            for l, feat in enumerate(frame_feats):
                all_feats[l].append(feat)

        layer_feats = [torch.stack(feats, dim=2) for feats in all_feats]

        fused = []
        for l in range(self.n_layers):
            fused.append(self.ltaes[l](layer_feats[l], batch_positions=doy))

        return self.decoder(fused, output_shape=(H, W))


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL: ViT + UPerNet + LTAE (LTAE AFTER full encode-decode)
# ═══════════════════════════════════════════════════════════════════════

class ViTUPerNetLTAE(nn.Module):
    """
    ViT + UPerNet + LTAE for multi-temporal segmentation.

    LTAE placement: AFTER full UPerNet decode (at FPN's native output
    resolution), BEFORE the final classification head.

    Pipeline:
        [B, T, C, H, W]
            → per-frame ViT encoder (shared)        → list of [B, T, D, H', W']
            → per-frame UPerNet decoder (features)  → [B, T, dec_ch, H_fpn, W_fpn]
            → SpatioTemporalLTAE (per-pixel)         → [B, dec_ch, H_fpn, W_fpn]
            → 1×1 classification head                → [B, num_classes, H_fpn, W_fpn]
            → upsample to (H, W)                     → [B, num_classes, H, W]

    This matches the same temporal-aggregation-at-output-resolution
    template as UNetLTAE and AtomiserLTAE.

    Note on cost: LTAE here runs at FPN's native resolution (typically
    H/patch_size for ViT, e.g. 8×8 for img=128, patch=16). Cheap.

    Args:
        in_channels:      Bands per frame (e.g. 10 for S2, 12 for S2+S1)
        num_classes:      Output segmentation classes
        img_size:         Input spatial size
        embed_dim:        ViT hidden dim
        depth:            ViT depth
        num_heads:        ViT attention heads
        patch_size:       ViT patch size
        output_layers:    Which ViT block indices to tap for FPN features
        decoder_channels: UPerNet hidden channels (also LTAE input channels)
        ltae_n_head:      LTAE attention heads
        ltae_d_k:         LTAE key dim per head
        ltae_d_model:     LTAE internal projection dim
        ltae_dropout:     LTAE attention dropout
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 256,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
        ltae_n_head: int = 16,
        ltae_d_k: int = 4,
        ltae_d_model: int = 256,
        ltae_dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_layers = len(output_layers)

        # Per-frame ViT encoder (shared weights across frames)
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )

        # Per-frame UPerNet decoder (shared weights, returns pre-classifier features)
        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * self.n_layers,
            channels=decoder_channels,
            num_classes=num_classes,
        )

        # Temporal aggregation at FPN's output resolution (canonical LTAE)
        self.temporal = SpatioTemporalLTAE(
            in_channels=decoder_channels,
            n_heads=ltae_n_head,
            d_k=ltae_d_k,
            d_model=ltae_d_model,
            dropout=ltae_dropout,
        )

        # Final classification head (after temporal aggregation)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   [B, T, C, H, W]
            doy: [B, T] day-of-year (optional, for LTAE positional encoding)
        Returns:
            logits: [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # Per-frame encode-decode (shared weights, batched via merging B and T).
        # This is equivalent to a Python loop over T but uses Conv2d's natural
        # batch parallelism instead.
        x_flat = x.reshape(B * T, C, H, W)

        feats_list = self.encoder(x_flat)  # list of [B*T, D, H', W']
        feat_flat = self.decoder(
            feats_list, output_shape=None, return_features=True,
        )  # [B*T, dec_ch, H_fpn, W_fpn]

        # Reshape back to temporal: [B, T, dec_ch, H_fpn, W_fpn]
        _, dec_ch, H_fpn, W_fpn = feat_flat.shape
        feat_t = feat_flat.reshape(B, T, dec_ch, H_fpn, W_fpn)

        # Per-pixel LTAE temporal aggregation
        agg, _attn = self.temporal(feat_t, doy=doy)  # [B, dec_ch, H_fpn, W_fpn]

        # Classify, then upsample to input resolution
        logits = self.head(agg)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False,
            )
        return logits


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION: ViT with global-pooled token + linear head
# ═══════════════════════════════════════════════════════════════════════

class ViTClassifier(nn.Module):
    """
    Vanilla ViT classifier for single-label image classification.

    Standalone (no UPerNet), uses its own patch embed + transformer blocks
    + a CLS token, ends with LayerNorm and a linear classifier.

    Input:  [B, C, H, W]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 320,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] (H, W must equal img_size)
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        x = self.patch_embed(x)                              # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)               # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                       # [B, N+1, D]
        x = x + self.pos_embed                               # add pos enc

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                    # [B, D]
        cls_out = self.dropout(cls_out)
        return self.head(cls_out)                            # [B, num_classes]