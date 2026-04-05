"""
ViT + LTAE + UPerNet — Standalone Temporal Segmentation
=========================================================

Same architecture as PANGAEA benchmark:
  - ViT encoder: shared weights, processes each frame independently
  - LTAE: lightweight temporal attention, fuses T frames per feature layer
  - UPerNet: FPN + PPM decoder → segmentation logits

Standalone tensor interface (no PANGAEA base classes):

    model = ViTLTAEUPerNet(in_channels=12, num_classes=2, img_size=256)
    logits = model(x, doy=doy)
    # x:      [B, T, C, H, W]
    # doy:    [B, T] (day-of-year for positional encoding)
    # logits: [B, num_classes, H, W]

For non-temporal use (channel stacking):
    model = ViTUPerNet(in_channels=36, num_classes=2, img_size=256)
    logits = model(x)  # [B, 36, 256, 256] → [B, 2, 256, 256]
"""

import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed


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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            list of [B, embed_dim, H', W'] feature maps at output_layers
        """
        B = x.shape[0]
        S = self.spatial_size

        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.output_layers:
                # Remove CLS token, reshape to spatial
                feat = x[:, 1:]  # [B, N, D]
                feat = feat.transpose(1, 2).reshape(B, -1, S, S)  # [B, D, H', W']
                output.append(feat.contiguous())

        return output


# ═══════════════════════════════════════════════════════════════════════
# LTAE (Lightweight Temporal Attention Encoder)
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
        """
        Args:
            positions: [B*H*W, T] or [B, T]
        Returns:
            [B*H*W, T, d_model] or [B, T, d_model]
        """
        if not self._device_set:
            self.denom = self.denom.to(positions.device)
            self._device_set = True

        # [*, T, 1] / [1, 1, D] → [*, T, D]
        table = positions.unsqueeze(-1) / self.denom.unsqueeze(0).unsqueeze(0)
        table[..., 0::2] = torch.sin(table[..., 0::2])
        table[..., 1::2] = torch.cos(table[..., 1::2])
        return table


class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder for feature maps.

    Takes [B, D, T, H, W] → [B, D_out, H, W] via temporal attention.

    Args:
        in_channels: input feature dimension
        n_head: number of attention heads
        d_k: key/query dimension per head
        d_model: projection dimension (if different from in_channels)
        mlp_dims: MLP widths after attention
        positional_encoding: use sinusoidal temporal PE
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

        # Learnable query per head
        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))

        # Key projection
        self.fc_k = nn.Linear(self.d_model, n_head * d_k)
        nn.init.normal_(self.fc_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        # MLP after attention
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

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, D, T, H, W] — multi-temporal features
            batch_positions: [B, T] — temporal positions (DOY)

        Returns:
            [B, D_out, H, W] — temporally fused features
        """
        B, D, T, H, W = x.shape

        # Reshape: [B, D, T, H, W] → [B*H*W, T, D]
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, T, D]
        x_flat = x_perm.view(B * H * W, T, D)

        # Group norm on channel dim
        x_normed = self.in_norm(x_flat.permute(0, 2, 1)).permute(0, 2, 1)

        # Project to d_model
        if self.inconv is not None:
            x_proj = self.inconv(x_normed.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x_proj = x_normed

        # Add temporal positional encoding
        if self.pe is not None and batch_positions is not None:
            # Expand positions: [B, T] → [B*H*W, T]
            bp = batch_positions.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
            bp = bp.reshape(B * H * W, T)
            pe = self.pe(bp)  # [B*H*W, T, d_model//n_head]
            # Repeat for all heads
            pe = pe.repeat(1, 1, self.n_head)  # [B*H*W, T, d_model]
            x_proj = x_proj + pe

        # Multi-head attention with learned query
        # Keys: [B*H*W, T, n_head*d_k]
        K = self.fc_k(x_proj)  # [BHW, T, n_head*d_k]
        K = K.view(B * H * W, T, self.n_head, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, T, self.d_k)  # [n*BHW, T, d_k]

        # Query: [n_head, d_k] → [n*BHW, 1, d_k]
        Q = self.Q.unsqueeze(1).repeat(1, B * H * W, 1).view(-1, 1, self.d_k)

        # Attention: [n*BHW, 1, T]
        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Values: split x_proj into heads (d_model space, not raw D)
        d_v = self.d_model // self.n_head
        V = x_proj.view(B * H * W, T, self.n_head, d_v)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, T, d_v)

        # Weighted sum: [n*BHW, 1, d_v]
        out = torch.bmm(attn, V)
        out = out.view(self.n_head, B * H * W, d_v)
        out = out.permute(1, 0, 2).contiguous().view(B * H * W, -1)  # [BHW, d_model]

        # MLP
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)

        # Reshape: [BHW, D_out] → [B, D_out, H, W]
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

    Takes multi-scale feature list → segmentation logits.
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

        # Feature2Pyramid: scale features to consistent spatial sizes
        # For ViT, all layers have same spatial size, so just use identity
        self.rescale = nn.ModuleList([nn.Identity() for _ in in_channels])

        # PSP module on deepest features
        self.psp = PPM(pool_scales, in_channels[-1], channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Lateral convs (for all but deepest)
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

        # Final fusion
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: list, output_shape=None) -> torch.Tensor:
        """
        Args:
            features: list of [B, D, H', W'] from encoder
            output_shape: (H, W) target spatial size
        Returns:
            [B, num_classes, H, W]
        """
        # Rescale (identity for ViT)
        features = [self.rescale[i](f) for i, f in enumerate(features)]

        # PSP on deepest
        psp_outs = [features[-1]]
        psp_outs.extend(self.psp(features[-1]))
        psp_out = self.bottleneck(torch.cat(psp_outs, dim=1))

        # Lateral connections
        laterals = [lconv(features[i]) for i, lconv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False,
            )

        # FPN convs
        fpn_outs = [self.fpn_convs[i](laterals[i])
                     for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])

        # Resize all to same spatial size
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size,
                mode='bilinear', align_corners=False,
            )

        # Fuse
        feat = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        feat = self.dropout(feat)
        logits = self.conv_seg(feat)

        # Upsample to output shape
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

    Input:  [B, C, H, W]  (e.g. C = 36 = 12 bands × 3 timesteps)
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
# TEMPORAL: ViT + LTAE + UPerNet
# ═══════════════════════════════════════════════════════════════════════

class ViTLTAEUPerNet(nn.Module):
    """
    ViT + LTAE + UPerNet for multi-temporal segmentation.

    ViT processes each frame independently (shared weights).
    LTAE fuses temporal features per layer.
    UPerNet decodes to segmentation.

    Input:  [B, T, C, H, W] (sequence mode)
    DOY:    [B, T] (temporal positions for LTAE)
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
        ltae_n_head: int = 16,
        ltae_d_k: int = 4,
        ltae_d_model: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_layers = len(output_layers)

        # Shared ViT encoder
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )

        # One LTAE per feature layer
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

        # UPerNet decoder
        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * self.n_layers,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
            doy: [B, T] temporal positions (optional)
        Returns:
            [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # Encode each frame independently
        # Collect features per layer: layer_feats[l] = [B, D, T, H', W']
        all_feats = [[] for _ in range(self.n_layers)]

        for t in range(T):
            frame_feats = self.encoder(x[:, t])  # list of [B, D, H', W']
            for l, feat in enumerate(frame_feats):
                all_feats[l].append(feat)

        # Stack temporal: [B, D, T, H', W']
        layer_feats = [torch.stack(feats, dim=2) for feats in all_feats]

        # LTAE: fuse temporal → [B, D, H', W']
        fused = []
        for l in range(self.n_layers):
            fused.append(self.ltaes[l](layer_feats[l], batch_positions=doy))

        # UPerNet decode
        return self.decoder(fused, output_shape=(H, W))