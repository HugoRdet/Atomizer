"""
RAMEN Classifier — Standalone Multi-modal Classification
============================================================

RAMEN encoder + linear classification head, mirroring ViTClassifier's
contract (CLS token -> LayerNorm -> Linear) but for RAMEN's native
multi-modal input.

Differences from RAMENBackbone (used by RAMENUPerNet for segmentation):
  - No output_layers / multi-scale feature extraction — classification
    only needs the final CLS token, so this runs full depth and returns
    one vector, not a list of spatial feature maps.
  - No windowing/tiling. Built for small single-date tiles (e.g.
    EuroSAT's 64x64) that fit in one forward pass — unlike Sen1Floods11's
    512x512, sliding-window inference is unnecessary here. If your tiles
    are large enough to need tiling, that's a sign you actually want a
    dense/segmentation-style RAMEN model instead of this one.
  - No temporal (LTAE2d) path — EuroSAT-style classification datasets are
    single-date, so this class only handles [B, C, H, W] per modality,
    not [B, C, T, H, W]. If you need multi-temporal classification, port
    RAMENBackbone's LTAE2d fusion step in before the ViT blocks.

Reused verbatim from RAMEN/ramen_encoder.py: SpectralProjector,
RadarProjector, DemProjector, ScaleResampler.

Input:
    x: dict[modality] -> Tensor [B, C, H, W]   (e.g. {"optical": ..., "sar": ...})
Output:
    logits: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

# Adjust these import paths to wherever RAMEN/ actually lives in your repo.
from .pos_embed import get_2d_sincos_pos_embed_with_resolution
from .ramen_encoder import (
    SpectralProjector,
    RadarProjector,
    DemProjector,
    ScaleResampler,
)


class RAMENClassifier(nn.Module):
    """
    Args mirror RAMENBackbone where they overlap (input_bands, wavelengths,
    input_size, embed_dim, depth, num_heads, input_res, res), plus
    ViTClassifier-style classification args (num_classes, dropout).

    Args:
        input_bands: dict[modality] -> list of band names, e.g.
            {"optical": ["Blue","Green",...], "sar": ["VV","VH"]}
        wavelengths: dict[modality][band] -> float (nm) for optical, or
            -> polarization/product string for sar/dem.
        num_classes: number of output classes.
        input_size: spatial size (H=W) of each modality's input tensor.
        input_res: native ground sampling distance (m/px) of the input.
        res: working resolution (m/px) all modalities are resampled to.
            Leave equal to input_res for no resampling (full native detail
            — reasonable default for small tiles like EuroSAT's 64x64,
            where token count is already modest even at native resolution).
    """

    expects_full_image_dict = True

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        wavelengths: dict,
        num_classes: int,
        input_size: int,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        input_res: float = 10.0,
        res: float = 10.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.modalities = list(input_bands.keys())
        self.input_bands = input_bands
        self.input_size = input_size
        self.input_res = input_res
        self.res = res
        self.embed_dim = embed_dim

        self.wavelengths = {
            m: [wavelengths[m][b] for b in self.input_bands[m]]
            for m in self.modalities
        }

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim * 2)
        if "sar" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim * 2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim * 2)

        self.resampler = ScaleResampler(embed_dim)
        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.effective_size = int(self.input_size * (self.input_res / self.res))
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim,
            self.effective_size,
            torch.tensor([self.res]),
            cls_token=True,
        )
        # Non-persistent: fixed sin-cos table, not learned, not saved in
        # state_dict — recomputed fresh at construction for whatever `res`
        # is passed, so a checkpoint trained at one `res` loads cleanly
        # into a model built at a different `res` (same as RAMENBackbone).
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _spectral_encoding(self, modality: str, device, dtype) -> torch.Tensor:
        if modality == "sar":
            return self.radar_projector(self.wavelengths[modality], device)
        if modality == "dem":
            return self.dem_projector(self.wavelengths[modality], device)
        return self.spectral_projector(
            torch.tensor(self.wavelengths[modality], device=device, dtype=dtype)
        )

    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: dict[modality] -> Tensor [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        ref = x[self.modalities[0]]
        device, dtype = ref.device, ref.dtype
        batch_size = ref.shape[0]
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        S = self.effective_size

        out = {}
        for modality in self.modalities:
            x_mod = x[modality]                                   # [B,C,H,W]
            x_mod = x_mod.permute(0, 2, 3, 1).contiguous()         # [B,H,W,C]

            spectral_encoding = self._spectral_encoding(modality, device, dtype)
            out_mod = x_mod @ spectral_encoding                    # [B,H,W,D]
            out_mod = out_mod.permute(0, 3, 1, 2).contiguous()     # [B,D,H,W]

            scale = self.input_res / self.res
            out_mod = self.resampler(out_mod, scale)
            if out_mod.shape[-2:] != (S, S):
                out_mod = nn.functional.interpolate(
                    out_mod, size=(S, S), mode="bilinear"
                )

            out_mod = out_mod.flatten(2).transpose(1, 2)           # [B,S*S,D]
            out_mod = self.in_norm(out_mod)
            out[modality] = out_mod

        n_mod = len(self.modalities)
        tokens = torch.cat([out[m] for m in self.modalities], dim=1)
        tokens = tokens + pos_embed[:, 1:, :].repeat(1, n_mod, 1)

        cls = (self.cls_token + pos_embed[:, :1, :]).expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]              # [B, D]
        cls_out = self.dropout(cls_out)
        return self.head(cls_out)           # [B, num_classes]


def build_ramen_classifier(
    input_bands: dict[str, list[str]],
    wavelengths: dict,
    num_classes: int,
    **kwargs,
) -> RAMENClassifier:
    """Factory mirroring build_ramen_upernet for registry consistency."""
    return RAMENClassifier(
        input_bands=input_bands,
        wavelengths=wavelengths,
        num_classes=num_classes,
        **kwargs,
    )
