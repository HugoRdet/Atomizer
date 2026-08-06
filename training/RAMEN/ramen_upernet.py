"""
RAMEN + UPerNet — Standalone Multi-modal Segmentation
=======================================================

Wraps RAMEN's multi-modal, multi-temporal spatial encoder to match the
same all-in-one convention as ViTUPerNet / ViTUPerNetMT / ViTLTAEUPerNet
in this codebase: a single nn.Module owning both encoder and decoder,
built with plain constructor args (no Pangaea `base.Encoder` / registry
machinery, no checkpoint auto-download).

  RAMENBackbone:  dict[modality] -> Tensor  =>  list[Tensor] (multi-scale)
                  Mirrors ViTEncoder's contract: forward() returns one
                  feature map per `output_layers` index, and exposes
                  `.output_dim` for the decoder to consume.

  RAMENUPerNet:   RAMENBackbone + UPerNetDecoder (reused from the ViT file).
                  dict[modality] -> Tensor  =>  [B, num_classes, H, W]

Differences from the original Pangaea `ramen_encoder.py`:

  - Not a `base.Encoder` subclass. No `input_size`/`download_url`/
    `load_encoder_weights` bookkeeping — plug weights in however the
    rest of this pipeline does it (e.g. `load_state_dict` directly).

  - RAMEN_Encoder and RAMEN_Encoder_MonoTemporal are merged into one
    class. Each modality is inspected per-forward-call: a 4D tensor
    [B,C,H,W] is treated as single-frame (LTAE2d skipped for that
    modality), a 5D tensor [B,C,T,H,W] is temporally fused via LTAE2d
    before the shared ViT stack. Mixing temporal and non-temporal
    modalities in the same forward pass is supported.

  - Fixes a bug in the original `RAMEN_Encoder_MonoTemporal`:
    `self.modalities = self.input_bands.keys()` is a `dict_keys` object,
    but `forward()` does `x[self.modalities[0]]`, which raises
    `TypeError: 'dict_keys' object is not subscriptable`. Always
    `list(...)` here.

  - `pos_embed` is registered as a (non-persistent) buffer instead of a
    plain float tensor attribute, so it follows the module across
    `.to(device)` / `.cuda()` calls automatically.

  - `output_dim` is `embed_dim * n_modalities` (modalities are
    concatenated channel-wise at every tapped layer — this matches the
    original `forward()`'s reshape logic, and is NOT `embed_dim *
    n_total_bands`). Your decoder's `in_channels` must use this value.

Reused verbatim from the original ramen_encoder.py (unchanged, just
imported): SpectralProjector, RadarProjector, DemProjector,
ScaleResampler, LTAE2d and its internals (PositionalEncoder,
MultiHeadAttention, ScaledDotProductAttention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# Adjust these two import paths to wherever RAMEN/ actually lives relative
# to this file in your repo.
from .pos_embed import get_2d_sincos_pos_embed_with_resolution
from .ramen_encoder import (
    SpectralProjector,
    RadarProjector,
    DemProjector,
    ScaleResampler,
    LTAE2d,
)

# Reused from the ViT file — same UPerNetDecoder, no changes needed.
from training.VIT.vit_upernet import UPerNetDecoder  # adjust import path as needed


# ═══════════════════════════════════════════════════════════════════════
# RAMEN BACKBONE (encoder-only, ViTEncoder-compatible contract)
# ═══════════════════════════════════════════════════════════════════════

class RAMENBackbone(nn.Module):
    """
    RAMEN-style multi-modal spatial encoder returning multi-scale feature
    maps, matching ViTEncoder's contract:

        forward(x, dates=None) -> list[Tensor]   # [B, D_total, S, S] each
        .output_dim                              # list[int], D_total per layer

    Each modality's bands are spectrally projected (SpectralProjector /
    RadarProjector / DemProjector depending on modality name), resampled
    to a common physical resolution `res` via ScaleResampler, optionally
    fused over time via LTAE2d (only if that modality's input is 5D),
    then all modalities' tokens are concatenated and passed through a
    shared ViT stack.

    Args:
        input_bands: dict[modality] -> list of band names, e.g.
            {"optical": ["B02","B03","B04",...], "sar": ["VV","VH"]}
        wavelengths: dict[modality][band] -> float (nm) for optical, or
            -> polarization/product string for sar/dem (looked up in
            RadarProjector.pol_map / DemProjector.dem_map).
        input_size: spatial size (H=W) of each modality's *input* tensor.
        input_res: native ground sampling distance (m/px) of the input.
        res: common working resolution (m/px) all modalities are resampled
            to before the shared ViT stack — this is what fixes
            `effective_size` and therefore the shared token grid.
        output_layers: ViT block indices to tap for multi-scale features.
    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        wavelengths: dict,
        input_size: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        input_res: float = 10.0,
        res: float = 40.0,
        output_layers: tuple = (2, 5, 8, 11),
        class_token: bool = True,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.modalities = list(input_bands.keys())
        self.input_bands = input_bands
        self.input_size = input_size
        self.input_res = input_res
        self.res = res
        self.embed_dim = embed_dim
        self.output_layers = list(output_layers)

        # Modalities are concatenated channel-wise at every tapped layer.
        self.output_dim = [embed_dim * len(self.modalities) for _ in output_layers]

        self.wavelengths = {
            m: [wavelengths[m][b] for b in self.input_bands[m]]
            for m in self.modalities
        }

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim * 2)
        if "sar" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim * 2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim * 2)

        self.resampler = ScaleResampler(embed_dim)

        self.ltae = LTAE2d(
            in_channels=embed_dim,
            n_head=16,
            d_k=16,
            mlp=[embed_dim, embed_dim * 2, embed_dim],
            dropout=0.0,
            mlp_in=[embed_dim],
            T=367,
            in_norm=True,
            return_att=False,
            positional_encoding=True,
        )

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

        self.effective_size = int(self.input_size * (self.input_res / self.res))
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim,
            self.effective_size,
            torch.tensor([self.res]),
            cls_token=True,
        )
        # Non-persistent buffer: follows .to(device)/.cuda(), not saved in
        # state_dict (it's a fixed sin-cos table, not a learned parameter).
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _spectral_encoding(self, modality: str, device, dtype) -> torch.Tensor:
        if modality == "sar":
            return self.radar_projector(self.wavelengths[modality], device)
        if modality == "dem":
            return self.dem_projector(self.wavelengths[modality], device)
        return self.spectral_projector(
            torch.tensor(self.wavelengths[modality], device=device, dtype=dtype)
        )

    def forward(self, x: dict, dates: dict | None = None) -> list:
        """
        Args:
            x: dict[modality] -> Tensor, either
                 [B, C, H, W]     (single frame), or
                 [B, C, T, H, W]  (multi-temporal; needs dates[modality])
            dates: dict[modality] -> Tensor [B, T] day-of-year. Required
                   only for modalities whose tensor is 5D.
        Returns:
            list of [B, embed_dim * n_modalities, S, S], one per
            self.output_layers entry (same length/order as ViTEncoder).
        """
        ref = x[self.modalities[0]]
        device, dtype = ref.device, ref.dtype
        batch_size = ref.shape[0]
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        S = self.effective_size

        out = {}
        for modality in self.modalities:
            x_mod = x[modality]
            temporal = x_mod.dim() == 5

            if temporal:
                B, C, T, H, W = x_mod.shape
                x_mod = x_mod.permute(0, 2, 3, 4, 1).contiguous()  # [B,T,H,W,C]
            else:
                B, C, H, W = x_mod.shape
                x_mod = x_mod.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]

            spectral_encoding = self._spectral_encoding(modality, device, dtype)
            out_mod = x_mod @ spectral_encoding  # [...,C] @ [C,D] -> [...,D]

            if temporal:
                out_mod = (
                    out_mod.permute(0, 1, 4, 2, 3)
                    .reshape(B * T, -1, H, W)
                    .contiguous()
                )
            else:
                out_mod = out_mod.permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]

            scale = self.input_res / self.res
            out_mod = self.resampler(out_mod, scale)

            if out_mod.shape[-2:] != (S, S):
                out_mod = F.interpolate(out_mod, size=(S, S), mode="bilinear")

            if temporal:
                out_mod = out_mod.view(B, T, -1, S, S)  # [B,T,D,S,S]
                if dates is None or modality not in dates:
                    raise ValueError(
                        f"Modality '{modality}' has a temporal dimension "
                        f"(T={T}) but no positions were given in `dates`."
                    )
                out_mod = self.ltae(
                    out_mod, batch_positions=dates[modality], pad_mask=None
                )  # [B,D,S,S]

            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, S*S, D]
            out_mod = self.in_norm(out_mod)
            out[modality] = out_mod

        n_mod = len(self.modalities)
        n_tok = S * S

        tokens = torch.cat([out[m] for m in self.modalities], dim=1)
        tokens = tokens + pos_embed[:, 1:, :].repeat(1, n_mod, 1)

        if self.cls_token is not None:
            cls = (self.cls_token + pos_embed[:, :1, :]).expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        output = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i == len(self.blocks) - 1:
                tokens = self.norm(tokens)
            if i in self.output_layers:
                patch_tokens = tokens[:, 1:] if self.cls_token is not None else tokens
                per_mod = [
                    patch_tokens[:, k * n_tok : (k + 1) * n_tok] for k in range(n_mod)
                ]
                feat = torch.cat(per_mod, dim=-1)  # [B, S*S, D*n_mod]
                feat = (
                    feat.permute(0, 2, 1)
                    .reshape(batch_size, -1, S, S)
                    .contiguous()
                )
                output.append(feat)

        return output


# ═══════════════════════════════════════════════════════════════════════
# RAMEN + UPERNET (all-in-one, ViTUPerNet-style)
# ═══════════════════════════════════════════════════════════════════════

class RAMENUPerNet(nn.Module):
    """
    RAMENBackbone + UPerNetDecoder, mirroring ViTUPerNet's all-in-one
    contract but for RAMEN's native multi-modal (and optionally
    multi-temporal) input.

    Unlike ViTLTAEUPerNet / ViTUPerNetLTAE, there's no separate temporal
    stage bolted onto the decoder: RAMEN fuses time per-modality *before*
    the shared ViT stack (see RAMENBackbone), so this single class
    already covers both single-date and multi-temporal use without a
    "...MT" variant.

    Input:
        x:     dict[modality] -> [B,C,H,W]  or  [B,C,T,H,W]
        dates: dict[modality] -> [B,T]      (required for temporal modalities)
    Output:
        logits: [B, num_classes, H, W]  (H, W taken from the first
                 modality's input spatial size)
    """

    # Tells BaselineTrainer._get_image (and anything else duck-typing on
    # this) to pass the whole batch["image"] dict through untouched,
    # rather than indexing a single modality's tensor. Any wrapper built
    # around this model (e.g. a modality-drop wrapper for ablations)
    # should also set this attribute for the same reason.
    expects_full_image_dict = True

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        wavelengths: dict,
        num_classes: int,
        input_size: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        input_res: float = 10.0,
        res: float = 40.0,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.input_size = input_size

        self.encoder = RAMENBackbone(
            input_bands=input_bands,
            wavelengths=wavelengths,
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            input_res=input_res,
            res=res,
            output_layers=output_layers,
        )

        self.decoder = UPerNetDecoder(
            in_channels=self.encoder.output_dim,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: dict, dates: dict | None = None) -> torch.Tensor:
        ref = x[self.encoder.modalities[0]]
        H, W = ref.shape[-2], ref.shape[-1]
        features = self.encoder(x, dates=dates)
        return self.decoder(features, output_shape=(H, W))


def build_ramen_upernet(
    input_bands: dict[str, list[str]],
    wavelengths: dict,
    num_classes: int,
    **kwargs,
) -> RAMENUPerNet:
    """Factory mirroring `build_vit_upernet_mt` for registry consistency."""
    return RAMENUPerNet(
        input_bands=input_bands,
        wavelengths=wavelengths,
        num_classes=num_classes,
        **kwargs,
    )
