"""
ViT + UPerNet Per-Modality (FLAIR-HUB)
========================================

Per-modality fusion baseline using ViT encoders. Mirrors the structure of
ResNetUPerNetPerModality but with ViT branches.

Architecture
------------

    Optical (4ch @ 512×512)  ───► ViT (img=512, patch=16) ───► [F1..F4 @ 32×32]
                                  (single-frame)
    DEM (2ch @ 512×512)      ───► ViT (img=512, patch=16) ───► [F1..F4 @ 32×32]
                                  (single-frame)
    S2 (10ch×T @ 10×10)
        per-frame ViT (img=10, patch=2, shared)
        → per-FPN-layer LTAE temporal aggregation
                                                        ───► [F1..F4 @ 5×5]
    S1 (4ch×T @ 10×10) [ASC+DESC fused]
        per-frame ViT (img=10, patch=2, shared)
        → per-FPN-layer LTAE temporal aggregation
                                                        ───► [F1..F4 @ 5×5]

    Per-scale fusion (4 scales):
        bilinear-align all branch features to optical's scale (32×32)
        → concat along channel dim
        → 1×1 Conv to standard ViT embed_dim

    UPerNet decoder → [B, num_classes, H, W]

Design rationale
----------------
- Native input sizes throughout; no upsampling of S2/S1. The 10×10 native
  resolution for satellite carries no information at higher resolutions,
  so upsampling is unjustified architecturally.
- Optical branch at 512×512 receives either VHR (native) or SPOT
  (already upsampled 64→512 in the dataset). Same architecture either
  way → cross-sensor transfer is a flag flip.
- patch_size=2 for satellite branches gives 5×5 patches per frame
  (25 tokens). Coarse but preserves some spatial structure for fusion.
  Smaller (patch=1) would be wasteful; larger (patch=5/10) collapses
  spatial info entirely.
- Per-FPN-LTAE temporal aggregation matches the existing ViTLTAEUPerNet
  pattern. Each FPN layer has its own LTAE that aggregates T frames
  into one feature map at that layer.
- 1×1 fusion conv mirrors FLAIR-HUB's official FusionHandler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse canonical ViT building blocks (ViTEncoder, LTAE, UPerNetDecoder).
from .model_vit_upernet import (
    ViTEncoder,
    LTAE,
    UPerNetDecoder,
)


# ═══════════════════════════════════════════════════════════════════════
# BRANCH WRAPPERS
# ═══════════════════════════════════════════════════════════════════════

class _MonoTempViTBranch(nn.Module):
    """
    Single-frame ViT branch (used for VHR/SPOT optical and DEM).

    Input:  [B, C, H, W]
    Output: list of K feature maps from ViT's tapped layers.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        output_layers: tuple,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )
        # Standard ViT outputs: all K layers have the same channel dim.
        self.output_dim = self.encoder.output_dim   # [embed_dim] * K

    def forward(self, x: torch.Tensor) -> list:
        if x.dim() == 5:  # [B, T=1, C, H, W]
            B, T, C, H, W = x.shape
            if T != 1:
                raise RuntimeError(
                    f"_MonoTempViTBranch received T={T} but expects T=1."
                )
            x = x.squeeze(1)
        return self.encoder(x)


class _MultiTempViTBranch(nn.Module):
    """
    Multi-temporal ViT branch with per-FPN-layer LTAE (for S2/S1).

    Input:  [B, T, C, H, W]
    Output: list of K feature maps; each is the LTAE-aggregated form of
            the per-frame features at that layer.

    Pipeline:
        for each frame t:
            ViT(x[:, t])  →  list of K [B, D, H', W'] features
        for each FPN layer l:
            stack T frames along T-dim → [B, D, T, H', W']
            LTAE_l → [B, D, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        output_layers: tuple,
        ltae_n_head: int = 16,
        ltae_d_k: int = 4,
        ltae_d_model: int = 256,
        ltae_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )
        self.n_layers = len(output_layers)

        # One LTAE per FPN layer. Output back to embed_dim so all four
        # feature maps have a consistent channel count for fusion.
        self.ltaes = nn.ModuleList([
            LTAE(
                in_channels=embed_dim,
                n_head=ltae_n_head,
                d_k=ltae_d_k,
                d_model=ltae_d_model,
                mlp_dims=[ltae_d_model, embed_dim],
                positional_encoding=True,
                dropout=ltae_dropout,
            )
            for _ in range(self.n_layers)
        ])

        self.output_dim = [embed_dim] * self.n_layers

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> list:
        """
        Args:
            x:   [B, T, C, H, W]
            doy: [B, T] day-of-year (optional, for LTAE positional encoding)
        Returns:
            list of K feature maps, each [B, embed_dim, H', W'].
        """
        if x.dim() != 5:
            raise RuntimeError(
                f"_MultiTempViTBranch expects 5D input [B, T, C, H, W], "
                f"got {x.dim()}D."
            )

        B, T, C, H, W = x.shape

        # Per-frame ViT — batched via merging B and T into one dim,
        # so the encoder runs T*B samples in a single forward.
        x_flat = x.reshape(B * T, C, H, W)
        feats_flat = self.encoder(x_flat)   # list of [B*T, D, H', W']

        # Reshape back to [B, T, D, H', W'] per layer, then permute for LTAE
        # which expects [B, D, T, H', W'].
        per_layer_BTDHW = []
        for f in feats_flat:
            _, D, Hp, Wp = f.shape
            f_BTDHW = f.reshape(B, T, D, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()
            per_layer_BTDHW.append(f_BTDHW)

        # Per-FPN-layer LTAE temporal aggregation.
        agg = []
        for l in range(self.n_layers):
            agg.append(self.ltaes[l](per_layer_BTDHW[l], batch_positions=doy))
        # Each LTAE output is [B, embed_dim, H', W']
        return agg


# ═══════════════════════════════════════════════════════════════════════
# PER-SCALE FUSION (same shape as the ResNet version)
# ═══════════════════════════════════════════════════════════════════════

class _PerScaleFusion(nn.Module):
    """
    Multi-scale fusion module mirroring FLAIR-HUB's `FusionHandler`.

    For each FPN scale:
      1. Bilinear-align all branch features to a target shape
         (taken from the first branch's feature shape, conventionally optical).
      2. Concat along channel dim.
      3. 1×1 Conv to project to a standard target channel width.
    """

    def __init__(
        self,
        branch_channels_per_scale: list,    # [K][num_branches]
        target_channels_per_scale: list,    # [K]
    ):
        super().__init__()
        self.fusion_convs = nn.ModuleList()
        for branch_chs, target_ch in zip(
            branch_channels_per_scale, target_channels_per_scale
        ):
            in_ch = sum(branch_chs)
            self.fusion_convs.append(
                nn.Conv2d(in_ch, target_ch, kernel_size=1)
            )

    def forward(self, branch_features: list) -> list:
        num_branches = len(branch_features)
        K = len(branch_features[0])
        target_shapes = [branch_features[0][k].shape[-2:] for k in range(K)]

        fused = []
        for k in range(K):
            target_h, target_w = target_shapes[k]
            aligned = []
            for b in range(num_branches):
                fmap = branch_features[b][k]
                if fmap.shape[-1] != target_w or fmap.shape[-2] != target_h:
                    fmap = F.interpolate(
                        fmap,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                aligned.append(fmap)
            cat = torch.cat(aligned, dim=1)
            fused.append(self.fusion_convs[k](cat))
        return fused


# ═══════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════

class ViTUPerNetPerModality(nn.Module):
    """
    Per-modality fusion ViT+UPerNet for FLAIR-HUB.

    Forward expects a dict of per-modality tensors; only those with
    matching `use_*` flags will be processed. Branch order:
        Optical → DEM → S2 → S1
    Optical handles either VHR (native 512×512) or SPOT (upsampled to
    512×512 by the dataset). Cross-sensor transfer = flip dataset flags.

    Args:
        num_classes:          output classes (FLAIR-HUB COSIA = 19)
        use_vhr_or_spot:      include the optical branch
        use_dem:              include the DEM branch
        use_s2:               include the S2 branch
        use_s1:               include the S1 branch
        num_frames:           T for satellite branches
        embed_dim:            ViT hidden dim (uniform across all branches)
        depth:                ViT depth (uniform)
        num_heads:            ViT heads (uniform)
        decoder_channels:     UPerNet hidden channels
        optical_dem_img_size: input size for optical/DEM branches (default 512)
        optical_dem_patch:    patch size for optical/DEM branches (default 16)
        sat_img_size:         input size for satellite branches (default 10)
        sat_patch:            patch size for satellite branches (default 2 → 5×5 patches)
    """

    def __init__(
        self,
        num_classes: int,
        use_vhr_or_spot: bool = True,
        use_dem: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        num_frames: int = 6,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
        optical_dem_img_size: int = 512,
        optical_dem_patch: int = 16,
        sat_img_size: int = 10,
        sat_patch: int = 2,
    ):
        super().__init__()
        self.use_vhr_or_spot = use_vhr_or_spot
        self.use_dem = use_dem
        self.use_s2 = use_s2
        self.use_s1 = use_s1

        # Sanity: img_size must be divisible by patch_size for ViT.
        if optical_dem_img_size % optical_dem_patch != 0:
            raise ValueError(
                f"optical_dem_img_size ({optical_dem_img_size}) must be "
                f"divisible by optical_dem_patch ({optical_dem_patch})."
            )
        if sat_img_size % sat_patch != 0:
            raise ValueError(
                f"sat_img_size ({sat_img_size}) must be divisible by "
                f"sat_patch ({sat_patch})."
            )

        # ── Build branches in canonical order ─────────────────────
        self.branches = nn.ModuleDict()
        self.branch_keys: list[str] = []
        self.temporal_branch_keys: set[str] = set()

        if use_vhr_or_spot:
            self.branches["optical"] = _MonoTempViTBranch(
                in_channels=4,
                img_size=optical_dem_img_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=optical_dem_patch,
                output_layers=output_layers,
            )
            self.branch_keys.append("optical")

        if use_dem:
            self.branches["dem"] = _MonoTempViTBranch(
                in_channels=2,
                img_size=optical_dem_img_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=optical_dem_patch,
                output_layers=output_layers,
            )
            self.branch_keys.append("dem")

        if use_s2:
            self.branches["s2"] = _MultiTempViTBranch(
                in_channels=10,
                img_size=sat_img_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=sat_patch,
                output_layers=output_layers,
            )
            self.branch_keys.append("s2")
            self.temporal_branch_keys.add("s2")

        if use_s1:
            # 4 channels per timestep = 2 ASC + 2 DESC concatenated.
            self.branches["s1"] = _MultiTempViTBranch(
                in_channels=4,
                img_size=sat_img_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=sat_patch,
                output_layers=output_layers,
            )
            self.branch_keys.append("s1")
            self.temporal_branch_keys.add("s1")

        if not self.branch_keys:
            raise ValueError("At least one modality must be enabled.")

        # ── Per-scale fusion ──────────────────────────────────────
        # All ViT branches output `embed_dim` channels at every layer.
        K = len(output_layers)
        branch_channels_per_scale = [
            [self.branches[bk].output_dim[k] for bk in self.branch_keys]
            for k in range(K)
        ]
        # Output widths uniformly = embed_dim, matching what UPerNet expects
        # from a vanilla ViT backbone.
        target_channels_per_scale = [embed_dim] * K

        self.fusion = _PerScaleFusion(
            branch_channels_per_scale=branch_channels_per_scale,
            target_channels_per_scale=target_channels_per_scale,
        )

        # ── UPerNet decoder ───────────────────────────────────────
        self.decoder = UPerNetDecoder(
            in_channels=target_channels_per_scale,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys among {"optical", "dem", "s2", "s1"}.
                   - "optical" / "dem" are 4D [B, C, H, W].
                   - "s2" / "s1" are 5D [B, T, C, H, W].
                   Spatial sizes are native (the model does not upsample
                   inputs; it aligns features at fusion time).
                   Optionally a "doy" key with a dict of [B, T] tensors
                   per temporal modality for LTAE positional encoding.

        Returns:
            logits: [B, num_classes, H_out, W_out], where (H_out, W_out)
                    matches the optical (or first) branch's input shape.
        """
        branch_features: list = []

        for key in self.branch_keys:
            if key not in batch:
                raise KeyError(
                    f"Branch '{key}' is enabled but missing from batch. "
                    f"Available keys: {list(batch.keys())}"
                )
            x = batch[key]

            if key in self.temporal_branch_keys:
                # Optional day-of-year positional encoding for LTAE.
                doy = None
                if "doy" in batch and isinstance(batch["doy"], dict):
                    doy = batch["doy"].get(key)
                feats = self.branches[key](x, doy=doy)
            else:
                feats = self.branches[key](x)

            branch_features.append(feats)

        fused = self.fusion(branch_features)

        # Output spatial size = first branch's input spatial size.
        first_input = batch[self.branch_keys[0]]
        if first_input.dim() == 5:
            H_out, W_out = first_input.shape[-2], first_input.shape[-1]
        else:
            H_out, W_out = first_input.shape[-2], first_input.shape[-1]

        return self.decoder(fused, output_shape=(H_out, W_out))


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_vit_upernet_per_modality(
    num_classes: int = 19,
    use_vhr_or_spot: bool = True,
    use_dem: bool = True,
    use_s2: bool = True,
    use_s1: bool = True,
    num_frames: int = 6,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    output_layers: tuple = (2, 5, 8, 11),
    decoder_channels: int = 256,
    optical_dem_img_size: int = 512,
    optical_dem_patch: int = 16,
    sat_img_size: int = 10,
    sat_patch: int = 2,
) -> ViTUPerNetPerModality:
    """Construct a per-modality ViT+UPerNet for FLAIR-HUB."""
    return ViTUPerNetPerModality(
        num_classes=num_classes,
        use_vhr_or_spot=use_vhr_or_spot,
        use_dem=use_dem,
        use_s2=use_s2,
        use_s1=use_s1,
        num_frames=num_frames,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        output_layers=output_layers,
        decoder_channels=decoder_channels,
        optical_dem_img_size=optical_dem_img_size,
        optical_dem_patch=optical_dem_patch,
        sat_img_size=sat_img_size,
        sat_patch=sat_patch,
    )