"""
ResNet + UPerNet Per-Modality (FLAIR-HUB)
==========================================

Per-modality fusion baseline: each input modality has its own ResNet50
branch, and features are fused per-scale before the UPerNet decoder.

Architecture
------------

    VHR (4ch @ 512×512)  ───► ResNet50 ───► [F1_v, F2_v, F3_v, F4_v]
    DEM (2ch @ 512×512)  ───► ResNet50 ───► [F1_d, F2_d, F3_d, F4_d]
    S2  (10ch×T @ 10×10)
        upsample to 512×512
        DoubleConv(10·T → 10)        ───► ResNet50 ───► [F1_s, F2_s, F3_s, F4_s]
    S1  (4ch×T @ 10×10) [ASC+DESC]
        upsample to 512×512
        DoubleConv(4·T → 4)          ───► ResNet50 ───► [F1_r, F2_r, F3_r, F4_r]

    Per-scale fusion (4 scales):
        concat([F_v_i, F_d_i, F_s_i, F_r_i]) → 1×1 Conv → standard ResNet width

    UPerNet decoder → [B, num_classes, H, W]

Fusion strategy mirrors FLAIR-HUB's official `FusionHandler`:
    1. Bilinear-align all branch features to a target shape per scale
       (target shape = first branch's, conventionally VHR)
    2. Concat along channel dim
    3. 1×1 conv to project back to a standard width

Memory note
-----------
Four ResNet50 branches at 512×512 input, batch 1, bf16 ≈ 8-10 GB of
activations during forward. Fits H100. If memory is tight, see
`branch_variants` argument in `build_resnet_upernet_per_modality` to
downsize satellite branches to ResNet_small.

Cross-sensor transfer
---------------------
At test time, swap VHR (4ch) for SPOT (4ch) in the VHR branch. Channel
count matches → architecture works with the same checkpoint. The model
sees SPOT data through a branch trained on VHR statistics, which is
exactly the cross-sensor degradation we want to measure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse canonical building blocks from the existing single-encoder model.
from .model_resnet_upernet import (
    Bottleneck,
    ResNetEncoder,
    TimeMerge,
    _LAYER_CONFIGS,
)
from training.VIT.model_vit_upernet import UPerNetDecoder



# ═══════════════════════════════════════════════════════════════════════
# PER-MODALITY BRANCH WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class _ModalityBranch(nn.Module):
    """
    One modality branch = optional TimeMerge + spatial upsampler + ResNet.

    Pre-encoder pipeline (in order, all optional except the encoder):
      1. If T > 1, channel-stack timesteps: [B, T, C, H, W] → [B, T*C, H, W]
      2. If input H ≠ target_size, bilinear upsample to target_size.
      3. If TimeMerge active (num_frames > 1), DoubleConv: [B, T*C, H, W] → [B, C, H, W]
      4. ResNet encoder → 4 multi-scale feature maps.

    The ordering "upsample BEFORE TimeMerge" matters: TimeMerge mixes
    time and channel info; doing it at full spatial resolution lets the
    DoubleConv kernels see the upsampled spatial structure already, which
    matches how single-encoder ViTUPerNetMT/ResNetUPerNetMT do it for
    consistency.

    Args:
        in_channels:  channels per timestep (e.g. 4 for VHR, 10 for S2)
        num_frames:   T (1 for mono-temporal modalities; e.g. 6 for S2/S1)
        target_size:  spatial size before encoder (typically 512 for FLAIR-HUB)
        resnet_variant: one of _LAYER_CONFIGS keys (e.g. "resnet50")
    """

    def __init__(
        self,
        in_channels: int,
        num_frames: int = 1,
        target_size: int = 512,
        resnet_variant: str = "resnet50",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.target_size = target_size

        # TimeMerge only when T > 1 (e.g. for S2 / S1).
        if num_frames > 1:
            self.time_merge = TimeMerge(in_channels, num_frames)
        else:
            self.time_merge = None

        if resnet_variant not in _LAYER_CONFIGS:
            raise ValueError(
                f"Unknown ResNet variant '{resnet_variant}'. "
                f"Available: {list(_LAYER_CONFIGS.keys())}"
            )

        self.encoder = ResNetEncoder(
            ResBlock=Bottleneck,
            layer_list=_LAYER_CONFIGS[resnet_variant],
            num_channels=in_channels,
        )
        # Standard Bottleneck widths: [256, 512, 1024, 2048].
        self.output_dim = self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]      (single-frame, num_frames must be 1)
               OR
               [B, T, C, H, W]   (multi-frame, T must equal num_frames)
        Returns:
            list of 4 feature maps from the ResNet stages.
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            if self.time_merge is None:
                if T == 1:
                    x = x.squeeze(1)              # [B, C, H, W]
                else:
                    raise RuntimeError(
                        f"Branch built with num_frames=1 but received T={T}. "
                        f"Set num_frames={T} when constructing the branch."
                    )
            else:
                if T != self.num_frames:
                    raise RuntimeError(
                        f"Branch built with num_frames={self.num_frames} but "
                        f"received T={T}."
                    )
                # Stack timesteps along channel dim BEFORE upsampling.
                x = x.reshape(B, T * C, H, W)
        else:
            B, C, H, W = x.shape
            if self.time_merge is not None:
                raise RuntimeError(
                    f"Branch built for T={self.num_frames} but received 4D "
                    f"input [B, C, H, W]. Use 5D [B, T, C, H, W]."
                )

        # 1. Spatial upsample to target_size (bilinear).
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        # 2. TimeMerge collapses T*C → C (only when num_frames > 1).
        if self.time_merge is not None:
            x = self.time_merge(x)                # [B, C, target, target]

        # 3. ResNet encoder → 4 feature maps at standard scales.
        return self.encoder(x)                    # list of 4 tensors


# ═══════════════════════════════════════════════════════════════════════
# PER-SCALE FUSION
# ═══════════════════════════════════════════════════════════════════════

class _PerScaleFusion(nn.Module):
    """
    Multi-scale fusion module mirroring FLAIR-HUB's `FusionHandler`.

    For each of the K scales output by the encoders:
      1. Optionally bilinear-align all branch features to a target shape
         (taken from the first branch's feature at that scale, which is
         conventionally VHR).
      2. Concat along channel dim.
      3. 1×1 Conv to project to a standard target channel width.

    Output: list of K fused feature maps at the standard widths, ready
    for UPerNet's decoder.

    Args:
        branch_channels_per_scale: shape [K][num_branches], channel counts
                                   each branch produces at each scale.
        target_channels_per_scale: list of K, standard widths for the
                                   decoder (e.g., [256, 512, 1024, 2048]).
    """

    def __init__(
        self,
        branch_channels_per_scale: list,   # [K][num_branches]
        target_channels_per_scale: list,   # [K]
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
        """
        Args:
            branch_features: list of length num_branches; each element is
                             a list of K feature maps from one branch.
                             branch_features[b][k] = [B, C_bk, H_k, W_k]
        Returns:
            list of K fused feature maps at standard widths.
        """
        num_branches = len(branch_features)
        K = len(branch_features[0])

        # First branch's feature shapes are the target shapes (mirrors
        # FLAIR-HUB's convention of using mono-key shapes as targets).
        target_shapes = [branch_features[0][k].shape[-2:] for k in range(K)]

        fused = []
        for k in range(K):
            target_h, target_w = target_shapes[k]
            # Spatial-align each branch's feature at scale k.
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

            # Concat across channel dim, project with 1×1 conv.
            cat = torch.cat(aligned, dim=1)        # [B, sum_C, H, W]
            fused.append(self.fusion_convs[k](cat))

        return fused


# ═══════════════════════════════════════════════════════════════════════
# MAIN MODEL: 4-BRANCH ResNet+UPerNet WITH PER-MODALITY FUSION
# ═══════════════════════════════════════════════════════════════════════

class ResNetUPerNetPerModality(nn.Module):
    """
    Per-modality fusion ResNet+UPerNet for FLAIR-HUB.

    Forward expects a dict of per-modality tensors; only those flagged
    `True` in `use_*` are processed. The branch order is:
        VHR / SPOT  →  DEM  →  S2  →  S1
    SPOT is routed through the VHR branch (same channel count, 4 RGBI),
    which makes the cross-sensor transfer experiment a single-flag flip.

    Args:
        num_classes:          output classes (FLAIR-HUB COSIA = 19)
        use_vhr_or_spot:      include the 4-channel optical branch
        use_dem:              include the 2-channel DEM branch
        use_s2:               include the S2 branch (T frames, 10 ch)
        use_s1:               include the S1 branch (T frames, 4 ch ASC+DESC)
        num_frames:           T for satellite branches (e.g. 6)
        resnet_variant:       backbone variant for *all* branches
        branch_target_size:   spatial size to upsample inputs to (default 512)
        decoder_channels:     UPerNet hidden channels
    """

    def __init__(
        self,
        num_classes: int,
        use_vhr_or_spot: bool = True,
        use_dem: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        num_frames: int = 6,
        resnet_variant: str = "resnet50",
        branch_target_size: int = 512,
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.use_vhr_or_spot = use_vhr_or_spot
        self.use_dem        = use_dem
        self.use_s2         = use_s2
        self.use_s1         = use_s1
        self.branch_target_size = branch_target_size

        # ── Build branches in canonical order ────────────────────────
        self.branches = nn.ModuleDict()
        self.branch_keys: list[str] = []  # keeps insertion order

        if use_vhr_or_spot:
            self.branches["optical"] = _ModalityBranch(
                in_channels=4, num_frames=1,
                target_size=branch_target_size,
                resnet_variant=resnet_variant,
            )
            self.branch_keys.append("optical")

        if use_dem:
            self.branches["dem"] = _ModalityBranch(
                in_channels=2, num_frames=1,
                target_size=branch_target_size,
                resnet_variant=resnet_variant,
            )
            self.branch_keys.append("dem")

        if use_s2:
            self.branches["s2"] = _ModalityBranch(
                in_channels=10, num_frames=num_frames,
                target_size=branch_target_size,
                resnet_variant=resnet_variant,
            )
            self.branch_keys.append("s2")

        if use_s1:
            # 4 channels per timestep = 2 (ASC: VV, VH) + 2 (DESC: VV, VH)
            self.branches["s1"] = _ModalityBranch(
                in_channels=4, num_frames=num_frames,
                target_size=branch_target_size,
                resnet_variant=resnet_variant,
            )
            self.branch_keys.append("s1")

        if not self.branch_keys:
            raise ValueError("At least one modality must be enabled.")

        # ── Per-scale fusion ─────────────────────────────────────────
        # All branches produce ResNet's standard widths at each scale.
        per_branch_dim = self.branches[self.branch_keys[0]].output_dim
        K = len(per_branch_dim)
        # Shape: [K][num_branches]
        branch_channels_per_scale = [
            [self.branches[bk].output_dim[k] for bk in self.branch_keys]
            for k in range(K)
        ]
        # Output widths == standard ResNet widths (to feed UPerNet
        # designed for a single ResNet of this variant).
        target_channels_per_scale = list(per_branch_dim)

        self.fusion = _PerScaleFusion(
            branch_channels_per_scale=branch_channels_per_scale,
            target_channels_per_scale=target_channels_per_scale,
        )

        # ── UPerNet decoder ──────────────────────────────────────────
        self.decoder = UPerNetDecoder(
            in_channels=target_channels_per_scale,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys among {"optical", "dem", "s2", "s1"}.
                   "optical" / "dem" are 4D [B, C, H, W].
                   "s2" / "s1" are 5D [B, T, C, H, W].
                   Spatial sizes can vary per modality (input upsampling
                   happens inside the branches).

        Returns:
            logits: [B, num_classes, H_out, W_out] where (H_out, W_out)
                    matches the optical branch's input spatial size if
                    "optical" is provided, else the largest input.
        """
        # ── 1. Per-branch encoding ───────────────────────────────────
        branch_features: list = []  # [num_branches][K]
        for key in self.branch_keys:
            if key not in batch:
                raise KeyError(
                    f"Branch '{key}' is enabled but missing from batch. "
                    f"Available keys: {list(batch.keys())}"
                )
            feats = self.branches[key](batch[key])
            branch_features.append(feats)

        # ── 2. Per-scale fusion ──────────────────────────────────────
        fused = self.fusion(branch_features)

        # ── 3. Determine output spatial size ─────────────────────────
        # Convention: match the FIRST branch's input H, W. For FLAIR-HUB
        # this is VHR/SPOT at 512×512, which is also the target the
        # branch upsamples to internally.
        first_key = self.branch_keys[0]
        first_input = batch[first_key]
        if first_input.dim() == 5:
            H_out, W_out = first_input.shape[-2], first_input.shape[-1]
        else:
            H_out, W_out = first_input.shape[-2], first_input.shape[-1]
        # If branch upsampled to target_size, the corresponding labels
        # are at original res; just return at branch's target.
        # The trainer interpolates logits → label shape if needed.
        return self.decoder(fused, output_shape=(H_out, W_out))


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_resnet_upernet_per_modality(
    num_classes: int = 19,
    use_vhr_or_spot: bool = True,
    use_dem: bool = True,
    use_s2: bool = True,
    use_s1: bool = True,
    num_frames: int = 6,
    resnet_variant: str = "resnet50",
    branch_target_size: int = 512,
    decoder_channels: int = 256,
) -> ResNetUPerNetPerModality:
    """Construct a per-modality ResNet+UPerNet for FLAIR-HUB."""
    return ResNetUPerNetPerModality(
        num_classes=num_classes,
        use_vhr_or_spot=use_vhr_or_spot,
        use_dem=use_dem,
        use_s2=use_s2,
        use_s1=use_s1,
        num_frames=num_frames,
        resnet_variant=resnet_variant,
        branch_target_size=branch_target_size,
        decoder_channels=decoder_channels,
    )