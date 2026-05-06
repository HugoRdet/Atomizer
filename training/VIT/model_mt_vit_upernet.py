"""
ViT + UPerNet — Multi-Task variant
==================================

Mirror of MultiTaskResNetUPerNet but with a ViT spatial encoder. Shared
ViT encoder + shared UPerNet decoder + per-task TimeMerge adapters and
per-task heads.

Architecture (one model, dispatched per task):

                        ┌─ adapters[task] (TimeMerge, only PASTIS)
                        │   only for tasks with num_frames > 1
    image ──────────────┤
        [B, C, H, W]    │
        or              │   T=1 path (no adapter)
        [B, T, C, H, W] └──> [B, C, H, W]
                                │
                                ▼
                        ViTEncoder (SHARED)
                        4 multi-scale features
                                │
                ┌───────────────┴───────────────┐
                │                                │
        seg path (any seg task)         cls path (any cls task)
                │                                │
                ▼                                ▼
        UPerNetDecoder (SHARED,         CLS token via attention
        num_classes=decoder_channels     pooling over patch tokens
        → produces a 256-d feat map)             │
                │                                ▼
                ▼                        cls_heads[task]:
        seg_heads[task]:                  LayerNorm + (Dropout) + Linear
        2-layer MLP (1x1 convs):                  │
          Conv(256 -> hidden) -> ReLU             ▼
          -> [Dropout2d]                    [B, K_task]
          -> Conv(hidden -> K_task)
                │
                ▼
        [B, K_task, H, W]

Sharing:
    - encoder: shared across all tasks.
    - decoder: shared across all SEGMENTATION tasks. Built only if
      task_specs contains at least one seg task. Cls tasks bypass it
      and read the encoder's pooled CLS token instead.
    - per-task: TimeMerge adapter (only when num_frames > 1), seg head
      (2-layer MLP via 1x1 convs), cls head (norm + linear over the
      mean-pooled patch tokens, see note below).

Cls path note:
    The ViTEncoder used here exposes 4 multi-scale spatial feature maps
    (matching ResNet's FPN output) but does NOT expose a separate CLS
    token in its output. To get a single vector per image for cls heads,
    we mean-pool the deepest spatial feature [B, D, H/P, W/P] over the
    spatial dims. This is the standard "GAP-style" ViT classifier
    (e.g. ViT-S in DINO and many fine-tuning recipes), and matches the
    pooling style of ResNet's avgpool head — keeps the multi-task
    comparison apples-to-apples.

    For tasks padded from a smaller native size (e.g. EuroSAT 64 -> 512,
    ForestNet 332 -> 512), the encoder's deepest feature includes
    activations derived from zero padding. The current cls head pools
    over the full feature map without using the batch's valid_mask.
    Same caveat as the ResNet MT head — a known limitation worth
    revisiting if cls tasks underperform.
"""

import torch
import torch.nn as nn

# Reuse ViT building blocks from the single-task / MT module.
from .model_vit_upernet import (
    UPerNetDecoder,
    ViTEncoder,
    TimeMerge,
)


# ═══════════════════════════════════════════════════════════════════════
# Multi-task ViT + UPerNet
# ═══════════════════════════════════════════════════════════════════════

class MultiTaskViTUPerNet(nn.Module):
    """
    Multi-task ViT + UPerNet.

    Args:
        in_channels:       channel count per frame (15 in our canonical layout:
                           13 S2 optical + 2 SAR).
        task_specs:        dict {task_name: {"type": "seg"|"cls",
                                              "num_classes": int,
                                              "num_frames": int (default 1)}}.
                           Order is preserved for ModuleDict iteration.
        img_size:          ViT spatial input size. Must match the canonical
                           multi-task spatial size (512). Patch positional
                           embeddings are sized accordingly.
        embed_dim:         ViT hidden dim
        depth:             ViT depth
        num_heads:         ViT attention heads
        patch_size:        ViT patch size. img_size / patch_size determines
                           the spatial-feature grid (32x32 for 512/16).
        output_layers:     Block indices to extract spatial features from.
                           Defaults to (2, 5, 8, 11) — standard 4-tier FPN.
        decoder_channels:  hidden width of UPerNet (default 256). Also the
                           feature dimension consumed by per-task seg heads.
        seg_head_hidden_dim: hidden dim of the per-task seg MLP head. None
                           defaults to decoder_channels.
        seg_head_dropout:  dropout in the seg MLP (default 0.0).
        cls_dropout:       dropout in the cls head (default 0.0).

    Forward signature:
        forward(image, task) -> logits

        image: [B, C_in, H, W]            for tasks with num_frames=1
            or [B, T, C_in, H, W]         for tasks with num_frames>1 (PASTIS)
        task:  string key — must be in task_specs

        Returns:
            [B, K_task, H, W]              if task is seg
            [B, K_task]                    if task is cls
    """

    def __init__(
        self,
        in_channels: int,
        task_specs: dict,
        img_size: int = 512,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 16,
        output_layers: tuple = (2, 5, 8, 11),
        decoder_channels: int = 256,
        seg_head_hidden_dim: int = None,
        seg_head_dropout: float = 0.0,
        cls_dropout: float = 0.0,
    ):
        super().__init__()
        if not task_specs:
            raise ValueError("task_specs must contain at least one task.")

        self.in_channels = in_channels
        self.task_specs = dict(task_specs)
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels

        # ── Shared ViT encoder ────────────────────────────────
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            output_layers=output_layers,
        )
        # ViT's deepest feature has `embed_dim` channels — we pool that for cls.
        self.feature_dim_cls = embed_dim
        self.feature_dim_seg = decoder_channels

        # ── Shared UPerNet decoder (built only if any seg task exists) ──
        # Trick: pass num_classes=decoder_channels so the decoder's internal
        # final 1x1 conv outputs a feature map. Per-task seg heads project
        # from this feature map to per-task num_classes.
        any_seg = any(s["type"] == "seg" for s in self.task_specs.values())
        if any_seg:
            self.decoder = UPerNetDecoder(
                in_channels=self.encoder.output_dim,
                channels=decoder_channels,
                num_classes=decoder_channels,
            )
        else:
            self.decoder = None

        # ── Per-task TimeMerge adapters (only when num_frames > 1) ──
        # In our 5-task pool this is just PASTIS (T=6).
        self.adapters = nn.ModuleDict()
        for task, spec in self.task_specs.items():
            T = int(spec.get("num_frames", 1))
            if T > 1:
                self.adapters[task] = TimeMerge(in_channels, T)

        # ── Per-task heads ────────────────────────────────────
        seg_hidden = seg_head_hidden_dim or self.feature_dim_seg
        self.seg_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for task, spec in self.task_specs.items():
            if spec["type"] == "seg":
                self.seg_heads[task] = self._build_seg_head(
                    in_dim=self.feature_dim_seg,
                    hidden_dim=seg_hidden,
                    num_classes=spec["num_classes"],
                    dropout=seg_head_dropout,
                )
            elif spec["type"] == "cls":
                self.cls_heads[task] = self._build_cls_head(
                    in_dim=self.feature_dim_cls,
                    num_classes=spec["num_classes"],
                    dropout=cls_dropout,
                )
            else:
                raise ValueError(
                    f"Unknown task type '{spec['type']}' for task '{task}'."
                )

    @staticmethod
    def _build_seg_head(
        in_dim: int, hidden_dim: int, num_classes: int, dropout: float,
    ) -> nn.Sequential:
        """
        Per-task MLP segmentation head — a 2-layer per-pixel MLP via 1x1 convs.
        Same as MultiTaskResNetUPerNet so heads are directly comparable.
        """
        layers = [
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(hidden_dim, num_classes, kernel_size=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_cls_head(in_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        """
        Cls head: GAP-style pooling already done in `forward`, so this is just
        LayerNorm + (Dropout) + Linear. The LayerNorm is standard for ViT
        classification heads (mirrors `nn.LayerNorm` before the linear in the
        single-task ViTClassifier and the MAE recipe).
        """
        layers = [nn.LayerNorm(in_dim, eps=1e-6)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, num_classes))
        return nn.Sequential(*layers)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(self, image: torch.Tensor, task: str) -> torch.Tensor:
        # ── Adapter (TimeMerge) for multi-temporal tasks ──
        x = self._apply_adapter(image, task)

        # ── Spatial-size guard ─────────────────────────────────
        H, W = x.shape[-2], x.shape[-1]
        if (H, W) != (self.img_size, self.img_size):
            raise RuntimeError(
                f"ViT encoder was built for img_size={self.img_size}x{self.img_size} "
                f"but received [B, C, {H}, {W}]. ViT positional embeddings are "
                f"size-bound; the canonical multi-task pipeline pads everything "
                f"to {self.img_size}x{self.img_size}."
            )

        # ── Shared encoder ──
        features = self.encoder(x)   # 4 multi-scale [B, embed_dim, H/P, W/P]

        # ── Per-task head dispatch ──
        if task in self.seg_heads:
            if self.decoder is None:
                raise RuntimeError(
                    "Seg task requested but decoder was not built "
                    "(no seg tasks in task_specs at construction time)."
                )
            feat_map = self.decoder(features, output_shape=(H, W))
            return self.seg_heads[task](feat_map)            # [B, K, H, W]

        if task in self.cls_heads:
            # Mean-pool the deepest spatial feature: [B, D, H/P, W/P] -> [B, D]
            pooled = features[-1].mean(dim=(2, 3))
            return self.cls_heads[task](pooled)              # [B, K]

        raise KeyError(
            f"Unknown task '{task}'. Available: "
            f"seg={list(self.seg_heads.keys())}, "
            f"cls={list(self.cls_heads.keys())}"
        )

    def _apply_adapter(self, image: torch.Tensor, task: str) -> torch.Tensor:
        """Apply per-task TimeMerge if the task needs one, else pass through."""
        if task in self.adapters:
            # Multi-temporal task: image must be 5D.
            if image.dim() != 5:
                raise RuntimeError(
                    f"Task '{task}' has a TimeMerge adapter and expects 5D input "
                    f"[B, T, C, H, W], got {image.dim()}D shape {tuple(image.shape)}."
                )
            B, T, C, H, W = image.shape
            expected_T = self.adapters[task].num_frames
            if T != expected_T:
                raise RuntimeError(
                    f"Task '{task}' expects T={expected_T} but got T={T}. "
                    f"Mismatch — pad/trim the input or rebuild the model."
                )
            return self.adapters[task](image.reshape(B, T * C, H, W))

        # Single-frame task — accept 4D directly, or 5D with T=1.
        if image.dim() == 5:
            B, T, C, H, W = image.shape
            if T != 1:
                raise RuntimeError(
                    f"Task '{task}' has no adapter (single-frame) but received "
                    f"T={T}. Either set num_frames>1 on this task or feed T=1."
                )
            return image.squeeze(1)
        return image


# ═══════════════════════════════════════════════════════════════════════
# Convenience builder
# ═══════════════════════════════════════════════════════════════════════

def build_multitask_vit_upernet(
    in_channels: int = 15,
    task_specs: dict = None,
    img_size: int = 512,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    patch_size: int = 16,
    output_layers: tuple = (2, 5, 8, 11),
    decoder_channels: int = 256,
    seg_head_hidden_dim: int = None,
    seg_head_dropout: float = 0.0,
    cls_dropout: float = 0.0,
) -> MultiTaskViTUPerNet:
    """
    Build a multi-task ViT+UPerNet for the multi-task baselines.

    Default config matches the single-task ViT (embed_dim=384, depth=12,
    num_heads=6, patch_size=16) so single-task and multi-task results are
    directly comparable. img_size defaults to 512 (the canonical
    multi-task spatial size).

    Example:
        task_specs = {
            "burnscars": {"type": "seg", "num_classes": 2,  "num_frames": 1},
            "senflood":  {"type": "seg", "num_classes": 2,  "num_frames": 1},
            "pastis":    {"type": "seg", "num_classes": 19, "num_frames": 6},
            "eurosat":   {"type": "cls", "num_classes": 10, "num_frames": 1},
            "forestnet": {"type": "cls", "num_classes": 12, "num_frames": 1},
        }
        model = build_multitask_vit_upernet(
            in_channels=15, task_specs=task_specs,
        )
    """
    return MultiTaskViTUPerNet(
        in_channels=in_channels,
        task_specs=task_specs or {},
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        output_layers=output_layers,
        decoder_channels=decoder_channels,
        seg_head_hidden_dim=seg_head_hidden_dim,
        seg_head_dropout=seg_head_dropout,
        cls_dropout=cls_dropout,
    )