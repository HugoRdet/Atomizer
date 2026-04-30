"""
ResNet + UPerNet — Multi-Task variant
=====================================

Shared encoder, shared UPerNet decoder, per-task heads and adapters.
Drop-in replacement for the placeholder TinyMTModel in the multi-task
sanity-check script — exposes the same forward(image, task) -> logits
interface that MultiTaskTrainer expects.

Architecture (one model, dispatched per task):

                        ┌─ adapters[task] (TimeMerge, only PASTIS)
                        │   only for tasks with num_frames > 1
    image ──────────────┤
        [B, C, H, W]    │
        or              │   T=1 path (no adapter)
        [B, T, C, H, W] └──> [B, C, H, W]
                                │
                                ▼
                        ResNetEncoder (SHARED)
                                │
                ┌───────────────┴───────────────┐
                │                                │
        seg path (any seg task)         cls path (any cls task)
                │                                │
                ▼                                ▼
        UPerNetDecoder (SHARED,          last encoder feat
        num_classes=decoder_channels      [B, 2048, H/32, W/32]
        → produces a 256-d feat map)             │
                │                                ▼
                ▼                        AdaptiveAvgPool2d
        seg_heads[task]:                  Flatten + (Dropout) + Linear
        2-layer MLP (1x1 convs):          (mirrors single-task
          Conv(256 -> hidden) -> ReLU      ResNetClassifier)
          -> [Dropout2d]                          │
          -> Conv(hidden -> K_task)               ▼
                │                          [B, K_task]
                ▼
        [B, K_task, H, W]

Sharing:
    - encoder: shared across all tasks (PASTIS, BurnScars, Sen1Floods11,
      MADOS, EuroSAT, ForestNet).
    - decoder: shared across all SEGMENTATION tasks. Built only if
      task_specs contains at least one seg task. Cls tasks bypass it.
    - per-task: TimeMerge adapter (only when num_frames > 1), seg head
      (2-layer MLP via 1x1 convs), cls head (avgpool + linear).

Trick: UPerNetDecoder is instantiated with num_classes=decoder_channels.
Its internal final 1x1 conv then produces a feature map at
decoder_channels (256 by default), and per-task seg heads project from
there to per-task num_classes. This avoids modifying UPerNetDecoder.

Note on classification + spatial padding:
    For tasks padded from a smaller native size (e.g. EuroSAT 64 -> 512,
    ForestNet 332 -> 512), the encoder's deepest feature has only a
    small region of "real" cells, the rest are activations from zero
    padding. The current cls head does naive AdaptiveAvgPool over the
    full feature map, matching single-task ResNetClassifier behaviour.
    For real cls multi-task runs we should switch to masked pooling
    using batch["valid_mask"]. Not implemented here — keep an eye on
    EuroSAT/ForestNet val numbers when added.
"""

import torch
import torch.nn as nn

# Adjust these imports to match your repo layout.
# UPerNetDecoder lives in the ViT file in the existing single-task code.
from training.VIT.model_vit_upernet import UPerNetDecoder
# Shared ResNet building blocks live in the single-task ResNet file.
from .model_resnet_upernet import (
    Bottleneck,
    ResNetEncoder,
    TimeMerge,
    _LAYER_CONFIGS,
)


# ═══════════════════════════════════════════════════════════════════════
# Multi-task ResNet + UPerNet
# ═══════════════════════════════════════════════════════════════════════

class MultiTaskResNetUPerNet(nn.Module):
    """
    Multi-task ResNet + UPerNet.

    Args:
        variant:           ResNet variant key into _LAYER_CONFIGS
                           ('resnet_super_small', 'resnet_small', 'resnet50',
                            'resnet101', 'resnet152').
        in_channels:       channel count per frame (15 in our canonical layout:
                           13 S2 optical + 2 SAR).
        task_specs:        dict {task_name: {"type": "seg"|"cls",
                                              "num_classes": int,
                                              "num_frames": int (default 1)}}.
                           Order is preserved for ModuleDict iteration.
        decoder_channels:  hidden width of UPerNet (default 256). Also the
                           feature dimension consumed by per-task seg heads.
        seg_head_hidden_dim: hidden dim of the per-task seg MLP head. None
                           defaults to decoder_channels.
        seg_head_dropout:  dropout in the seg MLP (default 0.0).
        cls_dropout:       dropout in the cls head (default 0.0). Same
                           semantics as single-task ResNetClassifier.

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
        variant: str,
        in_channels: int,
        task_specs: dict,
        decoder_channels: int = 256,
        seg_head_hidden_dim: int = None,
        seg_head_dropout: float = 0.0,
        cls_dropout: float = 0.0,
    ):
        super().__init__()
        if variant not in _LAYER_CONFIGS:
            raise ValueError(
                f"Unknown ResNet variant: {variant}. "
                f"Available: {list(_LAYER_CONFIGS.keys())}"
            )
        if not task_specs:
            raise ValueError("task_specs must contain at least one task.")

        self.variant = variant
        self.in_channels = in_channels
        self.task_specs = dict(task_specs)
        self.decoder_channels = decoder_channels

        # ── Shared ResNet encoder ─────────────────────────────
        self.encoder = ResNetEncoder(
            ResBlock=Bottleneck,
            layer_list=_LAYER_CONFIGS[variant],
            num_channels=in_channels,
        )
        self.feature_dim_cls = self.encoder.output_dim[-1]   # 2048 for resnet50
        self.feature_dim_seg = decoder_channels              # 256 by default

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
        # In our 6-task pool this is just PASTIS (T=6).
        self.adapters = nn.ModuleDict()
        for task, spec in self.task_specs.items():
            T = int(spec.get("num_frames", 1))
            if T > 1:
                self.adapters[task] = TimeMerge(in_channels, T)

        # ── Per-task heads ────────────────────────────────────
        # Seg head: 2-layer per-pixel MLP applied as 1x1 convs:
        #     Conv2d(d_seg -> hidden) -> ReLU -> [Dropout2d] -> Conv2d(hidden -> K)
        # Cls head: pool + flatten + (drop) + linear, mirroring single-task.
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

        Operates on the shared decoder's feature map [B, in_dim, H, W] and
        produces [B, num_classes, H, W]. Hidden dimension defaults to in_dim.
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
        """Mirror of single-task ResNetClassifier head: pool + flatten + (drop) + linear."""
        layers = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
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

        # ── Shared encoder ──
        H, W = x.shape[-2], x.shape[-1]
        features = self.encoder(x)   # 4 multi-scale feature maps

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
            return self.cls_heads[task](features[-1])        # [B, K]

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

def build_multitask_resnet_upernet(
    variant: str = "resnet50",
    in_channels: int = 15,
    task_specs: dict = None,
    decoder_channels: int = 256,
    seg_head_hidden_dim: int = None,
    seg_head_dropout: float = 0.0,
    cls_dropout: float = 0.0,
) -> MultiTaskResNetUPerNet:
    """
    Build a multi-task ResNet+UPerNet for the multi-task baselines.

    Example:
        task_specs = {
            "burnscars": {"type": "seg", "num_classes": 2,  "num_frames": 1},
            "senflood":  {"type": "seg", "num_classes": 2,  "num_frames": 1},
            "mados":     {"type": "seg", "num_classes": 15, "num_frames": 1},
            "pastis":    {"type": "seg", "num_classes": 19, "num_frames": 6},
            "eurosat":   {"type": "cls", "num_classes": 10, "num_frames": 1},
            "forestnet": {"type": "cls", "num_classes": 12, "num_frames": 1},
        }
        model = build_multitask_resnet_upernet(
            variant="resnet50", in_channels=15, task_specs=task_specs,
        )
    """
    return MultiTaskResNetUPerNet(
        variant=variant,
        in_channels=in_channels,
        task_specs=task_specs or {},
        decoder_channels=decoder_channels,
        seg_head_hidden_dim=seg_head_hidden_dim,
        seg_head_dropout=seg_head_dropout,
        cls_dropout=cls_dropout,
    )