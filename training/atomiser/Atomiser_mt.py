"""
Multi-Task Atomiser Wrapper
============================

Subclass of Atomiser_Senflood for multi-task training across heterogeneous
EO benchmarks (BurnScars, Sen1Floods11, PASTIS, EuroSAT, ForestNet, ...).

Design
------
The base Atomiser_Senflood has:
  - one `reconstruction_head` (seg MLP, latent_dim → num_classes)
  - one `to_logits` (cls pool + Linear, latent_dim → num_classes)

Both are baked to a single `num_classes` at construction time. For MT
across tasks with different class counts (2, 10, 12, 19, ...) we need
per-task heads on top of a shared encoder/decoder backbone.

This subclass:
  - Inherits all of Atomiser_Senflood's machinery (input processor, latents,
    geographic pruning, encoder/processor layers, decoder cross-attn,
    pooled-decoder logic).
  - Overrides _init_decoder to build a `seg_heads` ModuleDict instead of a
    single `reconstruction_head`. Each seg task gets its own MLP.
  - Overrides _init_classifier to build a `cls_heads` ModuleDict instead of
    a single `to_logits`. Each cls task gets its own pool+Linear.
  - Overrides `forward` to dispatch on `batch["task"]`.
  - Disables error predictor + refinement by construction (clean MT comparison;
    these are output-side mechanisms that interact awkwardly with cls and
    aren't part of the core MT contribution claim).

Round-robin assumption
----------------------
All samples in a batch have the same task (set by the dataset, propagated by
the collate via batch["task"]). Same as MultiTaskResNetUPerNet /
MultiTaskViTUPerNet. The ResNet/ViT MT trainer infrastructure works as-is.

Forward signature
-----------------
    forward(batch, training=True) -> logits

    batch must contain:
        "task":          str — task name in task_specs
        "groups":        per-resolution token dict (Atomiser standard)
        "queries":       [B, M, 8] (only used for seg tasks)
        "queries_mask":  [B, M]    (only used for seg tasks)
        "label":         [B] (only used for cls tasks)

    Returns:
        seg task: [B, M, K_task]  (per-query logits — trainer reshapes if needed)
        cls task: [B, K_task]     (single logit vector per sample)
"""

import torch
import torch.nn as nn

from .Atomiser_SENFLOOD import Atomiser_Senflood
from .nn_comp import LatentAttentionPooling


# ────────────────────────────────────────────────────────────────────
# MultiTaskAtomiser
# ────────────────────────────────────────────────────────────────────

class MultiTaskAtomiser(Atomiser_Senflood):
    """
    Atomiser_Senflood with per-task output heads for multi-task training.

    All encoder/processor/decoder-attention weights are shared across tasks.
    Only the final output projection differs per task.

    Args:
        config:        Atomiser config dict (passed through to base).
                       config["trainer"]["num_classes"] is irrelevant in MT —
                       per-task num_classes come from `task_specs`. We set
                       num_classes=2 in the base just so it doesn't crash;
                       the base's `reconstruction_head` and `to_logits` are
                       overwritten anyway.
        lookup_table:  Atomiser lookup table (passed through).
        task_specs:    {task_name: {"type": "seg"|"cls", "num_classes": int}}
                       Order is preserved for ModuleDict iteration.
    """

    def __init__(self, *, config, lookup_table, task_specs: dict):
        if not task_specs:
            raise ValueError("task_specs must contain at least one task.")

        # ── Force-disable error predictor + refinement for MT ─────────
        # The MT story is about input-flexibility under a stress test;
        # error-guided refinement is a separate output-side mechanism
        # that's a follow-up paper. Including it muddies what the MT
        # comparison is testing. We mutate a copy of the config so we
        # don't surprise upstream users of the same dict.
        config = self._sanitize_config_for_mt(config, task_specs)

        # Base __init__ builds:
        #   - input_processor, latents, geo_pruning
        #   - encoder_layers (cross+self attention stack)
        #   - decoder_cross_attn, reconstruction_head  (← we'll discard)
        #   - to_logits                                (← we'll discard)
        # We initialize with a placeholder num_classes (the smallest one in
        # task_specs) just to make the base's _init_decoder/_init_classifier
        # build something coherent. Their output heads are immediately
        # replaced below.
        super().__init__(config=config, lookup_table=lookup_table)

        self.task_specs = dict(task_specs)
        self._task_names = list(self.task_specs.keys())

        # ── Replace the single seg head with per-task seg heads ───────
        # Base built `self.reconstruction_head` as a single MLP. Drop it
        # and replace with a ModuleDict; the same MLP shape is reused per
        # task, just with the correct num_classes.
        del self.reconstruction_head
        self.seg_heads = nn.ModuleDict({
            task: self._build_seg_head(spec["num_classes"])
            for task, spec in self.task_specs.items()
            if spec["type"] == "seg"
        })

        # ── Replace the cls path with per-task cls heads ──────────────
        # Base built `self.to_logits` as a single Sequential. Drop it
        # and replace with a ModuleDict of LatentAttentionPooling +
        # LayerNorm + Linear per cls task.
        del self.to_logits
        self.cls_heads = nn.ModuleDict({
            task: self._build_cls_head(spec["num_classes"])
            for task, spec in self.task_specs.items()
            if spec["type"] == "cls"
        })

        n_seg = len(self.seg_heads)
        n_cls = len(self.cls_heads)
        print(f"[MT-Atomiser] Built {n_seg} seg head(s) and {n_cls} cls head(s).")
        for task, spec in self.task_specs.items():
            print(f"[MT-Atomiser]   {task:<14} type={spec['type']:<3} "
                  f"K={spec['num_classes']}")

    # ─────────────────────────────────────────────────────────────────
    # Config sanitization
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_config_for_mt(config: dict, task_specs: dict) -> dict:
        """
        Return a shallow-copied config with:
          - error predictor / refinement / targeted_depth2 disabled
          - num_classes set to a placeholder (1) for the base's init
            (we replace the heads anyway)
        Preserves the original config dict for any caller that's still
        using it.
        """
        import copy
        cfg = copy.deepcopy(config)
        cfg.setdefault("Atomiser", {})
        cfg["Atomiser"]["use_error_predictor"] = False
        cfg["Atomiser"]["use_refinement"] = False
        cfg["Atomiser"]["use_targeted_depth2"] = False
        cfg.setdefault("trainer", {})
        # Placeholder — real per-task class counts live in task_specs.
        # We just need *something* so the base's _init_decoder/_init_classifier
        # can construct (even though their heads get discarded).
        cfg["trainer"]["num_classes"] = 1
        return cfg

    # ─────────────────────────────────────────────────────────────────
    # Per-task head builders
    # ─────────────────────────────────────────────────────────────────

    def _build_seg_head(self, num_classes: int) -> nn.Sequential:
        """
        Same MLP shape as Atomiser_Senflood._init_decoder's
        reconstruction_head: latent_dim → 2*latent_dim → 2*latent_dim
        → num_classes, with LayerNorm + GELU between layers.

        Per-task seg head output: [B, M, num_classes].
        """
        hidden_dim = self.latent_dim * 2
        return nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_cls_head(self, num_classes: int) -> nn.Sequential:
        """
        Same shape as Atomiser_Senflood._init_classifier's to_logits:
        LatentAttentionPooling + LayerNorm + Linear.

        Per-task cls head output: [B, num_classes].
        """
        return nn.Sequential(
            LatentAttentionPooling(
                self.latent_dim,
                heads=self.latent_heads,
                dim_head=self.latent_dim_head,
                dropout=self.attn_dropout,
            ),
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, num_classes),
        )

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch, training=True, **kwargs):
        """
        Dispatch by batch["task"]. Reuses the base's encode() to build
        latents, then routes to the appropriate per-task head.

        Notes on kwargs:
          The base's forward takes a bunch of extras (return_trajectory,
          mask_ratio, return_for_error, return_features, etc.) that are
          relevant to single-task pretraining and visualization. For MT
          training we ignore them — MT is a clean train/eval setup with
          no MAE, no error supervision, no refinement.
          We accept **kwargs silently so the trainer can pass typical
          single-task signatures (`task=...`, `training=...`) without
          breaking.
        """
        if "task" not in batch:
            raise KeyError(
                "MultiTaskAtomiser expects batch['task'] to be set "
                "(by the per-task dataset's collate)."
            )
        task = batch["task"]
        if task not in self.task_specs:
            raise KeyError(
                f"Unknown task '{task}'. Available: {list(self.task_specs)}"
            )
        spec = self.task_specs[task]

        # ── Build latents (shared encoder path) ───────────────────────
        # Replicates the resolution-grid + encode logic at the top of
        # Atomiser_Senflood.forward. We don't call super().forward()
        # because the base then dispatches to its task branches with
        # baked-in heads — we want our own dispatch.
        from training.utils.datasets.token_grouping import compute_grid_config

        groups = batch["groups"]
        tpl, batch_cross_k = self.sample_config(training)
        resolutions = sorted(groups.keys())
        geo_k_budget = batch_cross_k * 2

        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.sigma_factor,
                max_k=geo_k_budget,
            )
            for res in resolutions
        }

        encoder_output = self.encode(
            groups=groups,
            grid_configs=grid_configs,
            training=training,
            return_trajectory=False,
            mask_ratio=0.0,             # no MAE during MT
            cross_k=batch_cross_k,
        )
        latents_per_res = encoder_output.latents_per_res
        coords_per_res  = encoder_output.coords_per_res

        # ── Route per task type ────────────────────────────────────────
        if spec["type"] == "seg":
            return self._forward_seg(
                latents_per_res, coords_per_res, batch, task, training,
            )
        if spec["type"] == "cls":
            return self._forward_cls(latents_per_res, task)

        raise ValueError(f"Unknown task type '{spec['type']}' for '{task}'")

    # ─────────────────────────────────────────────────────────────────
    # Per-type forward paths
    # ─────────────────────────────────────────────────────────────────

    def _forward_seg(self, latents_per_res, coords_per_res, batch, task, training):
        """
        Seg path: decode through pooled cross-attention to per-pixel queries,
        then run the task-specific seg head.

        Mirrors Atomiser_Senflood.forward's task in {"reconstruction",
        "visualization"} branch but stops short of `reconstruction_head`
        and instead applies `seg_heads[task]`.
        """
        queries      = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)

        # ── Decode (chunked, same as base) ─────────────────────────────
        # Base's reconstruct() returns pre-head features when
        # return_features=True. We use that path so our per-task seg head
        # can do the final projection.
        chunk_size = 10_000
        N = queries.shape[1]

        if N > chunk_size:
            feats_list = []
            for i in range(0, N, chunk_size):
                feats = self.reconstruct(
                    latents_per_res, coords_per_res,
                    queries[:, i:i + chunk_size],
                    queries_mask[:, i:i + chunk_size],
                    target_resolution=target_resolution,
                    training=training,
                    return_features=True,
                    return_topk=False,
                )
                feats_list.append(feats)
            features = torch.cat(feats_list, dim=1)         # [B, M, latent_dim]
        else:
            features = self.reconstruct(
                latents_per_res, coords_per_res,
                queries, queries_mask,
                target_resolution=target_resolution,
                training=training,
                return_features=True,
                return_topk=False,
            )                                                # [B, M, latent_dim]

        return self.seg_heads[task](features)                # [B, M, K_task]

    def _forward_cls(self, latents_per_res, task):
        """
        Cls path: pool latents via per-task cls head (LatentAttentionPooling
        + LayerNorm + Linear), return [B, K_task].

        Mirrors `Atomiser_Senflood.classify`, which concats latents across
        resolutions before running them through to_logits.
        """
        all_latents = torch.cat(
            [latents_per_res[res]
             for res in sorted(latents_per_res.keys(), key=str)],
            dim=1,
        )
        return self.cls_heads[task](all_latents)             # [B, K_task]


# ────────────────────────────────────────────────────────────────────
# Convenience builder
# ────────────────────────────────────────────────────────────────────

def build_multitask_atomiser(*, config, lookup_table, task_specs):
    """
    Construct a MultiTaskAtomiser. Thin wrapper for symmetry with
    build_multitask_resnet_upernet / build_multitask_vit_upernet.

    Example:
        task_specs = {
            "burnscars": {"type": "seg", "num_classes": 2},
            "senflood":  {"type": "seg", "num_classes": 2},
            "pastis":    {"type": "seg", "num_classes": 19},
            "eurosat":   {"type": "cls", "num_classes": 10},
            "forestnet": {"type": "cls", "num_classes": 12},
        }
        model = build_multitask_atomiser(
            config=cfg, lookup_table=lut, task_specs=task_specs,
        )
    """
    return MultiTaskAtomiser(
        config=config, lookup_table=lookup_table, task_specs=task_specs,
    )