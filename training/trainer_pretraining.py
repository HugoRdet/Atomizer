"""
Multi-Task Pre-training Trainer — Encode-Once
===============================================

Encodes input tokens ONCE, then decodes per-task for all tasks
in the batch. ~3× faster than separate per-task datasets because
encoding (geographic pruning + cross-attention) is the bottleneck.

Batch format (from MMEarthMultiTask + collate_multitask):
    {
        "groups": {res: {"tokens": [B,N,8], "mask": [B,N], "shape": ...}},
        "tasks": {
            "esa_worldcover":  {"queries": [B,M1,8], "queries_mask": [B,M1]},
            "dynamic_world":   {"queries": [B,M2,8], "queries_mask": [B,M2]},
            "reconstruction":  {"queries": [B,M3,8], "queries_mask": [B,M3]},
        },
        "target_resolution": 10.0,
    }

Architecture:
    1. encode(groups) → latents, coords           [ONCE — expensive]
    2. for each task:
       reconstruct(latents, coords, queries) → features  [cheap]
       head(features) → predictions                      [cheap]
       loss(predictions, targets)

Backward compatible: also handles old single-task format with
batch["task"] key.
"""

import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.utils.datasets.token_grouping import compute_grid_config


# =============================================================================
# TASK REGISTRY
# =============================================================================

SEGMENTATION_TASKS = ("dynamic_world", "esa_worldcover", "flairhub_cosia", "flairhub_lpis")
RECONSTRUCTION_TASKS = ("reconstruction",)
ALL_TASKS = SEGMENTATION_TASKS + RECONSTRUCTION_TASKS


class Model_Pretrain(pl.LightningModule):
    """
    Encode-once multi-task pre-training trainer.

    Shared encoder + task-specific MLP heads.
    Supports both:
        - Multi-task batches (batch["tasks"] dict → encode once)
        - Single-task batches (batch["task"] str → legacy path)
    """

    IGNORE_INDEX = 255

    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table

        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # =====================================================================
        # TASK CONFIGURATION
        # =====================================================================
        self.task_configs = {
            "esa_worldcover":  {"num_classes": 11, "type": "segmentation"},
            "dynamic_world":   {"num_classes": 9,  "type": "segmentation"},
            "flairhub_cosia":  {"num_classes": 18, "type": "segmentation"},
            "flairhub_lpis":   {"num_classes": 23, "type": "segmentation"},
            "reconstruction":  {"num_classes": 1,  "type": "reconstruction"},
        }

        # =====================================================================
        # ENCODER (shared)
        # =====================================================================
        self.encoder = Atomiser_Senflood(
            config=self.config, lookup_table=self.lookup_table
        )

        self.feature_dim = config["Atomiser"].get("latent_dim", 256)

        # =====================================================================
        # TASK-SPECIFIC HEADS
        # =====================================================================
        self.heads = nn.ModuleDict()
        for task_name, task_cfg in self.task_configs.items():
            if task_cfg["type"] == "segmentation":
                self.heads[task_name] = nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, task_cfg["num_classes"]),
                )
            else:
                self.heads[task_name] = nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, 1),
                )

        # =====================================================================
        # LOSSES
        # =====================================================================
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.recon_loss = nn.MSELoss()

        # =====================================================================
        # HOMOSCEDASTIC UNCERTAINTY WEIGHTING (Kendall et al., 2018)
        # =====================================================================
        # Learn log(σ²) per task. Loss becomes: exp(-s)*L + s
        # where s = log(σ²). This auto-balances loss magnitudes.
        #
        # Init values:
        #   seg tasks → s=0 → weight=1.0 (CE losses are ~1.0)
        #   recon     → s=0 → weight=1.0
        #
        # FIX: previously recon was init at -3 → exp(3)≈20× amplification
        # of early high MSE → gradient explosion → NaN by step 30.
        # Starting at 0 lets the optimizer find the right balance safely.
        self.log_vars = nn.ParameterDict({
            "esa_worldcover":  nn.Parameter(torch.tensor(0.0)),
            "dynamic_world":   nn.Parameter(torch.tensor(0.0)),
            "flairhub_cosia":  nn.Parameter(torch.tensor(0.0)),
            "flairhub_lpis":   nn.Parameter(torch.tensor(0.0)),
            "reconstruction":  nn.Parameter(torch.tensor(0.0)),
        })

        # =====================================================================
        # METRICS
        # =====================================================================
        self._init_metrics()
        self._train_tasks_seen = set()
        self._val_tasks_seen = set()

        print(f"[Pretrain] Encode-once multi-task trainer:")
        for task_name, task_cfg in self.task_configs.items():
            print(f"  {task_name}: {task_cfg['type']} "
                  f"({'classes=' + str(task_cfg['num_classes']) if task_cfg['type'] == 'segmentation' else 'MSE'})")

    # =========================================================================
    # METRICS
    # =========================================================================

    def _init_metrics(self):
        for task_name in SEGMENTATION_TASKS:
            if task_name not in self.task_configs:
                continue
            nc = self.task_configs[task_name]["num_classes"]
            for split in ["train", "val"]:
                setattr(self, f"{split}_{task_name}_mIoU", torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=nc,
                    average="macro", ignore_index=self.IGNORE_INDEX,
                ))
                setattr(self, f"{split}_{task_name}_acc", torchmetrics.Accuracy(
                    task="multiclass", num_classes=nc,
                    average="macro", ignore_index=self.IGNORE_INDEX,
                ))

        for split in ["train", "val"]:
            setattr(self, f"{split}_recon_mse", torchmetrics.MeanSquaredError())
            setattr(self, f"{split}_recon_mae", torchmetrics.MeanAbsoluteError())

    # =========================================================================
    # ENCODE-ONCE FORWARD
    # =========================================================================

    def _encode(self, batch, training=True, tpl_override=None):
        """
        Encode groups → latents + coords + grid_configs.

        Returns:
            latents_per_res: dict {res: [B, L, D]}
            coords_per_res: dict {res: [B, L, 2]}
            grid_configs: dict {res: config}
        """
        groups = batch["groups"]
        tpl = tpl_override or self.encoder.tokens_per_latent

        # ── DIAGNOSTIC: check input tokens for NaN ──
        if self.global_step < 3:
            for res, g in groups.items():
                tok = g["tokens"]
                msk = g["mask"]
                n_valid = (~msk).sum().item() if msk.dim() == 1 else (~msk).sum(dim=-1).float().mean().item()
                has_nan = torch.isnan(tok).any().item()
                has_inf = torch.isinf(tok).any().item()
                if has_nan or has_inf:
                    nan_cols = torch.isnan(tok).any(dim=0).any(dim=0) if tok.dim() == 3 else torch.isnan(tok).any(dim=0)
                    print(f"[DIAG encode] res={res}: NaN={has_nan} Inf={has_inf} "
                          f"shape={list(tok.shape)} valid={n_valid} "
                          f"nan_columns={nan_cols.nonzero().flatten().tolist()}")
                    # Per-column stats for non-NaN values
                    for col in range(tok.shape[-1]):
                        col_data = tok[..., col]
                        finite = col_data[torch.isfinite(col_data)]
                        if finite.numel() > 0:
                            print(f"    col[{col}]: range=[{finite.min():.4f}, {finite.max():.4f}] "
                                  f"std={finite.std():.6f} nan_frac={torch.isnan(col_data).float().mean():.3f}")
                        else:
                            print(f"    col[{col}]: ALL NaN/Inf")
                else:
                    # Still print basic stats for first few steps
                    print(f"[DIAG encode] res={res}: OK shape={list(tok.shape)} "
                          f"valid={n_valid} range=[{tok.min():.4f}, {tok.max():.4f}]")

        resolutions = sorted(groups.keys())
        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.encoder.sigma_factor,
                max_k=self.encoder.max_k,
            )
            for res in resolutions
        }

        encoder_output = self.encoder.encode(
            groups=groups,
            grid_configs=grid_configs,
            training=training,
        )

        # ── DIAGNOSTIC: check encoder output for NaN ──
        if self.global_step < 3:
            for res in encoder_output.latents_per_res:
                lat = encoder_output.latents_per_res[res]
                has_nan = torch.isnan(lat).any().item()
                nan_frac = torch.isnan(lat).float().mean().item()
                if has_nan:
                    print(f"[DIAG encode] LATENTS res={res}: NaN! "
                          f"shape={list(lat.shape)} nan_frac={nan_frac:.3f} "
                          f"finite_range=[{lat[torch.isfinite(lat)].min():.4f}, "
                          f"{lat[torch.isfinite(lat)].max():.4f}]"
                          if torch.isfinite(lat).any() else
                          f"[DIAG encode] LATENTS res={res}: ALL NaN!")
                else:
                    print(f"[DIAG encode] LATENTS res={res}: OK "
                          f"range=[{lat.min():.4f}, {lat.max():.4f}]")

        return encoder_output.latents_per_res, encoder_output.coords_per_res, grid_configs

    def _decode(self, latents_per_res, coords_per_res, queries, queries_mask,
                task_name, target_resolution=None, training=True):
        """
        Decode: latents + queries → pre-head features [B, M, D].

        Handles chunked decoding for large query sets.
        """
        chunk_size = 10000
        N = queries.shape[1]

        if N > chunk_size:
            feats = []
            for i in range(0, N, chunk_size):
                f = self.encoder.reconstruct(
                    latents_per_res, coords_per_res,
                    queries[:, i:i + chunk_size],
                    queries_mask[:, i:i + chunk_size],
                    target_resolution=target_resolution,
                    task_name=task_name,
                    training=training,
                    return_features=True,
                )
                feats.append(f)
            return torch.cat(feats, dim=1)
        else:
            return self.encoder.reconstruct(
                latents_per_res, coords_per_res,
                queries, queries_mask,
                target_resolution=target_resolution,
                task_name=task_name,
                training=training,
                return_features=True,
            )

    def forward_multitask(self, batch, training=True):
        """
        Encode-once, decode per-task.
        """
        # Variable latent density for reconstruction
        tpl_override = None
        if training and "reconstruction" in batch.get("tasks", {}):
            tpl_override = self._sample_tokens_per_latent()

        # ENCODE ONCE
        latents_per_res, coords_per_res, grid_configs = self._encode(
            batch, training=training, tpl_override=tpl_override,
        )

        target_resolution = batch.get("target_resolution", None)

        # DECODE PER TASK
        predictions = {}
        for task_name, task_data in batch["tasks"].items():
            if task_name not in self.heads:
                continue

            features = self._decode(
                latents_per_res, coords_per_res,
                task_data["queries"], task_data["queries_mask"],
                task_name=task_name,
                target_resolution=target_resolution,
                training=training,
            )

            predictions[task_name] = self.heads[task_name](features)

        return predictions

    # =========================================================================
    # LEGACY SINGLE-TASK FORWARD (backward compat)
    # =========================================================================

    def forward(self, batch, task_name=None, training=False):
        """
        Legacy forward: single task.

        If batch has "tasks" key → dispatch to forward_multitask.
        If batch has "task" key → single-task path.
        """
        if "tasks" in batch:
            return self.forward_multitask(batch, training=training)

        # Legacy single-task path
        if task_name is None:
            task_name = batch.get("task", "reconstruction")

        tpl_override = None
        if task_name == "reconstruction" and training:
            tpl_override = self._sample_tokens_per_latent()

        result = self.encoder(
            batch,
            training=training,
            task="reconstruction",
            return_features=True,
            tokens_per_latent_override=tpl_override,
        )

        features = result["features"]
        return self.heads[task_name](features)

    def _sample_tokens_per_latent(self) -> int:
        """
        Sample tokens_per_latent for variable-density training.

        DDP CRITICAL: Must be identical across ranks. Use global_step as
        seed so both ranks pick the same value deterministically.
        """
        choices = self.config.get("pretrain", {}).get(
            "tokens_per_latent_choices", [512, 768, 1024, 1500, 2000]
        )
        # Deterministic: index by global_step so all ranks agree
        idx = self.global_step % len(choices)
        return choices[idx]

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================

    def _seg_warmup_factor(self) -> float:
        """
        Linear warmup for segmentation losses.

        Ramps from seg_warmup_floor → 1.0 over the first seg_warmup_frac
        of total training. Keeps seg heads receiving *some* gradient from
        the start (floor=0.05) while letting reconstruction dominate early.

        Composes with homoscedastic uncertainty:
            effective_seg_loss = warmup_α * exp(-s) * L_raw + s
        """
        warmup_frac = self.config.get("pretrain", {}).get("seg_warmup_frac", 0.1)
        warmup_floor = self.config.get("pretrain", {}).get("seg_warmup_floor", 0.05)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(warmup_frac * total_steps)

        if warmup_steps <= 0:
            return 1.0

        progress = min(self.global_step / warmup_steps, 1.0)
        return warmup_floor + (1.0 - warmup_floor) * progress

    def _weighted_loss(self, task_name, raw_loss):
        """
        Homoscedastic uncertainty weighting (Kendall et al., 2018).

        L_weighted = exp(-s) * L_raw + s
        where s = log(σ²) is a learnable log-variance.

        Clamp s to [-6, 6]:
          - exp(-(-6)) = exp(6)  ≈ 403  — max amplification
          - exp(-6)              ≈ 0.002 — min amplification
        Previous [-10, 10] allowed exp(10) ≈ 22026× which caused
        gradient explosion with early high reconstruction MSE.
        """
        s = self.log_vars[task_name].clamp(-6, 6)
        return torch.exp(-s) * raw_loss + s

    def _compute_seg_loss(self, predictions, queries):
        """Segmentation loss from query col 4. Returns None if no valid labels."""
        labels = queries[:, :, 4].long()
        pred_flat = rearrange(predictions, "b m c -> (b m) c")
        label_flat = rearrange(labels, "b m -> (b m)")

        # Guard: if all labels are ignore_index, CE returns NaN (0/0)
        valid_count = (label_flat != self.seg_loss.ignore_index).sum()
        if valid_count == 0:
            return None, None, None

        loss = self.seg_loss(pred_flat, label_flat)
        preds = torch.argmax(predictions, dim=-1)
        return loss, preds, labels

    def _compute_recon_loss(self, predictions, queries, queries_mask):
        """
        Reconstruction loss from query col 4, masked.

        Returns (None, None, None) if no valid queries — mirrors
        _compute_seg_loss behavior to prevent NaN from mse_loss
        on empty tensors.
        """
        targets = queries[:, :, 4]
        preds = predictions.squeeze(-1)

        # FIX: Guard against empty valid set.
        # When ALL queries are masked (dummy sample or entire batch of
        # dummies), preds[valid] is empty → mse_loss returns NaN.
        valid = ~queries_mask
        valid_count = valid.sum()
        if valid_count == 0:
            return None, None, None

        loss = nn.functional.mse_loss(preds[valid], targets[valid])
        return loss, preds, targets

    # =========================================================================
    # TRAINING STEP
    # =========================================================================

    def training_step(self, batch, batch_idx):
        # Multi-task path (encode-once)
        if "tasks" in batch:
            return self._training_step_multitask(batch)

        # Legacy single-task path
        return self._training_step_single(batch)

    def _training_step_multitask(self, batch):
        """Encode once, decode + loss per task."""

        all_predictions = self.forward_multitask(batch, training=True)

        # ── DIAGNOSTIC PRINT BLOCK (keep your existing diagnostic code here) ──
        any_nan = any(torch.isnan(p).any().item() for p in all_predictions.values())
        if any_nan:
            print(f"\n[DIAG step={self.global_step}] === NaN DETECTED ===", flush=True)
            for res, g in batch["groups"].items():
                tok = g["tokens"]
                msk = g["mask"]
                n_valid = (~msk).sum().item()
                print(f"  INPUT res={res}: shape={list(tok.shape)} valid={n_valid}", flush=True)
                col_names = ["value", "x", "y", "spec_idx", "label", "query_idx", "res_idx", "time_idx"]
                for col in range(min(tok.shape[-1], len(col_names))):
                    col_data = tok[..., col]
                    finite = col_data[torch.isfinite(col_data)]
                    if finite.numel() > 0:
                        print(f"    col[{col}] {col_names[col]:>9s}: "
                              f"min={finite.min().item():>10.2f}  "
                              f"max={finite.max().item():>10.2f}  "
                              f"nan={torch.isnan(col_data).any().item()}  "
                              f"inf={torch.isinf(col_data).any().item()}",
                              flush=True)
            for task_name, pred in all_predictions.items():
                has_nan = torch.isnan(pred).any().item()
                if has_nan:
                    print(f"  PRED {task_name}: ALL NaN", flush=True)
                else:
                    print(f"  PRED {task_name}: OK range=[{pred.min():.4f}, {pred.max():.4f}]", flush=True)

        total_loss = 0.0
        num_tasks = 0
        seg_alpha = self._seg_warmup_factor()

        for task_name, predictions in all_predictions.items():
            task_data = batch["tasks"][task_name]
            queries = task_data["queries"]
            queries_mask = task_data["queries_mask"]

            self._train_tasks_seen.add(task_name)

            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

            is_dummy = False

            if task_name in SEGMENTATION_TASKS:
                loss, preds, labels = self._compute_seg_loss(predictions, queries)

                # No valid labels → mark as dummy
                if loss is None or not torch.isfinite(loss):
                    loss = predictions.sum() * 0.0
                    is_dummy = True
                else:
                    metric_mIoU = getattr(self, f"train_{task_name}_mIoU")
                    metric_acc = getattr(self, f"train_{task_name}_acc")
                    metric_mIoU.update(preds, labels)
                    metric_acc.update(preds, labels)
                    
                    # UPDATED: prog_bar=True so you see individual seg losses in the progress bar
                    self.log(f"train_{task_name}_loss", loss,
                             on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    loss = loss * seg_alpha

            elif task_name in RECONSTRUCTION_TASKS:
                loss, preds, targets = self._compute_recon_loss(
                    predictions, queries, queries_mask
                )

                # No valid queries → mark as dummy
                if loss is None or not torch.isfinite(loss):
                    loss = predictions.sum() * 0.0
                    is_dummy = True
                else:
                    self.train_recon_mse.update(preds, targets)
                    self.train_recon_mae.update(preds, targets)

                    # UPDATED: prog_bar=True so you see recon loss in the progress bar
                    self.log("train_recon_loss", loss,
                             on_step=True, on_epoch=True, prog_bar=True, logger=True)
            else:
                loss = predictions.sum() * 0.0
                is_dummy = True

            # Always log raw loss (0.0 for dummy tasks), keeping it off the progress bar to avoid clutter
            self.log(f"train_{task_name}_raw_loss", 0.0 if is_dummy else loss,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # ── THE DDP-SAFE HOMOSCEDASTIC FIX ──
            if is_dummy:
                # Add MLP head (loss) and log_var to the graph with 0.0 gradient
                total_loss = total_loss + loss + (self.log_vars[task_name] * 0.0)
            else:
                weighted = self._weighted_loss(task_name, loss)
                total_loss = total_loss + weighted
                num_tasks += 1

        # UPDATED: ensure total train_loss is on the progress bar
        self.log("train_loss", total_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("seg_warmup_alpha", seg_alpha, on_step=False, on_epoch=True, logger=True)
        for task_name in batch["tasks"]:
            if task_name in self.log_vars:
                s = self.log_vars[task_name]
                self.log(f"weight/{task_name}", torch.exp(-s), on_step=False, on_epoch=True, logger=True)
                self.log(f"log_var/{task_name}", s, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def _training_step_single(self, batch):
        """Legacy single-task training step."""
        task_name = batch["task"]
        self._train_tasks_seen.add(task_name)

        predictions = self.forward(batch, task_name, training=True)

        if task_name in SEGMENTATION_TASKS:
            loss, preds, labels = self._compute_seg_loss(
                predictions, batch["queries"]
            )
            if loss is None:
                return sum(p.sum() for p in self.parameters()) * 0.0
            metric_mIoU = getattr(self, f"train_{task_name}_mIoU")
            metric_acc = getattr(self, f"train_{task_name}_acc")
            metric_mIoU.update(preds, labels)
            metric_acc.update(preds, labels)
            self.log(f"train_{task_name}_loss", loss,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)

        elif task_name in RECONSTRUCTION_TASKS:
            loss, preds, targets = self._compute_recon_loss(
                predictions, batch["queries"], batch["queries_mask"]
            )
            if loss is None:
                return sum(p.sum() for p in self.parameters()) * 0.0
            self.train_recon_mse.update(preds, targets)
            self.train_recon_mae.update(preds, targets)
            self.log("train_recon_loss", loss,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            raise ValueError(f"Unknown task: {task_name}")

        self.log("train_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # =========================================================================
    # VALIDATION STEP
    # =========================================================================

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if "tasks" in batch:
            return self._validation_step_multitask(batch)
        return self._validation_step_single(batch, batch_idx, dataloader_idx)

    def _validation_step_multitask(self, batch):
        """Encode once, validate per task."""
        all_predictions = self.forward_multitask(batch, training=False)

        total_loss = 0.0
        num_tasks = 0

        for task_name, predictions in all_predictions.items():
            task_data = batch["tasks"][task_name]
            queries = task_data["queries"]
            queries_mask = task_data["queries_mask"]

            self._val_tasks_seen.add(task_name)

            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

            is_dummy = False

            if task_name in SEGMENTATION_TASKS:
                loss, preds, labels = self._compute_seg_loss(predictions, queries)

                if loss is None or not torch.isfinite(loss):
                    loss = torch.tensor(0.0, device=predictions.device)
                    is_dummy = True
                else:
                    metric_mIoU = getattr(self, f"val_{task_name}_mIoU")
                    metric_acc = getattr(self, f"val_{task_name}_acc")
                    metric_mIoU.update(preds, labels)
                    metric_acc.update(preds, labels)

                    # UPDATED: prog_bar=True
                    self.log(f"val_{task_name}_loss", loss,
                             on_step=False, on_epoch=True, prog_bar=True, logger=True)

            elif task_name in RECONSTRUCTION_TASKS:
                loss, preds, targets = self._compute_recon_loss(
                    predictions, queries, queries_mask
                )
                if loss is None or not torch.isfinite(loss):
                    loss = torch.tensor(0.0, device=predictions.device)
                    is_dummy = True
                else:
                    self.val_recon_mse.update(preds, targets)
                    self.val_recon_mae.update(preds, targets)

                    # UPDATED: prog_bar=True
                    self.log("val_recon_loss", loss,
                             on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                loss = torch.tensor(0.0, device=predictions.device)
                is_dummy = True

            # Only accumulate valid losses for an accurate validation curve
            if not is_dummy:
                total_loss = total_loss + loss
                num_tasks += 1

        if num_tasks > 0:
            total_loss = total_loss / num_tasks

        # UPDATED: prog_bar=True
        self.log("val_loss", total_loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    # =========================================================================
    # EPOCH END
    # =========================================================================

    def on_train_epoch_end(self):
        miou_values = []
        for task_name in SEGMENTATION_TASKS:
            if task_name not in self._train_tasks_seen:
                continue
            if not hasattr(self, f"train_{task_name}_mIoU"):
                continue

            metric_mIoU = getattr(self, f"train_{task_name}_mIoU")
            metric_acc = getattr(self, f"train_{task_name}_acc")

            miou = metric_mIoU.compute()
            self.log(f"train_{task_name}_mIoU", miou, on_epoch=True, logger=True)
            self.log(f"train_{task_name}_acc", metric_acc.compute(), on_epoch=True, logger=True)
            miou_values.append(miou)
            metric_mIoU.reset()
            metric_acc.reset()

        if miou_values:
            self.log("train_avg_mIoU", torch.stack(miou_values).mean(),
                     on_epoch=True, prog_bar=True, logger=True)

        if "reconstruction" in self._train_tasks_seen:
            self.log("train_recon_mse", self.train_recon_mse.compute(), on_epoch=True, logger=True)
            self.log("train_recon_mae", self.train_recon_mae.compute(), on_epoch=True, logger=True)
            self.train_recon_mse.reset()
            self.train_recon_mae.reset()

        self._train_tasks_seen.clear()

    def on_validation_epoch_end(self):
        miou_values = []
        for task_name in SEGMENTATION_TASKS:
            if task_name not in self._val_tasks_seen:
                continue
            if not hasattr(self, f"val_{task_name}_mIoU"):
                continue

            metric_mIoU = getattr(self, f"val_{task_name}_mIoU")
            metric_acc = getattr(self, f"val_{task_name}_acc")

            miou = metric_mIoU.compute()
            self.log(f"val_{task_name}_mIoU", miou, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val_{task_name}_acc", metric_acc.compute(), on_epoch=True, logger=True)
            miou_values.append(miou)
            metric_mIoU.reset()
            metric_acc.reset()

        if miou_values:
            self.log("val_avg_mIoU", torch.stack(miou_values).mean(),
                     on_epoch=True, prog_bar=True, logger=True)

        if "reconstruction" in self._val_tasks_seen:
            self.log("val_recon_mse", self.val_recon_mse.compute(),
                     on_epoch=True, prog_bar=True, logger=True)
            self.log("val_recon_mae", self.val_recon_mae.compute(), on_epoch=True, logger=True)
            self.val_recon_mse.reset()
            self.val_recon_mae.reset()

        self._val_tasks_seen.clear()

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_{self.name}{suffix}.pth"
        state = {
            "encoder": self.encoder.state_dict(),
            "heads": self.heads.state_dict(),
        }
        torch.save(state, file_path)
        print(f"[Pretrain] Model saved to {file_path}")

    def load_model(self, name=None, encoder_only=False):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_{self.name}{suffix}.pth"
        state = torch.load(file_path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[Pretrain] Encoder loaded from {file_path}")
        if not encoder_only and "heads" in state:
            self.heads.load_state_dict(state["heads"])
            print(f"[Pretrain] Heads loaded from {file_path}")

    def load_encoder_for_downstream(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[Pretrain] Encoder loaded for downstream from {checkpoint_path}")