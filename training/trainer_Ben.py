"""
BigEarthNet Classification Trainer for Atomizer-IO Pretraining
================================================================

Pretrains the Atomizer encoder on BigEarthNet-S2 multi-label classification.
Uses attention pooling on latents (no decoder/queries needed).

Architecture:
    1. encode(groups) → latents [B, L, D]
    2. attention_pool(latents) → embedding [B, D]
    3. classification_head(embedding) → logits [B, 19]
    4. BCEWithLogitsLoss(logits, multi_hot_labels)

The pretrained encoder checkpoint transfers directly to C2Seg
fine-tuning via load_encoder_for_downstream().
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.utils.datasets.token_grouping import compute_grid_config


class AttentionPooling(nn.Module):
    """
    Attention pooling: [B, L, D] → [B, D].
    
    Learns a query vector that attends over latents to produce
    a single global representation per sample.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, L, D]
        Returns:
            pooled: [B, D]
        """
        B = latents.shape[0]
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        pooled, _ = self.attn(query, latents, latents)  # [B, 1, D]
        pooled = self.norm(pooled.squeeze(1))  # [B, D]
        return pooled


class BENPretrainTrainer(pl.LightningModule):
    """
    BigEarthNet multi-label classification trainer.
    
    Shared encoder (Atomiser) + attention pooling + linear head.
    BCEWithLogitsLoss for multi-label classification.
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
            "ben_classification": {"num_classes": 19, "type": "classification"},
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
        self.pool = AttentionPooling(self.feature_dim)
        self.heads = nn.ModuleDict()
        for task_name, task_cfg in self.task_configs.items():
            self.heads[task_name] = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, task_cfg["num_classes"]),
            )
        
        # =====================================================================
        # LOSS
        # =====================================================================
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # =====================================================================
        # METRICS
        # =====================================================================
        nc = self.task_configs["ben_classification"]["num_classes"]
        self.train_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=nc, average="macro",
        )
        self.val_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=nc, average="macro",
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=nc, average="macro",
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=nc, average="macro",
        )
        
        # ── Variable latent density ──
        ben_cfg = config.get("ben_pretrain", {})
        self.tpl_min = ben_cfg.get("tpl_min", 768)
        self.tpl_max = ben_cfg.get("tpl_max", 768)
        self.tpl_val = ben_cfg.get("tpl_val", 768)
        self.tpl_step = ben_cfg.get("tpl_step", 20)
        
        n_tpl_values = (self.tpl_max - self.tpl_min) // self.tpl_step + 1 if self.tpl_min < self.tpl_max else 1
        
        print(f"[BEN Pretrain] Single-task trainer:")
        for task_name, task_cfg in self.task_configs.items():
            print(f"  {task_name}: {task_cfg['type']} "
                  f"(classes={task_cfg['num_classes']})")
        print(f"[BEN Pretrain] lr={self.lr}, wd={self.weight_decay}")
        print(f"[BEN Pretrain] tokens_per_latent: train=[{self.tpl_min}:{self.tpl_step}:{self.tpl_max}] "
              f"({n_tpl_values} values), val={self.tpl_val}")
    
    # ═════════════════════════════════════════════════════════════════
    # ENCODE
    # ═════════════════════════════════════════════════════════════════
    
    def _encode(self, batch, training=True):
        """
        Encode groups → latents [B, L, D].
        
        During training, tokens_per_latent is derived from global_step
        to ensure all DDP ranks use the same value (preventing deadlocks
        from mismatched latent grid sizes). Quantized to steps of tpl_step.
        During validation, uses fixed tpl_val for consistent metrics.
        """
        groups = batch["groups"]
        
        if training and self.tpl_min < self.tpl_max:
            # Build quantized grid: [tpl_min, tpl_min+step, ..., tpl_max]
            n_steps = (self.tpl_max - self.tpl_min) // self.tpl_step
            # DDP-safe: global_step is synchronized across all ranks
            step = self.global_step
            idx = step * 2654435761 % (n_steps + 1)
            tpl = self.tpl_min + int(idx) * self.tpl_step
        else:
            tpl = self.tpl_val
        
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
        
        # Concatenate latents across resolutions → [B, L_total, D]
        all_latents = []
        for res in resolutions:
            if res in encoder_output.latents_per_res:
                all_latents.append(encoder_output.latents_per_res[res])
        
        if len(all_latents) == 0:
            raise RuntimeError("No latents produced by encoder")
        
        latents = torch.cat(all_latents, dim=1)  # [B, L_total, D]
        return latents
    
    # ═════════════════════════════════════════════════════════════════
    # FORWARD
    # ═════════════════════════════════════════════════════════════════
    
    def forward(self, batch, training=False):
        latents = self._encode(batch, training=training)  # [B, L, D]
        pooled = self.pool(latents)                        # [B, D]
        logits = self.heads["ben_classification"](pooled)  # [B, 19]
        return logits
    
    # ═════════════════════════════════════════════════════════════════
    # TRAINING STEP
    # ═════════════════════════════════════════════════════════════════
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch, training=True)  # [B, 19]
        labels = batch["label"]                       # [B, 19] multi-hot
        
        loss = self.loss_fn(logits, labels)
        
        # Metrics
        probs = torch.sigmoid(logits)
        self.train_map.update(probs, labels.long())
        self.train_f1.update(probs, labels.long())
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # ═════════════════════════════════════════════════════════════════
    # VALIDATION STEP
    # ═════════════════════════════════════════════════════════════════
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch, training=False)
        labels = batch["label"]
        
        loss = self.loss_fn(logits, labels)
        
        probs = torch.sigmoid(logits)
        self.val_map.update(probs, labels.long())
        self.val_f1.update(probs, labels.long())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    # ═════════════════════════════════════════════════════════════════
    # EPOCH END
    # ═════════════════════════════════════════════════════════════════
    
    def on_train_epoch_end(self):
        self.log("train_mAP", self.train_map.compute(), on_epoch=True, prog_bar=True)
        self.log("train_F1", self.train_f1.compute(), on_epoch=True)
        self.train_map.reset()
        self.train_f1.reset()
    
    def on_validation_epoch_end(self):
        self.log("val_mAP", self.val_map.compute(), on_epoch=True, prog_bar=True)
        self.log("val_F1", self.val_f1.compute(), on_epoch=True)
        self.val_map.reset()
        self.val_f1.reset()
    
    # ═════════════════════════════════════════════════════════════════
    # OPTIMIZER
    # ═════════════════════════════════════════════════════════════════
    
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
    
    # ═════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ═════════════════════════════════════════════════════════════════
    
    def save_model(self, name=None):
        """Save encoder + classification heads."""
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_ben_{self.name}{suffix}.pth"
        state = {
            "encoder": self.encoder.state_dict(),
            "pool": self.pool.state_dict(),
            "heads": self.heads.state_dict(),
        }
        torch.save(state, file_path)
        print(f"[BEN Pretrain] Model saved to {file_path}")
    
    def save_encoder_only(self, name=None):
        """Save only the encoder for downstream transfer."""
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_ben_encoder_{self.name}{suffix}.pth"
        state = {"encoder": self.encoder.state_dict()}
        torch.save(state, file_path)
        print(f"[BEN Pretrain] Encoder saved to {file_path}")
    
    def load_model(self, path: str, encoder_only: bool = False):
        """Load from checkpoint."""
        state = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[BEN Pretrain] Encoder loaded from {path}")
        if not encoder_only:
            if "pool" in state:
                self.pool.load_state_dict(state["pool"])
            if "heads" in state:
                self.heads.load_state_dict(state["heads"])
            print(f"[BEN Pretrain] Full model loaded from {path}")