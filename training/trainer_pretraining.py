"""
Multi-Task Pre-training Trainer for Atomizer
==============================================

Alternates between 3 tasks using CombinedLoader("sequential"):
    - esa_worldcover:  Segmentation (11 classes) — CrossEntropyLoss
    - dynamic_world:   Segmentation (9 classes)  — CrossEntropyLoss
    - reconstruction:  Reflectance prediction     — MSELoss

Also works with a single task (e.g., just esa_worldcover) — unused heads
sit idle and their metrics are skipped at epoch end.

Architecture:
    Shared encoder produces features [B, M, D] (pre-head).
    Task-specific MLP heads map features → predictions:
        - Segmentation: [B, M, D] → [B, M, num_classes]
        - Reconstruction: [B, M, D] → [B, M, 1]

Batch format:
    groups[res]["tokens"]: [B, N, 8]
    queries:               [B, M, 8]
    queries_mask:          [B, M]
    task:                  str ("esa_worldcover" | "dynamic_world" | "reconstruction")

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
    col 4 = class label (segmentation) or reflectance target (reconstruction)

Usage with CombinedLoader:
    from pytorch_lightning.trainer import Trainer
    from lightning.pytorch.utilities import CombinedLoader

    train_loaders = {
        "esa_worldcover": esa_loader,
        "dynamic_world": dw_loader,
        "reconstruction": recon_loader,
    }
    combined = CombinedLoader(train_loaders, mode="sequential")
    trainer.fit(model, train_dataloaders=combined, val_dataloaders=val_loader)

Usage with a single task:
    dm = UnifiedDataModule(dataset_class=MMEarthSegESA, ...)
    trainer.fit(model, datamodule=dm)
"""
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood


# =============================================================================
# TASK REGISTRY
# =============================================================================

SEGMENTATION_TASKS = ["esa_worldcover", "dynamic_world", "flairhub_cosia", "flairhub_lpis"]
RECONSTRUCTION_TASKS = ["reconstruction"]
ALL_TASKS = SEGMENTATION_TASKS + RECONSTRUCTION_TASKS


class Model_Pretrain(pl.LightningModule):
    """
    Multi-task pre-training trainer.

    Shared encoder + task-specific MLP heads.
    Dispatches loss/metrics based on batch["task"].

    Works with 1, 2, or 3 tasks — unused tasks are simply skipped
    at epoch end (no errors, no misleading averages).
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

        # Get feature dimension from encoder config
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
            else:  # reconstruction
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
        # METRICS — per task, per split
        # =====================================================================
        self._init_metrics()

        # Track which tasks actually received data this epoch
        self._train_tasks_seen = set()
        self._val_tasks_seen = set()

        # =====================================================================
        # LOGGING
        # =====================================================================
        print(f"[Pretrain] Multi-task trainer initialized:")
        for task_name, task_cfg in self.task_configs.items():
            print(f"  {task_name}: {task_cfg['type']} "
                  f"({'classes=' + str(task_cfg['num_classes']) if task_cfg['type'] == 'segmentation' else 'MSE'})")

    # =========================================================================
    # METRICS INITIALIZATION
    # =========================================================================

    def _init_metrics(self):
        """Create per-task metrics for train and val."""

        # Segmentation metrics
        for task_name in SEGMENTATION_TASKS:
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

        # Reconstruction metrics
        for split in ["train", "val"]:
            setattr(self, f"{split}_recon_mse", torchmetrics.MeanSquaredError())
            setattr(self, f"{split}_recon_mae", torchmetrics.MeanAbsoluteError())

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, task_name, training=False):
        """
        Forward pass: encoder (return_features) → task-specific head.

        Args:
            batch: Standard batch dict with groups, queries, etc.
            task_name: Which head to use.
            training: Whether to use training mode (pruning etc.)

        Returns:
            predictions: [B, M, num_classes] or [B, M, 1]
        """
        # Variable latent density for reconstruction during training
        tpl_override = None
        if task_name == "reconstruction" and training:
            tpl_override = self._sample_tokens_per_latent()

        # Get pre-head features from encoder
        result = self.encoder(
            batch,
            training=training,
            task="reconstruction",  # always use decoder path (not classifier)
            return_features=True,
            tokens_per_latent_override=tpl_override,
        )

        features = result["features"]  # [B, M, D]

 
        # Apply task-specific head
        predictions = self.heads[task_name](features)

        return predictions

    def _sample_tokens_per_latent(self) -> int:
        choices = self.config.get("pretrain", {}).get(
            "tokens_per_latent_choices", [768, 1200, 1700, 3000]
        )
        return random.choice(choices)

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================

    def _compute_seg_loss(self, predictions, batch):
        labels = batch["queries"][:, :, 4].long()

        pred_flat = rearrange(predictions, "b m c -> (b m) c").float()
        label_flat = rearrange(labels, "b m -> (b m)")

        valid = label_flat != self.IGNORE_INDEX
        if not valid.any():
            # Still flow through predictions so DDP can sync gradients
            dummy_loss = 0.0 * predictions.sum()
            return dummy_loss, torch.argmax(predictions, dim=-1), labels

        loss = self.seg_loss(pred_flat, label_flat)
        preds = torch.argmax(predictions, dim=-1)

        return loss, preds, labels

    def _compute_recon_loss(self, predictions, batch):
        targets = batch["queries"][:, :, 4]
        preds = predictions.squeeze(-1)

        # Upcast for stable loss
        preds = preds.float()       # ← same treatment
        targets = targets.float()

        mask = batch["queries_mask"]
        if mask.any():
            valid = ~mask
            loss = nn.functional.mse_loss(preds[valid], targets[valid])
        else:
            loss = self.recon_loss(preds, targets)

        return loss, preds, targets

    # =========================================================================
    # TRAINING STEP
    # =========================================================================

    def training_step(self, batch, batch_idx):
        task_name = batch["task"]

        self._train_tasks_seen.add(task_name)
        

        predictions = self.forward(batch, task_name, training=True)

        if task_name in SEGMENTATION_TASKS:
            loss, preds, labels = self._compute_seg_loss(predictions, batch)

            metric_mIoU = getattr(self, f"train_{task_name}_mIoU")
            metric_acc = getattr(self, f"train_{task_name}_acc")
            metric_mIoU.update(preds, labels)
            metric_acc.update(preds, labels)

            self.log(f"train_{task_name}_loss", loss,
                    on_step=False, on_epoch=True, prog_bar=False, logger=True)

        elif task_name in RECONSTRUCTION_TASKS:
            loss, preds, targets = self._compute_recon_loss(predictions, batch)

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
        """
        Validation — same dispatch logic.

        If using a single val loader (e.g., one seg task), dataloader_idx=0.
        If using CombinedLoader for val too, dispatch by task.
        """
        
        task_name = batch.get("task")

        # Fallback: if no task key, infer from dataloader_idx
        if task_name is None:
            task_names = list(self.task_configs.keys())
            task_name = task_names[dataloader_idx] if dataloader_idx < len(task_names) else task_names[0]

        self._val_tasks_seen.add(task_name)

        predictions = self.forward(batch, task_name, training=False)

        if task_name in SEGMENTATION_TASKS:
            loss, preds, labels = self._compute_seg_loss(predictions, batch)

            metric_mIoU = getattr(self, f"val_{task_name}_mIoU")
            metric_acc = getattr(self, f"val_{task_name}_acc")
            metric_mIoU.update(preds, labels)
            metric_acc.update(preds, labels)

            self.log(f"val_{task_name}_loss", loss,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)

        elif task_name in RECONSTRUCTION_TASKS:
            loss, preds, targets = self._compute_recon_loss(predictions, batch)

            self.val_recon_mse.update(preds, targets)
            self.val_recon_mae.update(preds, targets)

            self.log("val_recon_loss", loss,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log("val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # =========================================================================
    # EPOCH END — only log metrics for tasks that received data
    # =========================================================================

    def on_training_epoch_end(self):
        # Always compute ALL metrics to keep DDP sync consistent across ranks
        miou_values = []
        for task_name in SEGMENTATION_TASKS:
            metric_mIoU = getattr(self, f"train_{task_name}_mIoU")
            metric_acc = getattr(self, f"train_{task_name}_acc")

            miou = metric_mIoU.compute()
            acc = metric_acc.compute()

            self.log(f"train_{task_name}_mIoU", miou,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"train_{task_name}_acc", acc,
                    on_epoch=True, logger=True, sync_dist=True)

            if task_name in self._train_tasks_seen:
                miou_values.append(miou)

            metric_mIoU.reset()
            metric_acc.reset()

        if miou_values:
            avg_miou = torch.stack(miou_values).mean()
            self.log("train_avg_mIoU", avg_miou,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        mse = self.train_recon_mse.compute()
        mae = self.train_recon_mae.compute()
        self.log("train_recon_mse", mse,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon_mae", mae,
                on_epoch=True, logger=True, sync_dist=True)
        self.train_recon_mse.reset()
        self.train_recon_mae.reset()

        self._train_tasks_seen.clear()

    def on_validation_epoch_end(self):
        # Always compute ALL metrics to keep DDP sync consistent across ranks
        miou_values = []
        for task_name in SEGMENTATION_TASKS:
            metric_mIoU = getattr(self, f"val_{task_name}_mIoU")
            metric_acc = getattr(self, f"val_{task_name}_acc")

            miou = metric_mIoU.compute()
            acc = metric_acc.compute()

            self.log(f"val_{task_name}_mIoU", miou,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_{task_name}_acc", acc,
                    on_epoch=True, logger=True, sync_dist=True)

            if task_name in self._val_tasks_seen:
                miou_values.append(miou)

            metric_mIoU.reset()
            metric_acc.reset()

        # Average mIoU — only over tasks that actually ran
        if miou_values:
            avg_miou = torch.stack(miou_values).mean()
            self.log("val_avg_mIoU", avg_miou,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        mse = self.val_recon_mse.compute()
        mae = self.val_recon_mae.compute()
        self.log("val_recon_mse", mse,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recon_mae", mae,
                on_epoch=True, logger=True, sync_dist=True)
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        """Save encoder + all heads."""
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_{self.name}{suffix}.pth"
        state = {
            "encoder": self.encoder.state_dict(),
            "heads": self.heads.state_dict(),
        }
        torch.save(state, file_path)
        print(f"[Pretrain] Model saved to {file_path}")

    def load_model(self, name=None, encoder_only=False):
        """
        Load model weights.

        Args:
            name: Optional suffix for the checkpoint filename.
            encoder_only: If True, only load encoder weights (for downstream
                          fine-tuning where heads will be replaced).
        """
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/pretrain_{self.name}{suffix}.pth"
        state = torch.load(file_path, weights_only=True)

        self.encoder.load_state_dict(state["encoder"])
        print(f"[Pretrain] Encoder loaded from {file_path}")

        if not encoder_only and "heads" in state:
            self.heads.load_state_dict(state["heads"])
            print(f"[Pretrain] Heads loaded from {file_path}")

    def load_encoder_for_downstream(self, checkpoint_path: str):
        """
        Load only the encoder from a pre-training checkpoint.
        Useful for PANGAEA evaluation where encoder is frozen.
        """
        state = torch.load(checkpoint_path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[Pretrain] Encoder loaded for downstream from {checkpoint_path}")