"""
Segmentation Trainer — Single-Task, Encode-Once
=====================================================

Trainer for cross-sensor transfer experiments.
Single segmentation task, plain CrossEntropyLoss, no loss weighting.

Per-task weighted loss supported for imbalanced datasets:
  - murat_segmentation: weight=[1.0, 19.0] (5% buildings, 95% background)

Per-task Dice loss supported for class-imbalanced segmentation:
  - c2seg_segmentation: CE + Dice (matching the original C2Seg paper)

Batch format:
    {
        "groups": {res: {"tokens": [B,N,8], "mask": [B,N], "shape": ...}},
        "tasks": {
            "<task_name>": {"queries": [B,M,8], "queries_mask": [B,M]},
        },
        "target_resolution": float,
    }

Architecture:
    1. encode(groups) → latents, coords           [ONCE]
    2. reconstruct(latents, coords, queries) → features
    3. head(features) → predictions
    4. CrossEntropyLoss(predictions, labels) [+ DiceLoss for some tasks]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.utils.datasets.token_grouping import compute_grid_config


# =============================================================================
# SOFT DICE LOSS
# =============================================================================

class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    Matches the original C2Seg paper's DiceLoss(smooth=1e-5, p=1).
    Computes per-class Dice and averages over present classes.
    """

    def __init__(self, ignore_index=255, smooth=1e-5, p=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.p = p

    def forward(self, logits, labels):
        valid = labels != self.ignore_index
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid]
        labels = labels[valid]

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes).float()

        intersection = (probs * one_hot).sum(dim=0)

        if self.p == 1:
            cardinality = probs.sum(dim=0) + one_hot.sum(dim=0)
        else:
            cardinality = (probs ** self.p).sum(dim=0) + (one_hot ** self.p).sum(dim=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        present = one_hot.sum(dim=0) > 0
        if present.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return 1.0 - dice_per_class[present].mean()


# =============================================================================
# TASK REGISTRY
# =============================================================================

SEGMENTATION_TASKS = (
    "dynamic_world",
    "esa_worldcover",
    "flairhub_cosia",
    "flairhub_lpis",
    "mdas_segmentation",
    "murat_segmentation",
    "c2seg_segmentation",
    "pastis_segmentation",
    "multiearth_deforestation",
)
RECONSTRUCTION_TASKS = ("reconstruction",)
ALL_TASKS = SEGMENTATION_TASKS + RECONSTRUCTION_TASKS

# Per-task class names for logging
TASK_CLASS_NAMES = {
    "c2seg_segmentation": {
        0: "Background", 1: "Urban Fabric", 2: "Industrial",
        3: "Street Network", 4: "Mine/Dump", 5: "Artif. Vegetated",
        6: "Arable Land", 7: "Low Vegetation", 8: "Forests",
        9: "Water",
    },
    "pastis_segmentation": {
        0: "Background", 1: "Meadow", 2: "Soft Winter Wheat",
        3: "Corn", 4: "Winter Barley", 5: "Winter Rapeseed",
        6: "Spring Barley", 7: "Sunflower", 8: "Grapevine",
        9: "Beet", 10: "Soy", 11: "Sorghum",
        12: "Flax", 13: "Protein Crops", 14: "Other Cereals",
        15: "Fruits/Veg", 16: "Other Crops", 17: "Grassland",
        18: "Shrub/Forest",
    },
    "mdas_segmentation": {
        0: "Pavement", 1: "Soil", 2: "Roof",
        3: "Low vegetation", 4: "Tree", 5: "Water",
    },
    "esa_worldcover": {
        0: "Tree cover", 1: "Shrubland", 2: "Grassland",
        3: "Cropland", 4: "Built-up", 5: "Bare/sparse",
        6: "Snow/ice", 7: "Water", 8: "Wetland",
        9: "Mangroves", 10: "Moss/lichen",
    },
    "multiearth_deforestation": {
        0: "Forest", 1: "Deforested",
    },
}


class Model_Pretrain(pl.LightningModule):
    """
    Single-task segmentation trainer.

    Shared encoder + task-specific head.
    Plain CrossEntropyLoss by default; per-task weighted CE for
    imbalanced datasets. Optional Dice loss for class-imbalanced tasks.
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
            "esa_worldcover":     {"num_classes": 11, "type": "segmentation"},
            "dynamic_world":      {"num_classes": 9,  "type": "segmentation"},
            "flairhub_cosia":     {"num_classes": 18, "type": "segmentation"},
            "flairhub_lpis":      {"num_classes": 23, "type": "segmentation"},
            "mdas_segmentation":  {"num_classes": 6,  "type": "segmentation"},
            "murat_segmentation": {"num_classes": 2,  "type": "segmentation"},
            "c2seg_segmentation": {"num_classes": 10, "type": "segmentation"},
            "pastis_segmentation": {"num_classes": 20, "type": "segmentation"},
            "multiearth_deforestation": {"num_classes": 2, "type": "segmentation"},
            "reconstruction":     {"num_classes": 1,  "type": "reconstruction"},
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
        # LOSS
        # =====================================================================
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.seg_loss_weighted = nn.ModuleDict({
            "murat_segmentation": nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 19.0]),
                ignore_index=self.IGNORE_INDEX,
            ),
        })

        self.dice_tasks = set()
        self.dice_loss = SoftDiceLoss(
            ignore_index=self.IGNORE_INDEX,
            smooth=1e-5,
            p=1,
        )

        # =====================================================================
        # METRICS (only for tasks that exist)
        # =====================================================================
        self._init_metrics()
        self._train_tasks_seen = set()
        self._val_tasks_seen = set()

        print(f"[Pretrain] Single-task trainer:")
        for task_name, task_cfg in self.task_configs.items():
            extras = []
            if task_name in self.seg_loss_weighted:
                extras.append("weighted CE")
            if task_name in self.dice_tasks:
                extras.append("CE + Dice")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"  {task_name}: {task_cfg['type']} "
                  f"({'classes=' + str(task_cfg['num_classes']) if task_cfg['type'] == 'segmentation' else 'MSE'}"
                  f"{extra_str})")

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
                setattr(self, f"{split}_{task_name}_mF1", torchmetrics.F1Score(
                    task="multiclass", num_classes=nc,
                    average="macro", ignore_index=self.IGNORE_INDEX,
                ))
                # Per-class IoU for detailed logging
                setattr(self, f"{split}_{task_name}_iou_per_class", torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=nc,
                    average="none", ignore_index=self.IGNORE_INDEX,
                ))

        for split in ["train", "val"]:
            setattr(self, f"{split}_recon_mse", torchmetrics.MeanSquaredError())
            setattr(self, f"{split}_recon_mae", torchmetrics.MeanAbsoluteError())

    # =========================================================================
    # ENCODE / DECODE
    # =========================================================================

    def _encode(self, batch, training=True):
        """Encode groups → latents + coords."""
        groups = batch["groups"]
        tpl = self.encoder.tokens_per_latent

        resolutions = sorted(groups.keys())
        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                total_tokens=groups[res]["tokens"].shape[1],
                tokens_per_latent=tpl,
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

        return encoder_output.latents_per_res, encoder_output.coords_per_res

    def _decode(self, latents_per_res, coords_per_res, queries, queries_mask,
                target_resolution=None, training=True):
        """Decode: latents + queries → pre-head features [B, M, D]."""
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
                training=training,
                return_features=True,
            )

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward_multitask(self, batch, training=True):
        """Encode once, decode per task."""
        latents_per_res, coords_per_res = self._encode(batch, training=training)

        target_resolution = batch.get("target_resolution", None)

        predictions = {}
        for task_name, task_data in batch["tasks"].items():
            if task_name not in self.heads:
                continue

            features = self._decode(
                latents_per_res, coords_per_res,
                task_data["queries"], task_data["queries_mask"],
                target_resolution=target_resolution,
                training=training,
            )

            predictions[task_name] = self.heads[task_name](features)

        return predictions

    def forward(self, batch, task_name=None, training=False):
        """Dispatch to forward_multitask if batch has 'tasks' key."""
        if "tasks" in batch:
            return self.forward_multitask(batch, training=training)

        if task_name is None:
            task_name = batch.get("task", "reconstruction")

        result = self.encoder(
            batch, training=training,
            task="reconstruction", return_features=True,
        )
        return self.heads[task_name](result["features"])

    # =========================================================================
    # LOSS
    # =========================================================================

    def _compute_seg_loss(self, predictions, queries, task_name=None):
        """Segmentation loss from query col 4."""
        labels = queries[:, :, 4].long()
        pred_flat = rearrange(predictions, "b m c -> (b m) c")
        label_flat = rearrange(labels, "b m -> (b m)")

        valid_count = (label_flat != self.seg_loss.ignore_index).sum()
        if valid_count == 0:
            return None, None, None

        if task_name and task_name in self.seg_loss_weighted:
            loss_fn = self.seg_loss_weighted[task_name]
            loss = loss_fn(pred_flat, label_flat)
        else:
            loss = self.seg_loss(pred_flat, label_flat)

        if task_name and task_name in self.dice_tasks:
            dice = self.dice_loss(pred_flat, label_flat)
            loss = loss + dice

        preds = torch.argmax(predictions, dim=-1)
        return loss, preds, labels

    def _compute_recon_loss(self, predictions, queries, queries_mask):
        """Reconstruction loss from query col 4, masked."""
        targets = queries[:, :, 4]
        preds = predictions.squeeze(-1)

        valid = ~queries_mask
        valid_count = valid.sum()
        if valid_count == 0:
            return None, None, None

        loss = nn.functional.mse_loss(preds[valid], targets[valid])
        return loss, preds, targets

    # =========================================================================
    # SHARED STEP
    # =========================================================================

    def _step(self, batch, split):
        """Shared logic for training and validation steps."""
        all_predictions = self.forward_multitask(
            batch, training=(split == "train"))

        if all_predictions:
            total_loss = sum(p.sum() * 0.0 for p in all_predictions.values())
        else:
            total_loss = sum(p.sum() * 0.0 for p in self.parameters())

        tasks_seen = (self._train_tasks_seen if split == "train"
                      else self._val_tasks_seen)

        for task_name, predictions in all_predictions.items():
            task_data = batch["tasks"][task_name]
            queries = task_data["queries"]
            queries_mask = task_data["queries_mask"]

            tasks_seen.add(task_name)

            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                predictions = torch.nan_to_num(
                    predictions, nan=0.0, posinf=0.0, neginf=0.0)

            if task_name in SEGMENTATION_TASKS:
                loss, preds, labels = self._compute_seg_loss(
                    predictions, queries, task_name=task_name)

                if loss is None or not torch.isfinite(loss):
                    continue

                # Update metrics
                getattr(self, f"{split}_{task_name}_mIoU").update(preds, labels)
                getattr(self, f"{split}_{task_name}_acc").update(preds, labels)
                getattr(self, f"{split}_{task_name}_mF1").update(preds, labels)
                getattr(self, f"{split}_{task_name}_iou_per_class").update(preds, labels)

                self.log(f"{split}_{task_name}_loss", loss,
                         on_step=(split == "train"), on_epoch=True,
                         prog_bar=True, logger=True)

                total_loss = total_loss + loss

            elif task_name in RECONSTRUCTION_TASKS:
                loss, preds, targets = self._compute_recon_loss(
                    predictions, queries, queries_mask)

                if loss is None or not torch.isfinite(loss):
                    continue

                getattr(self, f"{split}_recon_mse").update(preds, targets)
                getattr(self, f"{split}_recon_mae").update(preds, targets)

                self.log(f"{split}_recon_loss", loss,
                         on_step=(split == "train"), on_epoch=True,
                         prog_bar=True, logger=True)

                total_loss = total_loss + loss

        self.log(f"{split}_loss", total_loss,
                 on_step=(split == "train"), on_epoch=True,
                 prog_bar=True, logger=True)

        return total_loss

    # =========================================================================
    # TRAINING / VALIDATION STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "val")

    # =========================================================================
    # EPOCH END
    # =========================================================================

    def _on_epoch_end(self, split):
        tasks_seen = (self._train_tasks_seen if split == "train"
                      else self._val_tasks_seen)

        miou_values = []
        for task_name in SEGMENTATION_TASKS:
            if task_name not in tasks_seen:
                continue
            if not hasattr(self, f"{split}_{task_name}_mIoU"):
                continue

            metric_mIoU = getattr(self, f"{split}_{task_name}_mIoU")
            metric_acc = getattr(self, f"{split}_{task_name}_acc")
            metric_mF1 = getattr(self, f"{split}_{task_name}_mF1")
            metric_per_class = getattr(self, f"{split}_{task_name}_iou_per_class")

            miou = metric_mIoU.compute()
            acc = metric_acc.compute()
            mf1 = metric_mF1.compute()
            per_class = metric_per_class.compute()

            self.log(f"{split}_{task_name}_mIoU", miou,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{split}_{task_name}_acc", acc,
                     on_epoch=True, logger=True)
            self.log(f"{split}_{task_name}_mF1", mf1,
                     on_epoch=True, logger=True)

            # Per-class IoU with proper names
            class_names = TASK_CLASS_NAMES.get(task_name, {})
            nc = self.task_configs[task_name]["num_classes"]
            for cls_idx in range(min(nc, len(per_class))):
                cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
                self.log(f"{split}_{task_name}_IoU_{cls_name}", per_class[cls_idx],
                         on_epoch=True, logger=True)

            miou_values.append(miou)
            metric_mIoU.reset()
            metric_acc.reset()
            metric_mF1.reset()
            metric_per_class.reset()

        if miou_values:
            self.log(f"{split}_avg_mIoU", torch.stack(miou_values).mean(),
                     on_epoch=True, prog_bar=True, logger=True)

        if "reconstruction" in tasks_seen:
            recon_mse = getattr(self, f"{split}_recon_mse")
            recon_mae = getattr(self, f"{split}_recon_mae")
            self.log(f"{split}_recon_mse", recon_mse.compute(),
                     on_epoch=True, logger=True)
            self.log(f"{split}_recon_mae", recon_mae.compute(),
                     on_epoch=True, logger=True)
            recon_mse.reset()
            recon_mae.reset()

        tasks_seen.clear()

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

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