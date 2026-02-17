"""
PASTIS Crop Classification Trainer
===================================

PyTorch Lightning module for training the Atomizer model on PASTIS dataset
for multi-temporal crop type classification.

Adapted from Model_SenFlood to share the same interface and conventions.

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]

Batch format (normal):
    groups[res]["tokens"]: [B, N, 8]
    queries:               [B, M, 8]
    label:                 [B, H, W]
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.atomiser.error_supervision import compute_error_supervision


class PASTISTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for PASTIS crop classification.

    Same interface as Model_SenFlood:
        PASTISTrainer(config, wand, name, transform, lookup_table)
    """

    # PASTIS crop type names (19 classes)
    CROP_NAMES = [
        "background",
        "wheat",
        "corn",
        "barley",
        "rapeseed",
        "sunflower",
        "orchards",
        "nuts",
        "permanent_meadows",
        "temporary_meadows",
        "soybean",
        "hard_wheat",
        "protein_crops",
        "rice",
        "potatoes_sugarbeet",
        "hops_hemp",
        "vineyards",
        "fruits_vegetables",
        "other",
    ]

    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table

        self.num_classes = config["trainer"]["num_classes"]
        self.ignore_index = 255

        # Sliding window inference for val/test
        self.use_sliding = config["trainer"].get("slide", False)

        # =================================================================
        # METRICS  (per-class for 19 crop types)
        # =================================================================
        self.metric_IoU_train = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_IoU_val = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_IoU_test = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index,
        )
        self.metric_acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index,
        )
        self.metric_acc_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index,
        )

        # =================================================================
        # MODEL
        # =================================================================
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # =================================================================
        # ERROR SUPERVISION (optional)
        # =================================================================
        self.use_error_guided_displacement = config["Atomiser"].get("use_error_guided_displacement", False)
        self.use_gravity_displacement = config["Atomiser"].get("use_gravity_displacement", False)
        self.use_error_supervision = (
            self.use_error_guided_displacement or self.use_gravity_displacement
        )

        if self.use_error_supervision:
            self.lambda_error = config["Atomiser"].get("lambda_error", 0.1)
            self.error_grid_size = config["Atomiser"].get("error_grid_size", 7)
            self.error_grid_spacing = config["Atomiser"].get("error_grid_spacing", 2)
            self.error_channels_to_sample = config["Atomiser"].get("error_channels_to_sample", 1)
            self.error_loss_type = config["Atomiser"].get("error_loss_type", "mse")
            self.error_normalize = config["Atomiser"].get("error_normalize", True)
            self.error_warmup = config["Atomiser"].get("error_supervision_warmup_epochs", 0)
            self.stable_depth = config["Atomiser"].get("stable_depth", 0)

    # =====================================================================
    # FORWARD
    # =====================================================================

    def forward(self, batch, training=False, return_trajectory=False,
                return_predicted_errors=False):
        """Forward pass — delegates entirely to encoder."""
        return self.encoder(
            batch,
            training=training,
            return_trajectory=return_trajectory,
            return_predicted_errors=return_predicted_errors,
        )

    def _should_supervise_error(self):
        return self.use_error_supervision and self.current_epoch >= self.error_warmup

    # =====================================================================
    # SHARED STEP LOGIC
    # =====================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        """
        Run forward, compute loss, return (loss, class_loss, preds, labels).
        """
        supervise_error = self._should_supervise_error() and training

        result = self.forward(
            batch,
            training=training,
            return_trajectory=supervise_error or (not training),
            return_predicted_errors=supervise_error or (not training),
        )

        if isinstance(result, dict):
            y_hat = result["predictions"]
        else:
            y_hat = result

        # Labels from column 4 of query tokens: [B, M, 8] -> [B, M]
        labels = batch["queries"][:, :, 4].long()

        # Flatten: [B, M, C] -> [B*M, C],  [B, M] -> [B*M]
        y_hat_flat = rearrange(y_hat, "b t c -> (b t) c")
        labels_flat = rearrange(labels, "b n -> (b n)")

        class_loss = self.loss(y_hat_flat, labels_flat)

        # Error supervision (training only)
        total_loss = class_loss
        if supervise_error and isinstance(result, dict) and result.get("predicted_errors") is not None:
            error_loss, _ = compute_error_supervision(
                model=self.encoder,
                trajectory=result["trajectory"],
                predicted_errors=result["predicted_errors"],
                latents=result["latents"],
                final_coords=result["final_coords"],
                image_err=batch["image"],
                geometry=self.encoder.input_processor.geometry,
                grid_size=self.error_grid_size,
                spacing=self.error_grid_spacing,
                num_channels_to_sample=self.error_channels_to_sample,
                loss_type=self.error_loss_type,
                normalize=self.error_normalize,
            )
            total_loss = class_loss + (self.lambda_error * error_loss)
            self.log("train_error_loss", error_loss, on_step=False, on_epoch=True, logger=True)

        preds = torch.argmax(y_hat, dim=-1)  # [B, M]

        return total_loss, class_loss, preds, labels

    # =====================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =====================================================================

    def training_step(self, batch, batch_idx):
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=True
        )

        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", class_loss, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False
        )

        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)

        self.log("val_loss", class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return class_loss

    def test_step(self, batch, batch_idx):
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=False
        )

        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)

        self.log("test_loss", class_loss, on_step=False, on_epoch=True, logger=True)

        return class_loss

    # =====================================================================
    # EPOCH END HOOKS
    # =====================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU", self.metric_IoU_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.metric_acc_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_mIoU", self.metric_IoU_val.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self.metric_acc_val.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou = self.metric_IoU_test.compute()
        test_acc = self.metric_acc_test.compute()

        self.log("test_mIoU", test_iou.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc.mean(), on_epoch=True, logger=True)

        # Log per-class metrics for all 19 crop types
        for i, name in enumerate(self.CROP_NAMES):
            if i < len(test_iou):
                self.log(f"test_IoU_{name}", test_iou[i], on_epoch=True, logger=True)
            if i < len(test_acc):
                self.log(f"test_acc_{name}", test_acc[i], on_epoch=True, logger=True)

        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()

    # =====================================================================
    # MODEL SAVE/LOAD
    # =====================================================================

    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[PASTIS] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[PASTIS] Model loaded from {file_path}")

    # =====================================================================
    # OPTIMIZER
    # =====================================================================

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