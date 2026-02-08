"""
Sen1Floods11 Trainer for Atomizer
=================================

Semantic segmentation trainer for flood detection.
- 2 classes: No Flood (0), Flood (1)
- Ignore index: 255
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.atomiser.error_supervision import compute_error_supervision


class Model_SenFlood(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table
        
        # Sen1Floods11: 2 classes + ignore index 255
        self.num_classes = 2
        self.ignore_index = 255
        
        # =====================================================================
        # METRICS
        # =====================================================================
        # mIoU (Jaccard Index) - ignoring class 255
        self.metric_IoU_train = torchmetrics.JaccardIndex(
            task="multiclass", 
            num_classes=self.num_classes, 
            average="macro",
            ignore_index=self.ignore_index
        )
        self.metric_IoU_val = torchmetrics.JaccardIndex(
            task="multiclass", 
            num_classes=self.num_classes, 
            average="macro",
            ignore_index=self.ignore_index
        )
        self.metric_IoU_test = torchmetrics.JaccardIndex(
            task="multiclass", 
            num_classes=self.num_classes, 
            average=None,  # Per-class IoU for test
            ignore_index=self.ignore_index
        )
        
        # Accuracy - ignoring class 255
        self.metric_acc_train = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
            ignore_index=self.ignore_index
        )
        self.metric_acc_val = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
            ignore_index=self.ignore_index
        )
        self.metric_acc_test = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average=None,  # Per-class accuracy for test
            ignore_index=self.ignore_index
        )
        
        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)
        
        # Loss with ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        
        # =====================================================================
        # ERROR SUPERVISION (optional)
        # =====================================================================
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
            

    # =========================================================================
    # FORWARD
    # =========================================================================
    
    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=False, task="reconstruction", return_trajectory=False,
                return_predicted_errors=False):
        return self.encoder(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=training, task=task,
            return_trajectory=return_trajectory,
            return_predicted_errors=return_predicted_errors,
        )

    def _should_supervise_error(self):
        return self.use_error_supervision and self.current_epoch >= self.error_warmup

    # =========================================================================
    # TRAINING STEP
    # =========================================================================
    
    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        supervise_error = self._should_supervise_error()
        
        # Forward pass
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=True,
            task="reconstruction",
            return_trajectory=supervise_error,
            return_predicted_errors=supervise_error,
        )
        
        # Handle output format
        if isinstance(result, dict):
            y_hat = result['predictions']
        else:
            y_hat = result
        
        # Labels are in column 4 of mae_tokens
        labels = mae_tokens[:, :, 4].long()  # [B, N]
        
        # Flatten for loss computation
        # y_hat: [B, N, num_classes] -> [B*N, num_classes]
        # labels: [B, N] -> [B*N]
        y_hat_flat = rearrange(y_hat, "b t c -> (b t) c")
        labels_flat = rearrange(labels, "b n -> (b n)")
        
        # CrossEntropyLoss with ignore_index=255 automatically ignores masked tokens
        class_loss = self.loss(y_hat_flat, labels_flat)
        
        # Error supervision (optional)
        total_loss = class_loss
        if supervise_error and isinstance(result, dict) and result.get('predicted_errors') is not None:
            error_loss, error_stats = compute_error_supervision(
                model=self.encoder,
                trajectory=result['trajectory'],
                predicted_errors=result['predicted_errors'],
                latents=result['latents'],
                final_coords=result['final_coords'],
                image_err=image_err,
                geometry=self.encoder.input_processor.geometry,
                grid_size=self.error_grid_size,
                spacing=self.error_grid_spacing,
                num_channels_to_sample=self.error_channels_to_sample,
                loss_type=self.error_loss_type,
                normalize=self.error_normalize,
            )
            total_loss = class_loss + (self.lambda_error * error_loss)
            self.log('train_error_loss', error_loss, on_step=False, on_epoch=True, logger=True)
        
        # Update metrics (they also ignore index 255)
        preds = torch.argmax(y_hat, dim=-1)  # [B, N]
        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)
        
        # Logging
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_class_loss', class_loss, on_step=False, on_epoch=True, logger=True)
        
        return total_loss

    # =========================================================================
    # VALIDATION STEP
    # =========================================================================
    
    def validation_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        # Forward pass (always get full output for monitoring)
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=False,
            task="reconstruction",
            return_trajectory=True,
            return_predicted_errors=True,
        )
        
        # Handle output format
        if isinstance(result, dict):
            y_hat = result['predictions']
        else:
            y_hat = result
        
        # Labels
        labels = mae_tokens[:, :, 4].long()
        
        # Flatten for loss
        y_hat_flat = rearrange(y_hat, "b t c -> (b t) c")
        labels_flat = rearrange(labels, "b n -> (b n)")
        
        # Loss (ignores masked tokens)
        class_loss = self.loss(y_hat_flat, labels_flat)
        
        # Update metrics
        preds = torch.argmax(y_hat, dim=-1)
        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)
        
        # Logging
        self.log('val_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return class_loss

    # =========================================================================
    # TEST STEP
    # =========================================================================
    
    def test_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        # Forward pass
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=False,
            task="reconstruction",
        )
        
        if isinstance(result, dict):
            y_hat = result['predictions']
        else:
            y_hat = result
        
        labels = mae_tokens[:, :, 4].long()
        
        # Loss
        y_hat_flat = rearrange(y_hat, "b t c -> (b t) c")
        labels_flat = rearrange(labels, "b n -> (b n)")
        class_loss = self.loss(y_hat_flat, labels_flat)
        
        # Update metrics
        preds = torch.argmax(y_hat, dim=-1)
        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)
        
        self.log('test_loss', class_loss, on_step=False, on_epoch=True, logger=True)
        
        return class_loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================
    
    def on_train_epoch_end(self):
        # Compute and log metrics
        train_iou = self.metric_IoU_train.compute()
        train_acc = self.metric_acc_train.compute()
        
        self.log("train_mIoU", train_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", train_acc, on_epoch=True, prog_bar=True, logger=True)
        
        # Reset metrics
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        # Compute and log metrics
        val_iou = self.metric_IoU_val.compute()
        val_acc = self.metric_acc_val.compute()
        
        self.log("val_mIoU", val_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        
        # Reset metrics
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        # Compute per-class metrics for detailed analysis
        test_iou_per_class = self.metric_IoU_test.compute()
        test_acc_per_class = self.metric_acc_test.compute()
        
        # Log overall (mean)
        self.log("test_mIoU", test_iou_per_class.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc_per_class.mean(), on_epoch=True, logger=True)
        
        # Log per-class
        class_names = ["no_flood", "flood"]
        for i, name in enumerate(class_names):
            self.log(f"test_IoU_{name}", test_iou_per_class[i], on_epoch=True, logger=True)
            self.log(f"test_acc_{name}", test_acc_per_class[i], on_epoch=True, logger=True)
        

        
        # Reset metrics
        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()

    # =========================================================================
    # MODEL SAVE/LOAD
    # =========================================================================
    
    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[SenFlood] Model saved to {file_path}")
        
    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[SenFlood] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
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