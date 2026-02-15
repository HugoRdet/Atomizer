"""
PASTIS Crop Classification Trainer
===================================

PyTorch Lightning module for training the Atomizer model on PASTIS dataset
for multi-temporal crop type classification.

Supports:
- Multi-resolution token-based input
- Sliding window inference for large images
- Multi-temporal satellite imagery
- 19 crop type classes
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


class PASTISTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for PASTIS crop classification.
    
    Expected batch format:
    ----------------------
    batch = {
        "groups": {
            resolution: {
                "tokens": [B, N, 8],  # [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]
            }
        },
        "queries": [B, M, 8],  # Query tokens in same format
        "sliding": bool,  # (optional) Whether this is a sliding window batch
        "crop_positions": [B, 2],  # (if sliding) Position of crop in full image
        "crop_size": tuple,  # (if sliding) Size of crop
        "full_size": tuple,  # (if sliding) Size of full image
    }
    
    Labels are extracted from column 4 of query tokens.
    
    Config structure:
    -----------------
    config = {
        "trainer": {
            "num_classes": 19,
            "ignore_index": -1,
            "slide": True,  # Enable sliding window inference
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        }
    }
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
        "other"
    ]
    
    def __init__(self, model, config):
        """
        Initialize the PASTIS trainer.
        
        Args:
            model: The Atomizer model instance
            config: Configuration dictionary
        """
        super().__init__()
        self.model = model
        self.config = config
        
        # Extract config parameters
        self.num_classes = config["trainer"]["num_classes"]
        self.ignore_index = config["trainer"].get("ignore_index", -1)
        self.learning_rate = config["trainer"].get("learning_rate", 1e-4)
        self.weight_decay = config["trainer"].get("weight_decay", 0.01)
        self.use_sliding = config["trainer"].get("slide", False)
        
        # Loss function
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        # Metrics for train/val/test
        self.metric_IoU_train = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        self.metric_acc_train = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        
        self.metric_IoU_val = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        self.metric_acc_val = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        
        self.metric_IoU_test = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        self.metric_acc_test = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None
        )
        
        # For sliding window inference
        self.sliding_predictions = []
        self.sliding_labels = []
        self.sliding_logits = []
    
    def forward(self, batch):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing "groups" and "queries"
            
        Returns:
            logits: [B, M, num_classes] classification logits
        """
        return self.model(batch)
    
    def _compute_loss_and_preds(self, batch, training=True):
        """
        Compute loss and predictions for a batch.
        
        Args:
            batch: Input batch dictionary
            training: Whether this is training mode
            
        Returns:
            total_loss: Combined loss
            class_loss: Classification loss
            preds: Predicted classes [N]
            labels: Ground truth labels [N]
        """
        # Forward pass
        logits = self(batch)  # [B, M, num_classes]
        
        # Extract labels from query tokens (column 4)
        labels = batch["queries"][:, :, 4].long()  # [B, M]
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.num_classes)  # [B*M, num_classes]
        labels_flat = labels.reshape(-1)  # [B*M]
        
        # Compute classification loss
        class_loss = self.loss(logits_flat, labels_flat)
        
        # Get predictions
        preds = torch.argmax(logits_flat, dim=-1)  # [B*M]
        
        # Filter out ignored indices for metrics
        valid_mask = labels_flat != self.ignore_index
        preds_valid = preds[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        total_loss = class_loss
        
        return total_loss, class_loss, preds_valid, labels_valid
    
    def _sliding_window_step(self, batch):
        """
        Process a sliding window batch by accumulating predictions.
        
        Args:
            batch: Batch with sliding=True, crop_positions, crop_size, full_size
            
        Returns:
            preds_full: Predictions for full image
            label_full: Labels for full image  
            logits_avg: Average logits for full image
        """
        # Forward pass on crop
        logits_crop = self(batch)  # [B, M, num_classes]
        
        # Extract labels and positions
        queries = batch["queries"]  # [B, M, 8]
        labels_crop = queries[:, :, 4].long()  # [B, M]
        
        # Extract spatial positions from queries (columns 1 and 2)
        x_positions = queries[:, :, 1]  # [B, M]
        y_positions = queries[:, :, 2]  # [B, M]
        
        # Get crop information
        crop_positions = batch["crop_positions"]  # [B, 2]
        full_size = batch["full_size"]  # (H, W)
        
        B = logits_crop.shape[0]
        
        # Initialize accumulators if first crop
        if len(self.sliding_logits) == 0:
            H, W = full_size
            self.sliding_logits = torch.zeros(
                (H, W, self.num_classes),
                device=logits_crop.device,
                dtype=logits_crop.dtype
            )
            self.sliding_predictions = torch.zeros(
                (H, W),
                device=logits_crop.device,
                dtype=torch.long
            )
            self.sliding_labels = torch.full(
                (H, W),
                self.ignore_index,
                device=labels_crop.device,
                dtype=torch.long
            )
        
        # Accumulate predictions for each batch element
        for b in range(B):
            crop_y, crop_x = crop_positions[b]
            
            # Get valid query mask (query_flag in column 5)
            query_mask = queries[b, :, 5] == 1
            
            if query_mask.sum() == 0:
                continue
            
            # Get positions and predictions for valid queries
            x_pos = x_positions[b, query_mask].long()
            y_pos = y_positions[b, query_mask].long()
            logits_queries = logits_crop[b, query_mask]  # [N, num_classes]
            labels_queries = labels_crop[b, query_mask]
            
            # Convert local positions to global positions
            x_global = x_pos + crop_x
            y_global = y_pos + crop_y
            
            # Accumulate logits
            self.sliding_logits[y_global, x_global] += logits_queries
            
            # Store labels (only once)
            self.sliding_labels[y_global, x_global] = labels_queries
        
        # Get final predictions from accumulated logits
        preds_full = torch.argmax(self.sliding_logits, dim=-1)
        label_full = self.sliding_labels
        logits_avg = self.sliding_logits.clone()
        
        return preds_full, label_full, logits_avg
    
    def _reset_sliding_accumulators(self):
        """Reset sliding window accumulators for next image."""
        self.sliding_predictions = []
        self.sliding_labels = []
        self.sliding_logits = []
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            loss: Training loss
        """
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=True)
        
        # Update metrics
        if len(preds) > 0:
            self.metric_IoU_train.update(preds, labels)
            self.metric_acc_train.update(preds, labels)
        
        # Log losses
        self.log("train_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with sliding window support.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            loss: Validation loss
        """
        # ── Sliding window path ─────────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            
            # Compute loss on full image
            loss = self.loss(
                logits_avg.unsqueeze(0).permute(0, 3, 1, 2),  # [1, C, H, W]
                label_full.unsqueeze(0),  # [1, H, W]
            )
            
            # Update metrics only on valid pixels
            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_val.update(preds_full[valid], label_full[valid])
                self.metric_acc_val.update(preds_full[valid], label_full[valid])
            
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        # ── Normal path ──────────────────────────────────────────
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        
        if len(preds) > 0:
            self.metric_IoU_val.update(preds, labels)
            self.metric_acc_val.update(preds, labels)
        
        self.log("val_loss", class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return class_loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step with sliding window support.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            loss: Test loss
        """
        # ── Sliding window path ─────────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)
            
            # Compute loss on full image
            loss = self.loss(
                logits_avg.unsqueeze(0).permute(0, 3, 1, 2),  # [1, C, H, W]
                label_full.unsqueeze(0),  # [1, H, W]
            )
            
            # Update metrics only on valid pixels
            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_test.update(preds_full[valid], label_full[valid])
                self.metric_acc_test.update(preds_full[valid], label_full[valid])
            
            self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
            return loss
        
        # ── Normal path ──────────────────────────────────────────
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        
        if len(preds) > 0:
            self.metric_IoU_test.update(preds, labels)
            self.metric_acc_test.update(preds, labels)
        
        self.log("test_loss", class_loss, on_step=False, on_epoch=True, logger=True)
        
        return class_loss
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        train_iou = self.metric_IoU_train.compute()
        train_acc = self.metric_acc_train.compute()
        
        self.log("train_mIoU", train_iou.mean(), on_epoch=True, logger=True)
        self.log("train_accuracy", train_acc.mean(), on_epoch=True, logger=True)
        
        # Log per-class metrics
        for i, name in enumerate(self.CROP_NAMES[:len(train_iou)]):
            if torch.isfinite(train_iou[i]):
                self.log(f"train_IoU_{name}", train_iou[i], on_epoch=True, logger=True)
            if i < len(train_acc) and torch.isfinite(train_acc[i]):
                self.log(f"train_acc_{name}", train_acc[i], on_epoch=True, logger=True)
        
        # Reset metrics
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        val_iou = self.metric_IoU_val.compute()
        val_acc = self.metric_acc_val.compute()
        
        self.log("val_mIoU", val_iou.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc.mean(), on_epoch=True, logger=True)
        
        # Log per-class metrics
        for i, name in enumerate(self.CROP_NAMES[:len(val_iou)]):
            if torch.isfinite(val_iou[i]):
                self.log(f"val_IoU_{name}", val_iou[i], on_epoch=True, logger=True)
            if i < len(val_acc) and torch.isfinite(val_acc[i]):
                self.log(f"val_acc_{name}", val_acc[i], on_epoch=True, logger=True)
        
        # Reset metrics and sliding accumulators
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()
        self._reset_sliding_accumulators()
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        test_iou = self.metric_IoU_test.compute()
        test_acc = self.metric_acc_test.compute()
        
        self.log("test_mIoU", test_iou.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc.mean(), on_epoch=True, logger=True)
        
        # Log per-class metrics for all crop types
        for i, name in enumerate(self.CROP_NAMES[:len(test_iou)]):
            if torch.isfinite(test_iou[i]):
                self.log(f"test_IoU_{name}", test_iou[i], on_epoch=True, logger=True)
            if i < len(test_acc) and torch.isfinite(test_acc[i]):
                self.log(f"test_acc_{name}", test_acc[i], on_epoch=True, logger=True)
        
        # Reset metrics and sliding accumulators
        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()
        self._reset_sliding_accumulators()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            optimizer: AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Optional: Add learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=self.trainer.max_epochs
        # )
        # return [optimizer], [scheduler]
        
        return optimizer