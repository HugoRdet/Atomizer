"""
MMEarth MAE Trainer for Atomizer
=================================

Reconstruction trainer (unmasked for now).

Loss: MSE between predicted scalar and query reflectance (col 4).
Metrics: MSE, MAE.

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]
    col 4 = reflectance copy (reconstruction target)

Batch format:
    groups[res]["tokens"]: [B, N, 8]
    queries:               [B, M, 8]
    queries_mask:          [B, M]
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from training.atomiser import Atomiser_Senflood


class Model_MMEarth(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)
        self.loss = nn.MSELoss()

        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # =====================================================================
        # METRICS
        # =====================================================================
        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False):
        return self.encoder(batch, training=training)

    # =========================================================================
    # SHARED STEP
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        result = self.forward(batch, training=training)

        if isinstance(result, dict):
            y_hat = result["predictions"]
        else:
            y_hat = result

        # [B, M, 1] → [B, M]
        y_hat = y_hat.squeeze(-1)

        # Target: reflectance stored in col 4
        targets = batch["queries"][:, :, 4]

        loss = self.loss(y_hat, targets)

        return loss, y_hat, targets

    # =========================================================================
    # TRAINING / VALIDATION / TEST
    # =========================================================================

    def training_step(self, batch, batch_idx):


        loss, preds, targets = self._compute_loss_and_preds(batch, training=True)

        # Debug: check predictions vs targets periodically
     
        self.train_mse.update(preds, targets)
        self.train_mae.update(preds, targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):


        loss, preds, targets = self._compute_loss_and_preds(batch, training=False)

        # Debug: check predictions vs targets periodically
     
        self.val_mse.update(preds, targets)
        self.val_mae.update(preds, targets)



        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
        return loss

    # =========================================================================
    # EPOCH END
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_mse.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", self.train_mae.compute(), on_epoch=True, logger=True)
        self.train_mse.reset()
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", self.val_mae.compute(), on_epoch=True, logger=True)
        self.val_mse.reset()
        self.val_mae.reset()

    def on_test_epoch_end(self):
        self.log("test_mse", self.test_mse.compute(), on_epoch=True, logger=True)
        self.log("test_mae", self.test_mae.compute(), on_epoch=True, logger=True)
        self.test_mse.reset()
        self.test_mae.reset()

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
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[MMEarth] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[MMEarth] Model loaded from {file_path}")