from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ScaleMae import*
from training.ResNet import *
from collections import defaultdict
from training import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import matplotlib.pyplot as plt
from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import torchmetrics
import warnings
import wandb
from transformers import get_cosine_schedule_with_warmup
import seaborn as sns
from pytorch_optimizer import Lamb

# Error supervision imports - UPDATED to v3
from training.atomiser.error_supervision import (
    compute_error_supervision,
)


class Model_FLAIR(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.name = name
        self.lookup_table = lookup_table
        
        # Classification Metrics
        self.metric_IoU_train = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro")
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")
        self.metric_IoU_test = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average=None)

        # Core Architecture
        # Using Atomiser_error to support the return of trajectories and predicted errors
        self.encoder = Atomiser_error(config=self.config, lookup_table=self.lookup_table)

        self.loss = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # Error Supervision Setup (v3)
        self.use_error_guided_displacement = config["Atomiser"].get("use_error_guided_displacement", False)
        self.use_gravity_displacement = config["Atomiser"].get("use_gravity_displacement", False)
        self.use_error_supervision = (self.use_error_guided_displacement or self.use_gravity_displacement)
        
        if self.use_error_supervision:
            self.lambda_error = config["Atomiser"].get("lambda_error", 0.1)
            self.error_grid_size = config["Atomiser"].get("error_grid_size", 7)
            self.error_grid_spacing = config["Atomiser"].get("error_grid_spacing", 2)
            self.error_channels_to_sample = config["Atomiser"].get("error_channels_to_sample", 1)
            self.error_loss_type = config["Atomiser"].get("error_loss_type", "mse")
            self.error_normalize = config["Atomiser"].get("error_normalize", True)
            self.error_warmup = config["Atomiser"].get("error_supervision_warmup_epochs", 0)
            self.stable_depth = config["Atomiser"].get("stable_depth", 0)

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

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        supervise_error = self._should_supervise_error()

        # Forward with potential trajectory/error prediction
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=True,
            task="reconstruction",
            return_trajectory=supervise_error,
            return_predicted_errors=supervise_error,
        )


        y_hat = result
        labels = mae_tokens[:, :, 4].long() # Class labels

        # Classification Loss
        y_hat_loss = rearrange(y_hat, "b t c -> (b t) c")
        labels_loss = rearrange(labels, "b p -> (b p)")
        class_loss = self.loss(y_hat_loss, labels_loss)

        # Error Supervision Loss
        total_loss = class_loss
        if supervise_error and result.get('predicted_errors') is not None:
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
            self.log('train_error_loss', error_loss, on_epoch=True)

        self.metric_IoU_train.update(torch.argmax(y_hat, dim=-1), labels)
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        # Validation always monitors full output
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=False, task="reconstruction",
            return_trajectory=True, return_predicted_errors=True
        )
        
        y_hat = result['predictions']
        
        labels = mae_tokens[:, :, 4].long()
        
        class_loss = self.loss(rearrange(y_hat, "b t c -> (b t) c"), labels.flatten())
        self.metric_IoU_val.update(torch.argmax(y_hat, dim=-1), labels)
        self.log('val_loss', class_loss, on_epoch=True, prog_bar=True)
        return class_loss

    def on_train_epoch_end(self):
        self.log("train_IoU", self.metric_IoU_train.compute(), on_epoch=True)
        self.metric_IoU_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_IoU", self.metric_IoU_val.compute(), on_epoch=True)
        self.metric_IoU_val.reset()

    # --- Persistence Methods ---
    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}