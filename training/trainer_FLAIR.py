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

import torch_optimizer as optim
class Model_FLAIR(pl.LightningModule):
    def __init__(self, config, wand, name, transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        self.labels_idx = load_json_to_dict("./data/Encoded-BigEarthNet/labels.json")
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table = False
        self.comment_log = ""
        
        # Metrics
        self.metric_IoU_train = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro")
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")
        self.metric_IoU_test = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average=None)
        
        # Model
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config, transform=self.transform)

        self.loss = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask, training=False, task="reconstruction"):
        return self.encoder(image, attention_mask, mae_tokens, mae_tokens_mask, training=training, task=task)

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch

        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        labels = mae_tokens[:,::5,4]
        
        labels_loss=rearrange(labels,"b p -> (b p)")
        y_hat_loss =rearrange(y_hat.clone() ,"b t c -> (b t) c")
        
        
        
        loss = self.loss(y_hat_loss, labels_loss.long())
        
        preds = torch.argmax(y_hat.clone(), dim=-1)
        self.metric_IoU_train.update(preds, labels)
        
        # Log the loss directly here instead of manually tracking
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss
    
    def on_fit_start(self):
        # Model setup
        self.encoder.unfreeze_encoder()
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
    
    def on_train_epoch_start(self):
        self.encoder.unfreeze_encoder()    
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
        
    def on_train_epoch_end(self):
        # Compute and log IoU
        train_iou = self.metric_IoU_train.compute()
        self.log("train_IoU", train_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_train.reset()
    
    def on_validation_epoch_start(self):
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch
        
        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        labels = mae_tokens[:,::5,4]
        
        
        
        labels_loss=rearrange(labels,"b p -> (b p)")
        y_hat_loss =rearrange(y_hat.clone() ,"b t c -> (b t) c")
        
        
        
        loss = self.loss(y_hat_loss, labels_loss.long())
        
        preds = torch.argmax(y_hat.clone(), dim=-1)
        self.metric_IoU_val.update(preds, labels)
        
        # Log the loss directly here instead of manually tracking
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss

    def on_validation_epoch_end(self):
        # Compute and log IoU
        val_iou = self.metric_IoU_val.compute()
        self.log("val_IoU", val_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_val.reset()
        
        # Reset dataset mode
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")

    def test_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch

        # Forward pass
        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        # Get labels and predictions - keep original format
        labels = mae_tokens[:, ::5, 4]
        preds = torch.argmax(y_hat.clone(), dim=-1)
        y_hat = y_hat.squeeze(-1)
        
        # CRITICAL: Clamp labels to valid range [0, num_classes-1]
        labels = torch.clamp(labels, 0, self.num_classes - 1).long()
        
        # Calculate loss
        loss = self.loss(y_hat, labels)
        
        # Update metrics
        self.metric_IoU_test.update(preds, labels)
        
        # Log loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_test_epoch_end(self):
        # Compute per-class IoU
        test_iou_per_class = self.metric_IoU_test.compute()
        
        # Log mean IoU
        mean_iou = test_iou_per_class.mean()
        self.log("test_IoU_mean", mean_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Log per-class IoU if needed
        for i, iou in enumerate(test_iou_per_class):
            self.log(f"test_IoU_class_{i}", iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_test.reset()
        
    def save_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))

    def debug_data_ranges(self, dataloader):
        """
        Debug method to check label ranges in your dataset.
        Call this before training to identify issues.
        """
        print("=== DEBUGGING DATA RANGES ===")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Check first 5 batches
                break
                
            image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch
            labels = mae_tokens[:, :, 4]
            
            print(f"Batch {batch_idx}:")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Labels min: {labels.min().item()}")
            print(f"  Labels max: {labels.max().item()}")
            print(f"  Unique labels: {torch.unique(labels).cpu().tolist()}")
            print(f"  Expected range: [0, {self.num_classes-1}]")
            
            # Check for problematic values
            invalid_labels = (labels < 0) | (labels >= self.num_classes)
            if invalid_labels.any():
                print(f"  ⚠️  FOUND {invalid_labels.sum().item()} INVALID LABELS!")
                invalid_values = labels[invalid_labels].unique()
                print(f"  Invalid values: {invalid_values.cpu().tolist()}")
            else:
                print("  ✅ All labels are in valid range")
            print()
        
        print("=== END DEBUG ===")

    def configure_optimizers(self):
        base_lr = self.lr
        wd = self.weight_decay

        if self.config["optimizer"]["name"] == "ADAM":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=wd)
        else:
            import torch_optimizer as optim
            optimizer = optim.Lamb(self.parameters(), lr=base_lr, weight_decay=wd,
                                betas=(0.9, 0.999), eps=1e-6)

        # total optimizer steps for the entire fit (already accounts for grad accumulation & epochs)
        total_steps = int(self.trainer.estimated_stepping_batches)
        
        # pick a % warmup or your fixed value, but keep it <= total_steps
        warmup_steps = min(self.config["optimizer"]["warmup_steps"], max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # per-step schedule
                # no 'monitor' here
            },
        }
