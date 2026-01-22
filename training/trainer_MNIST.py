"""
MNIST Classification Model (Pure Classification Mode)
This version removes the reconstruction objective to focus entirely on testing
whether self-attention and RPE improve global digit recognition.
"""

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

class Model_MNIST(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.name = name
        self.lookup_table = lookup_table
        
        # Configuration
        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        
        # Core Architecture: Encoder only
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser_error(config=self.config, lookup_table=self.lookup_table)

        # 1. IMMEDIATE FREEZING OF DECODER
        # Since this is MNIST Classification, we never want the decoder active
        self.encoder.freeze_decoder()

        # Pure Classification Loss
        self.class_loss_fn = nn.CrossEntropyLoss()

        self.encoder.input_processor.spectral_encoder.requires_grad_(False)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")

    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=False, task="classification"):
        return self.encoder(image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=training, task=task)

    def on_fit_start(self):
        """Double check freezing at the start of trainer.fit()"""
        self.encoder.freeze_decoder()
        # Ensure classifier is unfrozen
        self.encoder.unfreeze_classifier()
        self.encoder.unfreeze_encoder()

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        logits = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=True, task="classification")
        loss = self.class_loss_fn(logits, labels)
     
        self.train_acc.update(logits.argmax(dim=-1), labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, logger=True)
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        logits = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=False, task="classification")
        loss = self.class_loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        
        self.val_acc.update(preds, labels)
        self.metric_IoU_val.update(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True, logger=True)
        self.log('val_IoU', self.metric_IoU_val.compute(), prog_bar=True, logger=True)
        self.val_acc.reset()
        self.metric_IoU_val.reset()

    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))



    def configure_optimizers(self):
        # 2. FILTER PARAMETERS
        # Only pass parameters that require grad to the optimizer
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "cosine_warmup",
            },
        }