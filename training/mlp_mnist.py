
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

class Diagnostic_MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = float(config["trainer"]["lr"])
        self.num_classes = config["trainer"]["num_classes"]
        
        # Hardcoded for 64x64 based on your previous code
        # input_dim = 1 channel * 64 * 64
        self.input_dim = 64 * 64 
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

        self.class_loss_fn = nn.CrossEntropyLoss()
        
        # Metrics - Adding IoU to match your Model_MNIST keys
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")

    def forward(self, image, *args, **kwargs):
        # Taking the first channel/slice as per your requirement
        # image shape: [batch, C, H, W] -> Slicing logic:
        x = image[:, 0, :, :] if image.ndim == 4 else image[:, :, 0]
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        
        # Data Debugging Block
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"\n--- DATA DEBUG ---")
            print(f"Image Shape: {image.shape}")
            print(f"Image Range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Label Range: [{labels.min()}, {labels.max()}]")
            print(f"------------------\n")

        logits = self.forward(image)
        loss = self.class_loss_fn(logits, labels)
        
        self.train_acc.update(logits.detach().argmax(dim=-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, logger=True)
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        logits = self.forward(image)
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

    def configure_optimizers(self):
        # Using simple Adam for the diagnostic check
        return torch.optim.Adam(self.parameters(), lr=1e-3)