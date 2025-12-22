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

# Suppress benign warnings
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")


def print_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")


class Model_MNIST(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
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
        self.lookup_table = lookup_table
        
        # Loss weights
        self.recon_weight = config["trainer"].get("recon_weight", 1.0)
        self.class_weight = config["trainer"].get("class_weight", 0.5)
        self.use_classification = config["trainer"].get("use_classification", True)
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config, lookup_table=self.lookup_table)

        self.recon_loss_fn = nn.MSELoss(reduction='mean')
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        
        # Metrics for classification
        if self.use_classification:
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        
    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=False, task="reconstruction"):
        return self.encoder(image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, training=training, task=task)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        profiler = False
        if profiler and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()
            print_memory("A. Start of step")
        
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        
        # Get encoder output (latents + final coords)
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, 
            training=True, task="encoder"
        )
        
        latents = result['latents']
        final_coords = result['final_coords']
        
        if profiler and batch_idx == 0:
            print_memory("B. After encoder")
        
        # =====================================================================
        # Reconstruction Loss
        # =====================================================================
        y_hat = self.encoder.reconstruct(latents, final_coords, mae_tokens, mae_tokens_mask)
        
        target = mae_tokens[:, :, 0]  # First column is reflectance
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        recon_loss = self.recon_loss_fn(y_hat_flat, target)
        
        if profiler and batch_idx == 0:
            print_memory("C. After reconstruction")
        
        # =====================================================================
        # Classification Loss
        # =====================================================================
        if self.use_classification:
            logits = self.encoder.classify(latents)
            class_loss = self.class_loss_fn(logits, labels)
            
            # Update accuracy metric
            preds = logits.argmax(dim=-1)
            self.train_acc.update(preds, labels)
            
            if profiler and batch_idx == 0:
                print_memory("D. After classification")
        else:
            class_loss = torch.tensor(0.0, device=recon_loss.device)
        
        # =====================================================================
        # Combined Loss
        # =====================================================================
        loss = self.recon_weight * recon_loss + self.class_weight * class_loss
        
        # Diagnostic: Check what model is predicting (every 100 batches)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                pred_mean = y_hat_flat.mean().item()
                pred_std = y_hat_flat.std().item()
                pred_min = y_hat_flat.min().item()
                pred_max = y_hat_flat.max().item()
                
                tgt_mean = target.mean().item()
                tgt_white = (target > 0.1).float().mean().item()  # % white pixels
                
                print(f"[Batch {batch_idx}] Pred: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}] | Target: mean={tgt_mean:.3f}, %white={tgt_white:.1%}")
        
        # Logging (on_step=True shows in tqdm progress bar)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('recon', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if self.use_classification:
            self.log('cls', class_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.use_classification:
            self.log('train_acc', self.train_acc.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.train_acc.reset()
    
    def on_fit_start(self):
        # if starting with MAE
        return
        self.encoder.unfreeze_encoder()
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
    
    def on_train_epoch_start(self):
        return
        self.encoder.unfreeze_encoder()    
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
        self.train_losses = []  # Reset for new epoch
        
    def on_validation_epoch_start(self):
        pass
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels, latents_pos = batch
        
        # Get encoder output
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=False, task="encoder"
        )
        
        latents = result['latents']
        final_coords = result['final_coords']
        
        # Reconstruction
        y_hat = self.encoder.reconstruct(latents, final_coords, mae_tokens, mae_tokens_mask)
        
        target = mae_tokens[:, :, 0]
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        recon_loss = self.recon_loss_fn(y_hat_flat, target)
        
        # Classification
        if self.use_classification:
            logits = self.encoder.classify(latents)
            class_loss = self.class_loss_fn(logits, labels)
            
            # Update accuracy metric
            preds = logits.argmax(dim=-1)
            self.val_acc.update(preds, labels)
        else:
            class_loss = torch.tensor(0.0, device=recon_loss.device)
        
        # Combined loss
        loss = self.recon_weight * recon_loss + self.class_weight * class_loss
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recon', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if self.use_classification:
            self.log('val_cls', class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        if self.use_classification:
            self.log('val_acc', self.val_acc.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        pass
        
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
        

    def configure_optimizers(self):
        base_lr = self.lr
        wd = self.weight_decay
        
        if self.config["optimizer"] == "ADAM":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=wd)
        else:
            import torch_optimizer as optim
            optimizer = optim.Lamb(
                self.parameters(), 
                lr=base_lr, 
                weight_decay=wd,
                betas=(0.9, 0.999), 
                eps=1e-6
            )
        
        # Total optimizer steps for the entire fit (already accounts for grad accumulation & epochs)
        total_steps = int(self.trainer.estimated_stepping_batches)
        
        # Pick a % warmup or your fixed value, but keep it <= total_steps
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
                "frequency": 1,
                "strict": True,
                "name": "cosine_warmup",
            },
        }