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

#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model_MAE(pl.LightningModule):
    def __init__(self, config, wand, name,transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform=transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        self.labels_idx = load_json_to_dict("./data/Encoded-BigEarthNet/labels.json")
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table=False
        self.comment_log=""
        
        # Add manual loss tracking to avoid NaN/inf issues
        self.train_losses = []
        self.val_losses = []
        
        self.metric_MSE_train = torchmetrics.MeanSquaredError(squared=False)
        self.metric_MSE_val_mod_val = torchmetrics.MeanSquaredError(squared=False)
        self.metric_MSE_val_mod_train = torchmetrics.MeanSquaredError(squared=False)
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config,transform=self.transform)

        self.loss = nn.MSELoss(reduction='mean')  # Explicitly set reduction
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, image, attention_mask,mae_tokens,mae_tokens_mask, training=False,task="reconstruction"):
        return self.encoder(image, attention_mask,mae_tokens,mae_tokens_mask, training=training,task=task)

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch
        
        y_hat, y_mask = self.forward(image.clone(), attention_mask.clone(), mae_tokens.clone(), mae_tokens_mask.clone(), training=True)
        
        y_hat_masked = y_hat.clone()
        mae_tokens_masked = mae_tokens.clone()
        
        # Apply masking
        y_hat_masked[y_mask == 1.0] = 0.0
        mae_tokens_masked[y_mask == 1.0] = 0.0
        
        # Compute loss
        loss = self.loss(y_hat_masked[:, :, 0], mae_tokens_masked[:, :, 0])
        
        # Update metrics
        self.metric_MSE_train.update(y_hat_masked[:, :, 0].detach(), mae_tokens_masked[:,:,0].detach())
        
        # Store loss for epoch-end logging
        self.train_losses.append(loss.detach().cpu().item())
        
        # IMPORTANT: Return the loss so PyTorch Lightning can call backward()
        return loss   
    
    def on_fit_start(self):
        # if starting with MAE
        self.encoder.unfreeze_encoder()
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
    
    def on_train_epoch_start(self):
        self.encoder.unfreeze_encoder()    
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
        self.train_losses = []  # Reset for new epoch
        
    def on_train_epoch_end(self):
                
        # Calculate average training loss manually
        if len(self.train_losses) > 0:
            avg_train_loss = np.mean(self.train_losses)
            # Check for NaN/inf
            if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
                avg_train_loss = 0.0
        else:
            avg_train_loss = 0.0
            
        self.log("train_reconstruction_loss", avg_train_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        #self.check_gradients()
        # Compute MSE metric
        try:
            train_mse = self.metric_MSE_train.compute()
            if torch.isnan(train_mse) or torch.isinf(train_mse):
                train_mse = torch.tensor(0.0)
        except:
            train_mse = torch.tensor(0.0)
            
        self.log("train_MSE", train_mse, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_train.reset()
        
        return {"train_reconstruction_loss": avg_train_loss}
    
    def on_validation_epoch_start(self):

        self.trainer.datamodule.val_dataset.set_modality_mode("validation")
        self.val_losses = []  # Reset for new epoch
        
    
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch

        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        # Create copies to avoid in-place operations
        y_hat_masked = y_hat.clone()
        mae_tokens_masked = mae_tokens.clone()
        
        # Apply masking
        y_hat_masked[y_mask == 1.0] = 0.0
        mae_tokens_masked[mae_tokens_mask == 1.0] = 0.0

        # Only compute loss on non-masked tokens to avoid NaN
        valid_mask = (mae_tokens_mask == 0.0)
        
        if valid_mask.sum() == 0:
            # If no valid tokens, return a small loss to avoid NaN
            loss = torch.tensor(0.0, device=self.device)
        else:
            # Only compute loss on valid (non-masked) tokens
            y_hat_valid = y_hat_masked[:, :, 0][valid_mask]
            mae_tokens_valid = mae_tokens_masked[:, :, 0][valid_mask]
            
            loss = self.loss(y_hat_valid, mae_tokens_valid)
        
        # Check for NaN/inf and handle gracefully
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected in validation step {batch_idx}")
            loss = torch.tensor(0.0, device=self.device)
        
        # Update metrics only with valid data
        if valid_mask.sum() > 0:
            if dataloader_idx == 0:  # Validation set
                self.metric_MSE_val_mod_val.update(y_hat_valid.detach(), mae_tokens_valid.detach())
            elif dataloader_idx == 1:  # Training set
                self.metric_MSE_val_mod_train.update(y_hat_valid.detach(), mae_tokens_valid.detach())

        # Store loss for epoch-end logging
        self.val_losses.append(loss.detach().cpu().item())
        
        # Log step-wise (optional, can remove if too verbose)
        self.log("val_reconstruction_loss_step", loss, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        return loss    

    def on_validation_epoch_end(self):
        

        # Compute MSE metric
        val_mse_val = self.metric_MSE_val_mod_val.compute()
        self.log("val mod val MSE", val_mse_val, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_reconstruction_loss", val_mse_val, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_val_mod_val.reset()

        val_mse_train = self.metric_MSE_val_mod_train.compute()
        self.log("val mod train MSE", val_mse_train, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_val_mod_train.reset()

        return {"val_reconstruction_loss": val_mse_val}

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
            optimizer = optim.Lamb(self.parameters(), lr=base_lr, weight_decay=wd,
                                betas=(0.9, 0.999), eps=1e-6)

        # total optimizer steps for the entire fit (already accounts for grad accumulation & epochs)
        total_steps = int(self.trainer.estimated_stepping_batches)
        
        # pick a % warmup or your fixed value, but keep it <= total_steps
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
                "interval": "step",   # per-step schedule
                # no 'monitor' here
            },
        }
        
        
                
        
        
