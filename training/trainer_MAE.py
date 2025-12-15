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


def print_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")


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
        
    
        
        #self.metric_MSE_train = torchmetrics.MeanSquaredError(squared=False)
        #self.metric_MSE_val = torchmetrics.MeanSquaredError(squared=False)

        
        
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config,transform=self.transform)

        self.loss = nn.MSELoss(reduction='mean')  
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask,latents_pos, training=False, task="reconstruction"):
        return self.encoder(image, attention_mask, mae_tokens, mae_tokens_mask,latents_pos, training=training, task=task)

    

    

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        profiler=False
        if profiler and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()
            print_memory("A. Start of step")
        
        image, attention_mask, mae_tokens, mae_tokens_mask, _ , latents_pos = batch
        
        y_hat = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask,latents_pos, training=True)
        if profiler and batch_idx == 0:
            print_memory("B. After forward")
        
        #labels = mae_tokens[:,::5,4]
        target = mae_tokens[:,:,0]

        target=rearrange(target,"b p -> (b p)")
        y_hat =rearrange(y_hat.clone() ,"b t c -> (b t) c").squeeze(-1)

        loss = self.loss(y_hat, target)
        if profiler and batch_idx == 0:
            print_memory("C. After loss")
        
        #self.metric_MSE_val.update(y_hat.detach(), target.detach())
        
        # Log the loss directly here instead of manually tracking
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss
    
    #def on_after_backward(self):
       
        #print_memory("D. After backward")  # <-- Likely big jump here!
    
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
        
    def on_train_epoch_end(self):    
        pass
    
    def on_validation_epoch_start(self):
        pass
        
    
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        image, attention_mask, mae_tokens, mae_tokens_mask, _ , latents_pos = batch
        
        y_hat = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask,latents_pos,training=False)
        target = mae_tokens[:,:,0]

        target=rearrange(target,"b p -> (b p)")
        y_hat =rearrange(y_hat.clone() ,"b t c -> (b t) c").squeeze(-1)

        loss = self.loss(y_hat, target)
        
        #self.metric_MSE_val.update(y_hat.detach(), target.detach())
        
        # Log the loss directly here instead of manually tracking
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss

    def on_validation_epoch_end(self):
        pass

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
        
        
                
        
        
