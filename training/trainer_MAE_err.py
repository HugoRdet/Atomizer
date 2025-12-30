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

# Error supervision imports
from training.atomiser.error_supervision import (
    ErrorSupervisionModule,
    compute_error_predictor_loss,
)


def print_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")


#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model_MAE_err(pl.LightningModule):
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
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser_error(config=self.config, lookup_table=self.lookup_table)

        self.loss = nn.MSELoss(reduction='mean')  
        self.lr = float(config["trainer"]["lr"])
        
        # =====================================================================
        # DISPLACEMENT WITH ERROR SUPERVISION SETUP
        # =====================================================================
        self.use_error_guided_displacement = config["Atomiser"].get(
            "use_error_guided_displacement", False
        )
        self.use_gravity_displacement = config["Atomiser"].get(
            "use_gravity_displacement", False
        )
        
        # Either error-guided or gravity displacement uses error supervision
        self.use_error_supervision = (
            self.use_error_guided_displacement or self.use_gravity_displacement
        )
        
        # Get stable_depth from config (for logging purposes)
        self.stable_depth = config["Atomiser"].get("stable_depth", 0)
        
        if self.use_error_supervision:
            # Error supervision module for computing actual errors
            self.error_supervision = ErrorSupervisionModule(
                geometry=self.encoder.input_processor.geometry,
                grid_size=config["Atomiser"].get("error_grid_size", 3),
                spacing=config["Atomiser"].get("error_grid_spacing", 2),
                image_size=512,
                gsd=0.2,
                num_channels=5,  # B, G, R, NIR, Elevation
            )
            
            # Weight for error predictor loss
            self.lambda_error = config["Atomiser"].get("lambda_error", 0.1)
            
            # Optional: warmup epochs before enabling error supervision
            self.error_supervision_warmup_epochs = config["Atomiser"].get(
                "error_supervision_warmup_epochs", 0
            )
            
            displacement_type = "GRAVITY" if self.use_gravity_displacement else "ERROR-GUIDED"
            print(f"[Trainer] {displacement_type} displacement ENABLED")
            print(f"[Trainer]   lambda_error={self.lambda_error}")
            print(f"[Trainer]   warmup_epochs={self.error_supervision_warmup_epochs}")
            print(f"[Trainer]   stable_depth={self.stable_depth} (no displacement in last {self.stable_depth} layers)")
        else:
            self.error_supervision = None
            self.lambda_error = 0.0
            print(f"[Trainer] Error supervision DISABLED (no error-based displacement)")
        
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
        """Check if we should compute error supervision this epoch."""
        if not self.use_error_supervision:
            return False
        return self.current_epoch >= self.error_supervision_warmup_epochs

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        profiler = False
        if profiler and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()
            print_memory("A. Start of step")
        
        # Unpack batch - now includes image_err
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        # Check if we should use error supervision
        supervise_error = self._should_supervise_error()
        
        if supervise_error:
            # Forward with trajectory and predicted errors
            result = self.forward(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=True,
                task="reconstruction",
                return_trajectory=True,
                return_predicted_errors=True,
            )
            
            # Unpack result
            y_hat = result['predictions']
            trajectory = result['trajectory']
            predicted_errors = result['predicted_errors']
            latents = result['latents']
            final_coords = result['final_coords']
            
        else:
            # Standard forward pass
            y_hat = self.forward(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=True
            )
            trajectory = None
            predicted_errors = None
        
        if profiler and batch_idx == 0:
            print_memory("B. After forward")
        
        # Compute reconstruction loss
        target = mae_tokens[:, :, 0]
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        recon_loss = self.loss(y_hat_flat, target)
        
        if profiler and batch_idx == 0:
            print_memory("C. After recon loss")
        
        # Compute error supervision loss if enabled
        if supervise_error and predicted_errors is not None and len(predicted_errors) > 0:
            # Compute actual errors at trajectory positions
            # Pass stable_depth to only compute errors for displacement layers
            actual_errors = self.error_supervision.compute_actual_error(
                trajectory=trajectory,
                latents=latents,
                final_coords=final_coords,
                image_err=image_err,
                model=self.encoder,
                stable_depth=self.encoder.stable_depth,  # Only compute for displacement layers
            )  # [B, num_displacement_layers, L]
            
            # Compute error predictor supervision loss
            error_loss = compute_error_predictor_loss(predicted_errors, actual_errors)
            
            # Total loss
            total_loss = recon_loss + self.lambda_error * error_loss
            
            # Log error-related metrics
            self.log('train_error_loss', error_loss, on_step=False, on_epoch=True, 
                     prog_bar=False, logger=True, sync_dist=False)
            self.log('train_actual_error_mean', actual_errors.mean(), on_step=False, 
                     on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            
            # Log per-layer correlation between predicted and actual errors
            if batch_idx % 100 == 0:  # Log less frequently to reduce overhead
                for layer_idx, pred_err in enumerate(predicted_errors):
                    actual_err = actual_errors[:, layer_idx, :]
                    if pred_err.numel() > 1:
                        corr = torch.corrcoef(
                            torch.stack([pred_err.flatten(), actual_err.flatten()])
                        )[0, 1]
                        if not torch.isnan(corr):
                            self.log(f'train_error_corr_layer_{layer_idx}', corr,
                                     on_step=False, on_epoch=True, logger=True)
            
            if profiler and batch_idx == 0:
                print_memory("D. After error supervision")
        else:
            total_loss = recon_loss
            error_loss = torch.tensor(0.0)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=False)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, sync_dist=False)
        
        # Log displacement statistics periodically
        if supervise_error and trajectory is not None and batch_idx % 100 == 0:
            self._log_displacement_stats(trajectory, prefix='train')
        
        return total_loss
    
    def _log_displacement_stats(self, trajectory, prefix='train'):
        """Log displacement statistics from trajectory."""
        if trajectory is None or len(trajectory) < 2:
            return
        
        # Total displacement (initial to final)
        total_disp = (trajectory[-1] - trajectory[0]).norm(dim=-1)
        self.log(f'{prefix}_total_disp_mean', total_disp.mean(), 
                 on_step=False, on_epoch=True, logger=True)
        self.log(f'{prefix}_total_disp_max', total_disp.max(), 
                 on_step=False, on_epoch=True, logger=True)
        
        # Per-layer displacement (only for non-stable layers)
        num_displacement_layers = len(trajectory) - 1 - self.stable_depth
        for i in range(1, min(len(trajectory), num_displacement_layers + 1)):
            layer_disp = (trajectory[i] - trajectory[i-1]).norm(dim=-1)
            self.log(f'{prefix}_disp_layer_{i-1}_mean', layer_disp.mean(),
                     on_step=False, on_epoch=True, logger=True)
        
        # Log displacement scale (learnable parameter)
        if hasattr(self.encoder, 'error_displacement') and self.encoder.error_displacement is not None:
            scale = self.encoder.error_displacement.displacement_scale.item()
            self.log(f'{prefix}_displacement_scale', scale, 
                     on_step=False, on_epoch=True, logger=True)
    
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
        # Unpack batch - now includes image_err
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        # Check if we should use error supervision
        supervise_error = self._should_supervise_error()
        
        if supervise_error:
            # Forward with trajectory and predicted errors
            result = self.forward(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=False,
                task="reconstruction",
                return_trajectory=True,
                return_predicted_errors=True,
            )
            
            y_hat = result['predictions']
            trajectory = result['trajectory']
            predicted_errors = result['predicted_errors']
            latents = result['latents']
            final_coords = result['final_coords']
        else:
            y_hat = self.forward(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=False
            )
            trajectory = None
            predicted_errors = None
        
        # Compute reconstruction loss
        target = mae_tokens[:, :, 0]
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        recon_loss = self.loss(y_hat_flat, target)
        
        # Compute error supervision loss if enabled
        if supervise_error and predicted_errors is not None and len(predicted_errors) > 0:
            # Pass stable_depth to only compute errors for displacement layers
            actual_errors = self.error_supervision.compute_actual_error(
                trajectory=trajectory,
                latents=latents,
                final_coords=final_coords,
                image_err=image_err,
                model=self.encoder,
                stable_depth=self.encoder.stable_depth,  # Only compute for displacement layers
            )
            
            error_loss = compute_error_predictor_loss(predicted_errors, actual_errors)
            total_loss = recon_loss + self.lambda_error * error_loss
            
            self.log('val_error_loss', error_loss, on_step=False, on_epoch=True, 
                     prog_bar=False, logger=True, sync_dist=False)
            self.log('val_actual_error_mean', actual_errors.mean(), on_step=False, 
                     on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        else:
            total_loss = recon_loss
        
        # Log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=False)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, sync_dist=False)
        
        # Log displacement stats on first batch
        if supervise_error and trajectory is not None and batch_idx == 0:
            self._log_displacement_stats(trajectory, prefix='val')
        
        return total_loss

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