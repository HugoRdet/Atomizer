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
from torch.profiler import record_function
import gc

# Error supervision imports - UPDATED to v3
from training.atomiser.error_supervision import (
    compute_error_supervision,
)


# =============================================================================
# MEMORY PROFILING UTILITIES
# =============================================================================

def print_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[{label}]")
        print(f"    Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        print(f"    Peak Alloc: {max_allocated:.2f} GB | Peak Reserved: {max_reserved:.2f} GB")


def print_memory_short(label=""):
    """Print current GPU memory usage (short format)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Alloc: {allocated:.2f} GB | Peak: {peak:.2f} GB")


def get_tensor_memory_usage():
    """Get memory usage of all tensors on GPU."""
    if not torch.cuda.is_available():
        return {}
    
    tensor_sizes = defaultdict(int)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_bytes = obj.numel() * obj.element_size()
                tensor_sizes[str(obj.shape)] += size_bytes
        except:
            pass
    
    # Sort by size
    sorted_sizes = sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)
    return sorted_sizes[:10]  # Top 10


def print_top_tensors(label=""):
    """Print top GPU tensors by memory usage."""
    print(f"\n[{label}] Top GPU Tensors:")
    for shape, size_bytes in get_tensor_memory_usage():
        print(f"    {shape}: {size_bytes / 1024**2:.1f} MB")


# =============================================================================
# TRAINER CLASS
# =============================================================================

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
        
        # =====================================================================
        # MEMORY PROFILING CONFIG
        # =====================================================================
        self.memory_profiling = config.get("memory_profiling",False)
        self.profile_batches = config.get("profile_batches", 1)  # How many batches to profile
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser_error(config=self.config, lookup_table=self.lookup_table)

        self.loss = nn.MSELoss(reduction='mean')  
        self.lr = float(config["trainer"]["lr"])
        
        # =====================================================================
        # PREDICTOR-ONLY MODE
        # =====================================================================
        self.predictor_only = config["Atomiser"].get("predictor_only", False)
        
        if self.predictor_only:
            print(f"[Trainer] *** PREDICTOR-ONLY MODE ***")
            print(f"[Trainer]   Training: Only error predictor loss (no reconstruction)")
            print(f"[Trainer]   Validation: Full reconstruction for monitoring")
        
        # =====================================================================
        # DISPLACEMENT WITH ERROR SUPERVISION SETUP
        # =====================================================================
        self.use_error_guided_displacement = config["Atomiser"].get("use_error_guided_displacement", False)
        self.use_gravity_displacement = config["Atomiser"].get("use_gravity_displacement", False)
        
        self.use_error_supervision = (
            self.use_error_guided_displacement or self.use_gravity_displacement
        )
        
        self.stable_depth = config["Atomiser"].get("stable_depth", 0)
        
        if self.use_error_supervision:
            self.lambda_error = config["Atomiser"].get("lambda_error", 0.1)
            self.error_grid_size = config["Atomiser"].get("error_grid_size", 7)
            self.error_grid_spacing = config["Atomiser"].get("error_grid_spacing", 2)
            self.error_channels_to_sample = config["Atomiser"].get("error_channels_to_sample", 1)
            self.error_loss_type = config["Atomiser"].get("error_loss_type", "mse")
            self.error_normalize = config["Atomiser"].get("error_normalize", True)
            self.error_supervision_warmup_epochs = config["Atomiser"].get(
                "error_supervision_warmup_epochs", 0
            )
            
            displacement_type = "GRAVITY" if self.use_gravity_displacement else "ERROR-GUIDED"
            print(f"[Trainer] {displacement_type} displacement ENABLED")
            print(f"[Trainer]   lambda_error={self.lambda_error}")
            print(f"[Trainer]   error_grid_size={self.error_grid_size}")
            print(f"[Trainer]   error_channels_to_sample={self.error_channels_to_sample}")
            print(f"[Trainer]   error_loss_type={self.error_loss_type}")
            print(f"[Trainer]   warmup_epochs={self.error_supervision_warmup_epochs}")
            print(f"[Trainer]   stable_depth={self.stable_depth}")
            print(f"[Trainer]   === v3: Using INITIAL positions + FINAL latents ===")
        else:
            self.lambda_error = 0.0
            print(f"[Trainer] Error supervision DISABLED (no error-based displacement)")

    # =========================================================================
    # MEMORY PROFILING HOOKS
    # =========================================================================
    
    def _should_profile(self, batch_idx):
        """Check if we should profile this batch."""
        return self.memory_profiling and batch_idx < self.profile_batches
    
    def on_fit_start(self):
        """Called when training starts."""
        if self.memory_profiling:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
            print("\n" + "=" * 70)
            print("MEMORY PROFILING ENABLED")
            print("=" * 70)
            print_memory("on_fit_start (after reset)")
            
            # Print model parameter count
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"\nModel Parameters:")
            print(f"    Total: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB in FP32)")
            print(f"    Trainable: {trainable_params:,} ({trainable_params * 4 / 1024**3:.2f} GB in FP32)")
            print("=" * 70 + "\n")
    
    def on_train_start(self):
        """Called when training begins."""
        if self.memory_profiling:
            print_memory("on_train_start")
    
    def on_train_epoch_start(self):
        """Called at the start of each epoch."""
        if self.memory_profiling and self.current_epoch == 0:
            print_memory(f"on_train_epoch_start (epoch {self.current_epoch})")
    
    def on_train_batch_start(self, batch, batch_idx):
        """Called before each batch."""
        if self._should_profile(batch_idx):
            print(f"\n{'=' * 70}")
            print(f"BATCH {batch_idx} - START")
            print(f"{'=' * 70}")
            
            if batch_idx == 0:
                # Reset peak stats for accurate per-batch measurement
                torch.cuda.reset_peak_memory_stats()
            
            print_memory(f"on_train_batch_start (batch {batch_idx})")
    
    def on_after_backward(self):
        """Called after backward pass."""
        if self._should_profile(self.global_step):
            print_memory(f"on_after_backward (step {self.global_step})")
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step."""
        if self._should_profile(self.global_step):
            print_memory(f"on_before_optimizer_step (step {self.global_step})")
            
            # Check optimizer state memory
            self._check_optimizer_memory(optimizer)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after optimizer step and gradient zeroing."""
        if self._should_profile(batch_idx):
            print_memory(f"on_train_batch_end (batch {batch_idx})")
            
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
            
            print(f"\n{'=' * 70}")
            print(f"BATCH {batch_idx} - SUMMARY")
            print(f"    Peak Allocated: {peak_alloc:.2f} GB")
            print(f"    Peak Reserved:  {peak_reserved:.2f} GB")
            print(f"{'=' * 70}\n")
            
            # Clear cache after profiling batch
            torch.cuda.empty_cache()
            gc.collect()
    
    def _check_optimizer_memory(self, optimizer):
        """Check memory used by optimizer state."""
        total_bytes = 0
        num_states = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            total_bytes += val.numel() * val.element_size()
                            num_states += 1
        
        if total_bytes > 0:
            print(f"    Optimizer state: {total_bytes / 1024**3:.2f} GB ({num_states} tensors)")

    # =========================================================================
    # FORWARD
    # =========================================================================
        
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

    # =========================================================================
    # TRAINING STEP
    # =========================================================================

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        profiler = self._should_profile(batch_idx)
        
        if profiler:
            print_memory_short("A. training_step START")
        
        # Unpack batch
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        if profiler:
            print_memory_short("B. After unpack")
            print(f"    image: {image.shape}")
            print(f"    mae_tokens: {mae_tokens.shape}")

        # Check if we should use error supervision
        supervise_error = self._should_supervise_error()
        
        # =====================================================================
        # PREDICTOR-ONLY MODE
        # =====================================================================
        if self.predictor_only:
            return self._training_step_predictor_only(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, image_err,
                batch_idx, profiler
            )
        
        # =====================================================================
        # STANDARD MODE
        # =====================================================================
        if supervise_error:
            if profiler:
                print_memory_short("C. Before forward (with error supervision)")
            
            result = self.forward(
                image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                training=True,
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
            if profiler:
                print_memory_short("C. Before forward (no error supervision)")
            
            with record_function("training_step"):
                y_hat = self.forward(
                    image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
                    training=True
                )
            trajectory = None
            predicted_errors = None
        
        if profiler:
            print_memory_short("D. After forward")
            print(f"    y_hat: {y_hat.shape}")

        # =====================================================================
        # RECONSTRUCTION LOSS
        # =====================================================================
        target = mae_tokens[:, :, 0]
        
        if profiler:
            print_memory_short("E. After target extraction")
        
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        if profiler:
            print_memory_short("F. After rearrange")
            print(f"    target: {target.shape}")
            print(f"    y_hat_flat: {y_hat_flat.shape}")
        
        recon_loss = self.loss(y_hat_flat, target)
        
        if profiler:
            print_memory_short("G. After recon_loss")

        # =====================================================================
        # ERROR SUPERVISION LOSS
        # =====================================================================
        if supervise_error and predicted_errors is not None and len(predicted_errors) > 0:
            if profiler:
                print_memory_short("H. Before error supervision")
            
            error_loss, error_stats = compute_error_supervision(
                model=self.encoder,
                trajectory=trajectory,
                predicted_errors=predicted_errors,
                latents=latents,
                final_coords=final_coords,
                image_err=image_err,
                geometry=self.encoder.input_processor.geometry,
                grid_size=self.error_grid_size,
                spacing=self.error_grid_spacing,
                num_channels_to_sample=self.error_channels_to_sample,
                loss_type=self.error_loss_type,
                normalize=self.error_normalize,
            )
            
            total_loss = recon_loss + self.lambda_error * error_loss
            
            if profiler:
                print_memory_short("I. After error supervision")
            
            # Log error metrics
            self.log('train_error_loss', error_loss, on_step=False, on_epoch=True, 
                     prog_bar=False, logger=True, sync_dist=False)
            self.log('train_actual_error_mean', error_stats['actual_error_mean'], 
                     on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            
            if 'correlation' in error_stats:
                self.log('train_error_corr', error_stats['correlation'],
                         on_step=False, on_epoch=True, logger=True)
            if 'rank_correlation' in error_stats:
                self.log('train_error_rank_corr', error_stats['rank_correlation'],
                         on_step=False, on_epoch=True, logger=True)
            
            if 'movement_error_correlation' in error_stats and batch_idx % 100 == 0:
                self.log('train_movement_error_corr', error_stats['movement_error_correlation'],
                         on_step=False, on_epoch=True, logger=True)
        else:
            total_loss = recon_loss
            error_loss = torch.tensor(0.0)
        
        if profiler:
            print_memory_short("J. Before return (backward will happen next)")
        
        # Log losses
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=False)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, sync_dist=False)
        
        # Log displacement stats periodically
        if supervise_error and trajectory is not None and batch_idx % 100 == 0:
            self._log_displacement_stats(trajectory, prefix='train')
        
        # Clean up intermediate tensors
        if profiler:
            del y_hat_flat, target
            if supervise_error:
                del trajectory, predicted_errors, latents, final_coords
            gc.collect()
            print_memory_short("K. After cleanup")
        
        return total_loss
    
    # =========================================================================
    # PREDICTOR-ONLY TRAINING STEP
    # =========================================================================
    
    def _training_step_predictor_only(
        self, image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos, image_err,
        batch_idx, profiler=False
    ):
        """Training step for predictor-only mode."""
        
        if profiler:
            print_memory_short("P1. predictor_only: Before forward")
        
        result = self.forward(
            image, attention_mask, mae_tokens, mae_tokens_mask, latents_pos,
            training=True,
            task="encoder",
            return_trajectory=True,
            return_predicted_errors=True,
        )
        
        trajectory = result['trajectory']
        predicted_errors = result['predicted_errors']
        latents = result['latents']
        final_coords = result['final_coords']
        
        if profiler:
            print_memory_short("P2. predictor_only: After forward")
        
        error_loss, error_stats = compute_error_supervision(
            model=self.encoder,
            trajectory=trajectory,
            predicted_errors=predicted_errors,
            latents=latents,
            final_coords=final_coords,
            image_err=image_err,
            geometry=self.encoder.input_processor.geometry,
            grid_size=self.error_grid_size,
            spacing=self.error_grid_spacing,
            num_channels_to_sample=self.error_channels_to_sample,
            loss_type=self.error_loss_type,
            normalize=self.error_normalize,
        )
        
        if profiler:
            print_memory_short("P3. predictor_only: After error supervision")
        
        # Log metrics
        self.log('train_loss', error_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=False)
        self.log('train_error_loss', error_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, sync_dist=False)
        self.log('train_actual_error_mean', error_stats['actual_error_mean'], 
                 on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        
        if 'correlation' in error_stats:
            self.log('train_error_corr', error_stats['correlation'],
                     on_step=False, on_epoch=True, logger=True)
        if 'rank_correlation' in error_stats:
            self.log('train_error_rank_corr', error_stats['rank_correlation'],
                     on_step=False, on_epoch=True, logger=True)
        
        if 'movement_mean' in error_stats:
            self.log('train_movement_mean', error_stats['movement_mean'],
                     on_step=False, on_epoch=True, logger=True)
        if 'movement_error_correlation' in error_stats:
            self.log('train_movement_error_corr', error_stats['movement_error_correlation'],
                     on_step=False, on_epoch=True, logger=True)
        
        if trajectory is not None and batch_idx % 100 == 0:
            self._log_displacement_stats(trajectory, prefix='train')
        
        # Clean up
        if profiler:
            del trajectory, predicted_errors, latents, final_coords
            gc.collect()
            print_memory_short("P4. predictor_only: After cleanup")
        
        return error_loss

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _check_module_gradients(self, module, name):
        """Check gradients for a specific module."""
        if module is None:
            return
        
        for param_name, param in module.named_parameters():
            if param.grad is None:
                print(f"  {name}.{param_name}: NO GRADIENT")
            else:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                grad_has_nan = param.grad.isnan().any().item()
                grad_has_inf = param.grad.isinf().any().item()
                
                status = "✓" if not (grad_has_nan or grad_has_inf) else "⚠️"
                print(f"  {status} {name}.{param_name}: norm={grad_norm:.6f}, max={grad_max:.6f}")

    def _log_displacement_stats(self, trajectory, prefix='train'):
        """Log displacement statistics from trajectory."""
        if trajectory is None or len(trajectory) < 2:
            return
        
        total_disp = (trajectory[-1] - trajectory[0]).norm(dim=-1)
        self.log(f'{prefix}_total_disp_mean', total_disp.mean(), 
                 on_step=False, on_epoch=True, logger=True)
        self.log(f'{prefix}_total_disp_max', total_disp.max(), 
                 on_step=False, on_epoch=True, logger=True)
        
        num_displacement_layers = len(trajectory) - 1 - self.stable_depth
        for i in range(1, min(len(trajectory), num_displacement_layers + 1)):
            layer_disp = (trajectory[i] - trajectory[i-1]).norm(dim=-1)
            self.log(f'{prefix}_disp_layer_{i-1}_mean', layer_disp.mean(),
                     on_step=False, on_epoch=True, logger=True)
        
        if hasattr(self.encoder, 'error_displacement') and self.encoder.error_displacement is not None:
            scale = self.encoder.error_displacement.displacement_scale.item()
            self.log(f'{prefix}_displacement_scale', scale, 
                     on_step=False, on_epoch=True, logger=True)
    
    # =========================================================================
    # VALIDATION STEP
    # =========================================================================
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step - ALWAYS performs full reconstruction."""
        profiler = self._should_profile(batch_idx) and batch_idx == 0
        
        if profiler:
            print(f"\n{'=' * 70}")
            print("VALIDATION BATCH 0 - MEMORY PROFILE")
            print(f"{'=' * 70}")
            torch.cuda.reset_peak_memory_stats()
            print_memory_short("V1. validation_step START")
        
        image, attention_mask, mae_tokens, mae_tokens_mask, _, latents_pos, image_err = batch
        
        if profiler:
            print_memory_short("V2. After unpack")
            print(f"    image: {image.shape}")
            print(f"    mae_tokens: {mae_tokens.shape}")
        
        supervise_error = self._should_supervise_error()
        
        if supervise_error:
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
        
        if profiler:
            print_memory_short("V3. After forward")
            print(f"    y_hat: {y_hat.shape}")
        
        # Compute reconstruction loss
        target = mae_tokens[:, :, 0]
        target = rearrange(target, "b p -> (b p)")
        y_hat_flat = rearrange(y_hat.clone(), "b t c -> (b t) c").squeeze(-1)
        
        recon_loss = self.loss(y_hat_flat, target)
        
        if profiler:
            print_memory_short("V4. After recon_loss")
        
        # Error supervision
        if supervise_error and predicted_errors is not None and len(predicted_errors) > 0:
            error_loss, error_stats = compute_error_supervision(
                model=self.encoder,
                trajectory=trajectory,
                predicted_errors=predicted_errors,
                latents=latents,
                final_coords=final_coords,
                image_err=image_err,
                geometry=self.encoder.input_processor.geometry,
                grid_size=self.error_grid_size,
                spacing=self.error_grid_spacing,
                num_channels_to_sample=self.error_channels_to_sample,
                loss_type=self.error_loss_type,
                normalize=self.error_normalize,
            )
            
            if self.predictor_only:
                total_loss = error_loss
            else:
                total_loss = recon_loss + self.lambda_error * error_loss
            
            self.log('val_error_loss', error_loss, on_step=False, on_epoch=True, 
                     prog_bar=False, logger=True, sync_dist=False)
            self.log('val_actual_error_mean', error_stats['actual_error_mean'], 
                     on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            
            if 'correlation' in error_stats:
                self.log('val_error_corr', error_stats['correlation'],
                         on_step=False, on_epoch=True, logger=True)
            if 'rank_correlation' in error_stats:
                self.log('val_error_rank_corr', error_stats['rank_correlation'],
                         on_step=False, on_epoch=True, logger=True)
            
            if 'movement_mean' in error_stats:
                self.log('val_movement_mean', error_stats['movement_mean'],
                         on_step=False, on_epoch=True, logger=True)
            if 'movement_error_correlation' in error_stats:
                self.log('val_movement_error_corr', error_stats['movement_error_correlation'],
                         on_step=False, on_epoch=True, logger=True)
        else:
            total_loss = recon_loss
        
        if profiler:
            print_memory_short("V5. After error supervision")
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n    VALIDATION PEAK: {peak:.2f} GB")
            print(f"{'=' * 70}\n")
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=False)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, 
                 prog_bar=True if self.predictor_only else False, logger=True, sync_dist=False)
        
        if supervise_error and trajectory is not None and batch_idx == 0:
            self._log_displacement_stats(trajectory, prefix='val')
        
        return total_loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pass
        
    # =========================================================================
    # MODEL SAVE/LOAD
    # =========================================================================
    
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
        
    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def configure_optimizers(self):
        if self.memory_profiling:
            print_memory("BEFORE optimizer creation")
        
        base_lr = self.lr
        wd = self.weight_decay

        if self.predictor_only:
            params = self.parameters()
            print(f"[Trainer] Optimizer: Using all parameters (frozen ones have requires_grad=False)")
        else:
            params = self.parameters()

        if self.config["optimizer"] == "ADAM":
            optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=wd)
        else:
            import torch_optimizer as optim
            optimizer = optim.Lamb(params, lr=base_lr, weight_decay=wd,
                                betas=(0.9, 0.999), eps=1e-6)

        if self.memory_profiling:
            print_memory("AFTER optimizer creation")

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