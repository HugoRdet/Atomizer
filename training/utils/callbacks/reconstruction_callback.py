
import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import seaborn as sns
from typing import Optional, Union, List
from training.utils.image_utils import *

class CustomMAEReconstructionCallback(pl.Callback):

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: Configuration dict containing visualization parameters
        """
        super().__init__()
        self.config=config
        
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
        # Sentinel-2 band definitions (12 bands)
        self.sentinel2_bands = {
            'B01': 0,   # Coastal aerosol (443 nm)
            'B02': 1,   # Blue (490 nm)
            'B03': 2,   # Green (560 nm)
            'B04': 3,   # Red (665 nm)
            'B05': 4,   # Vegetation Red Edge (705 nm)
            'B06': 5,   # Vegetation Red Edge (740 nm)
            'B07': 6,   # Vegetation Red Edge (783 nm)
            'B08': 7,   # NIR (842 nm)
            'B8A': 8,   # Vegetation Red Edge (865 nm)
            'B09': 9,   # Water vapour (945 nm)
            'B11': 10,  # SWIR (1610 nm)
            'B12': 11,  # SWIR (2190 nm)
        }
        
        # Band combinations for visualization
        self.rgb_bands = [self.sentinel2_bands['B04'], self.sentinel2_bands['B03'], self.sentinel2_bands['B02']]  # Red, Green, Blue
        self.infrared_bands = [self.sentinel2_bands['B08'], self.sentinel2_bands['B11'], self.sentinel2_bands['B12']]  # NIR, SWIR1, SWIR2
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform reconstruction visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # **CRITICAL: Ensure we only run on GPU 0 (rank 0) to avoid multiple executions**
        if trainer.is_global_zero == False:
            return
            
        # Alternative way to check for single GPU execution
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            # Only execute on rank 0
            if wandb.run is not None:
                self._perform_custom_reconstruction(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping reconstruction logging")
        except Exception as e:
            print(f"Error in custom reconstruction callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_custom_reconstruction(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform reconstruction using your custom data loading."""
        
        # Get the dataset from the trainer
        dataset = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(sample_idx, i, dataset, pl_module, trainer.current_epoch)
            
    def _process_single_sample(self, dataset_idx: int, viz_idx: int, dataset, pl_module, epoch: int):
        """Process a single sample using your get_samples_to_viz function."""
        
        # Get data from your custom method
        image_tokens, attention_mask, mae_tokens, mask_MAE_res, latents_pos = dataset.get_samples_to_viz(dataset_idx)
        #image_to_return, image, attention_mask, queries, queries_mask, label, latent_pos
        print(latents_pos.shape)
        # Move to device
        mae_tokens_mask = torch.ones(mae_tokens.shape[0])
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mask_MAE_res = mask_MAE_res.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions to match batch format
                image_tokens_batch = image_tokens.unsqueeze(0) #image, , , ,
                image_tokens_mask = attention_mask.unsqueeze(0) #attention_mask
                mae_tokens_batch = mae_tokens.clone().unsqueeze(0) #mae_tokens
                mae_tokens_mask_batch = mae_tokens_mask.unsqueeze(0) #mae_tokens_mask
                #latents_pos
                # Perform reconstruction
                y_hat, y_mask = pl_module.forward(
                    image_tokens_batch,
                    image_tokens_mask,
                    mae_tokens_batch,
                    mae_tokens_mask_batch,
                    latents_pos.unsqueeze(0).to(device),
                    training=False,
                    task="visualization"
                )
                
                # Remove batch dimension for visualization
                y_hat = y_hat.squeeze(0)
                
                # Reshape to spatial format
                ground_truth = mae_tokens[:,0]
                ground_truth = rearrange(ground_truth, "(b h w) -> b h w", b=12, h=120, w=120)
                
                y_hat = rearrange(y_hat, "(b h w) c -> b h w c", b=12, h=120, w=120, c=1).squeeze(-1)
                
                print(f"✓ Successfully reconstructed sample {dataset_idx}")
                
            except Exception as e:
                print(f"Error in model forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_spatial_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                ground_truth=ground_truth,      # [12, 120, 120]
                prediction=y_hat,               # [12, 120, 120]
                spatial_mask=mask_MAE_res,      # [120, 120]
                epoch=epoch
            )
        except Exception as e:
            print(f"Error creating spatial visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_and_upload_spatial_visualization(self, sample_idx, viz_idx, ground_truth, prediction, spatial_mask, epoch):
        """Create and upload spatial reconstruction visualization to wandb."""
        
        # Convert to numpy for visualization
        gt_np = ground_truth.cpu().numpy()          # [12, 120, 120]
        pred_np = prediction.cpu().numpy()          # [12, 120, 120]
        mask_np = spatial_mask.cpu().numpy()        # [120, 120]
        
        # Calculate basic metrics for logging
        mse = np.mean((gt_np - pred_np) ** 2)
        mae = np.mean(np.abs(gt_np - pred_np))
        correlation = np.corrcoef(gt_np.flatten(), pred_np.flatten())[0, 1]
        
        # Create the main spatial visualization
        spatial_fig = self._create_spatial_reconstruction_figure(
            gt_np, pred_np, mask_np, sample_idx, epoch, mse, mae, correlation
        )
        
        # Prepare wandb data
        wandb_data = {
            # Main spatial visualization
            f"spatial_reconstruction/epoch_{epoch}_sample_{sample_idx}": wandb.Image(
                spatial_fig, 
                caption=f"Spatial MAE Reconstruction - Epoch {epoch}, Sample {sample_idx}, MSE: {mse:.6f}"
            ),
            
            # Basic metrics
            f"reconstruction_metrics/epoch": epoch,
            f"reconstruction_metrics/sample_{viz_idx}_mse": mse,
            f"reconstruction_metrics/sample_{viz_idx}_mae": mae,
            f"reconstruction_metrics/sample_{viz_idx}_correlation": correlation,
        }
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Successfully uploaded spatial reconstruction for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
        
        plt.close(spatial_fig)
    
    def _create_spatial_reconstruction_figure(self, ground_truth, prediction, spatial_mask, 
                                            sample_idx, epoch, mse, mae, correlation):
        """Create clean spatial reconstruction visualization with RGB and Infrared bands."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MAE Reconstruction - Sample {sample_idx}, Epoch {epoch}\n'
                    f'MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}', 
                    fontsize=16, fontweight='bold')
        
        # Row 1: RGB bands (Red, Green, Blue)
        rgb_gt = self._create_rgb_composite(ground_truth, self.rgb_bands)
        rgb_pred = self._create_rgb_composite(prediction, self.rgb_bands)
        
        # Ground Truth RGB
        axes[0, 0].imshow(rgb_gt)
        axes[0, 0].set_title('Ground Truth\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction RGB
        axes[0, 1].imshow(rgb_pred)
        axes[0, 1].set_title('Prediction\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Spatial Mask
        axes[0, 2].imshow(spatial_mask, cmap='RdYlBu_r', alpha=0.9)
        axes[0, 2].set_title(f'Spatial Mask\n(Masked: {np.mean(spatial_mask):.1%})', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Infrared bands (NIR, SWIR1, SWIR2)
        infrared_gt = self._create_rgb_composite(ground_truth, self.infrared_bands)
        infrared_pred = self._create_rgb_composite(prediction, self.infrared_bands)
        
        # Ground Truth Infrared
        axes[1, 0].imshow(infrared_gt)
        axes[1, 0].set_title('Ground Truth\n(False Color: B08-B11-B12)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Prediction Infrared
        axes[1, 1].imshow(infrared_pred)
        axes[1, 1].set_title('Prediction\n(False Color: B08-B11-B12)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Error Map (using RGB bands)
        rgb_error = np.abs(rgb_gt.astype(float) - rgb_pred.astype(float))
        rgb_error = rgb_error / rgb_error.max() if rgb_error.max() > 0 else rgb_error
        axes[1, 2].imshow(rgb_error)
        axes[1, 2].set_title('RGB Error Map\n(Absolute Difference)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_composite(self, data, band_indices):
        """Create RGB composite from selected bands."""
        # data: [12, 120, 120], band_indices: [R_idx, G_idx, B_idx]
        composite = np.stack([
            data[band_indices[0]],  # Red channel
            data[band_indices[1]],  # Green channel  
            data[band_indices[2]]   # Blue channel
        ], axis=-1)  # [120, 120, 3]
        
        # Normalize using the provided function
        composite_tensor = torch.from_numpy(composite).permute(2, 0, 1)  # [3, 120, 120]
        normalized = normalize(composite_tensor)  # [3, 120, 120]
        
        return normalized.permute(1, 2, 0).numpy()  # [120, 120, 3]
    
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange
import wandb


def normalize(img, min_val=0, max_val=1):
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 1e-6:
        return (img - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
    return img




class MAE_CustomVisualizationCallback(pl.Callback):

    def __init__(self, config):
        """
        Args:
            config: Configuration dict containing visualization parameters
        """
        super().__init__()
        self.config = config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
        # Band definitions for RGB visualization
        self.rgb_bands = [2, 1, 0]  # Red, Green, Blue indices (assuming BGR order)
        
        # Colors for trajectory visualization (layer by layer)
        self.trajectory_colors = [
            "#000000",  # Layer 0 (initial) - black
            "#3a86ff",  # Layer 1 - blue
            "#8338ec",  # Layer 2 - purple
            "#ff006e",  # Layer 3 - pink
            "#fb5607",  # Layer 4 - orange
            "#ffbe0b",  # Layer 5 - yellow (extra)
            "#06d6a0",  # Layer 6 - teal (extra)
            "#118ab2",  # Layer 7 - dark blue (extra)
        ]
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform MAE reconstruction visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # **CRITICAL: Ensure we only run on GPU 0 (rank 0) to avoid multiple executions**
        if trainer.is_global_zero == False:
            return
            
        # Alternative way to check for single GPU execution
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            # Only execute on rank 0
            if wandb.run is not None:
                self._perform_custom_reconstruction(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping MAE reconstruction logging")
        except Exception as e:
            print(f"Error in MAE visualization callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_custom_reconstruction(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform MAE reconstruction visualization using custom data loading."""
        
        # Get the dataset from the trainer
        dataset_val = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(
                sample_idx, i, dataset_val, pl_module, 
                trainer.current_epoch, id="validation"
            )
            
        dataset_train = trainer.datamodule.train_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(
                sample_idx, i, dataset_train, pl_module, 
                trainer.current_epoch, id="train"
            )
    
    def _process_single_sample(self, dataset_idx: int, viz_idx: int, dataset, 
                               pl_module, epoch: int, id="val"):
        """Process a single sample for MAE reconstruction visualization."""
        
        # Get data from your custom method
        image, image_tokens, attention_mask, mae_tokens, mask_MAE_res, _, latent_pos = dataset.get_samples_to_viz(dataset_idx)
        
        # Move to device
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mae_tokens_mask = torch.ones(mae_tokens.shape[0]).to(device)
        latent_pos = latent_pos.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions to match batch format
                image_tokens_batch = image_tokens.unsqueeze(0) 
                image_tokens_mask = attention_mask.unsqueeze(0)
                mae_tokens_batch = mae_tokens.clone().unsqueeze(0)
                mae_tokens_mask_batch = mae_tokens_mask.unsqueeze(0)
                latent_pos_batch = latent_pos.unsqueeze(0)
                
                # Forward pass with trajectory (single call)
                result = pl_module.forward(
                    image_tokens_batch,
                    image_tokens_mask,
                    mae_tokens_batch,
                    mae_tokens_mask_batch,
                    latent_pos_batch,
                    training=False,
                    task="visualization",
                )
                
                # Unpack result
                if isinstance(result, tuple):
                    y_hat, trajectory = result
                else:
                    y_hat = result
                    trajectory = None
                
                # Remove batch dimension for visualization
                y_hat = y_hat.squeeze(0)  # [num_tokens, num_channels]
                
                # Get original values (target)
                target = mae_tokens[:, 0]  # [num_tokens] - reflectance values
                
                # Get reconstruction
                reconstruction = y_hat.squeeze(-1) if y_hat.dim() > 1 else y_hat  # [num_tokens]
                
                print(f"✓ Successfully reconstructed sample {dataset_idx}")
                print(f"  Target shape: {target.shape}, Reconstruction shape: {reconstruction.shape}")
                if trajectory is not None:
                    print(f"  Trajectory: {len(trajectory)} layers, shape {trajectory[0].shape}")
                
            except Exception as e:
                print(f"Error in model forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_mae_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                original_image=image,
                target=target,
                reconstruction=reconstruction,
                mask=mask_MAE_res,
                trajectory=trajectory,
                epoch=epoch,
                id=id
            )
        except Exception as e:
            print(f"Error creating MAE visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_and_upload_mae_visualization(self, sample_idx, viz_idx, original_image,
                                            target, reconstruction, mask, trajectory, epoch, id):
        """Create and upload MAE reconstruction visualization to wandb."""
        
        # Calculate reconstruction metrics
        target_np = target.cpu().numpy()
        reconstruction_np = reconstruction.cpu().numpy()
        mse = np.mean((target_np - reconstruction_np) ** 2)
        mae = np.mean(np.abs(target_np - reconstruction_np))
        
        # Create the main MAE visualization
        mae_fig = self._create_mae_figure(
            original_image, target, reconstruction, mask,
            sample_idx, epoch, mse, mae
        )
        
        # Prepare wandb data
        wandb_data = {
            # Main MAE visualization
            f"mae_reconstruction/sample_{sample_idx}_{id}": wandb.Image(
                mae_fig, 
                caption=f"MAE Reconstruction - Epoch {epoch}, Sample {sample_idx}, MSE: {mse:.6f}, MAE: {mae:.6f}"
            ),
            
            # Reconstruction metrics
            f"mae_metrics/epoch": epoch,
            f"mae_metrics/sample_{viz_idx}_mse_{id}": mse,
            f"mae_metrics/sample_{viz_idx}_mae_{id}": mae,
        }
        
        # Create trajectory visualization if available
        if trajectory is not None and len(trajectory) > 1:
            try:
                trajectory_fig = self._create_latent_trajectory_figure(
                    trajectory, sample_idx, epoch, original_image=original_image
                )
                wandb_data[f"latent_trajectory/sample_{sample_idx}_{id}"] = wandb.Image(
                    trajectory_fig,
                    caption=f"Latent Trajectories - Epoch {epoch}, Sample {sample_idx}"
                )

               
                traj_stats = self._compute_trajectory_stats(trajectory)
                wandb_data[f"trajectory_stats/sample_{viz_idx}_mean_displacement_{id}"] = traj_stats['mean_total']
                wandb_data[f"trajectory_stats/sample_{viz_idx}_max_displacement_{id}"] = traj_stats['max_total']
                

                
               
                plt.close(trajectory_fig)
            except Exception as e:
                print(f"Error creating trajectory visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Successfully uploaded MAE reconstruction for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
        
        plt.close(mae_fig)
    
    def _create_latent_trajectory_figure(
        self, 
        trajectory: list, 
        sample_idx: int, 
        epoch: int,
        original_image: np.ndarray = None,
        num_latents: int = None,
        figsize: tuple = (12, 12)
    ) -> plt.Figure:
        """
        Create visualization of latent position trajectories.
        
        Each layer gets a different color:
        - Layer 0 (initial): black dots
        - Layer 1: blue (#3a86ff)
        - Layer 2: purple (#8338ec)
        - Layer 3: pink (#ff006e)
        - Layer 4: orange (#fb5607)
        
        Args:
            trajectory: List of [B, L, 2] tensors (one per layer)
            sample_idx: Sample index for title
            epoch: Current epoch for title
            original_image: Optional RGB image to show as background [H, W, C]
            num_latents: Number of latents to plot (None = all)
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Extract positions for batch 0
        traj_np = [t[0].detach().cpu().numpy() for t in trajectory]
        num_layers = len(traj_np)
        L = traj_np[0].shape[0]
        
        # Select subset of latents if needed
        if num_latents is not None and num_latents < L:
            indices = np.linspace(0, L - 1, num_latents, dtype=int)
        else:
            indices = np.arange(L)
            num_latents = L
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get coordinate bounds from trajectory
        all_coords = np.concatenate(traj_np, axis=0)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        # Add margin
        margin = max(x_max - x_min, y_max - y_min) * 0.05
        extent = [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
        
        # Show RGB background if provided
        if original_image is not None:
            try:
                rgb_background = self._create_rgb_from_image(original_image)
                # Display image with extent matching latent coordinate system
                ax.imshow(
                    rgb_background, 
                    extent=extent,
                    aspect='auto',
                    alpha=0.5,
                    origin='lower',
                    zorder=1
                )
            except Exception as e:
                print(f"Warning: Could not display RGB background: {e}")
        
        # Plot each latent's trajectory
        for lat_idx in indices:
            # Get path for this latent
            path = np.array([traj_np[layer][lat_idx] for layer in range(num_layers)])
            
            # Plot initial position (black dot)
            ax.scatter(
                path[0, 0], path[0, 1],
                c=self.trajectory_colors[0],
                s=15,
                zorder=10,
                alpha=0.8
            )
            
            # Plot each subsequent layer
            for layer in range(1, num_layers):
                color = self.trajectory_colors[min(layer, len(self.trajectory_colors) - 1)]
                
                # Draw line from previous position to current
                ax.plot(
                    [path[layer - 1, 0], path[layer, 0]],
                    [path[layer - 1, 1], path[layer, 1]],
                    color=color,
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=5
                )
                
                # Draw dot at current position
                ax.scatter(
                    path[layer, 0], path[layer, 1],
                    c=color,
                    s=12,
                    zorder=10,
                    alpha=0.9
                )
        
        # Create legend
        legend_elements = []
        layer_names = ['Initial (L0)', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 
                       'Layer 5', 'Layer 6', 'Layer 7']
        for layer in range(min(num_layers, len(self.trajectory_colors))):
            color = self.trajectory_colors[layer]
            name = layer_names[layer] if layer < len(layer_names) else f'Layer {layer}'
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, label=name)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Compute stats for title
        total_disp = traj_np[-1] - traj_np[0]
        mean_disp = np.mean(np.linalg.norm(total_disp, axis=-1))
        max_disp = np.max(np.linalg.norm(total_disp, axis=-1))
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(
            f'Latent Trajectories - Sample {sample_idx}, Epoch {epoch}\n'
            f'{num_latents} latents, {num_layers} layers | '
            f'Mean Δ: {mean_disp:.3f}m, Max Δ: {max_disp:.3f}m',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _compute_trajectory_stats(self, trajectory: list) -> dict:
        """Compute statistics about latent trajectories."""
        traj = [t[0].detach().cpu() for t in trajectory]
        
        # Total displacement
        total_delta = traj[-1] - traj[0]
        total_magnitude = torch.norm(total_delta, dim=-1)
        
        stats = {
            'mean_total': total_magnitude.mean().item(),
            'max_total': total_magnitude.max().item(),
            'min_total': total_magnitude.min().item(),
            'std_total': total_magnitude.std().item(),
        }
        
        # Per-layer stats
        for layer in range(1, len(traj)):
            delta = traj[layer] - traj[layer - 1]
            magnitude = torch.norm(delta, dim=-1)
            stats[f'mean_layer_{layer}'] = magnitude.mean().item()
        
        return stats
    
    def _create_mae_figure(self, original_image, target, reconstruction, mask,
                          sample_idx, epoch, mse, mae):
        """Create MAE reconstruction visualization figure with RGB, NIR, and Elevation."""
        
        # Spatial dimensions
        h, w, c = 512, 512, 5
        
        # Reshape target and reconstruction using einops rearrange
        try:
            target_spatial = rearrange(target, "(c h w) -> c h w", h=h, w=w, c=c)
            reconstruction_spatial = rearrange(reconstruction, "(c h w) -> c h w", h=h, w=w, c=c)
            
        except Exception as e:
            print(f"Warning: Could not reshape target/reconstruction: {e}")
            print(f"Target shape: {target.shape}, Reconstruction shape: {reconstruction.shape}")
            device = target.device if hasattr(target, 'device') else 'cpu'
            target_spatial = torch.zeros((c, h, w), device=device)
            reconstruction_spatial = torch.zeros((c, h, w), device=device)
        
        # Create figure with 8 subplots (2 rows x 4 columns)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(
            f'MAE Reconstruction - Sample {sample_idx}, Epoch {epoch}\n'
            f'MSE: {mse:.6f}, MAE: {mae:.6f}', 
            fontsize=16, fontweight='bold'
        )
        
        # 1. Original RGB composite
        rgb_composite = self._create_rgb_from_image(original_image)
        axes[0, 0].imshow(rgb_composite)
        axes[0, 0].set_title('Original RGB Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. NIR channel (index 3) - Original
        nir_original = original_image[:, :, 3]
        nir_vmin, nir_vmax = nir_original.min(), nir_original.max()
        im1 = axes[0, 1].imshow(nir_original, cmap='RdYlGn', vmin=nir_vmin, vmax=nir_vmax)
        axes[0, 1].set_title('Original NIR Channel', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Elevation channel (index 4) - Original
        elevation_original = original_image[:, :, 4]
        elevation_vmin, elevation_vmax = elevation_original.min(), elevation_original.max()
        im2 = axes[0, 2].imshow(elevation_original, cmap='gray', vmin=elevation_vmin, vmax=elevation_vmax)
        axes[0, 2].set_title('Original Elevation', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Combined Error
        error_all = torch.abs(target_spatial - reconstruction_spatial)
        error_mean = torch.mean(error_all, dim=0)
        im3 = axes[0, 3].imshow(error_mean.cpu().numpy(), cmap='hot')
        axes[0, 3].set_title('Mean Absolute Error (All Channels)', fontsize=14, fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # 5. Reconstructed RGB composite
        reconstructed_rgb = self._create_rgb_from_reconstruction(reconstruction_spatial)
        axes[1, 0].imshow(reconstructed_rgb)
        axes[1, 0].set_title('Reconstructed RGB Image', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 6. NIR channel - Reconstruction
        nir_reconstruction = reconstruction_spatial[3]
        im4 = axes[1, 1].imshow(nir_reconstruction.cpu().numpy(), cmap='RdYlGn', vmin=nir_vmin, vmax=nir_vmax)
        axes[1, 1].set_title('Reconstructed NIR Channel', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 7. Elevation channel - Reconstruction
        elevation_reconstruction = reconstruction_spatial[4]
        im5 = axes[1, 2].imshow(elevation_reconstruction.cpu().numpy(), cmap='gray', vmin=elevation_vmin, vmax=elevation_vmax)
        axes[1, 2].set_title('Reconstructed Elevation', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 8. RGB Error
        rgb_error = np.abs(rgb_composite - reconstructed_rgb)
        rgb_error_magnitude = np.mean(rgb_error, axis=-1)
        im6 = axes[1, 3].imshow(rgb_error_magnitude, cmap='hot')
        axes[1, 3].set_title('RGB Reconstruction Error', fontsize=14, fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(im6, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_from_image(self, image):
        """Create RGB composite from image array."""
        if len(image.shape) == 3 and image.shape[2] >= 3:
            rgb_composite = np.stack([
                image[:, :, 2],  # Red channel
                image[:, :, 1],  # Green channel  
                image[:, :, 0]   # Blue channel
            ], axis=-1)
        else:
            print(f"Warning: Cannot create RGB from image shape {image.shape}")
            single_channel = image[:, :, 0] if len(image.shape) == 3 else image
            rgb_composite = np.stack([single_channel] * 3, axis=-1)
        
        composite_tensor = torch.from_numpy(rgb_composite).permute(2, 0, 1)
        normalized = normalize(composite_tensor)
        
        return normalized.permute(1, 2, 0).numpy()
    
    def _create_rgb_from_reconstruction(self, reconstruction_spatial):
        """Create RGB composite from reconstructed channels."""
        rgb_composite = torch.stack([
            reconstruction_spatial[2],  # Red
            reconstruction_spatial[1],  # Green
            reconstruction_spatial[0]   # Blue
        ], dim=0)
        
        normalized = normalize(rgb_composite)
        
        return normalized.permute(1, 2, 0).cpu().numpy()

import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from einops import rearrange


class FLAIR_CustomSegmentationCallback(pl.Callback):

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: Configuration dict containing visualization parameters
        """
        super().__init__()
        self.config = config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
        # FLAIR-2 class definitions (13 classes as shown in the table)
        self.class_names = {
            1: 'building',
            2: 'pervious surface', 
            3: 'impervious surface',
            4: 'bare soil',
            5: 'water',
            6: 'coniferous',
            7: 'deciduous',
            8: 'brushwood',
            9: 'vineyard',
            10: 'herbaceous vegetation',
            11: 'agricultural land',
            12: 'plowed land',
            13: 'other'  # Classes >13 are grouped as "other"
        }
        
        # Define colors for each class (RGB values 0-1) - matching the colors in your table
        self.class_colors = {
            0: [0.0, 0.0, 0.0],        # background - black
            1: [0.86, 0.05, 0.60],     # building - magenta (as shown in table)
            2: [0.58, 0.56, 0.48],     # pervious surface - brown/gray (as shown in table)
            3: [0.97, 0.05, 0.0],      # impervious surface - red (as shown in table)
            4: [0.66, 0.44, 0.0],      # bare soil - brown (as shown in table)
            5: [0.08, 0.33, 0.68],     # water - blue (as shown in table)
            6: [0.10, 0.29, 0.15],     # coniferous - dark green (as shown in table)
            7: [0.27, 0.89, 0.51],     # deciduous - light green (as shown in table)
            8: [0.95, 0.65, 0.05],     # brushwood - orange (as shown in table)
            9: [0.40, 0.0, 0.51],      # vineyard - purple (as shown in table)
            10: [0.33, 1.0, 0.0],      # herbaceous vegetation - bright green (as shown in table)
            11: [1.0, 0.95, 0.05],     # agricultural land - yellow (as shown in table)
            12: [0.89, 0.87, 0.49],    # plowed land - pale yellow (as shown in table)
            13: [0.0, 0.0, 0.0]        # other - black (as shown in table)
        }
        
        # Band definitions for RGB visualization (5 channels total)
        self.rgb_bands = [2, 1, 0]  # Red, Green, Blue (assuming channels 0,1,2 are RGB-like)
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform segmentation visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # **CRITICAL: Ensure we only run on GPU 0 (rank 0) to avoid multiple executions**
        if trainer.is_global_zero == False:
            return
            
        # Alternative way to check for single GPU execution
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            # Only execute on rank 0
            if wandb.run is not None:
                self._perform_custom_segmentation(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping segmentation logging")
        except Exception as e:
            print(f"Error in custom segmentation callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_custom_segmentation(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform segmentation using your custom data loading."""
        
        # Get the dataset from the trainer
        dataset_train = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(sample_idx, i, dataset_train, pl_module, trainer.current_epoch,id="validation")
            
        dataset_val = trainer.datamodule.train_dataset
        
        
            
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(sample_idx, i, dataset_val, pl_module, trainer.current_epoch,id="train")
            
            
            
    def _process_single_sample(self, dataset_idx: int, viz_idx: int, dataset, pl_module, epoch: int,id="val"):
        """Process a single sample using your get_samples_to_viz function."""
        
        # Get data from your custom method
        image,image_tokens, attention_mask, mae_tokens, mask_MAE_res,label_res = dataset.get_samples_to_viz(dataset_idx)
        
        # Move to device
        mae_tokens_mask = torch.ones(mae_tokens.shape[0])
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mask_MAE_res = mask_MAE_res.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions to match batch format
                image_tokens_batch = image_tokens.unsqueeze(0) 
                image_tokens_mask = attention_mask.unsqueeze(0)
                mae_tokens_batch = mae_tokens.clone().unsqueeze(0)
                mae_tokens_mask_batch = mae_tokens_mask.unsqueeze(0)
                
                y_hat, (bias_matrix,attention_matrix) = pl_module.forward(
                    image_tokens_batch,
                    image_tokens_mask,
                    mae_tokens_batch,
                    mae_tokens_mask_batch,
                    training=False,
                    task="vizualisation"
                )


                
                # Remove batch dimension for visualization
                y_hat = y_hat.squeeze(0)
                mae_tokens=mae_tokens.squeeze(0)
                labels = label_res
                mae_tokens = rearrange(mae_tokens, "(c h w) b -> c h w b", h=512, w=512, c=5, b=6)
                
                # Reshape to spatial format
                bandvalues = image# mae_tokens[:, :, :, 0]  # [h, w, channels] - B,G,R,NIR,elevation
                classes = labels


               
                
                
                y_hat = rearrange(y_hat, "(c h w ) l -> c h w l", h=512, w=512 ,c=5)
                #y_hat = y_hat.mean(dim=0)
                y_hat  = y_hat[0]
                y_hat = torch.argmax(y_hat.clone(), dim=-1)
                #y_hat=y_hat[0,:,:]
                
                labels= mae_tokens[0,:,:,4]#rearrange(labels,"c h w -> h w c").squeeze(-1)
                
                
                print(f"✓ Successfully reconstructed sample {dataset_idx}")
                
            except Exception as e:
                print(f"Error in model forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_bias_matrix_slices(
                bias_matrix,
                slices=[0,110,378],
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                epoch=epoch,
                id=id,
                title_prefix = "Bias Matrix Slices",
                title="Bias"
            )

            self._create_and_upload_bias_matrix_slices(
                attention_matrix,
                slices=[0,110,378],
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                epoch=epoch,
                id=id,
                title_prefix = "Attention Matrix Slices",
                title="Attention"
            )
                
            self._create_and_upload_segmentation_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                bandvalues=bandvalues,        # [h, w, 5] - B,G,R,NIR,elevation
                ground_truth=labels,        # [h, w] - ground truth classes
                prediction=y_hat,            # [h, w] - predicted classes
                epoch=epoch,
                id=id
            )
        except Exception as e:
            print(f"Error creating segmentation visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_iou_per_class(self, ground_truth, predictions):
        """Calculate IoU for each class."""
        iou_scores = {}
        
        for class_id in range(1, 14):  # Classes 1-13 (including "other" as class 13)
            if class_id in self.class_names:
                gt_mask = (ground_truth == class_id)
                pred_mask = (predictions == class_id)
                
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0 if gt_mask.sum() == 0 else float('nan')
                
                iou_scores[class_id] = iou
        
        return iou_scores
    
    def _create_and_upload_segmentation_visualization(self, sample_idx, viz_idx, bandvalues, 
                                                    ground_truth, prediction, epoch,id):
        """Create and upload segmentation visualization to wandb."""
        
        # Convert to numpy for visualization
        bandvalues_np = bandvalues.cpu().numpy()   # [h, w, 5] - B,G,R,NIR,elevation
        gt_np = ground_truth.cpu().numpy()         # [h, w] - ground truth classes
        pred_np = prediction.cpu().numpy()         # [h, w] - predicted classes
        
        # Calculate simple accuracy
        accuracy = (pred_np == gt_np).mean()
        
        # Create the main segmentation visualization
        segmentation_fig = self._create_segmentation_figure(
            bandvalues_np, gt_np, pred_np, sample_idx, epoch, accuracy
        )
        
        # Prepare wandb data
        wandb_data = {
            # Main segmentation visualization
            f"segmentation/sample_{sample_idx} {id}": wandb.Image(
                segmentation_fig, 
                caption=f"Segmentation - Epoch {epoch}, Sample {sample_idx}, Accuracy: {accuracy:.3f}"
            ),
            
            # Basic metrics
            f"segmentation_metrics/epoch": epoch,
            f"segmentation_metrics/sample_{viz_idx}_accuracy": accuracy,
        }
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Successfully uploaded segmentation for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
        
        plt.close(segmentation_fig)
    
    def _create_segmentation_figure(self, bandvalues, ground_truth, prediction, 
                                  sample_idx, epoch, accuracy):
        """Create simplified segmentation visualization figure."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Segmentation Results - Sample {sample_idx}, Epoch {epoch}\n'
                    f'Accuracy: {accuracy:.3f}', 
                    fontsize=16, fontweight='bold')
        
        # 1. RGB composite from bandvalues (B,G,R,NIR,elevation -> RGB)
        rgb_composite = self._create_rgb_from_bandvalues(bandvalues)
        axes[0].imshow(rgb_composite)
        axes[0].set_title('RGB Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Ground truth segmentation
        gt_colored = self._labels_to_rgb(ground_truth)
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Classes', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Predicted segmentation
        pred_colored = self._labels_to_rgb(prediction)
        axes[2].imshow(pred_colored)
        axes[2].set_title('Predicted Classes', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_from_bandvalues(self, bandvalues):
        """Create RGB composite from bandvalues tensor."""
        # bandvalues: [h, w, 5] where channels are B,G,R,NIR,elevation
        # We want RGB, so we need to extract channels 2,1,0 (R,G,B)
        
        rgb_composite = np.stack([
            bandvalues[:, :, 2],  # Red channel (index 2)
            bandvalues[:, :, 1],  # Green channel (index 1)  
            bandvalues[:, :, 0]   # Blue channel (index 0)
        ], axis=-1)  # [h, w, 3]
        
        # Normalize using the provided function
        composite_tensor = torch.from_numpy(rgb_composite).permute(2, 0, 1)  # [3, h, w]
        normalized = normalize(composite_tensor)  # [3, h, w]
        
        return normalized.permute(1, 2, 0).numpy()  # [h, w, 3]

    def _create_and_upload_bias_matrix_slices(
        self,
        bias_matrix: torch.Tensor,
        slices: list | tuple | None,
        sample_idx: int,
        viz_idx: int,
        epoch: int,
        id: str,
        title_prefix: str = "Bias Matrix Slices",
        title: str = "Bias",
    ):
        """
        Visualize 3 slices from a [S, H, W] bias_matrix (e.g., [400,512,512]) in a single row,
        with each subplot independently normalized and its own aligned colorbar.

        Args:
            bias_matrix: torch.Tensor, shape [S,H,W] (or [1,S,H,W]) on any device
            slices: list/tuple of 3 integers indicating which S indices to plot.
                    If None/short, we auto-pick evenly spaced valid indices.
            sample_idx, viz_idx, epoch, id: metadata to show in title & W&B keys.
            title_prefix: optional figure title prefix.
            title: series name for W&B logging / colorbar labels.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # ---- Sanity + shape handling
        if bias_matrix is None:
            print("Warning: bias_matrix is None; skipping bias visualization.")
            return

        # Remove possible batch dim: [1,S,H,W] -> [S,H,W]
        if bias_matrix.dim() == 4 and bias_matrix.shape[0] == 1:
            bias_matrix = bias_matrix.squeeze(0)

        if bias_matrix.dim() != 3:
            print(f"Warning: Expected bias_matrix [S,H,W], got {tuple(bias_matrix.shape)}; skipping.")
            return

        S, H, W = bias_matrix.shape

        # Move to CPU numpy
        bm_np = bias_matrix.detach().float().cpu().numpy()  # [S,H,W]

        # ---- Pick/validate slices
        if not slices:
            slices = [0, max(0, S // 2), S - 1]
        else:
            slices = [int(max(0, min(S - 1, s))) for s in slices]

        if len(slices) < 3:
            needed = 3 - len(slices)
            candidates = [0, max(0, S // 2), S - 1]
            for c in candidates:
                if len(slices) >= 3:
                    break
                if c not in slices:
                    slices.append(c)
            slices = slices[:3]
        elif len(slices) > 3:
            slices = slices[:3]

        # Extract images
        imgs = [bm_np[s] for s in slices]  # list of [H,W]

        # ---- Plot: per-slice normalization + one colorbar per axes
        # constrained_layout keeps a tight layout and aligns colorbars
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        fig.suptitle(
            f"{title_prefix} – Sample {sample_idx} (viz {viz_idx}) – {id}\n"
            f"Epoch {epoch} • slices {slices[0]}, {slices[1]}, {slices[2]} • shape [{S},{H},{W}]",
            fontsize=14,
            fontweight="bold",
        )

        ims = []
        for ax, img, s in zip(axes, imgs, slices):
            # Robust local normalization (per slice)
            vmin = np.percentile(img, 2.0)
            vmax = np.percentile(img, 98.0)

            # Fallback if near-constant
            if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                img_min = float(np.nanmin(img))
                img_max = float(np.nanmax(img))
                if np.isclose(img_min, img_max):
                    # Degenerate image: expand a touch to avoid singular colormap
                    vmin, vmax = img_min - 1e-6, img_max + 1e-6
                else:
                    vmin, vmax = img_min, img_max

            im = ax.imshow(img, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"Slice {s}", fontsize=12)
            ax.axis("off")
            ims.append(im)

            # Individual colorbar (aligned via constrained_layout)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"{title} value", rotation=90)

        # ---- Log to W&B
        try:
            import wandb  # local import in case environment varies
            wandb.log({
                f"{title}/sample_{sample_idx} {id}": wandb.Image(
                    fig,
                    caption=f"{title} slices {slices} – epoch={epoch} – sample={sample_idx}",
                ),
                f"{title}/epoch": epoch,
                f"{title}/sample_{viz_idx}_indices": slices,
            })
            print(f"✓ Uploaded {title} slices {slices} for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload {title} slices to wandb: {e}")

        plt.close(fig)


    
    def _labels_to_rgb(self, labels):
        """Convert label map to RGB visualization."""
        rgb_image = np.zeros((labels.shape[0], labels.shape[1], 3))
        
        for class_id, color in self.class_colors.items():
            mask = (labels == class_id)
            rgb_image[mask] = color
        
        return rgb_image