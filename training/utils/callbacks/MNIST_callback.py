"""
Visualization Callback for MNIST Sparse Canvas Experiment

Visualizes:
1. Reconstruction: Ground truth, prediction, error (simple 3-panel)
2. Latent trajectories: How latents move through layers within a forward pass
3. Classification: Predicted vs actual digit class with confidence

Uses purple gradient for trajectory colors (dark = early layer, light = late layer).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange
from typing import List, Optional
import wandb


def normalize(img, min_val=0, max_val=1):
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 1e-6:
        return (img - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
    return img


class MNISTVisualizationCallback(pl.Callback):
    """
    Callback for visualizing MNIST sparse canvas reconstruction and latent trajectories.
    """
    
    # Purple gradient: dark (early layer) → light (late layer)
    TRAJECTORY_COLORS = [
        '#10002b',  # Layer 0 (initial) - darkest
        '#240046',  # Layer 1
        '#3c096c',  # Layer 2
        '#5a189a',  # Layer 3
        '#7b2cbf',  # Layer 4
        '#9d4edd',  # Layer 5
        '#c77dff',  # Layer 6
        '#e0aaff',  # Layer 7 - lightest
    ]
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict containing:
                - debug.viz_every_n_epochs: How often to visualize
                - debug.idxs_to_viz: Which sample indices to visualize
        """
        super().__init__()
        self.config = config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # Only run on rank 0
        if not trainer.is_global_zero:
            return
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            if wandb.run is not None:
                self._perform_visualization(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping visualization")
        except Exception as e:
            print(f"Error in MNIST visualization callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_visualization(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform reconstruction and trajectory visualization."""
        
        # Validation set
        dataset_val = trainer.datamodule.val_dataset
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(
                sample_idx, i, dataset_val, pl_module,
                trainer.current_epoch, split="val"
            )
        
        # Training set
        dataset_train = trainer.datamodule.train_dataset
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(
                sample_idx, i, dataset_train, pl_module,
                trainer.current_epoch, split="train"
            )
    
    def _process_single_sample(
        self,
        dataset_idx: int,
        viz_idx: int,
        dataset,
        pl_module,
        epoch: int,
        split: str = "val"
    ):
        """Process a single sample for visualization."""
        
        # Get data from dataset
        result = dataset.get_samples_to_viz(dataset_idx)
        
        # Unpack: image, image_tokens, attention_mask, queries, queries_mask, label, latent_pos, bbox
        if len(result) == 8:
            image, image_tokens, attention_mask, queries, queries_mask, label, latent_pos, bbox = result
        else:
            image, image_tokens, attention_mask, queries, queries_mask, label, latent_pos = result[:7]
            bbox = {'x': 0, 'y': 0, 'w': 64, 'h': 64}
        
        # Extract ground truth label
        if isinstance(label, torch.Tensor):
            gt_label = label.item() if label.numel() == 1 else int(label[0].item())
        else:
            gt_label = int(label)
        
        # Move to device
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        queries = queries.to(device)
        queries_mask = torch.ones(queries.shape[0]).to(device)
        latent_pos = latent_pos.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions for batch
                image_tokens_batch = image_tokens.unsqueeze(0)
                attention_mask_batch = attention_mask.unsqueeze(0)
                queries_batch = queries.unsqueeze(0)
                queries_mask_batch = queries_mask.unsqueeze(0)
                latent_pos_batch = latent_pos.unsqueeze(0)
                
                # Forward pass with encoder to get latents + trajectory
                encoder_result = pl_module.encoder(
                    image_tokens_batch,
                    attention_mask_batch,
                    queries_batch,
                    queries_mask_batch,
                    latent_pos_batch,
                    training=False,
                    task="encoder_viz",
                    return_trajectory=True  # Explicitly request trajectory
                )
                
                latents = encoder_result['latents']
                final_coords = encoder_result['final_coords']
                trajectory = encoder_result.get('trajectory', None)
                
                # Get reconstruction
                y_hat = pl_module.encoder.reconstruct(
                    latents, final_coords, queries_batch, queries_mask_batch
                )
                
                # Get classification prediction
                pred_label = None
                pred_probs = None
                if hasattr(pl_module.encoder, 'to_logits') and pl_module.encoder.to_logits is not None:
                    try:
                        logits = pl_module.encoder.classify(latents)
                        pred_probs = torch.softmax(logits, dim=-1).squeeze(0)  # [num_classes]
                        pred_label = logits.argmax(dim=-1).item()
                    except Exception as e:
                        print(f"Warning: Could not get classification prediction: {e}")
                
                # Remove batch dimension
                y_hat = y_hat.squeeze(0)
                
                # Get target and reconstruction
                target = queries[:, 0]  # Reflectance values
                reconstruction = y_hat.squeeze(-1) if y_hat.dim() > 1 else y_hat
                
                print(f"✓ Reconstructed sample {dataset_idx} ({split})")
                if trajectory is not None:
                    print(f"  Trajectory: {len(trajectory)} layers")
                if pred_label is not None:
                    correct = "✓" if pred_label == gt_label else "✗"
                    conf = pred_probs[pred_label].item() * 100 if pred_probs is not None else 0
                    print(f"  Classification: GT={gt_label}, Pred={pred_label} {correct} ({conf:.1f}%)")
                
            except Exception as e:
                print(f"Error in forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                original_image=image,
                target=target,
                reconstruction=reconstruction,
                trajectory=trajectory,
                bbox=bbox,
                epoch=epoch,
                split=split,
                gt_label=gt_label,
                pred_label=pred_label,
                pred_probs=pred_probs
            )
        except Exception as e:
            print(f"Error creating visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_and_upload_visualization(
        self,
        sample_idx: int,
        viz_idx: int,
        original_image,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        trajectory: Optional[List[torch.Tensor]],
        bbox: dict,
        epoch: int,
        split: str,
        gt_label: int = None,
        pred_label: int = None,
        pred_probs: torch.Tensor = None
    ):
        """Create and upload reconstruction + trajectory visualization."""
        
        # Calculate metrics
        target_np = target.cpu().numpy()
        reconstruction_np = reconstruction.cpu().numpy()
        mse = np.mean((target_np - reconstruction_np) ** 2)
        mae_val = np.mean(np.abs(target_np - reconstruction_np))
        
        # Create reconstruction figure
        recon_fig = self._create_reconstruction_figure(
            original_image=original_image,
            target=target,
            reconstruction=reconstruction,
            sample_idx=sample_idx,
            epoch=epoch,
            mse=mse,
            mae_val=mae_val,
            gt_label=gt_label,
            pred_label=pred_label,
            pred_probs=pred_probs
        )
        
        # Build caption
        caption = f"Epoch {epoch}, Sample {sample_idx}, MSE: {mse:.6f}"
        if gt_label is not None and pred_label is not None:
            correct = "✓" if pred_label == gt_label else "✗"
            caption += f", GT: {gt_label}, Pred: {pred_label} {correct}"
        
        # Prepare wandb data
        wandb_data = {
            f"reconstruction/sample_{sample_idx}_{split}": wandb.Image(
                recon_fig,
                caption=caption
            ),
            f"metrics/sample_{viz_idx}_mse_{split}": mse,
            f"metrics/sample_{viz_idx}_mae_{split}": mae_val,
        }
        
        # Log classification metrics
        if pred_label is not None and gt_label is not None:
            is_correct = 1.0 if pred_label == gt_label else 0.0
            wandb_data[f"classification/sample_{viz_idx}_correct_{split}"] = is_correct
            wandb_data[f"classification/sample_{viz_idx}_gt_{split}"] = gt_label
            wandb_data[f"classification/sample_{viz_idx}_pred_{split}"] = pred_label
            
            if pred_probs is not None:
                # Log confidence for predicted class and ground truth class
                wandb_data[f"classification/sample_{viz_idx}_confidence_{split}"] = pred_probs[pred_label].item()
                wandb_data[f"classification/sample_{viz_idx}_gt_prob_{split}"] = pred_probs[gt_label].item()
        
        plt.close(recon_fig)


        
        
        # Create trajectory figure if available
        if trajectory is not None and len(trajectory) > 1:
            try:
                traj_fig = self._create_trajectory_figure(
                    trajectory=trajectory,
                    original_image=original_image,
                    bbox=bbox,
                    sample_idx=sample_idx,
                    epoch=epoch,
                    gt_label=gt_label,
                    pred_label=pred_label
                )
                
                traj_caption = f"Latent Trajectories - Epoch {epoch}, Sample {sample_idx}"
                if gt_label is not None and pred_label is not None:
                    correct = "✓" if pred_label == gt_label else "✗"
                    traj_caption += f", GT: {gt_label}, Pred: {pred_label} {correct}"
                
                wandb_data[f"trajectory/sample_{sample_idx}_{split}"] = wandb.Image(
                    traj_fig,
                    caption=traj_caption
                )
                
                # Trajectory stats
                traj_stats = self._compute_trajectory_stats(trajectory)
                wandb_data[f"trajectory_stats/sample_{viz_idx}_mean_displacement_{split}"] = traj_stats['mean_total']
                wandb_data[f"trajectory_stats/sample_{viz_idx}_max_displacement_{split}"] = traj_stats['max_total']
                
                plt.close(traj_fig)
                
            except Exception as e:
                print(f"Error creating trajectory visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Uploaded visualization for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
    
    def _create_reconstruction_figure(
        self,
        original_image,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        sample_idx: int,
        epoch: int,
        mse: float,
        mae_val: float,
        gt_label: int = None,
        pred_label: int = None,
        pred_probs: torch.Tensor = None
    ) -> plt.Figure:
        """Create simple 3-panel reconstruction figure: GT, Reconstruction, Error."""
        
        # Get image dimensions
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.cpu().numpy()
        else:
            img_np = np.array(original_image)
        
        # Handle different formats
        if img_np.ndim == 3:
            if img_np.shape[0] in [1, 3]:  # [C, H, W]
                img_np = img_np.transpose(1, 2, 0)
            img_2d = img_np[:, :, 0] if img_np.shape[2] >= 1 else img_np
        else:
            img_2d = img_np
        
        h, w = img_2d.shape[:2]
        
        # Reshape target and reconstruction to spatial format
        try:
            target_spatial = rearrange(target, "(h w) -> h w", h=h, w=w)
            recon_spatial = rearrange(reconstruction, "(h w) -> h w", h=h, w=w)
        except Exception as e:
            print(f"Warning: Could not reshape: {e}, target shape: {target.shape}")
            # Fallback
            target_spatial = img_2d
            recon_spatial = torch.zeros_like(torch.from_numpy(img_2d))
        
        target_np = target_spatial.cpu().numpy() if isinstance(target_spatial, torch.Tensor) else target_spatial
        recon_np = recon_spatial.cpu().numpy() if isinstance(recon_spatial, torch.Tensor) else recon_spatial
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Build title with classification info
        title_lines = [f'MNIST Reconstruction - Sample {sample_idx}, Epoch {epoch}']
        title_lines.append(f'MSE: {mse:.6f}, MAE: {mae_val:.6f}')
        
        if gt_label is not None and pred_label is not None:
            correct = "✓" if pred_label == gt_label else "✗"
            class_info = f'GT: {gt_label}, Pred: {pred_label} {correct}'
            if pred_probs is not None:
                conf = pred_probs[pred_label].item() * 100
                class_info += f' (Confidence: {conf:.1f}%)'
            title_lines.append(class_info)
        
        # Color title based on correctness
        title_color = 'green' if (pred_label is not None and pred_label == gt_label) else 'red' if pred_label is not None else 'black'
        
        fig.suptitle('\n'.join(title_lines), fontsize=12, fontweight='bold', color=title_color)
        
        # 1. Ground Truth
        axes[0].imshow(target_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Reconstruction
        axes[1].imshow(recon_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Reconstruction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Error Map
        error = np.abs(target_np - recon_np)
        im = axes[2].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 1e-6))
        axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def _create_trajectory_figure(
        self,
        trajectory: List[torch.Tensor],
        original_image,
        bbox: dict,
        sample_idx: int,
        epoch: int,
        gt_label: int = None,
        pred_label: int = None,
        figsize: tuple = (10, 10)
    ) -> plt.Figure:
        """
        Create visualization of latent position trajectories through layers.
        
        Uses purple gradient: dark (#10002b) = early layer → light (#e0aaff) = late layer
        
        Args:
            trajectory: List of [B, L, 2] tensors (one per layer)
            original_image: Background image
            bbox: Digit bounding box
            sample_idx: Sample index
            epoch: Current epoch
            gt_label: Ground truth digit class
            pred_label: Predicted digit class
        """
        # Extract positions for batch 0
        traj_np = [t[0].detach().cpu().numpy() for t in trajectory]
        num_layers = len(traj_np)
        num_latents = traj_np[0].shape[0]
        
        # Get colors - interpolate if more layers than colors
        if num_layers <= len(self.TRAJECTORY_COLORS):
            colors = self.TRAJECTORY_COLORS[:num_layers]
        else:
            # Interpolate colors
            indices = np.linspace(0, len(self.TRAJECTORY_COLORS) - 1, num_layers).astype(int)
            colors = [self.TRAJECTORY_COLORS[i] for i in indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get image as 2D numpy for background
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.cpu().numpy()
        else:
            img_np = np.array(original_image)
        
        if img_np.ndim == 3:
            if img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)
            img_2d = img_np[:, :, 0]
        else:
            img_2d = img_np
        
        h, w = img_2d.shape[:2]
        
        # Show image as background
        ax.imshow(img_2d, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        
        # Determine coordinate scale (positions might be in meters or pixels)
        # Check if positions are in meter scale (small values) or pixel scale
        sample_pos = traj_np[0][0]
        if sample_pos.max() < 10:  # Likely in meters, need to convert
            gsd = 0.2  # meters per pixel
            scale = 1.0 / gsd
        else:
            scale = 1.0
        
        # Plot each latent's trajectory
        for lat_idx in range(num_latents):
            # Get path for this latent across all layers
            path = np.array([traj_np[layer][lat_idx] * scale for layer in range(num_layers)])
            
            # Draw trajectory lines between consecutive layers
            for layer in range(num_layers - 1):
                ax.plot(
                    [path[layer, 0], path[layer + 1, 0]],
                    [path[layer, 1], path[layer + 1, 1]],
                    color=colors[layer + 1],
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=5
                )
            
            # Draw points at each layer position
            for layer in range(num_layers):
                # Larger marker for first and last layer
                if layer == 0:
                    marker_size = 40
                    edge_color = 'white'
                    edge_width = 1.5
                elif layer == num_layers - 1:
                    marker_size = 80
                    edge_color = 'white'
                    edge_width = 2
                else:
                    marker_size = 25
                    edge_color = 'none'
                    edge_width = 0
                
                ax.scatter(
                    path[layer, 0], path[layer, 1],
                    c=colors[layer],
                    s=marker_size,
                    marker='o',
                    edgecolors=edge_color,
                    linewidths=edge_width,
                    zorder=10,
                    alpha=0.9
                )
        
        # Create legend
        legend_elements = []
        for layer in range(num_layers):
            label = f'Layer {layer}' if layer > 0 else 'Initial'
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[layer],
                          markersize=8 if layer in [0, num_layers-1] else 6,
                          label=label)
            )
       
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        # Compute displacement stats
        total_disp = traj_np[-1] - traj_np[0]
        mean_disp = np.mean(np.linalg.norm(total_disp, axis=-1))
        max_disp = np.max(np.linalg.norm(total_disp, axis=-1))
        
        # Build title with classification info
        title_lines = [f'Latent Trajectories - Sample {sample_idx}, Epoch {epoch}']
        title_lines.append(f'{num_latents} latents, {num_layers} layers | Mean Δ: {mean_disp:.3f}, Max Δ: {max_disp:.3f}')
        
        if gt_label is not None and pred_label is not None:
            correct = "✓" if pred_label == gt_label else "✗"
            title_lines.append(f'GT: {gt_label}, Pred: {pred_label} {correct}')
        
        # Color title based on correctness
        title_color = 'green' if (pred_label is not None and pred_label == gt_label) else 'red' if pred_label is not None else 'black'
        
        ax.set_title('\n'.join(title_lines), fontsize=12, fontweight='bold', color=title_color)
        
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Flip y-axis for image coordinates
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _compute_trajectory_stats(self, trajectory: List[torch.Tensor]) -> dict:
        """Compute statistics about latent trajectories."""
        traj = [t[0].detach().cpu() for t in trajectory]
        
        # Total displacement (first to last layer)
        total_delta = traj[-1] - traj[0]
        total_magnitude = torch.norm(total_delta, dim=-1)
        
        stats = {
            'mean_total': total_magnitude.mean().item(),
            'max_total': total_magnitude.max().item(),
            'min_total': total_magnitude.min().item(),
            'std_total': total_magnitude.std().item(),
        }
        
        # Per-layer displacement stats
        for layer in range(1, len(traj)):
            delta = traj[layer] - traj[layer - 1]
            magnitude = torch.norm(delta, dim=-1)
            stats[f'mean_layer_{layer}'] = magnitude.mean().item()
            stats[f'max_layer_{layer}'] = magnitude.max().item()
        
        return stats