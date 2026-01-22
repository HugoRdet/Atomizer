"""
Token-to-Latent Assignment Visualization Callback

Visualizes which tokens are assigned to which latents by:
1. Assigning a random color to each latent
2. Coloring each token's pixel position with its assigned latent's color
3. Creating a side-by-side visualization: RGB image + token assignment map
4. (Evolution) Showing how assignments change as latents move across layers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Tuple
from einops import repeat

try:
    import wandb
except ImportError:
    wandb = None


def generate_distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate n visually distinct colors using HSV space.
    """
    np.random.seed(seed)
    
    # Use golden ratio for well-distributed hues
    golden_ratio = 0.618033988749895
    hues = np.mod(np.arange(n) * golden_ratio + np.random.random(), 1.0)
    
    # High saturation and value for visibility
    saturations = np.random.uniform(0.7, 1.0, n)
    values = np.random.uniform(0.7, 1.0, n)
    
    colors = np.zeros((n, 3))
    for i in range(n):
        colors[i] = mcolors.hsv_to_rgb([hues[i], saturations[i], values[i]])
    
    return colors


class TokenAssignmentVisualizationCallback(pl.Callback):
    """
    Callback for visualizing token-to-latent assignments.
    
    Creates two types of visualizations:
    1. Static: Side-by-side RGB + token assignment map
    2. Evolution: Multi-panel showing how assignments change across layers
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Visualization frequency
        self.log_every_n_epochs = config["debug"].get("viz_every_n_epochs", 5)
        self.sample_indices = config["debug"].get("idxs_to_viz", [0])
        
        # Image parameters
        self.image_size = config.get("image_size", 512)
        self.gsd = config.get("gsd", 0.2)
        self.half_extent = (self.image_size * self.gsd) / 2.0
        
        # Latent parameters
        atomiser_config = config.get("Atomiser", {})
        self.spatial_latents_per_row = atomiser_config.get("spatial_latents", 16)
        self.num_spatial_latents = self.spatial_latents_per_row ** 2
        self.geo_k = atomiser_config.get("geo_k", 1500)
        self.depth = atomiser_config.get("depth", 4)
        
        # Visualization settings
        self.show_latent_positions = config["debug"].get("show_latent_positions", True)
        self.latent_marker_size = config["debug"].get("latent_marker_size", 50)
        self.color_seed = config["debug"].get("color_seed", 42)
        self.max_layers_to_show = config["debug"].get("max_assignment_layers", 6)
        self.show_evolution = config["debug"].get("show_token_assignment_evolution", True)
        
        # Pre-generate colors for latents
        self.latent_colors = generate_distinct_colors(self.num_spatial_latents, self.color_seed)
        
        print(f"[TokenAssignmentCallback] Initialized:")
        print(f"  Samples: {self.sample_indices}")
        print(f"  Image: {self.image_size}px, Extent: ±{self.half_extent:.1f}m")
        print(f"  Latents: {self.num_spatial_latents} ({self.spatial_latents_per_row}x{self.spatial_latents_per_row})")
        print(f"  Tokens per latent (geo_k): {self.geo_k}")
        print(f"  Show evolution: {self.show_evolution}")
        if self.show_evolution:
            print(f"  Max layers to show: {self.max_layers_to_show}")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate visualizations at end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        try:
            if wandb is not None and wandb.run is not None:
                self._generate_visualizations(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping token assignment visualization")
        except Exception as e:
            print(f"Error in TokenAssignmentVisualizationCallback: {e}")
            import traceback
            traceback.print_exc()
        pl_module.train()
    
    def _generate_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate visualizations for all sample indices."""
        dataset_val = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            try:
                self._visualize_sample(
                    sample_idx=sample_idx,
                    viz_idx=i,
                    dataset=dataset_val,
                    pl_module=pl_module,
                    epoch=trainer.current_epoch,
                    split="val"
                )
            except Exception as e:
                print(f"Error visualizing token assignment for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    def _visualize_sample(
        self,
        sample_idx: int,
        viz_idx: int,
        dataset,
        pl_module: pl.LightningModule,
        epoch: int,
        split: str
    ):
        """Visualize token assignments for a single sample."""
        # Get sample data
        image, image_tokens, attention_mask, mae_tokens, mask_MAE_res, _, latent_pos, _ = \
            dataset.get_samples_to_viz(sample_idx)
        
        device = pl_module.device
        
        # Prepare batch
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        
        with torch.no_grad():
            # Get encoder
            encoder = pl_module.encoder
            
            # Get tokens and mask
            tokens = image_tokens.unsqueeze(0)
            mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
            latent_pos=latent_pos.unsqueeze(0)
            
            # =================================================================
            # Run forward pass to get trajectory
            # =================================================================
            result = pl_module.forward(
                tokens, mask,
                mae_tokens.unsqueeze(0),
                torch.ones(mae_tokens.shape[0]).to(device).unsqueeze(0),
                latents_pos=latent_pos,
                training=False,
                task="visualization"
            )
            
            trajectory = result.get('trajectory', None)
            
            # If no trajectory (no displacement), create one with just initial positions
            if trajectory is None or len(trajectory) == 0:
                current_coords = encoder._get_default_latent_coords(1, device)
                trajectory = [current_coords]
            
            # =================================================================
            # Get token assignments for final positions (static visualization)
            # =================================================================
            final_coords = trajectory[-1]
            geo_tokens, geo_masks, _ = encoder.geo_pruning(tokens, mask, final_coords)
            
            # Extract token centers using geometry
            token_centers_np, geo_masks_np = self._extract_token_centers(
                encoder, geo_tokens, geo_masks
            )
            latent_coords_np = final_coords[0].cpu().numpy()
        
        # Create RGB background
        rgb_background = self._create_rgb_background(image)
        
        # =================================================================
        # 1. Static visualization (final state)
        # =================================================================
        assignment_map = self._create_assignment_map(token_centers_np, geo_masks_np)
        
        fig_static = self._create_static_figure(
            rgb_background=rgb_background,
            assignment_map=assignment_map,
            latent_coords=latent_coords_np,
            sample_idx=sample_idx,
            epoch=epoch
        )
        
        wandb_key = f"token_assignment/sample_{sample_idx}_{split}"
        wandb.log({wandb_key: wandb.Image(fig_static)})
        plt.close(fig_static)
        
        # =================================================================
        # 2. Evolution visualization (if enabled and we have trajectory)
        # =================================================================
        if self.show_evolution and len(trajectory) > 1:
            fig_evolution = self._create_evolution_figure(
                encoder=encoder,
                tokens=tokens,
                mask=mask,
                trajectory=trajectory,
                rgb_background=rgb_background,
                sample_idx=sample_idx,
                epoch=epoch
            )
            
            wandb_key_evolution = f"token_assignment_evolution/sample_{sample_idx}_{split}"
            wandb.log({wandb_key_evolution: wandb.Image(fig_evolution)})
            plt.close(fig_evolution)
        
        print(f"✓ Uploaded token assignment viz for sample {sample_idx}")
    
    def _extract_token_centers(
        self,
        encoder,
        geo_tokens: torch.Tensor,  # [B, L_spatial, k, feat_dim]
        geo_masks: torch.Tensor,   # [B, L_spatial, k]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract token centers from geo_tokens using geometry module.
        
        Returns:
            token_centers: [L_spatial, k, 2] numpy array
            geo_masks: [L_spatial, k] numpy array
        """
        B_size, L_spatial, k, feat_dim = geo_tokens.shape
        
        # Reshape to [B, L_spatial * k, feat_dim] for geometry processing
        geo_tokens_flat = geo_tokens.reshape(B_size, L_spatial * k, feat_dim)
        
        # Get token centers using the proper geometry method
        token_centers_flat = encoder.input_processor.geometry.get_token_centers(geo_tokens_flat)
        # token_centers_flat: [B, L_spatial * k, 2]
        
        # Reshape back to [B, L_spatial, k, 2]
        token_centers = token_centers_flat.reshape(B_size, L_spatial, k, 2)
        
        # Convert to numpy (take first batch)
        token_centers_np = token_centers[0].cpu().numpy()  # [L_spatial, k, 2]
        geo_masks_np = geo_masks[0].cpu().numpy()          # [L_spatial, k]
        
        return token_centers_np, geo_masks_np
    
    def _create_assignment_map(
        self,
        token_centers: np.ndarray,  # [L_spatial, k, 2]
        geo_masks: np.ndarray,      # [L_spatial, k]
        alpha: float = 0.8,         # High alpha for clarity
    ) -> np.ndarray:
        """
        Modified to prevent color bleeding. 
        Shows the 'primary' latent for each pixel to identify misassignments.
        """
        H, W = self.image_size, self.image_size
        assignment_map = np.ones((H, W, 3), dtype=np.float32)
        
        # Track assignment to avoid 'dot' artifacts
        # We store the latent_idx that 'owns' this pixel
        owner_map = np.full((H, W), -1, dtype=np.int32)
        coverage_count = np.zeros((H, W), dtype=np.int32)
        
        L_spatial, k, _ = token_centers.shape
        
        # Sort latents by index so rendering is deterministic
        for latent_idx in range(L_spatial):
            color = self.latent_colors[latent_idx]
            
            for token_idx in range(k):
                if geo_masks[latent_idx, token_idx]:
                    continue
                
                x_m, y_m = token_centers[latent_idx, token_idx, 0], token_centers[latent_idx, token_idx, 1]
                
                px = int((x_m + self.half_extent) / (2 * self.half_extent) * (W - 1))
                py = int((y_m + self.half_extent) / (2 * self.half_extent) * (H - 1))
                px, py = np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)
                
                # Update map: No blending, just direct assignment or z-buffering
                # This removes the 'pink dots in yellow zone' artifact
                assignment_map[py, px] = color
                owner_map[py, px] = latent_idx
                coverage_count[py, px] += 1
        
        self._last_coverage_count = coverage_count
        return assignment_map
    
    def _create_static_figure(
        self,
        rgb_background: np.ndarray,
        assignment_map: np.ndarray,
        latent_coords: np.ndarray,
        sample_idx: int,
        epoch: int
    ) -> plt.Figure:
        """
        Create side-by-side visualization figure with coverage stats.
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # Added third panel for coverage
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        # Left: RGB Image
        ax1 = axes[0]
        ax1.imshow(rgb_background, extent=extent, aspect='equal', origin='lower')
        
        if self.show_latent_positions:
            ax1.scatter(
                latent_coords[:, 0], latent_coords[:, 1],
                c=self.latent_colors,
                s=self.latent_marker_size,
                edgecolors='white',
                linewidths=1.5,
                zorder=10
            )
        
        ax1.set_title(f'RGB Image\nSample {sample_idx}', fontsize=14)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        
        # Middle: Token Assignment Map (with alpha blending)
        ax2 = axes[1]
        ax2.imshow(assignment_map, extent=extent, aspect='equal', origin='lower')
        
        if self.show_latent_positions:
            ax2.scatter(
                latent_coords[:, 0], latent_coords[:, 1],
                c=self.latent_colors,
                s=self.latent_marker_size,
                edgecolors='black',
                linewidths=1.5,
                zorder=10
            )
        
        ax2.set_title(f'Token-to-Latent Assignment\n(geo_k={self.geo_k}, α=0.15)', fontsize=14)
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        
        # Right: Coverage heatmap
        ax3 = axes[2]
        coverage = self._last_coverage_count if hasattr(self, '_last_coverage_count') else np.zeros((self.image_size, self.image_size))
        
        # Coverage statistics
        total_pixels = self.image_size * self.image_size
        uncovered = (coverage == 0).sum()
        single_covered = (coverage == 1).sum()
        multi_covered = (coverage > 1).sum()
        max_coverage = coverage.max()
        mean_coverage = coverage[coverage > 0].mean() if (coverage > 0).any() else 0
        
        im = ax3.imshow(coverage, extent=extent, aspect='equal', origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax3, label='# latents covering pixel')
        
        if self.show_latent_positions:
            ax3.scatter(
                latent_coords[:, 0], latent_coords[:, 1],
                c='cyan',
                s=self.latent_marker_size // 2,
                edgecolors='white',
                linewidths=1,
                zorder=10,
                alpha=0.7
            )
        
        ax3.set_title(
            f'Coverage Heatmap\n'
            f'Uncovered: {uncovered} ({100*uncovered/total_pixels:.1f}%), '
            f'Max: {max_coverage}, Mean: {mean_coverage:.1f}',
            fontsize=12
        )
        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)')
        
        fig.suptitle(
            f'Token Assignment Visualization - Epoch {epoch}\n'
            f'{self.num_spatial_latents} latents, {self.geo_k} tokens/latent',
            fontsize=16
        )
        
        plt.tight_layout()
        return fig
    
    def _create_evolution_figure(
        self,
        encoder,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        trajectory: List[torch.Tensor],
        rgb_background: np.ndarray,
        sample_idx: int,
        epoch: int
    ) -> plt.Figure:
        """
        Create multi-panel evolution figure showing token assignments at different layers.
        """
        n_trajectory = len(trajectory)
        
        # Select which trajectory points to show
        if n_trajectory <= self.max_layers_to_show:
            indices = list(range(n_trajectory))
        else:
            # Sample evenly, always including first and last
            indices = np.linspace(0, n_trajectory - 1, self.max_layers_to_show, dtype=int).tolist()
            # Ensure unique indices
            indices = sorted(list(set(indices)))
        
        n_panels = len(indices) + 1  # +1 for RGB
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        # =================================================================
        # First panel: RGB Image
        # =================================================================
        ax = axes.flat[0]
        ax.imshow(rgb_background, extent=extent, aspect='equal', origin='lower')
        
        # Show initial latent positions
        initial_coords = trajectory[0][0].cpu().numpy()
        ax.scatter(
            initial_coords[:, 0], initial_coords[:, 1],
            c=self.latent_colors,
            s=30,
            edgecolors='white',
            linewidths=1,
            zorder=10
        )
        ax.set_title('RGB Image', fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # =================================================================
        # Remaining panels: Assignment maps at different trajectory points
        # =================================================================
        with torch.no_grad():
            for panel_idx, traj_idx in enumerate(indices):
                ax = axes.flat[panel_idx + 1]
                
                # Get coordinates at this trajectory point
                coords = trajectory[traj_idx]
                
                # Run geographic pruning with these coordinates
                geo_tokens, geo_masks, _ = encoder.geo_pruning(tokens, mask, coords)
                
                # Extract token centers
                token_centers_np, geo_masks_np = self._extract_token_centers(
                    encoder, geo_tokens, geo_masks
                )
                coords_np = coords[0].cpu().numpy()
                
                # Create assignment map
                assignment_map = self._create_assignment_map(token_centers_np, geo_masks_np)
                
                # Plot
                ax.imshow(assignment_map, extent=extent, aspect='equal', origin='lower')
                ax.scatter(
                    coords_np[:, 0], coords_np[:, 1],
                    c=self.latent_colors,
                    s=30,
                    edgecolors='black',
                    linewidths=1,
                    zorder=10
                )
                
                # Title based on position in trajectory
                if traj_idx == 0:
                    title = 'Layer 0 (Initial)'
                elif traj_idx == n_trajectory - 1:
                    title = f'Layer {traj_idx} (Final)'
                else:
                    title = f'Layer {traj_idx}'
                
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
        
        # =================================================================
        # Hide unused axes
        # =================================================================
        for j in range(len(indices) + 1, len(axes.flat)):
            axes.flat[j].axis('off')
        
        fig.suptitle(
            f'Token Assignment Evolution - Sample {sample_idx}, Epoch {epoch}\n'
            f'{self.num_spatial_latents} latents, {self.geo_k} tokens/latent, {n_trajectory} trajectory points',
            fontsize=14
        )
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_background(self, image: np.ndarray) -> np.ndarray:
        """Convert image to RGB for visualization."""
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Assume BGR or similar, take first 3 channels as RGB
            rgb = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)
        else:
            # Single channel
            single = image[:, :, 0] if len(image.shape) == 3 else image
            rgb = np.stack([single] * 3, axis=-1)
        
        # Normalize to [0, 1]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb
    
class TokenAssignmentVisualizationCallbackMNIST(pl.Callback):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.log_every_n_epochs = config["debug"].get("viz_every_n_epochs", 5)
        self.sample_indices = config["debug"].get("idxs_to_viz", [0])
        self.image_size = config.get("image_size", 64) 
        
        atomiser_config = config.get("Atomiser", {})
        self.num_spatial_latents = atomiser_config.get("spatial_latents", 16) ** 2
        
        self.latent_colors = generate_distinct_colors(self.num_spatial_latents)
        self.marker_size = config["debug"].get("latent_marker_size", 40)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Triggers the visualization at the end of the validation loop."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
            
        pl_module.eval()
        dataset_val = trainer.datamodule.val_dataset
        for i, sample_idx in enumerate(self.sample_indices):
            try:
                self._visualize_sample(sample_idx, i, dataset_val, pl_module, trainer.current_epoch)
            except Exception as e:
                print(f"Token Assignment Viz Error: {e}")
        pl_module.train()

    def _visualize_sample(self, sample_idx, viz_idx, dataset, pl_module, epoch):
        # 1. Fetch data
        result = dataset.get_samples_to_viz(sample_idx)
        image, tokens, mask, queries, q_mask, label, latent_pos, bbox = result[:8]
        
        device = pl_module.device
        with torch.no_grad():
            # 2. Forward pass for trajectory
            out = pl_module.encoder(
                tokens.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device) if mask is not None else None,
                queries.unsqueeze(0).to(device),
                torch.ones(1, queries.shape[0]).to(device),
                latent_pos.unsqueeze(0).to(device),
                training=False, task="visualization", return_trajectory=True
            )
            
            trajectory = out.get('trajectory', [latent_pos.unsqueeze(0)])
            final_coords = trajectory[-1]

            # 3. Get Assignment via Geographic Pruning
            # geo_masks: [1, L, k] -> 0 means token k is assigned to latent L
            _, geo_masks, _ = pl_module.encoder.geo_pruning(
                tokens.unsqueeze(0).to(device), 
                mask.unsqueeze(0).to(device) if mask is not None else None, 
                final_coords
            )
            
            # 4. Map back to 64x64 grid
            x_coords = tokens[:, 1].cpu().numpy()
            y_coords = tokens[:, 2].cpu().numpy()
            geo_masks_np = geo_masks[0].cpu().numpy()
            
            assignment_img = np.zeros((self.image_size, self.image_size, 3))
            coverage_map = np.zeros((self.image_size, self.image_size))

            for l_idx in range(self.num_spatial_latents):
                color = self.latent_colors[l_idx]
                assigned_indices = np.where(geo_masks_np[l_idx] == 0)[0]
                for t_idx in assigned_indices:
                    ix, iy = int(x_coords[t_idx]), int(y_coords[t_idx])
                    if 0 <= ix < self.image_size and 0 <= iy < self.image_size:
                        assignment_img[iy, ix] = color
                        coverage_map[iy, ix] += 1

        # 5. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Original Image
        axes[0].imshow(image.cpu().numpy()[:,:,0], cmap='gray')
        axes[0].set_title(f"Original Digit (Label: {label})")
        
        # Panel 2: Assignment Map
        axes[1].imshow(assignment_img)
        axes[1].set_title("Latent Assignment Map")
        coords_np = final_coords[0].cpu().numpy()
        axes[1].scatter(coords_np[:, 0], coords_np[:, 1], c=self.latent_colors, edgecolors='white', s=self.marker_size)
        
        # Panel 3: Coverage Heatmap (Crucial for debugging 25% accuracy)
        im = axes[2].imshow(coverage_map, cmap='hot')
        axes[2].set_title("Coverage (Latents per Pixel)")
        plt.colorbar(im, ax=axes[2])
        
        for ax in axes: ax.axis('off')
        
        wandb.log({f"assignment/sample_{sample_idx}": wandb.Image(fig)})
        plt.close(fig)