"""
Error Landscape and Gravity Visualization Callback

Visualizes:
1. Predicted error landscape (512x512) - interpolated from latent predictions
2. Gravity magnitude map - strength of gravitational pull (Continuous Field)
3. Gravity direction map - Middlebury color wheel (Continuous Field)

Uses bilinear interpolation from latent positions to create smooth fields,
then computes gravity treating the landscape as a continuous mass density.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Tuple
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter

try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# MIDDLEBURY COLOR WHEEL
# Ported from the classic C implementation used for optical flow visualization
# Color transitions are non-uniform based on perceptual similarity
# =============================================================================

def make_colorwheel() -> np.ndarray:
    """
    Create Middlebury color wheel for optical flow / vector field visualization.
    """
    # Segment lengths (from Middlebury)
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    # Red to Yellow
    for i in range(RY): colorwheel[col] = [255, 255 * i / RY, 0]; col += 1
    # Yellow to Green
    for i in range(YG): colorwheel[col] = [255 - 255 * i / YG, 255, 0]; col += 1
    # Green to Cyan
    for i in range(GC): colorwheel[col] = [0, 255, 255 * i / GC]; col += 1
    # Cyan to Blue
    for i in range(CB): colorwheel[col] = [0, 255 - 255 * i / CB, 255]; col += 1
    # Blue to Magenta
    for i in range(BM): colorwheel[col] = [255 * i / BM, 0, 255]; col += 1
    # Magenta to Red
    for i in range(MR): colorwheel[col] = [255, 0, 255 - 255 * i / MR]; col += 1
    return colorwheel / 255.0

def compute_middlebury_color(fx: np.ndarray, fy: np.ndarray, max_magnitude: float = None) -> np.ndarray:
    """
    Compute Middlebury color encoding for a vector field.
    """
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    if max_magnitude is None:
        max_magnitude = np.percentile(magnitude, 99) + 1e-8
    rad = np.clip(magnitude / max_magnitude, 0, 1)
    angle = np.arctan2(-fy, -fx) / np.pi
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    h, w = fx.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for channel in range(3):
        col0 = colorwheel[k0, channel]
        col1 = colorwheel[k1, channel]
        col = (1 - f) * col0 + f * col1
        col = 1 - rad * (1 - col)
        rgb[:, :, channel] = col
    return rgb

def create_colorwheel_legend(size: int = 100) -> np.ndarray:
    """Create a circular color wheel legend image."""
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    rgb = compute_middlebury_color(x, y, max_magnitude=1.0)
    mask = (x**2 + y**2) > 1
    rgb[mask] = 1.0
    return rgb


class ErrorLandscapeVisualizationCallback(pl.Callback):
    """
    Callback for visualizing error landscape and gravity fields.
    
    UPDATED: Now uses Continuous Field Gravity calculation.
    It interpolates latents to a dense source grid and computes gravity
    from that continuous mass, fixing "voids" between latents.
    
    Supports error_clamp_percentile to handle outliers in visualization.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        
        # Image parameters
        self.image_size = config.get("image_size", 512)
        self.gsd = config.get("gsd", 0.2)
        self.half_extent = (self.image_size * self.gsd) / 2.0
        
        # Gravity parameters
        atomiser_config = config.get("Atomiser", {})
        self.gravity_power = atomiser_config.get("gravity_power", 2.0)
        self.error_offset = atomiser_config.get("error_offset", 0.1)
        self.error_clamp_percentile = atomiser_config.get("error_clamp_percentile", 100.0)
        
        # Viz parameters
        debug_config = config.get("debug", {})
        self.arrow_density = debug_config.get("gravity_arrow_density", 20)
        self.interpolation_method = debug_config.get("interpolation_method", "rbf")
        
        # Continuous Gravity settings
        # Resolution of the "Source Grid" (Mass Density). 64x64 is enough for smooth gravity.
        self.gravity_source_res = 64
        
        print(f"[ErrorLandscapeCallback] Initialized (Continuous Field Mode):")
        print(f"  Samples: {self.sample_indices}")
        print(f"  Image: {self.image_size}px, Extent: ±{self.half_extent:.1f}m")
        print(f"  Source Grid for Gravity: {self.gravity_source_res}x{self.gravity_source_res}")
        if self.error_clamp_percentile < 100.0:
            print(f"  Error Clamp Percentile: {self.error_clamp_percentile}%")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        if not trainer.is_global_zero:
            return
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        if not hasattr(pl_module, 'encoder') or not hasattr(pl_module.encoder, 'gravity_displacement'):
            return
        
        pl_module.eval()
        try:
            if wandb is not None and wandb.run is not None:
                self._generate_visualizations(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping error landscape visualization")
        except Exception as e:
            print(f"Error in ErrorLandscapeVisualizationCallback: {e}")
            import traceback
            traceback.print_exc()
        pl_module.train()
    
    def _generate_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        dataset_val = trainer.datamodule.val_dataset
        for i, sample_idx in enumerate(self.sample_indices):
            try:
                self._visualize_sample(
                    sample_idx=sample_idx, viz_idx=i, dataset=dataset_val,
                    pl_module=pl_module, epoch=trainer.current_epoch, split="val"
                )
            except Exception as e:
                print(f"Error visualizing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    def _visualize_sample(self, sample_idx, viz_idx, dataset, pl_module, epoch, split):
        image, image_tokens, attention_mask, mae_tokens, mask_MAE_res, _, latent_pos, _ = dataset.get_samples_to_viz(sample_idx)
        device = pl_module.device
        
        # Prep batch
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mae_tokens_mask = torch.ones(mae_tokens.shape[0]).to(device)
        latent_pos = latent_pos.to(device)
        
        with torch.no_grad():
            result = pl_module.forward(
                image_tokens.unsqueeze(0),
                attention_mask.unsqueeze(0) if attention_mask is not None else None,
                mae_tokens.clone().unsqueeze(0),
                mae_tokens_mask.unsqueeze(0),
                latent_pos.unsqueeze(0),
                training=False, task="visualization",
            )
            
            trajectory = result.get('trajectory', None)
            predicted_errors = result.get('predicted_errors', None)
            final_coords = result.get('final_coords', None)
            
            if trajectory is None or predicted_errors is None or len(predicted_errors) == 0:
                return
            
            final_positions = final_coords[0].cpu().numpy()
            final_errors = predicted_errors[-1][0].cpu().numpy()
            all_layer_errors = [e[0].cpu().numpy() for e in predicted_errors]
            all_layer_positions = [t[0].cpu().numpy() for t in trajectory]
        
        rgb_background = self._create_rgb_background(image)
        wandb_data = {}
        
        # =====================================================================
        # CLAMP ERRORS ONCE FOR ALL VISUALIZATIONS, THEN NORMALIZE TO [0,1]
        # =====================================================================
        clamped_errors = self._clamp_errors_percentile(final_errors)
        normalized_errors = self._normalize_errors_for_viz(clamped_errors)
        
        all_layer_errors_clamped = [self._clamp_errors_percentile(e) for e in all_layer_errors]
        all_layer_errors_normalized = [self._normalize_errors_for_viz(e) for e in all_layer_errors_clamped]
        
        # 1. Error Landscape (Interpolated Heatmap)
        error_landscape_fig = self._create_error_landscape(
            final_positions, normalized_errors, rgb_background, sample_idx, epoch, len(predicted_errors)-1
        )
        wandb_data[f"error_landscape/sample_{sample_idx}_{split}"] = wandb.Image(error_landscape_fig)
        plt.close(error_landscape_fig)
        
        # 2. Gravity Magnitude (Continuous Field)
        gravity_mag_fig = self._create_gravity_magnitude_map(
            final_positions, normalized_errors, rgb_background, sample_idx, epoch
        )
        wandb_data[f"gravity_magnitude/sample_{sample_idx}_{split}"] = wandb.Image(gravity_mag_fig)
        plt.close(gravity_mag_fig)
        
        # 3. Gravity Direction (Continuous Field)
        gravity_dir_fig = self._create_gravity_direction_map(
            final_positions, normalized_errors, rgb_background, sample_idx, epoch
        )
        wandb_data[f"gravity_direction/sample_{sample_idx}_{split}"] = wandb.Image(gravity_dir_fig)
        plt.close(gravity_dir_fig)
        
        # 4. Combined
        combined_fig = self._create_combined_visualization(
            final_positions, normalized_errors, all_layer_positions, all_layer_errors_normalized,
            rgb_background, sample_idx, epoch
        )
        wandb_data[f"error_gravity_combined/sample_{sample_idx}_{split}"] = wandb.Image(combined_fig)
        plt.close(combined_fig)
        
        wandb.log(wandb_data)
        print(f"✓ Uploaded continuous field viz for sample {sample_idx}")
    
    def _create_pixel_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-self.half_extent, self.half_extent, self.image_size)
        y = np.linspace(-self.half_extent, self.half_extent, self.image_size)
        return np.meshgrid(x, y)
    
    def _clamp_errors_percentile(self, errors: np.ndarray) -> np.ndarray:
        """
        Clamp errors above a given percentile to handle outliers.
        
        Args:
            errors: [L] raw predicted errors
            
        Returns:
            clamped_errors: [L] errors with outliers clamped
        """
        if self.error_clamp_percentile >= 100.0:
            return errors
        
        threshold = np.percentile(errors, self.error_clamp_percentile)
        return np.clip(errors, None, threshold)
    
    def _normalize_errors_for_viz(self, errors: np.ndarray) -> np.ndarray:
        """
        Normalize errors to [0, 1] for visualization after clamping.
        
        Args:
            errors: [L] clamped errors
            
        Returns:
            normalized_errors: [L] errors in [0, 1]
        """
        e_min, e_max = errors.min(), errors.max()
        if e_max - e_min < 1e-8:
            return np.zeros_like(errors)
        return (errors - e_min) / (e_max - e_min)
    
    def _interpolate_to_grid(self, positions, values, grid_size=None, method=None):
        """Interpolate discrete values to a grid."""
        if method is None: method = self.interpolation_method
        if grid_size is None: grid_size = self.image_size
        
        x = np.linspace(-self.half_extent, self.half_extent, grid_size)
        y = np.linspace(-self.half_extent, self.half_extent, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        if method == 'rbf':
            try:
                rbf = RBFInterpolator(positions, values, kernel='thin_plate_spline', smoothing=0.1)
                grid_values = rbf(grid_points)
            except Exception:
                grid_values = griddata(positions, values, grid_points, method='linear', fill_value=values.mean())
        else:
            grid_values = griddata(positions, values, grid_points, method=method, fill_value=values.mean())
        
        return np.nan_to_num(grid_values, nan=values.mean()).reshape(grid_size, grid_size)

    def _compute_continuous_gravity(
        self,
        positions: np.ndarray,  # Discrete latents [L, 2]
        errors: np.ndarray,     # Discrete errors [L] - SHOULD BE CLAMPED AND NORMALIZED TO [0,1]
        query_points: np.ndarray, # Visualization pixels [N, 2]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes gravity treating the error landscape as a CONTINUOUS sheet of mass.
        
        NOTE: errors should already be clamped and normalized to [0, 1].
        
        Strategy:
        1. Interpolate discrete latents to a medium-res 'Source Grid' (e.g., 64x64).
        2. Treat every pixel in the Source Grid as a mass point.
        3. Calculate the sum of forces from all Source pixels to all Query pixels.
        """
        # 1. Create Source Grid (The Masses)
        res = self.gravity_source_res # e.g. 64x64
        
        # Interpolate error landscape to this grid (Linear is fast/stable for sources)
        source_grid_errors = self._interpolate_to_grid(
            positions, errors, grid_size=res, method='linear'
        )
        
        # Create coordinates for the source grid
        x_src = np.linspace(-self.half_extent, self.half_extent, res)
        y_src = np.linspace(-self.half_extent, self.half_extent, res)
        xx_src, yy_src = np.meshgrid(x_src, y_src)
        source_pos = np.stack([xx_src.flatten(), yy_src.flatten()], axis=-1) # [M, 2]
        
        # Normalize mass (Log-MinMax consistent with physics engine)
        source_mass = np.log1p(source_grid_errors.flatten())
        source_mass = (source_mass - source_mass.min()) / (source_mass.max() - source_mass.min() + 1e-8)
        source_mass = source_mass + self.error_offset
        
        # 2. Compute Gravity (Batched to prevent OOM)
        # N queries (262k) vs M sources (4k)
        N = query_points.shape[0]
        
        total_fx = np.zeros(N, dtype=np.float32)
        total_fy = np.zeros(N, dtype=np.float32)
        
        batch_size = 10000 
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            q_batch = query_points[i:end] # [B, 2]
            
            # delta[b, m, 2] = source[m] - query[b]
            delta = source_pos[np.newaxis, :, :] - q_batch[:, np.newaxis, :] # [B, M, 2]
            dist = np.linalg.norm(delta, axis=-1) # [B, M]
            
            # Softening parameter (approx pixel size)
            eps = (2 * self.half_extent / res) * 0.5
            dist = np.clip(dist, eps, None) 
            
            # Gravity Magnitude: Mass / r^p
            force_mag = source_mass[np.newaxis, :] / (dist ** self.gravity_power)
            
            direction = delta / dist[:, :, np.newaxis]
            force_batch = (direction * force_mag[:, :, np.newaxis]).sum(axis=1) # [B, 2]
            
            total_fx[i:end] = force_batch[:, 0]
            total_fy[i:end] = force_batch[:, 1]
            
        magnitude = np.linalg.norm(np.stack([total_fx, total_fy], axis=-1), axis=-1)
        
        return magnitude, total_fx, total_fy

    def _create_error_landscape(self, positions, errors, rgb_background, sample_idx, epoch, layer_idx):
        """
        Create error landscape visualization.
        
        NOTE: errors should already be clamped AND normalized to [0, 1].
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        error_grid = self._interpolate_to_grid(positions, errors)
        error_grid_smooth = gaussian_filter(error_grid, sigma=2)
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        ax.imshow(rgb_background, extent=extent, aspect='equal', alpha=0.3, origin='lower')
        im = ax.imshow(error_grid_smooth, extent=extent, aspect='equal', cmap='hot', alpha=0.7, origin='lower',
                       vmin=0, vmax=1)
        ax.scatter(positions[:, 0], positions[:, 1], c=errors, cmap='hot', s=50, edgecolors='white',
                   vmin=0, vmax=1, zorder=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Pred. Error (normalized)')
        
        title = f'Error Landscape\nSample {sample_idx}, Epoch {epoch}'
        if self.error_clamp_percentile < 100.0:
            title += f' (clamped at {self.error_clamp_percentile}%)'
        ax.set_title(title)
        return fig

    def _create_gravity_magnitude_map(self, positions, errors, rgb_background, sample_idx, epoch):
        """
        Create gravity magnitude visualization.
        
        NOTE: errors should already be clamped and normalized to [0, 1].
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        xx, yy = self._create_pixel_grid()
        query_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        # USE CONTINUOUS GRAVITY
        magnitude, _, _ = self._compute_continuous_gravity(positions, errors, query_points)
        magnitude_grid = magnitude.reshape(self.image_size, self.image_size)
        
        p90 = np.percentile(magnitude_grid, 90)
        magnitude_clamped = np.clip(magnitude_grid, 0, p90)
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        ax.imshow(rgb_background, extent=extent, aspect='equal', alpha=0.3, origin='lower')
        im = ax.imshow(magnitude_clamped, extent=extent, aspect='equal', cmap='plasma', alpha=0.7, origin='lower')
        ax.scatter(positions[:, 0], positions[:, 1], c='white', s=30, edgecolors='black', alpha=0.5, zorder=10)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Gravity (Continuous Field)')
        ax.set_title(f'Gravity Magnitude (Continuous)\nSample {sample_idx}')
        return fig

    def _create_gravity_direction_map(self, positions, errors, rgb_background, sample_idx, epoch):
        """
        Create gravity direction visualization with Middlebury color wheel.
        
        NOTE: errors should already be clamped and normalized to [0, 1].
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        xx, yy = self._create_pixel_grid()
        query_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        # USE CONTINUOUS GRAVITY
        magnitude, fx, fy = self._compute_continuous_gravity(positions, errors, query_points)
        
        fx_grid = fx.reshape(self.image_size, self.image_size)
        fy_grid = fy.reshape(self.image_size, self.image_size)
        mag_grid = magnitude.reshape(self.image_size, self.image_size)
        
        max_mag = np.percentile(mag_grid, 90)
        flow_rgb = compute_middlebury_color(fx_grid, fy_grid, max_magnitude=max_mag)
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        ax.imshow(flow_rgb, extent=extent, aspect='equal', origin='lower')
        
        # Sparse arrows
        n_arrows = self.arrow_density
        step = self.image_size // n_arrows
        sl = slice(step // 2, None, step)
        xx_s, yy_s = xx[sl, sl], yy[sl, sl]
        fx_s, fy_s = fx_grid[sl, sl], fy_grid[sl, sl]
        mag_s = np.sqrt(fx_s**2 + fy_s**2).clip(1e-8, None)
        
        ax.quiver(xx_s, yy_s, fx_s/mag_s, fy_s/mag_s, color='black', scale=30, width=0.003, alpha=0.6)
        ax.scatter(positions[:, 0], positions[:, 1], c='black', s=40, edgecolors='white', marker='o', zorder=10)
        ax.set_title(f'Gravity Direction (Continuous)\nSample {sample_idx}')
        return fig

    def _create_combined_visualization(self, positions, errors, trajectory, all_errors, rgb_background, sample_idx, epoch):
        """
        Create combined 2x2 visualization.
        
        NOTE: errors and all_errors should already be clamped and normalized to [0, 1].
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        extent = [-self.half_extent, self.half_extent, -self.half_extent, self.half_extent]
        
        # 1. Error
        ax1 = axes[0, 0]
        error_grid = self._interpolate_to_grid(positions, errors)
        error_grid_smooth = gaussian_filter(error_grid, sigma=2)
        ax1.imshow(rgb_background, extent=extent, alpha=0.3, origin='lower')
        im1 = ax1.imshow(error_grid_smooth, extent=extent, cmap='hot', alpha=0.7, origin='lower', vmin=0, vmax=1)
        ax1.scatter(positions[:, 0], positions[:, 1], c=errors, cmap='hot', s=40, edgecolors='white', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Error Landscape')

        # Compute Continuous Gravity ONCE
        xx, yy = self._create_pixel_grid()
        query_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        magnitude, fx, fy = self._compute_continuous_gravity(positions, errors, query_points)
        magnitude_grid = magnitude.reshape(self.image_size, self.image_size)
        fx_grid = fx.reshape(self.image_size, self.image_size)
        fy_grid = fy.reshape(self.image_size, self.image_size)

        # 2. Magnitude
        ax2 = axes[0, 1]
        p90 = np.percentile(magnitude_grid, 90)
        mag_clamped = np.clip(magnitude_grid, 0, p90)
        ax2.imshow(rgb_background, extent=extent, alpha=0.3, origin='lower')
        im2 = ax2.imshow(mag_clamped, extent=extent, cmap='plasma', alpha=0.7, origin='lower')
        ax2.scatter(positions[:, 0], positions[:, 1], c='white', s=20, alpha=0.4)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Gravity Magnitude (Continuous)')

        # 3. Direction
        ax3 = axes[1, 0]
        flow_rgb = compute_middlebury_color(fx_grid, fy_grid, max_magnitude=p90)
        ax3.imshow(flow_rgb, extent=extent, origin='lower')
        # Sparse arrows
        n_arrows = self.arrow_density
        step = self.image_size // n_arrows
        sl = slice(step // 2, None, step)
        xx_s, yy_s = xx[sl, sl], yy[sl, sl]
        fx_s, fy_s = fx_grid[sl, sl], fy_grid[sl, sl]
        mag_s = np.sqrt(fx_s**2 + fy_s**2).clip(1e-8, None)
        ax3.quiver(xx_s, yy_s, fx_s/mag_s, fy_s/mag_s, color='black', scale=30, width=0.003, alpha=0.5)
        ax3.set_title('Gravity Direction (Continuous)')

        # 4. Trajectory
        ax4 = axes[1, 1]
        colors = plt.cm.Purples(np.linspace(0.3, 1.0, len(trajectory)))
        ax4.imshow(rgb_background, extent=extent, alpha=0.4, origin='lower')
        num_latents = trajectory[0].shape[0]
        skip = 1 if num_latents < 500 else 5
        for lat_idx in range(0, num_latents, skip):
            path = np.array([traj[lat_idx] for traj in trajectory])
            ax4.plot(path[:, 0], path[:, 1], color='purple', linewidth=0.5, alpha=0.3)
        ax4.scatter(trajectory[-1][:, 0], trajectory[-1][:, 1], c='white', s=20, edgecolors='black')
        ax4.set_title('Trajectories')

        fig.suptitle(f'Continuous Gravity Analysis\nSample {sample_idx}', fontsize=16)
        plt.tight_layout()
        return fig

    def _create_rgb_background(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] >= 3:
            rgb = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)
        else:
            single = image[:, :, 0] if len(image.shape) == 3 else image
            rgb = np.stack([single] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb