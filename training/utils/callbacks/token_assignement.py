"""
Token-to-Latent Assignment Visualization for Sen1Floods11
Updated for dynamic grid configuration (no modality strings)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytorch_lightning as pl
from typing import List, Optional
import os
import gc
import math

try:
    import wandb
except ImportError:
    wandb = None

# Import the grid config function
from training.utils.datasets.token_grouping import compute_grid_config


def generate_distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate n visually distinct colors."""
    np.random.seed(seed)
    golden_ratio = 0.618033988749895
    hues = np.mod(np.arange(n) * golden_ratio + np.random.random(), 1.0)
    saturations = np.random.uniform(0.7, 1.0, n)
    values = np.random.uniform(0.7, 1.0, n)
    
    colors = np.zeros((n, 3))
    for i in range(n):
        colors[i] = mcolors.hsv_to_rgb([hues[i], saturations[i], values[i]])
    return colors


class TokenAssignmentCallbackSenFlood(pl.Callback):
    """
    Visualizes token-to-latent assignments.
    
    Creates a 6-panel figure:
    1. RGB composite (S2)
    2. SAR composite (VV/VH)
    3. Ground truth labels
    4. Voronoi cells (theoretical 100% coverage)
    5. Sampled tokens (actual geo_k tokens selected)
    6. Overlap heatmap
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 5,
        sample_indices: List[int] = [0],
        save_dir: str = "./viz_token_assignment",
        use_wandb: bool = True,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_indices = sample_indices
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.latent_colors = None
        self.num_latents = None
        
        print(f"[TokenAssignmentCallback] Dynamic grid mode")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if trainer.global_rank != 0:
            return
        
        try:
            self._run_safe(trainer, pl_module)
        except Exception as e:
            print(f"[TokenAssignmentCallback] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_safe(self, trainer, pl_module):
        """Run visualization with proper cleanup."""
        pl_module.eval()
        
        try:
            val_loader = trainer.val_dataloaders
            if val_loader is None:
                print("[TokenAssignmentCallback] No validation dataloader found")
                return
            
            batch = next(iter(val_loader))
            
            # Handle both old (tuple) and new (dict) batch formats
            if isinstance(batch, dict):
                # New dictionary format
                groups = batch["groups"]
                first_res = sorted(groups.keys())[0]
                group = groups[first_res]
                
                tokens = group["tokens"]
                masks = group["mask"]
                
                B = tokens.shape[0]
                sample_idx = self.sample_indices[0] if self.sample_indices else 0
                if sample_idx >= B:
                    sample_idx = 0
                
                image_tokens = tokens[sample_idx].clone()
                attention_mask = masks[sample_idx].clone()
                label = batch["label"][sample_idx].clone().cpu()
                image = batch["image"][sample_idx].clone().cpu()
                resolution = batch.get("target_resolution", first_res)
                
                if isinstance(resolution, torch.Tensor):
                    resolution = resolution.item()
            else:
                # Old tuple format (fallback)
                sample_idx = self.sample_indices[0] if self.sample_indices else 0
                if sample_idx >= batch[0].shape[0]:
                    sample_idx = 0
                
                image_tokens = batch[0][sample_idx].clone()
                attention_mask = batch[1][sample_idx].clone()
                label = batch[4][sample_idx].clone().cpu()
                image = batch[6][sample_idx].clone().cpu()
                resolution = batch[7][sample_idx].item() if len(batch) > 7 else 10.0
            
            del batch
            gc.collect()
            
            self._visualize(
                sample_idx, image_tokens, attention_mask,
                label, image, resolution, pl_module, trainer.current_epoch
            )
            
        finally:
            pl_module.train()
            gc.collect()
            torch.cuda.empty_cache()
    
    def _visualize(
        self,
        sample_idx: int,
        image_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        image: torch.Tensor,
        resolution: float,
        pl_module: pl.LightningModule,
        epoch: int,
    ):
        """Create visualization using dynamic grid config."""
        device = pl_module.device
        
        # =========================================================================
        # Find the actual Atomiser model (may be nested)
        # =========================================================================
        model = None
        
        # Try common attribute names
        for attr in ['model', 'encoder', 'atomiser', 'net']:
            if hasattr(pl_module, attr):
                candidate = getattr(pl_module, attr)
                if hasattr(candidate, 'geo_pruning') and hasattr(candidate, '_compute_latent_grid'):
                    model = candidate
                    break
        
        # If not found, check if pl_module itself is the model
        if model is None:
            if hasattr(pl_module, 'geo_pruning') and hasattr(pl_module, '_compute_latent_grid'):
                model = pl_module
            else:
                print("[TokenAssignmentCallback] Could not find Atomiser model")
                return
        
        # =========================================================================
        # Get grid parameters (with fallbacks)
        # =========================================================================
        pixels_per_latent = getattr(model, 'pixels_per_latent', 50)
        sigma_factor = getattr(model, 'sigma_factor', 1.5)
        max_k = getattr(model, 'max_k', 2000)
        hexagonal = getattr(model, 'hexagonal', False)
        
        # Alternative: try to get from config
        if hasattr(model, 'config'):
            latent_cfg = model.config.get('latent_grid', {})
            pixels_per_latent = latent_cfg.get('pixels_per_latent', pixels_per_latent)
            sigma_factor = latent_cfg.get('sigma_factor', sigma_factor)
            max_k = latent_cfg.get('max_k', max_k)
            hexagonal = latent_cfg.get('hexagonal', hexagonal)
        
        print(f"[DEBUG] Grid params: pixels_per_latent={pixels_per_latent}, "
            f"sigma_factor={sigma_factor}, max_k={max_k}, hexagonal={hexagonal}")
        
        # =========================================================================
        # Infer image dimensions from tokens
        # =========================================================================
        x_coords = image_tokens[:, 1]
        y_coords = image_tokens[:, 2]
        x_offset = int(x_coords.min().item())
        y_offset = int(y_coords.min().item())
        W = int(x_coords.max().item()) - x_offset + 1
        H = int(y_coords.max().item()) - y_offset + 1
        
        # Infer number of channels from unique spectral indices
        C = int(image_tokens[:, 3].max().item()) + 1
        
        print(f"[DEBUG] Image: {H}×{W}, Channels: {C}, Resolution: {resolution}m")
        print(f"[DEBUG] Token x range: [{x_coords.min():.0f}, {x_coords.max():.0f}]")
        print(f"[DEBUG] Token y range: [{y_coords.min():.0f}, {y_coords.max():.0f}]")
        print(f"[DEBUG] Offsets: x={x_offset}, y={y_offset}")
        
        with torch.no_grad():
            # Compute grid config dynamically
            grid_config = compute_grid_config(
                resolution=resolution,
                shape=(C, H, W),
                pixels_per_latent=pixels_per_latent,
                sigma_factor=sigma_factor,
                max_k=max_k,
                hexagonal=hexagonal,
            )
            
            num_latents = grid_config["L_spatial"]
            lx = grid_config["latents_x"]
            ly = grid_config["latents_y"]
            geo_k = grid_config["geo_k"]
            geo_sigma = grid_config["geo_sigma"]
            span_x = grid_config["span_x"]
            span_y = grid_config["span_y"]
            hexagonal = grid_config.get("hexagonal", False)
            
            print(f"[DEBUG] Grid: {lx}×{ly} = {num_latents} latents")
            print(f"[DEBUG] geo_k: {geo_k}, geo_sigma: {geo_sigma:.1f}m")
            print(f"[DEBUG] Span: {span_x:.1f}×{span_y:.1f}m, Hexagonal: {hexagonal}")
            
            # Generate colors for latents
            if self.latent_colors is None or self.num_latents != num_latents:
                self.num_latents = num_latents
                self.latent_colors = generate_distinct_colors(num_latents)
            
            # Compute latent grid using model's method
            latent_coords = model._compute_latent_grid(grid_config, 1, device)  # [1, L, 2]
            
            # Run geo_pruning with dynamic parameters
            geo_tokens, geo_masks, _ = model.geo_pruning(
                image_tokens.unsqueeze(0).to(device),
                attention_mask.unsqueeze(0).to(device),
                latent_coords,
                geo_k=geo_k,
                sigma=geo_sigma,
                L_spatial=num_latents,
                hexagonal=hexagonal,  # ← ADD THIS
            )
            
            # Extract results
            tx = geo_tokens[0, :, :, 1].cpu().numpy()  # [L, k]
            ty = geo_tokens[0, :, :, 2].cpu().numpy()  # [L, k]
            geo_masks_np = geo_masks[0].cpu().numpy()  # [L, k]
            
            # Convert latent coords from meters to pixels
            latent_meters = latent_coords[0].cpu().numpy()  # [L, 2]
            latent_px = (latent_meters[:, 0] + span_x / 2) / span_x * (W - 1)
            latent_py = (latent_meters[:, 1] + span_y / 2) / span_y * (H - 1)
            
            print(f"[DEBUG] Latent meters x: [{latent_meters[:, 0].min():.1f}, {latent_meters[:, 0].max():.1f}]")
            print(f"[DEBUG] Latent meters y: [{latent_meters[:, 1].min():.1f}, {latent_meters[:, 1].max():.1f}]")
            print(f"[DEBUG] Latent pixels x: [{latent_px.min():.1f}, {latent_px.max():.1f}]")
            print(f"[DEBUG] Latent pixels y: [{latent_py.min():.1f}, {latent_py.max():.1f}]")
            
            # Compute Voronoi cells for ALL pixels
            voronoi_assignment = self._compute_voronoi_assignment(
                H, W, latent_meters, span_x, span_y, resolution
            )
            
            del geo_tokens, geo_masks, latent_coords
            torch.cuda.empty_cache()
        
        # Build sampled token maps
        L, k = tx.shape
        
        sampled_count = np.zeros((H, W), dtype=np.int32)
        sampled_latent = np.full((H, W), -1, dtype=np.int32)
        
        valid_count = 0
        oob_count = 0
        
        for l_idx in range(L):
            for t_idx in range(k):
                if geo_masks_np[l_idx, t_idx]:
                    continue
                
                px = int(tx[l_idx, t_idx]) - x_offset
                py = int(ty[l_idx, t_idx]) - y_offset
                
                if 0 <= px < W and 0 <= py < H:
                    sampled_count[py, px] += 1
                    if sampled_latent[py, px] == -1:
                        sampled_latent[py, px] = l_idx
                    valid_count += 1
                else:
                    oob_count += 1
        
        print(f"[DEBUG] Valid tokens: {valid_count}, Out of bounds: {oob_count}")
        
        # Create figure
        self._create_figure(
            image.numpy(), label.numpy(),
            voronoi_assignment, sampled_latent, sampled_count,
            latent_px, latent_py,
            grid_config, H, W,
            sample_idx, epoch
        )
    
    def _compute_voronoi_assignment(
        self,
        H: int,
        W: int,
        latent_meters: np.ndarray,
        span_x: float,
        span_y: float,
        resolution: float,
    ) -> np.ndarray:
        """
        Compute Voronoi cell assignment for every pixel.
        
        Returns:
            assignment: [H, W] with latent index for each pixel
        """
        # Create pixel coordinate grid in meters
        px_x = np.linspace(-span_x / 2 + resolution / 2, span_x / 2 - resolution / 2, W)
        px_y = np.linspace(-span_y / 2 + resolution / 2, span_y / 2 - resolution / 2, H)
        grid_x, grid_y = np.meshgrid(px_x, px_y)  # [H, W]
        
        # Compute distance to each latent and find nearest
        assignment = np.zeros((H, W), dtype=np.int32)
        min_dist = np.full((H, W), np.inf)
        
        for l_idx in range(latent_meters.shape[0]):
            lx, ly = latent_meters[l_idx]
            dist_sq = (grid_x - lx) ** 2 + (grid_y - ly) ** 2
            
            closer = dist_sq < min_dist
            assignment[closer] = l_idx
            min_dist[closer] = dist_sq[closer]
        
        return assignment
    
    def _create_figure(
        self,
        image_np: np.ndarray,
        label_np: np.ndarray,
        voronoi_assignment: np.ndarray,
        sampled_latent: np.ndarray,
        sampled_count: np.ndarray,
        latent_px: np.ndarray,
        latent_py: np.ndarray,
        grid_config: dict,
        H: int,
        W: int,
        sample_idx: int,
        epoch: int,
    ):
        """Create and save the 6-panel figure."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        num_latents = grid_config["L_spatial"]
        lx = grid_config["latents_x"]
        ly = grid_config["latents_y"]
        geo_k = grid_config["geo_k"]
        hexagonal = grid_config.get("hexagonal", False)
        
        # =====================================================================
        # Row 1: Input data
        # =====================================================================
        
        # 1. RGB (S2 bands B4, B3, B2 = indices 3, 2, 1)
        ax = axes[0]
        if image_np.shape[0] >= 4:
            rgb = np.stack([image_np[3], image_np[2], image_np[1]], axis=-1)
        else:
            rgb = np.stack([image_np[0]] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title('RGB (S2 B4-B3-B2)', fontsize=11)
        ax.axis('off')
        
        # 2. SAR (VV, VH = indices 13, 14)
        ax = axes[1]
        if image_np.shape[0] >= 15:
            vv = image_np[13]
            vh = image_np[14]
            vv_n = (vv - vv.min()) / (vv.max() - vv.min() + 1e-8)
            vh_n = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)
            sar = np.stack([vv_n, vh_n, (vv_n + vh_n) / 2], axis=-1)
            ax.imshow(np.clip(sar, 0, 1))
        ax.set_title('SAR (VV/VH)', fontsize=11)
        ax.axis('off')
        
        # 3. Labels
        ax = axes[2]
        label_rgb = np.zeros((H, W, 3))
        label_rgb[label_np == 0] = [0.2, 0.4, 0.8]   # Blue = no flood
        label_rgb[label_np == 1] = [0.9, 0.2, 0.2]   # Red = flood
        label_rgb[label_np == 255] = [0.5, 0.5, 0.5] # Gray = ignore
        ax.imshow(label_rgb)
        n_flood = (label_np == 1).sum()
        ax.set_title(f'Ground Truth (Flood: {n_flood} px)', fontsize=11)
        ax.axis('off')
        
        # =====================================================================
        # Row 2: Token assignment visualization
        # =====================================================================
        
        # 4. Voronoi cells (100% coverage, theoretical)
        ax = axes[3]
        voronoi_color = self.latent_colors[voronoi_assignment]
        ax.imshow(voronoi_color)
        ax.scatter(latent_px, latent_py, c='white', s=30, edgecolors='black', linewidths=0.8, zorder=10)
        ax.set_title(f'Voronoi Cells (100% coverage)', fontsize=11)
        ax.axis('off')
        
        # 5. Sampled tokens (actual geo_k tokens)
        ax = axes[4]
        sampled_color = np.ones((H, W, 3), dtype=np.float32) * 0.9  # Light gray background
        mask = sampled_latent >= 0
        sampled_color[mask] = self.latent_colors[sampled_latent[mask]]
        ax.imshow(sampled_color)
        ax.scatter(latent_px, latent_py, c='white', s=30, edgecolors='black', linewidths=0.8, zorder=10)
        sampled_coverage = mask.sum() / (H * W) * 100
        ax.set_title(f'Sampled Tokens ({sampled_coverage:.1f}% coverage, k={geo_k})', fontsize=11)
        ax.axis('off')
        
        # 6. Overlap Heatmap
        ax = axes[5]
        im = ax.imshow(sampled_count, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(latent_px, latent_py, c='cyan', s=20, edgecolors='white', linewidths=0.5, zorder=10)
        ax.set_title(f'Overlap Heatmap (max: {sampled_count.max()})', fontsize=11)
        ax.axis('off')
        
        # =====================================================================
        # Title and save
        # =====================================================================
        
        grid_type = "Hexagonal" if hexagonal else "Square"
        fig.suptitle(
            f'Epoch {epoch}, Sample {sample_idx} | '
            f'{grid_type} Grid: {lx}×{ly} = {num_latents} latents | '
            f'Image: {H}×{W} | k={geo_k}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        save_path = os.path.join(self.save_dir, f'e{epoch:03d}_s{sample_idx}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[TokenAssignmentCallback] Saved: {save_path}")
        
        # Stats
        print(f"  Voronoi coverage: 100%")
        print(f"  Sampled coverage: {sampled_coverage:.1f}%")
        print(f"  Max overlap: {sampled_count.max()}")
        
        # Wandb
        if self.use_wandb and wandb is not None and wandb.run is not None:
            try:
                wandb.log({
                    f"token_assignment/sample_{sample_idx}": wandb.Image(fig),
                    f"token_assignment/sampled_coverage": sampled_coverage,
                    f"token_assignment/max_overlap": sampled_count.max(),
                })
            except:
                pass
        
        plt.close(fig)
        plt.close('all')
