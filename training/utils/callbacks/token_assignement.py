"""
Token-to-Latent Assignment Visualization for Sen1Floods11
Simplified version that avoids triggering spectral encoder
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytorch_lightning as pl
from typing import List
import os
import gc

try:
    import wandb
except ImportError:
    wandb = None


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
    
    Creates a 512x512 image where each pixel is colored by its assigned latent.
    """
    
    def __init__(
        self,
        modality: str = "SENFLOOD",
        log_every_n_epochs: int = 5,
        sample_indices: List[int] = [0],
        save_dir: str = "./viz_token_assignment",
        use_wandb: bool = True,
        image_size: int = 512,
    ):
        super().__init__()
        self.modality = modality
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_indices = sample_indices
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.image_size = image_size
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.latent_colors = None
        self.num_latents = None
        
        print(f"[TokenAssignmentCallback] Modality: {modality}, Image: {image_size}x{image_size}")
    
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
                return
            
            batch = next(iter(val_loader))
            
            sample_idx = self.sample_indices[0] if self.sample_indices else 0
            if sample_idx >= batch[0].shape[0]:
                sample_idx = 0
            
            # Extract single sample and clone to CPU
            image_tokens = batch[0][sample_idx].clone()
            attention_mask = batch[1][sample_idx].clone()
            label = batch[4][sample_idx].clone().cpu()
            image = batch[6][sample_idx].clone().cpu()
            
            del batch
            gc.collect()
            
            self._visualize(
                sample_idx, image_tokens, attention_mask,
                label, image, pl_module, trainer.current_epoch
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
        pl_module: pl.LightningModule,
        epoch: int,
    ):
        """Create visualization with simple index-to-pixel mapping."""
        device = pl_module.device
        encoder = pl_module.encoder
        
        H, W = self.image_size, self.image_size
        
        # Get x,y offset from original tokens (all tokens from same image share same offset)
        # Tokens have x,y in range [offset, offset + 511] e.g., [512, 1023]
        x_offset = int(image_tokens[:, 1].min().item())
        y_offset = int(image_tokens[:, 2].min().item())
        
        print(f"[DEBUG] Token x range: [{image_tokens[:, 1].min():.0f}, {image_tokens[:, 1].max():.0f}]")
        print(f"[DEBUG] Token y range: [{image_tokens[:, 2].min():.0f}, {image_tokens[:, 2].max():.0f}]")
        print(f"[DEBUG] Offsets: x={x_offset}, y={y_offset}")
        
        with torch.no_grad():
            grid_config = encoder.config["latent_grids"][self.modality]
            latent_coords = encoder.get_default_coords(1, device, grid_config)  # [1, L, 2] in meters
            num_latents = latent_coords.shape[1]
            
            if self.latent_colors is None or self.num_latents != num_latents:
                self.num_latents = num_latents
                self.latent_colors = generate_distinct_colors(num_latents)
            
            # Run geo_pruning
            geo_pruning_fn = encoder.geo_pruning[self.modality]
            geo_tokens, geo_masks, _ = geo_pruning_fn(
                image_tokens.unsqueeze(0).to(device),
                attention_mask.unsqueeze(0).to(device),
                latent_coords,
                id_modality=self.modality
            )
            
            # Get raw x,y indices from geo_tokens
            tx = geo_tokens[0, :, :, 1].cpu().numpy()  # [L, k]
            ty = geo_tokens[0, :, :, 2].cpu().numpy()  # [L, k]
            geo_masks_np = geo_masks[0].cpu().numpy()  # [L, k]
            
            # Latent coords are in meters - convert to pixels
            # Use the grid config to get physical extent
            span = grid_config["span"]  # Total span in meters
            latent_meters = latent_coords[0].cpu().numpy()  # [L, 2]
            
            # Latents are in [-span/2, +span/2], map to [0, 511]
            latent_px = (latent_meters[:, 0] + span / 2) / span * (W - 1)
            latent_py = (latent_meters[:, 1] + span / 2) / span * (H - 1)
            
            print(f"[DEBUG] Latent meters x range: [{latent_meters[:, 0].min():.1f}, {latent_meters[:, 0].max():.1f}]")
            print(f"[DEBUG] Latent meters y range: [{latent_meters[:, 1].min():.1f}, {latent_meters[:, 1].max():.1f}]")
            print(f"[DEBUG] Latent pixels x range: [{latent_px.min():.1f}, {latent_px.max():.1f}]")
            print(f"[DEBUG] Latent pixels y range: [{latent_py.min():.1f}, {latent_py.max():.1f}]")
            
            del geo_tokens, geo_masks, latent_coords
            torch.cuda.empty_cache()
        
        # Build assignment maps
        L, k = tx.shape
        
        assignment_count = np.zeros((H, W), dtype=np.int32)
        primary_latent = np.full((H, W), -1, dtype=np.int32)
        
        valid_count = 0
        oob_count = 0
        
        for l_idx in range(L):
            for t_idx in range(k):
                if geo_masks_np[l_idx, t_idx]:
                    continue
                
                # Simple: subtract offset to get pixel coordinates
                px = int(tx[l_idx, t_idx]) - x_offset
                py = int(ty[l_idx, t_idx]) - y_offset
                
                if 0 <= px < W and 0 <= py < H:
                    assignment_count[py, px] += 1
                    if primary_latent[py, px] == -1:
                        primary_latent[py, px] = l_idx
                    valid_count += 1
                else:
                    oob_count += 1
        
        print(f"[DEBUG] Valid tokens: {valid_count}, Out of bounds: {oob_count}")
        
        # Build color map
        assignment_color = np.ones((H, W, 3), dtype=np.float32) * 0.9
        for y in range(H):
            for x in range(W):
                if primary_latent[y, x] >= 0:
                    assignment_color[y, x] = self.latent_colors[primary_latent[y, x]]
        
        # Create figure
        self._create_figure(
            image.numpy(), label.numpy(),
            assignment_color, assignment_count,
            latent_px, latent_py,
            num_latents, k, H, W,
            sample_idx, epoch
        )
    
    def _create_figure(
        self,
        image_np, label_np,
        assignment_color, assignment_count,
        latent_px, latent_py,
        num_latents, k, H, W,
        sample_idx, epoch
    ):
        """Create and save the 5-panel figure."""
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # 1. RGB (S2 bands B4, B3, B2 = indices 3, 2, 1)
        ax = axes[0]
        if image_np.shape[0] >= 4:
            rgb = np.stack([image_np[3], image_np[2], image_np[1]], axis=-1)
        else:
            rgb = np.stack([image_np[0]] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title('RGB (S2)')
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
        ax.set_title('SAR (VV/VH)')
        ax.axis('off')
        
        # 3. Labels
        ax = axes[2]
        label_rgb = np.zeros((H, W, 3))
        label_rgb[label_np == 0] = [0.2, 0.4, 0.8]   # Blue = no flood
        label_rgb[label_np == 1] = [0.9, 0.2, 0.2]   # Red = flood
        label_rgb[label_np == 255] = [0.5, 0.5, 0.5] # Gray = ignore
        ax.imshow(label_rgb)
        n_flood = (label_np == 1).sum()
        ax.set_title(f'Labels (Flood: {n_flood})')
        ax.axis('off')
        
        # 4. Assignment Map (512x512, colored by latent)
        ax = axes[3]
        ax.imshow(assignment_color)
        ax.scatter(latent_px, latent_py, c=self.latent_colors,
                  s=20, edgecolors='white', linewidths=0.5)
        covered = (assignment_count > 0).sum()
        total = H * W
        ax.set_title(f'Assignment ({100*covered/total:.1f}% covered)')
        ax.axis('off')
        
        # 5. Overlap Heatmap
        ax = axes[4]
        im = ax.imshow(assignment_count, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(latent_px, latent_py, c='cyan', s=15, edgecolors='white', linewidths=0.3)
        ax.set_title(f'Overlap (max: {assignment_count.max()})')
        ax.axis('off')
        
        # Title
        grid_size = int(np.sqrt(num_latents))
        fig.suptitle(
            f'Epoch {epoch}, Sample {sample_idx} | '
            f'Latents: {grid_size}×{grid_size}, k={k}, Image: {H}×{W}',
            fontsize=12
        )
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, f'e{epoch:03d}_s{sample_idx}.png')
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[TokenAssignmentCallback] Saved: {save_path}")
        
        # Stats
        print(f"  Coverage: {100*covered/total:.1f}% ({covered}/{total} pixels)")
        print(f"  Max overlap: {assignment_count.max()}")
        
        # Wandb
        if self.use_wandb and wandb is not None and wandb.run is not None:
            try:
                wandb.log({f"token_assignment/sample_{sample_idx}": wandb.Image(fig)})
            except:
                pass
        
        plt.close(fig)
        plt.close('all')