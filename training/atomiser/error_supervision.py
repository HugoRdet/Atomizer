"""
Error Supervision Module for Error-Guided Displacement (v2)

Changes from v1:
- Added channel subsampling to reduce VRAM while allowing larger spatial grids
- Fixed spatial_latents extraction from latents tensor
- Removed position from error predictor (features-only)

Usage:
    error_module = ErrorSupervisionModule(geometry, config)
    actual_errors = error_module.compute_actual_error(
        trajectory=trajectory,
        latents=latents,
        final_coords=final_coords,
        image_err=image_err,
        model=model,
        stable_depth=2,
        num_channels_to_sample=2,  # NEW: sample 2 random channels per position
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class ErrorSupervisionModule(nn.Module):
    """
    Module to compute actual reconstruction error for error predictor supervision.
    
    Supports channel subsampling to reduce memory usage while allowing
    larger spatial grids for better error estimation.
    """
    
    def __init__(
        self,
        geometry,  # SensorGeometry instance
        grid_size: int = 3,
        spacing: int = 2,
        image_size: int = 512,
        gsd: float = 0.2,
        num_channels: int = 5,
        num_channels_to_sample: Optional[int] = None,  # NEW
    ):
        """
        Args:
            geometry: SensorGeometry instance for coordinate conversion
            grid_size: Size of sampling grid (e.g., 3 → 3×3 = 9 samples)
            spacing: Spacing between grid points in pixels
            image_size: Image dimension in pixels
            gsd: Ground sample distance in meters/pixel
            num_channels: Total number of spectral channels
            num_channels_to_sample: If set, randomly sample this many channels.
                Allows larger grid_size with same VRAM budget.
        """
        super().__init__()
        self.geometry = geometry
        self.grid_size = grid_size
        self.spacing = spacing
        self.image_size = image_size
        self.gsd = gsd
        self.num_channels = num_channels
        self.num_channels_to_sample = num_channels_to_sample
        self.num_samples_per_position = grid_size ** 2
        
        # Log memory savings
        if num_channels_to_sample is not None:
            ratio = num_channels_to_sample / num_channels
            print(f"[ErrorSupervision] Channel subsampling: {num_channels_to_sample}/{num_channels} "
                  f"({ratio:.1%} of channels, {1/ratio:.1f}x larger grid possible)")
    
    def _subsample_channels(
        self,
        query_tokens: torch.Tensor,
        ground_truth: torch.Tensor,
        num_positions: int,
        num_channels: int,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly subsample channels from query tokens and ground truth.
        
        Assumes tokens are organized as:
            [pos0_ch0, pos0_ch1, ..., pos0_chC, pos1_ch0, pos1_ch1, ...]
        
        Args:
            query_tokens: [B, num_positions * num_channels, token_dim]
            ground_truth: [B, num_positions * num_channels]
            num_positions: Number of spatial positions (L * num_samples)
            num_channels: Total number of channels (C)
            num_to_sample: Number of channels to randomly select (k)
        
        Returns:
            query_tokens_sampled: [B, num_positions * num_to_sample, token_dim]
            ground_truth_sampled: [B, num_positions * num_to_sample]
        """
        B = query_tokens.shape[0]
        device = query_tokens.device
        token_dim = query_tokens.shape[-1]
        
        # Reshape to separate positions and channels
        # [B, num_positions * C, D] → [B, num_positions, C, D]
        query_tokens = query_tokens.view(B, num_positions, num_channels, token_dim)
        ground_truth = ground_truth.view(B, num_positions, num_channels)
        
        # Random channel indices (same for all positions in batch, different per batch item)
        # For simplicity, use same channels across batch (can be made per-item if needed)
        channel_indices = torch.randperm(num_channels, device=device)[:num_to_sample]
        channel_indices = channel_indices.sort().values  # Keep order for consistency
        
        # Select channels
        # [B, num_positions, k, D]
        query_tokens_sampled = query_tokens[:, :, channel_indices, :]
        ground_truth_sampled = ground_truth[:, :, channel_indices]
        
        # Reshape back to flat
        # [B, num_positions * k, D]
        query_tokens_sampled = query_tokens_sampled.reshape(B, num_positions * num_to_sample, token_dim)
        ground_truth_sampled = ground_truth_sampled.reshape(B, num_positions * num_to_sample)
        
        return query_tokens_sampled, ground_truth_sampled
    
    def compute_actual_error(
        self,
        trajectory: List[torch.Tensor],
        latents: torch.Tensor,
        final_coords: torch.Tensor,
        image_err: torch.Tensor,
        model,
        layers_to_supervise: Optional[List[int]] = None,
        stable_depth: int = 0,
        num_channels_to_sample: Optional[int] = None,  # Can override instance default
    ) -> torch.Tensor:
        """
        Compute actual reconstruction error at latent positions for each layer.
        
        Args:
            trajectory: List of [B, L, 2] positions in meters, one per layer
            latents: [B, L_total, D] final encoded latents (may include global latents)
            final_coords: [B, L_spatial, 2] final latent positions
            image_err: [B, C, H, W, 6] full image with metadata
            model: Atomiser model (used for decoding)
            layers_to_supervise: Optional list of layer indices to supervise
            stable_depth: Number of final layers without displacement
            num_channels_to_sample: Override instance setting for channel sampling
        
        Returns:
            actual_errors: [B, num_displacement_layers, L] reconstruction error per latent per layer
        """
        B = latents.shape[0]
        depth = len(trajectory) - 1
        L = trajectory[0].shape[1]  # Number of SPATIAL latents
        device = latents.device
        C = self.num_channels
        
        # Use instance default if not overridden
        if num_channels_to_sample is None:
            num_channels_to_sample = self.num_channels_to_sample
        
        # Effective channels for reshape
        C_effective = num_channels_to_sample if num_channels_to_sample else C
        
        # Extract spatial latents only (in case latents includes global latents)
        spatial_latents = latents[:, :L, :]  # [B, L, D]
        
        if layers_to_supervise is None:
            num_displacement_layers = depth - stable_depth
            layers_to_supervise = list(range(num_displacement_layers))
        
        all_layer_errors = []
        
        for layer_idx in layers_to_supervise:
            layer_coords = trajectory[layer_idx]  # [B, L, 2] in meters
            
            # Step 1: Convert meters to pixels
            positions_pixels = self.geometry.meters_to_pixels(
                layer_coords,
                image_size=self.image_size,
                gsd=self.gsd,
            )
            
            # Step 2: Sample grid around each latent position
            sample_coords = self.geometry.sample_grid_around_positions(
                positions_pixels,
                grid_size=self.grid_size,
                spacing=self.spacing,
                image_size=self.image_size,
            )  # [B, L, num_samples, 2]
            
            num_samples = sample_coords.shape[-2]
            num_positions = L * num_samples
            
            # Step 3: Extract query tokens from image
            sample_coords_flat = sample_coords.view(B, -1, 2)
            
            query_tokens, ground_truth, _ = self.geometry.extract_query_tokens_from_image(
                image_err,
                sample_coords_flat,
            )
            # query_tokens: [B, num_positions * C, 6]
            # ground_truth: [B, num_positions * C]
            
            # Step 3.5: Subsample channels if requested
            if num_channels_to_sample is not None and num_channels_to_sample < C:
                query_tokens, ground_truth = self._subsample_channels(
                    query_tokens,
                    ground_truth,
                    num_positions=num_positions,
                    num_channels=C,
                    num_to_sample=num_channels_to_sample,
                )
                C_effective = num_channels_to_sample
            else:
                C_effective = C
            
            # Step 4: Decode
            query_mask = torch.zeros(query_tokens.shape[:2], device=device)
            
            predictions = model.reconstruct(
                spatial_latents,  # Use spatial latents only
                layer_coords,
                query_tokens,
                query_mask,
            )
            
            # Handle output shape
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            elif predictions.dim() == 3:
                predictions = predictions[..., 0]
            
            # Step 5: Compute MSE per query
            errors = (predictions - ground_truth) ** 2
            
            # Step 6: Reshape and aggregate per latent
            errors = errors.view(B, L, num_samples, C_effective)
            layer_errors = errors.mean(dim=(-1, -2))  # [B, L]
            
            all_layer_errors.append(layer_errors)
        
        actual_errors = torch.stack(all_layer_errors, dim=1)
        
        return actual_errors
    
    def compute_error_at_single_layer(
        self,
        positions: torch.Tensor,
        latents: torch.Tensor,
        latent_coords: torch.Tensor,
        image_err: torch.Tensor,
        model,
        num_channels_to_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute error at a single set of positions (for debugging/visualization).
        
        Args:
            positions: [B, L, 2] positions in meters to sample around
            latents: [B, L, D] latents to use for decoding
            latent_coords: [B, L, 2] latent positions for k-NN lookup
            image_err: [B, C, H, W, 6]
            model: Atomiser model
            num_channels_to_sample: Override channel sampling
        
        Returns:
            errors: [B, L] error per latent
        """
        B, L, _ = positions.shape
        device = latents.device
        C = self.num_channels
        
        if num_channels_to_sample is None:
            num_channels_to_sample = self.num_channels_to_sample
        
        # Convert to pixels
        positions_pixels = self.geometry.meters_to_pixels(
            positions, image_size=self.image_size, gsd=self.gsd
        )
        
        # Sample grid
        sample_coords = self.geometry.sample_grid_around_positions(
            positions_pixels,
            grid_size=self.grid_size,
            spacing=self.spacing,
            image_size=self.image_size,
        )
        
        num_samples = sample_coords.shape[-2]
        num_positions = L * num_samples
        
        # Extract query tokens
        sample_coords_flat = sample_coords.view(B, -1, 2)
        query_tokens, ground_truth, _ = self.geometry.extract_query_tokens_from_image(
            image_err, sample_coords_flat
        )
        
        # Subsample channels
        if num_channels_to_sample is not None and num_channels_to_sample < C:
            query_tokens, ground_truth = self._subsample_channels(
                query_tokens, ground_truth,
                num_positions=num_positions,
                num_channels=C,
                num_to_sample=num_channels_to_sample,
            )
            C_effective = num_channels_to_sample
        else:
            C_effective = C
        
        # Decode
        query_mask = torch.zeros(query_tokens.shape[:2], device=device)
        predictions = model.reconstruct(latents, latent_coords, query_tokens, query_mask)
        
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1) if predictions.shape[-1] == 1 else predictions[..., 0]
        
        # Compute error
        errors = (predictions - ground_truth) ** 2
        
        # Reshape and aggregate
        errors = errors.view(B, L, num_samples, C_effective)
        errors_per_latent = errors.mean(dim=(-1, -2))
        
        return errors_per_latent


def compute_error_predictor_loss(
    predicted_errors: List[torch.Tensor],
    actual_errors: torch.Tensor,
    loss_type: str = 'mse',
) -> torch.Tensor:
    """
    Compute supervision loss for error predictor.
    
    Args:
        predicted_errors: List of [B, L] predicted errors, one per displacement layer
        actual_errors: [B, num_displacement_layers, L] actual errors
        loss_type: 'mse' or 'ranking' (ranking loss for relative ordering)
    
    Returns:
        loss: Scalar loss for error predictor supervision
    """
    if len(predicted_errors) == 0:
        return torch.tensor(0.0, device=actual_errors.device)
    
    total_loss = 0.0
    num_layers = len(predicted_errors)
    
    for layer_idx, pred_error in enumerate(predicted_errors):
        target = actual_errors[:, layer_idx, :].detach()
        
        if loss_type == 'mse':
            layer_loss = F.mse_loss(pred_error, target)
        elif loss_type == 'ranking':
            # Ranking loss: care about relative ordering, not absolute values
            layer_loss = ranking_loss(pred_error, target)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        total_loss += layer_loss
    
    return total_loss / num_layers


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    Pairwise ranking loss: if target_i > target_j, then pred_i should be > pred_j.
    
    This is useful when we care about relative ordering (which latents have higher error)
    rather than absolute error values.
    
    Args:
        pred: [B, L] predicted errors
        target: [B, L] actual errors
    
    Returns:
        loss: Scalar ranking loss
    """
    B, L = pred.shape
    
    # Pairwise differences
    pred_diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)  # [B, L, L]
    target_diff = target.unsqueeze(-1) - target.unsqueeze(-2)  # [B, L, L]
    
    # Sign of target difference (which should be larger)
    target_sign = torch.sign(target_diff)  # +1 if i > j, -1 if i < j, 0 if equal
    
    # Hinge loss: if target_i > target_j, then pred_i - pred_j should be > margin
    # loss = max(0, margin - target_sign * pred_diff)
    loss = F.relu(margin - target_sign * pred_diff)
    
    # Mask diagonal (self-comparison)
    mask = ~torch.eye(L, dtype=torch.bool, device=pred.device)
    loss = loss[:, mask].mean()
    
    return loss


def compute_error_supervision(
    model,
    trajectory: List[torch.Tensor],
    predicted_errors: List[torch.Tensor],
    latents: torch.Tensor,
    final_coords: torch.Tensor,
    image_err: torch.Tensor,
    geometry,
    grid_size: int = 7,  # Larger default since we subsample channels
    spacing: int = 2,
    stable_depth: int = 0,
    num_channels_to_sample: int = 1,  # NEW: default to 2 channels
    loss_type: str = 'mse',
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convenience function to compute error supervision loss in training step.
    
    Args:
        model: Atomiser model
        trajectory: List of [B, L, 2] positions
        predicted_errors: List of [B, L] predicted errors from error predictor
        latents: [B, L, D] final latents
        final_coords: [B, L, 2] final positions
        image_err: [B, C, H, W, 6]
        geometry: SensorGeometry instance
        grid_size: Sampling grid size (default 5 for 5×5=25 samples)
        spacing: Grid spacing in pixels
        stable_depth: Number of final layers without displacement
        num_channels_to_sample: Number of channels to randomly sample
        loss_type: 'mse' or 'ranking'
    
    Returns:
        loss: Error predictor supervision loss
        stats: Dictionary with debugging statistics
    """
    if len(predicted_errors) == 0:
        return torch.tensor(0.0, device=latents.device), {
            'actual_error_mean': 0.0,
            'actual_error_std': 0.0,
            'predicted_error_mean': 0.0,
        }
    
    # Infer num_channels from image_err
    num_channels = image_err.shape[1]
    
    error_module = ErrorSupervisionModule(
        geometry=geometry,
        grid_size=grid_size,
        spacing=spacing,
        num_channels=num_channels,
        num_channels_to_sample=num_channels_to_sample,
    )
    
    actual_errors = error_module.compute_actual_error(
        trajectory=trajectory,
        latents=latents,
        final_coords=final_coords,
        image_err=image_err,
        model=model,
        stable_depth=stable_depth,
    )
    
    loss = compute_error_predictor_loss(predicted_errors, actual_errors, loss_type=loss_type)
    
    # Statistics
    stats = {
        'actual_error_mean': actual_errors.mean().item(),
        'actual_error_std': actual_errors.std().item(),
        'actual_error_max': actual_errors.max().item(),
        'predicted_error_mean': torch.stack(predicted_errors).mean().item(),
        'predicted_error_std': torch.stack(predicted_errors).std().item(),
        'num_displacement_layers': len(predicted_errors),
        'grid_size': grid_size,
        'num_channels_sampled': num_channels_to_sample,
    }
    
    # Per-layer correlation (useful diagnostic)
    for i, pred in enumerate(predicted_errors):
        actual = actual_errors[:, i, :]
        if pred.numel() > 1:
            # Compute Spearman-like rank correlation
            pred_flat = pred.flatten()
            actual_flat = actual.flatten()
            correlation = torch.corrcoef(
                torch.stack([pred_flat, actual_flat])
            )[0, 1].item()
            stats[f'layer_{i}_correlation'] = correlation
    
    return loss, stats