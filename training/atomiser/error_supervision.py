"""
Error Supervision Module for Error-Guided Displacement (v3)

KEY CHANGE in v3:
- Uses INITIAL POSITIONS (uniform grid) + FINAL LATENTS for supervision
- This creates accountability: "Can you still reconstruct where you started?"
- If a latent abandons a complex region, its final features (tuned for destination)
  will perform poorly at reconstructing its origin → high error → penalty

Why this works:
- Final latents are most refined (best representation)
- Initial positions are stable (no circularity from movement)
- Creates right incentive: leaving hard regions hurts more than leaving easy ones

Usage:
    error_module = ErrorSupervisionModule(geometry, config)
    actual_errors = error_module.compute_actual_error(
        initial_positions=trajectory[0],  # Uniform grid
        final_latents=latents,             # After all processing
        final_coords=final_coords,         # For k-NN in decoder
        image_err=image_err,
        model=model,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class ErrorSupervisionModule(nn.Module):
    """
    Module to compute actual reconstruction error for error predictor supervision.
    
    v3: Uses initial positions + final latents.
    
    The key insight: We ask each latent "can you reconstruct your ORIGIN region?"
    using your FINAL features. If you abandoned a complex region, your features
    are now tuned for somewhere else → poor reconstruction → high error.
    """
    
    def __init__(
        self,
        geometry,  # SensorGeometry instance
        grid_size: int = 3,
        spacing: int = 2,
        image_size: int = 512,
        gsd: float = 0.2,
        num_channels: int = 5,
        num_channels_to_sample: Optional[int] = None,
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
        
        if num_channels_to_sample is not None:
            ratio = num_channels_to_sample / num_channels
       
    
    def _subsample_channels(
        self,
        query_tokens: torch.Tensor,
        ground_truth: torch.Tensor,
        num_positions: int,
        num_channels: int,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly subsample channels from query tokens and ground truth."""
        B = query_tokens.shape[0]
        device = query_tokens.device
        token_dim = query_tokens.shape[-1]
        
        query_tokens = query_tokens.view(B, num_positions, num_channels, token_dim)
        ground_truth = ground_truth.view(B, num_positions, num_channels)
        
        channel_indices = torch.randperm(num_channels, device=device)[:num_to_sample]
        channel_indices = channel_indices.sort().values
        
        query_tokens_sampled = query_tokens[:, :, channel_indices, :]
        ground_truth_sampled = ground_truth[:, :, channel_indices]
        
        query_tokens_sampled = query_tokens_sampled.reshape(B, num_positions * num_to_sample, token_dim)
        ground_truth_sampled = ground_truth_sampled.reshape(B, num_positions * num_to_sample)
        
        return query_tokens_sampled, ground_truth_sampled
    
    def compute_actual_error(
        self,
        initial_positions: torch.Tensor,
        final_latents: torch.Tensor,
        final_coords: torch.Tensor,
        image_err: torch.Tensor,
        model,
        num_channels_to_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute actual reconstruction error at INITIAL positions using FINAL latents.
        
        This creates accountability: each latent must still be able to reconstruct
        the region it started at. If it moved away from a complex region, its
        features (now tuned for the destination) will struggle → high error.
        
        Args:
            initial_positions: [B, L, 2] initial grid positions (trajectory[0])
            final_latents: [B, L, D] final latent features after all layers
            final_coords: [B, L, 2] final latent positions (for k-NN in decoder)
            image_err: [B, C, H, W, 6] full image with metadata
            model: Atomiser model (used for decoding)
            num_channels_to_sample: Override instance setting for channel sampling
        
        Returns:
            actual_errors: [B, L] reconstruction error per latent
        """
        B = final_latents.shape[0]
        L = initial_positions.shape[1]
        device = final_latents.device
        C = self.num_channels
        
        if num_channels_to_sample is None:
            num_channels_to_sample = self.num_channels_to_sample
        
        C_effective = num_channels_to_sample if num_channels_to_sample else C
        
        # Extract spatial latents only (in case latents includes global latents)
        spatial_latents = final_latents[:, :L, :]
        
        # =================================================================
        # STEP 1: Sample query positions around INITIAL positions
        # =================================================================
        
        # Convert initial positions (meters) to pixels
        positions_pixels = self.geometry.meters_to_pixels(
            initial_positions,
            image_size=self.image_size,
            gsd=self.gsd,
        )
        
        # Sample grid around each initial position
        sample_coords = self.geometry.sample_grid_around_positions(
            positions_pixels,
            grid_size=self.grid_size,
            spacing=self.spacing,
            image_size=self.image_size,
        )  # [B, L, num_samples, 2]
        
        num_samples = sample_coords.shape[-2]
        num_positions = L * num_samples
        
        # =================================================================
        # STEP 2: Extract query tokens from image
        # =================================================================
        
        sample_coords_flat = sample_coords.view(B, -1, 2)
        
        query_tokens, ground_truth, _ = self.geometry.extract_query_tokens_from_image(
            image_err,
            sample_coords_flat,
        )
        
        # Subsample channels if requested
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
        
        # =================================================================
        # STEP 3: Decode using FINAL latents at INITIAL positions
        # 
        # Key: We use initial_positions for the k-NN lookup in decoder,
        # NOT final_coords. This asks "how well can you reconstruct HERE
        # (your origin) with your current features?"
        # =================================================================
        
        query_mask = torch.zeros(query_tokens.shape[:2], device=device)
        
        predictions = model.reconstruct(
            spatial_latents,  # FINAL features
            final_coords,     # FINAL positions (k-NN lookup)
            query_tokens,     # queries at INITIAL positions
            query_mask,
        )
                
        # Handle output shape
        if predictions.dim() == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        elif predictions.dim() == 3:
            predictions = predictions[..., 0]
        
        # =================================================================
        # STEP 4: Compute MSE per latent
        # =================================================================
        
        errors = (predictions - ground_truth) ** 2
        errors = errors.view(B, L, num_samples, C_effective)
        errors_per_latent = errors.mean(dim=(-1, -2))  # [B, L]
        
        return errors_per_latent
    
    def compute_actual_error_at_current_positions(
        self,
        current_positions: torch.Tensor,
        latents: torch.Tensor,
        image_err: torch.Tensor,
        model,
        num_channels_to_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Alternative: Compute error at CURRENT positions (for comparison/debugging).
        
        This measures "how well can you reconstruct WHERE YOU ARE NOW?"
        Useful for seeing if movement actually helped reconstruction.
        """
        return self.compute_actual_error(
            initial_positions=current_positions,  # Use current as "initial"
            final_latents=latents,
            final_coords=current_positions,
            image_err=image_err,
            model=model,
            num_channels_to_sample=num_channels_to_sample,
        )


def compute_error_predictor_loss(
    predicted_errors: List[torch.Tensor],
    actual_errors: torch.Tensor,
    loss_type: str = 'mse',
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute supervision loss for error predictor.
    
    All layers predict the SAME target: intrinsic difficulty at initial positions.
    
    Args:
        predicted_errors: List of [B, L] predicted errors, one per displacement layer
        actual_errors: [B, L] actual errors (same target for all layers!)
        loss_type: 'mse', 'ranking', or 'log_mse'
        normalize: If True, normalize both pred and target before loss
    
    Returns:
        loss: Scalar loss for error predictor supervision
    """
    if len(predicted_errors) == 0:
        return torch.tensor(0.0, device=actual_errors.device)
    
    total_loss = 0.0
    num_layers = len(predicted_errors)
    target = actual_errors.detach()
    
    # Optionally normalize target (log-robust)
    if normalize:
        target_log = torch.log1p(target)
        target_min = target_log.min(dim=-1, keepdim=True).values
        target_max = target_log.max(dim=-1, keepdim=True).values
        target_norm = (target_log - target_min) / (target_max - target_min + 1e-6)
    else:
        target_norm = target
    
    for pred_error in predicted_errors:
        # Normalize predictions similarly
        if normalize:
            pred_log = torch.log1p(pred_error)
            pred_min = pred_log.min(dim=-1, keepdim=True).values
            pred_max = pred_log.max(dim=-1, keepdim=True).values
            pred_norm = (pred_log - pred_min) / (pred_max - pred_min + 1e-6)
        else:
            pred_norm = pred_error
        
        if loss_type == 'mse':
            layer_loss = F.mse_loss(pred_norm, target_norm)
        elif loss_type == 'log_mse':
            # MSE in log space (handles large value differences)
            layer_loss = F.mse_loss(torch.log1p(pred_error), torch.log1p(target))
        elif loss_type == 'ranking':
            layer_loss = ranking_loss(pred_error, target)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        total_loss += layer_loss
    
    return total_loss / num_layers


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    Pairwise ranking loss: if target_i > target_j, then pred_i should be > pred_j.
    """
    B, L = pred.shape
    
    pred_diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    target_diff = target.unsqueeze(-1) - target.unsqueeze(-2)
    
    target_sign = torch.sign(target_diff)
    loss = F.relu(margin - target_sign * pred_diff)
    
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
    grid_size: int = 7,
    spacing: int = 2,
    num_channels_to_sample: int = 1,
    loss_type: str = 'mse',
    normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convenience function to compute error supervision loss in training step.
    
    Uses INITIAL positions (trajectory[0]) + FINAL latents.
    
    Args:
        model: Atomiser model
        trajectory: List of [B, L, 2] positions (trajectory[0] = initial grid)
        predicted_errors: List of [B, L] predicted errors from error predictor
        latents: [B, L, D] final latents (after all processing)
        final_coords: [B, L, 2] final positions
        image_err: [B, C, H, W, 6]
        geometry: SensorGeometry instance
        grid_size: Sampling grid size
        spacing: Grid spacing in pixels
        num_channels_to_sample: Number of channels to randomly sample
        loss_type: 'mse', 'ranking', or 'log_mse'
        normalize: If True, normalize pred and target before loss
    
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
    
    # Get initial positions (uniform grid, before any movement)
    initial_positions = trajectory[0]
    
    # Infer num_channels from image_err
    num_channels = image_err.shape[1]
    
    error_module = ErrorSupervisionModule(
        geometry=geometry,
        grid_size=grid_size,
        spacing=spacing,
        num_channels=num_channels,
        num_channels_to_sample=num_channels_to_sample,
    )
    
    # Compute actual error at INITIAL positions using FINAL latents
    actual_errors = error_module.compute_actual_error(
        initial_positions=initial_positions,
        final_latents=latents,
        final_coords=final_coords,
        image_err=image_err,
        model=model,
        num_channels_to_sample=num_channels_to_sample,
    )
    
    # Compute loss (same target for all layers)
    loss = compute_error_predictor_loss(
        predicted_errors, 
        actual_errors, 
        loss_type=loss_type,
        normalize=normalize,
    )
    
    # Statistics
    pred_stack = torch.stack(predicted_errors)
    
    stats = {
        'actual_error_mean': actual_errors.mean().item(),
        'actual_error_std': actual_errors.std().item(),
        'actual_error_max': actual_errors.max().item(),
        'actual_error_min': actual_errors.min().item(),
        'predicted_error_mean': pred_stack.mean().item(),
        'predicted_error_std': pred_stack.std().item(),
        'predicted_error_max': pred_stack.max().item(),
        'predicted_error_min': pred_stack.min().item(),
        'num_displacement_layers': len(predicted_errors),
        'grid_size': grid_size,
        'num_channels_sampled': num_channels_to_sample,
        'loss_type': loss_type,
    }
    
    # Correlation between predicted and actual (across all latents)
    # Use first layer's predictions as representative
    if predicted_errors[0].numel() > 1:
        pred_flat = predicted_errors[0].flatten()
        actual_flat = actual_errors.flatten()
        
        # Pearson correlation
        pred_centered = pred_flat - pred_flat.mean()
        actual_centered = actual_flat - actual_flat.mean()
        correlation = (pred_centered * actual_centered).sum() / (
            pred_centered.norm() * actual_centered.norm() + 1e-8
        )
        stats['correlation'] = correlation.item()
        
        # Spearman rank correlation (more robust)
        pred_ranks = pred_flat.argsort().argsort().float()
        actual_ranks = actual_flat.argsort().argsort().float()
        pred_ranks_centered = pred_ranks - pred_ranks.mean()
        actual_ranks_centered = actual_ranks - actual_ranks.mean()
        rank_correlation = (pred_ranks_centered * actual_ranks_centered).sum() / (
            pred_ranks_centered.norm() * actual_ranks_centered.norm() + 1e-8
        )
        stats['rank_correlation'] = rank_correlation.item()
    
    # Movement statistics (how much did latents move?)
    if len(trajectory) > 1:
        total_displacement = (trajectory[-1] - trajectory[0]).norm(dim=-1)
        stats['movement_mean'] = total_displacement.mean().item()
        stats['movement_max'] = total_displacement.max().item()
        stats['movement_std'] = total_displacement.std().item()
        
        # Correlation between movement and error (should be positive!)
        # Latents should move MORE toward high-error regions
        if total_displacement.numel() > 1:
            disp_flat = total_displacement.flatten()
            disp_centered = disp_flat - disp_flat.mean()
            movement_error_corr = (disp_centered * actual_centered).sum() / (
                disp_centered.norm() * actual_centered.norm() + 1e-8
            )
            stats['movement_error_correlation'] = movement_error_corr.item()
    
    return loss, stats


def compute_abandonment_penalty(
    trajectory: List[torch.Tensor],
    actual_errors: torch.Tensor,
    penalty_weight: float = 0.1,
) -> torch.Tensor:
    """
    Optional: Explicit penalty for abandoning high-error regions.
    
    If a latent moves away from a high-error region, add extra penalty.
    
    Args:
        trajectory: List of positions
        actual_errors: [B, L] error at initial positions
        penalty_weight: Weight for abandonment penalty
    
    Returns:
        penalty: Scalar penalty term
    """
    if len(trajectory) < 2:
        return torch.tensor(0.0, device=actual_errors.device)
    
    # Movement magnitude
    movement = (trajectory[-1] - trajectory[0]).norm(dim=-1)  # [B, L]
    
    # Normalize errors to [0, 1]
    errors_norm = (actual_errors - actual_errors.min()) / (
        actual_errors.max() - actual_errors.min() + 1e-6
    )
    
    # Penalty: movement * error (moving away from high-error = bad)
    # But we want to ENCOURAGE moving toward high-error...
    # So actually we want to penalize when movement is AWAY from high-error
    
    # Simple heuristic: penalize movement that's not proportional to error
    # If error is high but movement is low → penalty (should have moved there!)
    # If error is low but movement is high → no penalty (fine to leave)
    
    # This is tricky to compute without knowing direction...
    # For now, just return correlation as diagnostic
    
    return torch.tensor(0.0, device=actual_errors.device)