"""
Error Supervision Module for Error-Guided Displacement

This module provides functions to compute actual reconstruction error
at latent positions for supervising the error predictor.

Pipeline:
1. Convert trajectory (meters) → pixels
2. Sample grid around each latent position
3. Extract query tokens from image_err
4. Decode using final latents
5. Compute MSE between predictions and ground truth
6. Aggregate error per latent

Usage:
    from error_supervision import ErrorSupervisionModule
    
    error_module = ErrorSupervisionModule(geometry, config)
    actual_errors = error_module.compute_actual_error(
        trajectory=trajectory,        # [depth+1] list of [B, L, 2]
        latents=latents,              # [B, L, D]
        final_coords=final_coords,    # [B, L, 2]
        image_err=image_err,          # [B, C, H, W, 6]
        model=model,                  # Atomiser model (for decoder)
        stable_depth=2,               # Number of final layers without displacement
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class ErrorSupervisionModule(nn.Module):
    """
    Module to compute actual reconstruction error for error predictor supervision.
    
    This module handles the full pipeline:
    1. Convert latent positions from meters to pixels
    2. Sample points around each latent
    3. Extract query tokens from the image
    4. Decode at those positions
    5. Compute reconstruction error
    6. Aggregate per latent
    """
    
    def __init__(
        self,
        geometry,  # SensorGeometry instance
        grid_size: int = 3,
        spacing: int = 2,
        image_size: int = 512,
        gsd: float = 0.2,
        num_channels: int = 5,
    ):
        """
        Args:
            geometry: SensorGeometry instance for coordinate conversion
            grid_size: Size of sampling grid (e.g., 3 → 3×3 = 9 samples)
            spacing: Spacing between grid points in pixels
            image_size: Image dimension in pixels
            gsd: Ground sample distance in meters/pixel
            num_channels: Number of spectral channels (e.g., 5 for B,G,R,NIR,Elev)
        """
        super().__init__()
        self.geometry = geometry
        self.grid_size = grid_size
        self.spacing = spacing
        self.image_size = image_size
        self.gsd = gsd
        self.num_channels = num_channels
        self.num_samples_per_position = grid_size ** 2
    
    def compute_actual_error(
        self,
        trajectory: List[torch.Tensor],
        latents: torch.Tensor,
        final_coords: torch.Tensor,
        image_err: torch.Tensor,
        model,  # Atomiser model
        layers_to_supervise: Optional[List[int]] = None,
        stable_depth: int = 0,
    ) -> torch.Tensor:
        """
        Compute actual reconstruction error at latent positions for each layer.
        
        IMPORTANT: For each layer i, we use trajectory[i] as BOTH:
        1. The positions to sample query tokens around
        2. The latent_coords for the decoder k-NN lookup
        
        This ensures we measure "error in latent's neighborhood at layer i"
        using the latent positions AT THAT LAYER, not final positions.
        
        Args:
            trajectory: List of [B, L, 2] positions in meters, one per layer
                Length is depth+1 (includes initial position)
            latents: [B, L_total, D] final encoded latents (may include global latents)
            final_coords: [B, L_spatial, 2] final latent positions (NOT USED - kept for API)
            image_err: [B, C, H, W, 6] full image with metadata
            model: Atomiser model (used for decoding)
            layers_to_supervise: Optional list of layer indices to supervise
                If None, supervise all displacement layers (excluding stable_depth final layers)
            stable_depth: Number of final layers without displacement.
                These layers won't be supervised.
        
        Returns:
            actual_errors: [B, num_displacement_layers, L] reconstruction error per latent per layer
        """
        B = latents.shape[0]
        depth = len(trajectory) - 1  # trajectory has depth+1 entries
        
        # Get L from trajectory, NOT from latents (latents may include global latents)
        L = trajectory[0].shape[1]  # Number of SPATIAL latents
        
        device = latents.device
        
        if layers_to_supervise is None:
            # Only supervise layers where displacement happens
            # With stable_depth, last stable_depth layers have no displacement
            num_displacement_layers = depth - stable_depth
            layers_to_supervise = list(range(num_displacement_layers))
        
        num_layers = len(layers_to_supervise)
        C = self.num_channels
        
        # Compute error for each layer separately to use correct latent_coords
        all_layer_errors = []
        
        for layer_idx in layers_to_supervise:
            # Get latent positions at this layer
            layer_coords = trajectory[layer_idx]  # [B, L, 2] in meters
            
            # Step 1: Convert meters to pixels
            positions_pixels = self.geometry.meters_to_pixels(
                layer_coords,
                image_size=self.image_size,
                gsd=self.gsd,
            )  # [B, L, 2]
            
            # Step 2: Sample grid around each latent position
            sample_coords = self.geometry.sample_grid_around_positions(
                positions_pixels,
                grid_size=self.grid_size,
                spacing=self.spacing,
                image_size=self.image_size,
            )  # [B, L, num_samples, 2]
            
            num_samples = sample_coords.shape[-2]  # grid_size²
            
            # Step 3: Extract query tokens from image
            sample_coords_flat = sample_coords.view(B, -1, 2)  # [B, L * num_samples, 2]
            
            query_tokens, ground_truth, _ = self.geometry.extract_query_tokens_from_image(
                image_err,
                sample_coords_flat,
            )
            # query_tokens: [B, L * num_samples * C, 6]
            # ground_truth: [B, L * num_samples * C]
            
            # Step 4: Decode using latent positions AT THIS LAYER (not final!)
            # This ensures k-NN finds the right latents for each query
            query_mask = torch.zeros(query_tokens.shape[:2], device=device)
            
            predictions = model.reconstruct(
                latents,
                layer_coords,  # USE THIS LAYER'S POSITIONS, not final_coords!
                query_tokens,
                query_mask,
            )
            
            # Handle output shape
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            elif predictions.dim() == 3:
                predictions = predictions[..., 0]
            
            # Step 5: Compute MSE per query
            errors = (predictions - ground_truth) ** 2  # [B, L * num_samples * C]
            
            # Step 6: Reshape and aggregate per latent
            errors = errors.view(B, L, num_samples, C)
            layer_errors = errors.mean(dim=(-1, -2))  # [B, L]
            
            all_layer_errors.append(layer_errors)
        
        # Stack: [B, num_layers, L]
        actual_errors = torch.stack(all_layer_errors, dim=1)
        
        return actual_errors
    
    def compute_error_at_single_layer(
        self,
        positions: torch.Tensor,
        latents: torch.Tensor,
        final_coords: torch.Tensor,
        image_err: torch.Tensor,
        model,
    ) -> torch.Tensor:
        """
        Compute error at a single set of positions (for debugging/visualization).
        
        Args:
            positions: [B, L, 2] positions in meters
            latents: [B, L, D] final encoded latents
            final_coords: [B, L, 2] final latent positions
            image_err: [B, C, H, W, 6]
            model: Atomiser model
        
        Returns:
            errors: [B, L] error per latent
        """
        B, L, _ = positions.shape
        device = latents.device
        
        # Convert to pixels
        positions_pixels = self.geometry.meters_to_pixels(
            positions, image_size=self.image_size, gsd=self.gsd
        )  # [B, L, 2]
        
        # Sample grid
        sample_coords = self.geometry.sample_grid_around_positions(
            positions_pixels,
            grid_size=self.grid_size,
            spacing=self.spacing,
            image_size=self.image_size,
        )  # [B, L, num_samples, 2]
        
        num_samples = sample_coords.shape[-2]
        
        # Extract query tokens
        sample_coords_flat = sample_coords.view(B, -1, 2)
        query_tokens, ground_truth, _ = self.geometry.extract_query_tokens_from_image(
            image_err, sample_coords_flat
        )
        
        # Decode
        query_mask = torch.zeros(query_tokens.shape[:2], device=device)
        predictions = model.reconstruct(latents, final_coords, query_tokens, query_mask)
        
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1) if predictions.shape[-1] == 1 else predictions[..., 0]
        
        # Compute error
        errors = (predictions - ground_truth) ** 2  # [B, L * num_samples * C]
        
        # Reshape and aggregate
        C = self.num_channels
        errors = errors.view(B, L, num_samples, C)
        errors_per_latent = errors.mean(dim=(-1, -2))  # [B, L]
        
        return errors_per_latent


def compute_error_predictor_loss(
    predicted_errors: List[torch.Tensor],
    actual_errors: torch.Tensor,
) -> torch.Tensor:
    """
    Compute supervision loss for error predictor.
    
    Args:
        predicted_errors: List of [B, L] predicted errors, one per displacement layer
        actual_errors: [B, num_displacement_layers, L] actual errors from compute_actual_error
    
    Returns:
        loss: Scalar loss for error predictor supervision
    """
    if len(predicted_errors) == 0:
        return torch.tensor(0.0, device=actual_errors.device)
    
    total_loss = 0.0
    num_layers = len(predicted_errors)
    
    for layer_idx, pred_error in enumerate(predicted_errors):
        # pred_error: [B, L]
        # actual_errors[:, layer_idx, :]: [B, L]
        target = actual_errors[:, layer_idx, :].detach()  # Detach to not backprop through actual error
        
        layer_loss = F.mse_loss(pred_error, target)
        total_loss += layer_loss
    
    return total_loss / num_layers


# =============================================================================
# Convenience function for training step
# =============================================================================

def compute_error_supervision(
    model,
    trajectory: List[torch.Tensor],
    predicted_errors: List[torch.Tensor],
    latents: torch.Tensor,
    final_coords: torch.Tensor,
    image_err: torch.Tensor,
    geometry,
    grid_size: int = 3,
    spacing: int = 2,
    stable_depth: int = 0,
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
        grid_size: Sampling grid size
        spacing: Grid spacing in pixels
        stable_depth: Number of final layers without displacement
    
    Returns:
        loss: Error predictor supervision loss
        stats: Dictionary with debugging statistics
    """
    # Handle empty predicted_errors (all layers are stable)
    if len(predicted_errors) == 0:
        return torch.tensor(0.0, device=latents.device), {
            'actual_error_mean': 0.0,
            'actual_error_std': 0.0,
            'actual_error_max': 0.0,
            'predicted_error_mean': 0.0,
            'predicted_error_std': 0.0,
        }
    
    # Create error supervision module
    error_module = ErrorSupervisionModule(
        geometry=geometry,
        grid_size=grid_size,
        spacing=spacing,
    )
    
    # Compute actual errors (only for displacement layers)
    actual_errors = error_module.compute_actual_error(
        trajectory=trajectory,
        latents=latents,
        final_coords=final_coords,
        image_err=image_err,
        model=model,
        stable_depth=stable_depth,
    )  # [B, num_displacement_layers, L]
    
    # Compute loss
    loss = compute_error_predictor_loss(predicted_errors, actual_errors)
    
    # Compute statistics for logging
    stats = {
        'actual_error_mean': actual_errors.mean().item(),
        'actual_error_std': actual_errors.std().item(),
        'actual_error_max': actual_errors.max().item(),
        'predicted_error_mean': torch.stack(predicted_errors).mean().item(),
        'predicted_error_std': torch.stack(predicted_errors).std().item(),
        'num_displacement_layers': len(predicted_errors),
        'stable_depth': stable_depth,
    }
    
    # Per-layer statistics
    for i, pred in enumerate(predicted_errors):
        actual = actual_errors[:, i, :]
        stats[f'layer_{i}_pred_mean'] = pred.mean().item()
        stats[f'layer_{i}_actual_mean'] = actual.mean().item()
        stats[f'layer_{i}_correlation'] = torch.corrcoef(
            torch.stack([pred.flatten(), actual.flatten()])
        )[0, 1].item() if pred.numel() > 1 else 0.0
    
    return loss, stats