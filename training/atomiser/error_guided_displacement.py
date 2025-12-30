"""
Error-Guided Displacement Module

This module implements displacement based on predicted reconstruction error.
The key idea: predict error as a function of (latent features, position),
then compute gradient of error w.r.t. position to determine movement direction.

The error predictor learns to predict reconstruction error at any position.
During inference, latents move toward positions with lower predicted error
by following the negative gradient.

Classes:
    ErrorPredictor: MLP that predicts error from (latent, position)
    ErrorGuidedDisplacement: Full displacement module with gradient computation

Usage:
    displacement_module = ErrorGuidedDisplacement(
        latent_dim=256,
        num_latents_per_row=20,
        max_displacement=5.0,
        depth=4,
        share_weights=True,
    )
    
    new_pos, displacement, pred_error = displacement_module(
        latents,      # [B, L, D]
        positions,    # [B, L, 2]
        layer_idx,    # int
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ErrorPredictor(nn.Module):
    """
    Predicts reconstruction error from latent features and position.
    
    Architecture:
        position → MLP → pos_features
        concat(latent, pos_features) → MLP → error (scalar)
    
    The position encoding is a simple MLP to allow gradient flow.
    """
    
    def __init__(
        self,
        latent_dim: int,
        pos_hidden_dim: int = 64,
        error_hidden_dim: int = 256,
        pos_scale: float = 51.5,  # Half of latent surface (103/2)
    ):
        """
        Args:
            latent_dim: Dimension of latent features
            pos_hidden_dim: Hidden dimension for position encoder
            error_hidden_dim: Hidden dimension for error predictor
            pos_scale: Scale for normalizing positions to [-1, 1]
        """
        super().__init__()
        
        self.pos_scale = pos_scale
        
        # Position encoder: (x, y) → features
        # Simple MLP allows direct gradient flow to positions
        self.position_encoder = nn.Sequential(
            nn.Linear(2, pos_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.GELU(),
        )
        
        # Error predictor: (latent_features, pos_features) → scalar error
        self.error_mlp = nn.Sequential(
            nn.Linear(latent_dim + pos_hidden_dim, error_hidden_dim),
            nn.LayerNorm(error_hidden_dim),
            nn.GELU(),
            nn.Linear(error_hidden_dim, error_hidden_dim // 2),
            nn.LayerNorm(error_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(error_hidden_dim // 2, 1),
            nn.Softplus(),  # Error is non-negative
        )
    
    def forward(
        self, 
        latents: torch.Tensor, 
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict error at given positions.
        
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] positions in meters (must have requires_grad=True
                       for gradient computation)
        
        Returns:
            predicted_error: [B, L] predicted reconstruction error per latent
        """
        # Normalize positions to [-1, 1] for numerical stability
        pos_normalized = positions / self.pos_scale
        
        # Encode positions
        pos_features = self.position_encoder(pos_normalized)  # [B, L, pos_hidden_dim]
        
        # Combine latent features and position features
        combined = torch.cat([latents, pos_features], dim=-1)  # [B, L, D + pos_hidden_dim]
        
        # Predict error
        predicted_error = self.error_mlp(combined).squeeze(-1)  # [B, L]
        
        return predicted_error


class ErrorGuidedDisplacement(nn.Module):
    """
    Displacement module that moves latents toward lower predicted error.
    
    The displacement is computed as:
        1. Predict error at current position: e = f(latent, position)
        2. Compute gradient: g = ∂e/∂position
        3. Scale gradient: displacement = g * scale
        4. Clamp magnitude: |displacement| ≤ max_displacement
        5. Update position: new_pos = pos - displacement (gradient descent)
    
    The scale is learnable, initialized based on latent spacing.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_latents_per_row: int,
        max_displacement: float = 5.0,
        min_displacement: float = 0.0,
        depth: int = 4,
        share_weights: bool = True,
        pos_hidden_dim: int = 64,
        error_hidden_dim: int = 256,
        latent_surface: float = 103.0,
        learnable_scale: bool = False,
        initial_scale_multiplier: float = 10.0,
    ):
        """
        Args:
            latent_dim: Dimension of latent features
            num_latents_per_row: Number of latents per row in grid
            max_displacement: Maximum displacement magnitude per layer (meters)
            min_displacement: Minimum displacement magnitude per layer (meters)
                If gradient-based displacement is smaller, scale up to this value.
                Set to 0 to disable. Recommended: 0.1-0.5m for exploration.
            depth: Number of encoder layers
            share_weights: If True, share error predictor across layers
            pos_hidden_dim: Hidden dimension for position encoder
            error_hidden_dim: Hidden dimension for error MLP
            latent_surface: Total spatial extent of latent grid (meters)
            learnable_scale: If True, displacement scale is learnable (nn.Parameter).
                If False, scale is fixed. Recommended: False (to prevent scale collapse).
            initial_scale_multiplier: Multiplier for initial scale (scale = latent_spacing * multiplier)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latents_per_row = num_latents_per_row
        self.max_displacement = max_displacement
        self.min_displacement = min_displacement
        self.depth = depth
        self.share_weights = share_weights
        self.latent_surface = latent_surface
        self.learnable_scale = learnable_scale
        
        # Compute latent spacing
        if num_latents_per_row > 1:
            self.latent_spacing = latent_surface / (num_latents_per_row - 1)
        else:
            self.latent_spacing = latent_surface
        
        # Initialize error predictor(s)
        if share_weights:
            # Single shared error predictor
            self.error_predictor = ErrorPredictor(
                latent_dim=latent_dim,
                pos_hidden_dim=pos_hidden_dim,
                error_hidden_dim=error_hidden_dim,
                pos_scale=latent_surface / 2.0,
            )
        else:
            # Separate error predictor per layer
            self.error_predictors = nn.ModuleList([
                ErrorPredictor(
                    latent_dim=latent_dim,
                    pos_hidden_dim=pos_hidden_dim,
                    error_hidden_dim=error_hidden_dim,
                    pos_scale=latent_surface / 2.0,
                )
                for _ in range(depth)
            ])
        
        # Learnable displacement scale
        # Initialize so that typical gradients (~0.01-0.1) produce reasonable displacements
        initial_scale = self.latent_spacing * initial_scale_multiplier
        
        if learnable_scale:
            self.displacement_scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            # Register as buffer (saved with model but not trained)
            self.register_buffer('displacement_scale', torch.tensor(initial_scale))
        
        # Log initialization
        print(f"[ErrorGuidedDisplacement] Initialized:")
        print(f"  latent_spacing={self.latent_spacing:.2f}m")
        print(f"  initial_scale={initial_scale:.2f}")
        print(f"  max_displacement={max_displacement}m")
        print(f"  min_displacement={min_displacement}m")
        print(f"  learnable_scale={learnable_scale}")
        print(f"  share_weights={share_weights}")
    
    def get_error_predictor(self, layer_idx: int) -> ErrorPredictor:
        """Get the error predictor for a given layer."""
        if self.share_weights:
            return self.error_predictor
        else:
            return self.error_predictors[layer_idx]
    
    def forward(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute displacement based on error gradient.
        
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] current positions in meters
            layer_idx: Current layer index
        
        Returns:
            new_positions: [B, L, 2] updated positions
            displacement: [B, L, 2] displacement vector applied
            predicted_error: [B, L] predicted error at original positions
        """
        B, L, D = latents.shape
        device = latents.device
        
        # =====================================================================
        # Handle validation mode (torch.no_grad context)
        # =====================================================================
        # During validation, we can't compute gradients, so we need to either:
        # 1. Return zero displacement, or
        # 2. Use torch.enable_grad() to force gradient computation
        # We use option 2 to get meaningful displacement during validation
        
        # Use enable_grad to allow gradient computation even in no_grad context
        with torch.enable_grad():
            # =====================================================================
            # STEP 1: Enable gradients on positions
            # =====================================================================
            # Clone and enable gradients for position
            # We need positions in the computation graph to compute gradients
            positions_for_grad = positions.detach().clone().requires_grad_(True)
            
            # =====================================================================
            # STEP 2: Predict error at current positions
            # =====================================================================
            error_predictor = self.get_error_predictor(layer_idx)
            
            # Detach latents to avoid backprop through encoder during grad computation
            # We only want gradient w.r.t. positions
            latents_detached = latents.detach()
            
            predicted_error = error_predictor(latents_detached, positions_for_grad)  # [B, L]
            
            # =====================================================================
            # STEP 3: Compute gradient of error w.r.t. position
            # =====================================================================
            # Sum over batch and latents to get scalar for grad computation
            gradient = torch.autograd.grad(
                outputs=predicted_error.sum(),
                inputs=positions_for_grad,
                create_graph=self.training,  # Only need computation graph during training
                retain_graph=True,
            )[0]  # [B, L, 2]
        
        # =====================================================================
        # STEP 4: Scale gradient to get displacement
        # =====================================================================
        displacement = gradient * self.displacement_scale
        
        # =====================================================================
        # STEP 5: Enforce MINIMUM displacement (for exploration)
        # =====================================================================
        # If gradient is too small, scale up to ensure minimum movement
        # This helps the error predictor learn position-dependent predictions
        if self.min_displacement > 0:
            disp_magnitude = displacement.norm(dim=-1, keepdim=True)  # [B, L, 1]
            
            # Handle near-zero gradients: use random direction
            # This prevents NaN and encourages exploration
            near_zero_mask = disp_magnitude < 1e-8
            if near_zero_mask.any():
                # Generate random unit vectors for zero-gradient positions
                random_direction = torch.randn_like(displacement)
                random_direction = random_direction / (random_direction.norm(dim=-1, keepdim=True) + 1e-8)
                random_displacement = random_direction * self.min_displacement
                
                # Replace zero-gradient displacements with random exploration
                displacement = torch.where(
                    near_zero_mask.expand_as(displacement),
                    random_displacement,
                    displacement
                )
                disp_magnitude = displacement.norm(dim=-1, keepdim=True)
            
            # Scale up small displacements to meet minimum
            # Only scale up, never scale down here
            scale_up = torch.clamp(
                self.min_displacement / (disp_magnitude + 1e-8),
                min=1.0  # Never shrink, only grow
            )
            displacement = displacement * scale_up
        
        # =====================================================================
        # STEP 6: Clamp displacement by MAXIMUM (not per-dimension)
        # =====================================================================
        disp_magnitude = displacement.norm(dim=-1, keepdim=True)  # [B, L, 1]
        
        # Scale factor: 1.0 if within limit, otherwise shrink proportionally
        scale_down = torch.clamp(
            self.max_displacement / (disp_magnitude + 1e-8),
            max=1.0
        )
        displacement = displacement * scale_down
        
        # =====================================================================
        # STEP 7: Update positions (gradient DESCENT = subtract)
        # =====================================================================
        # Move toward LOWER error = move AGAINST gradient direction
        # 
        # IMPORTANT: Detach displacement from reconstruction gradient path!
        # Otherwise, reconstruction_loss will learn to shrink displacement_scale
        # to zero (because "not moving" is easier for reconstruction).
        # The displacement_scale should only be updated by error_predictor_loss.
        new_positions = positions.detach() - displacement.detach()
        
        # =====================================================================
        # STEP 8: Re-compute predicted_error for training loss
        # =====================================================================
        # IMPORTANT: Detach latents here too!
        # If we don't, self-attention could learn to produce features that
        # "trick" the error predictor into outputting low error, regardless
        # of actual reconstruction quality.
        # 
        # With detached latents:
        #   - Error predictor learns to READ latent features and predict error
        #   - Self-attention is trained ONLY by reconstruction loss
        #   - No "cheating" possible
        if self.training:
            predicted_error = error_predictor(latents.detach(), positions.detach())
        
        return new_positions, displacement, predicted_error
    
    def predict_error_only(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Predict error without computing displacement.
        Useful for evaluation/visualization.
        
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] positions in meters
            layer_idx: Layer index
        
        Returns:
            predicted_error: [B, L]
        """
        error_predictor = self.get_error_predictor(layer_idx)
        with torch.no_grad():
            return error_predictor(latents, positions)
    
    def compute_error_landscape(
        self,
        latents: torch.Tensor,
        center_positions: torch.Tensor,
        layer_idx: int,
        grid_size: int = 21,
        grid_extent: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute error landscape around current positions for visualization.
        
        Args:
            latents: [B, L, D] latent features
            center_positions: [B, L, 2] center positions
            layer_idx: Layer index
            grid_size: Number of points per dimension
            grid_extent: Extent of grid in meters (±grid_extent from center)
        
        Returns:
            error_grid: [B, L, grid_size, grid_size] error at each grid point
            x_offsets: [grid_size] x offset values
            y_offsets: [grid_size] y offset values
        """
        B, L, D = latents.shape
        device = latents.device
        
        # Create offset grid
        offsets = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
        
        # Create all position combinations
        y_grid, x_grid = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_grid = torch.stack([x_grid, y_grid], dim=-1)  # [grid_size, grid_size, 2]
        
        # Expand for batch and latents
        # center_positions: [B, L, 2]
        # offset_grid: [grid_size, grid_size, 2]
        # positions: [B, L, grid_size, grid_size, 2]
        positions = center_positions.unsqueeze(2).unsqueeze(3) + offset_grid
        
        # Reshape for batch processing
        positions_flat = positions.view(B, L * grid_size * grid_size, 2)
        latents_expanded = latents.unsqueeze(2).unsqueeze(3).expand(-1, -1, grid_size, grid_size, -1)
        latents_flat = latents_expanded.reshape(B, L * grid_size * grid_size, D)
        
        # Predict errors
        error_predictor = self.get_error_predictor(layer_idx)
        with torch.no_grad():
            errors_flat = error_predictor(latents_flat, positions_flat)  # [B, L*grid_size*grid_size]
        
        # Reshape back
        error_grid = errors_flat.view(B, L, grid_size, grid_size)
        
        return error_grid, offsets, offsets


def create_error_guided_displacement(config: dict) -> ErrorGuidedDisplacement:
    """
    Factory function to create ErrorGuidedDisplacement from config.
    
    Config keys used:
        Atomiser.latent_dim: int
        Atomiser.spatial_latents: int (per row)
        Atomiser.max_displacement: float
        Atomiser.min_displacement: float (default 0.0)
        Atomiser.depth: int
        Atomiser.share_error_predictor_weights: bool (default True)
        Atomiser.error_pos_hidden_dim: int (default 64)
        Atomiser.error_hidden_dim: int (default 256)
        Atomiser.latent_surface: float (default 103.0)
        Atomiser.learnable_displacement_scale: bool (default False)
        Atomiser.displacement_scale_multiplier: float (default 10.0)
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        ErrorGuidedDisplacement instance
    """
    atomiser_config = config["Atomiser"]
    
    return ErrorGuidedDisplacement(
        latent_dim=atomiser_config.get("latent_dim", 256),
        num_latents_per_row=atomiser_config["spatial_latents"],
        max_displacement=atomiser_config.get("max_displacement", 5.0),
        min_displacement=atomiser_config.get("min_displacement", 0.0),
        depth=atomiser_config["depth"],
        share_weights=atomiser_config.get("share_error_predictor_weights", True),
        pos_hidden_dim=atomiser_config.get("error_pos_hidden_dim", 64),
        error_hidden_dim=atomiser_config.get("error_hidden_dim", 256),
        latent_surface=atomiser_config.get("latent_surface", 103.0),
        learnable_scale=atomiser_config.get("learnable_displacement_scale", False),
        initial_scale_multiplier=atomiser_config.get("displacement_scale_multiplier", 10.0),
    )