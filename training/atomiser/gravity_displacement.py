"""
Gravity-Based Latent Displacement

Latents are attracted toward high-error regions using a gravity model:
- Each latent has a predicted reconstruction error
- High-error latents exert gravitational pull on others
- Pull strength ∝ normalized_error / distance²
- Optional repulsion between latents to prevent piling up

Key insight: Uses all L=400 error predictions to build a global "gravity map"
without any extra forward passes.

Scale-invariant: Errors are normalized to [0, 1] so displacement magnitudes
stay meaningful throughout training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ErrorPredictor(nn.Module):
    """
    Predicts reconstruction error from (latent_features, position).
    
    Simple MLP architecture for fast inference.
    """
    
    def __init__(
        self,
        latent_dim: int,
        pos_hidden_dim: int = 64,
        error_hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Position encoder: (x, y) → features
        self.position_encoder = nn.Sequential(
            nn.Linear(2, pos_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
        )
        
        # Error predictor: concat(latent, pos_features) → scalar error
        self.error_mlp = nn.Sequential(
            nn.Linear(latent_dim + pos_hidden_dim, error_hidden_dim),
            nn.LayerNorm(error_hidden_dim),
            nn.GELU(),
            nn.Linear(error_hidden_dim, error_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(error_hidden_dim // 2, 1),
            nn.Softplus(),  # Error is non-negative
        )
    
    def forward(self, latents: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] latent positions (x, y)
        
        Returns:
            errors: [B, L] predicted reconstruction error per latent
        """
        pos_features = self.position_encoder(positions)  # [B, L, pos_hidden]
        combined = torch.cat([latents, pos_features], dim=-1)  # [B, L, D + pos_hidden]
        errors = self.error_mlp(combined).squeeze(-1)  # [B, L]
        return errors


class GravityDisplacement(nn.Module):
    """
    Gravity-based displacement: latents move toward high-error regions.
    
    How it works:
    1. Predict error at all latent positions (L predictions)
    2. Normalize errors to [0, 1] for scale invariance
    3. Each latent exerts "gravitational pull" proportional to its error
    4. Pull strength ∝ normalized_error / distance²
    5. Each latent moves according to net gravitational force
    6. Optional: repulsion between latents prevents piling up
    
    This gives a GLOBAL view - latents in easy regions can see distant hard regions.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_latents_per_row: int,
        max_displacement: float = 5.0,
        min_displacement: float = 0.5,
        repulsion_strength: float = 0.3,
        gravity_power: float = 2.0,
        depth: int = 4,
        share_weights: bool = True,
        pos_hidden_dim: int = 64,
        error_hidden_dim: int = 256,
        latent_surface: float = 103.0,
        error_offset: float = 0.1,
        danger_zone_divisor: float = 4.0,
    ):
        """
        Args:
            latent_dim: Dimension of latent features
            num_latents_per_row: Number of latents per row in grid
            max_displacement: Maximum displacement magnitude per layer (meters)
            min_displacement: Minimum displacement magnitude per layer (meters)
            repulsion_strength: Strength of repulsion between latents (0 = no repulsion)
                Recommended: 0.2-0.5 to prevent piling up
            gravity_power: Power for inverse distance law (2 = inverse square)
            depth: Number of encoder layers
            share_weights: If True, share error predictor across layers
            pos_hidden_dim: Hidden dimension for position encoder
            error_hidden_dim: Hidden dimension for error MLP
            latent_surface: Total spatial extent of latent grid (meters)
            error_offset: Offset added to normalized errors (prevents zero gravity)
            danger_zone_divisor: Repulsion activates when dist < latent_spacing / divisor
                Higher = smaller danger zone = allows tighter clustering
                Default 4.0 allows ~4x density before repulsion kicks in
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latents_per_row = num_latents_per_row
        self.max_displacement = max_displacement
        self.min_displacement = min_displacement
        self.repulsion_strength = repulsion_strength
        self.gravity_power = gravity_power
        self.depth = depth
        self.share_weights = share_weights
        self.latent_surface = latent_surface
        self.error_offset = error_offset
        self.danger_zone_divisor = danger_zone_divisor
        
        # Compute latent spacing for reference
        self.latent_spacing = latent_surface / (num_latents_per_row - 1)
        self.danger_zone = self.latent_spacing / danger_zone_divisor
        
        # Create error predictor(s)
        if share_weights:
            self.error_predictor = ErrorPredictor(
                latent_dim=latent_dim,
                pos_hidden_dim=pos_hidden_dim,
                error_hidden_dim=error_hidden_dim,
            )
        else:
            self.error_predictors = nn.ModuleList([
                ErrorPredictor(
                    latent_dim=latent_dim,
                    pos_hidden_dim=pos_hidden_dim,
                    error_hidden_dim=error_hidden_dim,
                )
                for _ in range(depth)
            ])
        
        # Log initialization
        print(f"[GravityDisplacement] Initialized:")
        print(f"  latent_spacing={self.latent_spacing:.2f}m")
        print(f"  danger_zone={self.danger_zone:.2f}m (repulsion activates below this)")
        print(f"  max_displacement={max_displacement}m")
        print(f"  min_displacement={min_displacement}m")
        print(f"  repulsion_strength={repulsion_strength}")
        print(f"  gravity_power={gravity_power}")
        print(f"  error_offset={error_offset}")
        print(f"  share_weights={share_weights}")
    
    def get_error_predictor(self, layer_idx: int) -> ErrorPredictor:
        """Get error predictor for given layer."""
        if self.share_weights:
            return self.error_predictor
        else:
            return self.error_predictors[layer_idx]
    
    def normalize_errors(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Normalize errors to [0, 1] range per batch for scale invariance.
        
        Args:
            errors: [B, L] raw predicted errors
        
        Returns:
            errors_norm: [B, L] normalized to [offset, 1 + offset]
        """
        error_min = errors.min(dim=-1, keepdim=True).values
        error_max = errors.max(dim=-1, keepdim=True).values
        
        # Normalize to [0, 1]
        errors_norm = (errors - error_min) / (error_max - error_min + 1e-8)
        
        # Add offset so minimum isn't zero (prevents zero gravity)
        errors_norm = errors_norm + self.error_offset
        
        return errors_norm
    
    def compute_gravity_forces(
        self,
        positions: torch.Tensor,
        errors_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gravitational forces between all latent pairs.
        
        Each latent j exerts a pull on latent i:
            force_ij = (error_j / distance_ij^power) × direction_ij
        
        Args:
            positions: [B, L, 2] latent positions
            errors_norm: [B, L] normalized errors
        
        Returns:
            total_force: [B, L, 2] net gravitational force on each latent
        """
        B, L, _ = positions.shape
        device = positions.device
        
        # Pairwise position differences: [B, L, L, 2]
        # delta[b, i, j] = position[b, j] - position[b, i] (direction from i to j)
        pos_i = positions.unsqueeze(2)  # [B, L, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, L, 2]
        delta = pos_j - pos_i           # [B, L, L, 2]
        
        # Distances: [B, L, L]
        dist = delta.norm(dim=-1).clamp(min=1e-6)
        
        # Normalized direction: [B, L, L, 2]
        direction = delta / dist.unsqueeze(-1)
        
        # Gravity magnitude: error_j / distance^power
        # [B, 1, L] / [B, L, L] → [B, L, L]
        error_j = errors_norm.unsqueeze(1)  # [B, 1, L]
        gravity_magnitude = error_j / (dist ** self.gravity_power)
        
        # Mask self-interaction (latent doesn't pull itself)
        mask = ~torch.eye(L, dtype=torch.bool, device=device)
        gravity_magnitude = gravity_magnitude * mask.unsqueeze(0)
        
        # Total gravitational force on each latent: sum over all j
        # [B, L, L, 2] * [B, L, L, 1] → sum over j → [B, L, 2]
        gravity_force = (direction * gravity_magnitude.unsqueeze(-1)).sum(dim=2)
        
        return gravity_force
    
    def compute_repulsion_forces(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute repulsion forces between latents to prevent piling up.
        
        Uses margin-based repulsion that only activates in "danger zone":
        - No repulsion when dist >= latent_spacing/4 (free to cluster)
        - Strong exponential repulsion when dist < latent_spacing/4 (prevent piling)
        
        This allows healthy clustering (2-4 latents covering hard region)
        while preventing destructive piling (10+ latents at same spot → blur).
        
        Args:
            positions: [B, L, 2] latent positions
        
        Returns:
            total_repulsion: [B, L, 2] net repulsion force on each latent
        """
        B, L, _ = positions.shape
        device = positions.device
        
        # Pairwise position differences: [B, L, L, 2]
        pos_i = positions.unsqueeze(2)  # [B, L, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, L, 2]
        delta = pos_j - pos_i           # [B, L, L, 2]
        
        # Distances: [B, L, L]
        dist = delta.norm(dim=-1).clamp(min=1e-6)
        
        # Normalized direction (from i toward j): [B, L, L, 2]
        direction = delta / dist.unsqueeze(-1)
        
        # =====================================================================
        # Margin-based repulsion: only activates in "danger zone"
        # =====================================================================
        # Danger zone is pre-computed: latent_spacing / danger_zone_divisor
        # This allows clustering while preventing piling
        danger_zone = self.danger_zone  # e.g., ~1.35m if spacing is 5.4m and divisor is 4
        
        # How much are we violating the danger zone? (positive = violation)
        violation = (danger_zone - dist).clamp(min=0)  # [B, L, L]
        
        # Exponential repulsion for violations
        # dist = 0:              violation = danger_zone → repulsion = e^1 - 1 ≈ 1.7
        # dist = danger_zone/2:  violation = danger_zone/2 → repulsion = e^0.5 - 1 ≈ 0.65
        # dist >= danger_zone:   violation = 0 → repulsion = 0 (FREE ZONE)
        repulsion_magnitude = torch.exp(violation / danger_zone) - 1  # [B, L, L]
        
        # Mask self-interaction
        mask = ~torch.eye(L, dtype=torch.bool, device=device)
        repulsion_magnitude = repulsion_magnitude * mask.unsqueeze(0)
        
        # Repulsion is AWAY from other latents (negative direction)
        # Sum over all j → [B, L, 2]
        repulsion_force = -(direction * repulsion_magnitude.unsqueeze(-1)).sum(dim=2)
        
        return repulsion_force
    
    def forward(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gravity-based displacement for latents.
        
        Args:
            latents: [B, L, D] latent features (DETACHED - no gradient through encoder)
            positions: [B, L, 2] current latent positions
            layer_idx: which encoder layer (for layer-specific predictors)
        
        Returns:
            new_positions: [B, L, 2] updated positions
            displacement: [B, L, 2] how much each latent moved
            predicted_errors: [B, L] predicted error at each position (for supervision)
        """
        B, L, D = latents.shape
        
        # Get error predictor for this layer
        error_predictor = self.get_error_predictor(layer_idx)
        
        # =====================================================================
        # STEP 1: Predict error at all latent positions
        # =====================================================================
        # Detach latents to prevent encoder from "cheating"
        predicted_errors = error_predictor(latents.detach(), positions.detach())  # [B, L]
        
        # =====================================================================
        # STEP 2: Normalize errors for scale invariance
        # =====================================================================
        errors_norm = self.normalize_errors(predicted_errors)  # [B, L] in [offset, 1+offset]
        
        # =====================================================================
        # STEP 3: Compute gravitational attraction toward high-error latents
        # =====================================================================
        gravity_force = self.compute_gravity_forces(positions, errors_norm)  # [B, L, 2]
        
        # =====================================================================
        # STEP 4: Compute repulsion between latents (optional)
        # =====================================================================
        if self.repulsion_strength > 0:
            repulsion_force = self.compute_repulsion_forces(positions)  # [B, L, 2]
        else:
            repulsion_force = torch.zeros_like(gravity_force)
        
        # =====================================================================
        # STEP 5: Combine forces
        # =====================================================================
        # Gravity pulls toward high-error, repulsion spreads latents out
        total_force = gravity_force + self.repulsion_strength * repulsion_force  # [B, L, 2]
        
        # =====================================================================
        # STEP 6: Convert force to displacement
        # =====================================================================
        force_magnitude = total_force.norm(dim=-1, keepdim=True)  # [B, L, 1]
        force_direction = total_force / (force_magnitude + 1e-8)  # [B, L, 2]
        
        # Normalize force magnitude across latents (relative strength)
        # High relative force → larger displacement
        force_mean = force_magnitude.mean(dim=1, keepdim=True)  # [B, 1, 1]
        force_relative = force_magnitude / (force_mean + 1e-8)  # [B, L, 1]
        
        # Map relative force to displacement magnitude
        # Clamp to [0, 2] range, then interpolate between min and max
        force_clamped = force_relative.clamp(0, 2)  # [B, L, 1]
        displacement_magnitude = (
            self.min_displacement + 
            (force_clamped / 2) * (self.max_displacement - self.min_displacement)
        )  # [B, L, 1]
        
        # Final displacement
        displacement = force_direction * displacement_magnitude  # [B, L, 2]
        
        # =====================================================================
        # STEP 7: Update positions
        # =====================================================================
        new_positions = positions.detach() + displacement.detach()
        
        return new_positions, displacement, predicted_errors


def create_gravity_displacement(config: dict) -> GravityDisplacement:
    """
    Factory function to create GravityDisplacement from config.
    
    Config keys used:
        Atomiser.latent_dim: int
        Atomiser.spatial_latents: int (per row)
        Atomiser.max_displacement: float (default 5.0)
        Atomiser.min_displacement: float (default 0.5)
        Atomiser.repulsion_strength: float (default 0.3)
        Atomiser.gravity_power: float (default 2.0)
        Atomiser.depth: int
        Atomiser.share_error_predictor_weights: bool (default True)
        Atomiser.error_pos_hidden_dim: int (default 64)
        Atomiser.error_hidden_dim: int (default 256)
        Atomiser.latent_surface: float (default 103.0)
        Atomiser.error_offset: float (default 0.1)
        Atomiser.danger_zone_divisor: float (default 4.0)
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        GravityDisplacement instance
    """
    atomiser_config = config["Atomiser"]
    
    return GravityDisplacement(
        latent_dim=atomiser_config.get("latent_dim", 256),
        num_latents_per_row=atomiser_config["spatial_latents"],
        max_displacement=atomiser_config.get("max_displacement", 5.0),
        min_displacement=atomiser_config.get("min_displacement", 0.5),
        repulsion_strength=atomiser_config.get("repulsion_strength", 0.3),
        gravity_power=atomiser_config.get("gravity_power", 2.0),
        depth=atomiser_config["depth"],
        share_weights=atomiser_config.get("share_error_predictor_weights", True),
        pos_hidden_dim=atomiser_config.get("error_pos_hidden_dim", 64),
        error_hidden_dim=atomiser_config.get("error_hidden_dim", 256),
        latent_surface=atomiser_config.get("latent_surface", 103.0),
        error_offset=atomiser_config.get("error_offset", 0.1),
        danger_zone_divisor=atomiser_config.get("danger_zone_divisor", 4.0),
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing GravityDisplacement...")
    
    # Config
    B, L, D = 2, 400, 256
    num_per_row = 20
    
    # Create module
    gravity = GravityDisplacement(
        latent_dim=D,
        num_latents_per_row=num_per_row,
        max_displacement=5.0,
        min_displacement=0.5,
        repulsion_strength=0.3,
        depth=4,
    )
    
    # Random inputs
    latents = torch.randn(B, L, D)
    
    # Grid positions
    x = torch.linspace(0, 103, num_per_row)
    y = torch.linspace(0, 103, num_per_row)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [L, 2]
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, L, 2]
    
    # Forward pass
    new_pos, displacement, errors = gravity(latents, positions, layer_idx=0)
    
    print(f"\nInput shapes:")
    print(f"  latents: {latents.shape}")
    print(f"  positions: {positions.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  new_positions: {new_pos.shape}")
    print(f"  displacement: {displacement.shape}")
    print(f"  predicted_errors: {errors.shape}")
    
    print(f"\nDisplacement stats:")
    disp_mag = displacement.norm(dim=-1)
    print(f"  mean magnitude: {disp_mag.mean().item():.4f}m")
    print(f"  max magnitude: {disp_mag.max().item():.4f}m")
    print(f"  min magnitude: {disp_mag.min().item():.4f}m")
    
    print(f"\nError stats:")
    print(f"  mean error: {errors.mean().item():.4f}")
    print(f"  std error: {errors.std().item():.4f}")
    
    print("\n✓ Test passed!")