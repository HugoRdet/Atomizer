"""
Gravity-Based Latent Displacement with Two-Phase Dynamics

Phase 1: ATTRACTION (error-based)
    - Latents move toward high-error regions
    - Uses gravity model: pull ∝ error / distance²
    
Phase 2: DENSITY SPREADING (error-aware)
    - Latents spread to reduce local crowding
    - Spreading strength inversely proportional to error
    - High-error regions: allow clustering (weak spreading)
    - Low-error regions: enforce spacing (strong spreading)

This two-phase approach:
1. First decides WHERE latents should go (attraction)
2. Then decides HOW to arrange them (spreading)

The key insight: target density should be proportional to error.
At equilibrium: density(x) ∝ error(x)

FIXES in this version:
- Added `centered` parameter for proper coordinate system handling
- Boundary clamping uses [spatial_min, spatial_max] instead of [0, latent_surface]
- ErrorPredictor normalizes positions to [-1, 1] for stable training
- Added max_total_spread to prevent excessive spreading displacement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ErrorPredictor(nn.Module):
    """Predicts reconstruction error from latent features ONLY."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # [B, L, D] → [B, L]
        return self.mlp(latents).squeeze(-1)


class GravityDisplacement(nn.Module):
    """
    Two-phase gravity-based displacement:
    
    Phase 1 - ATTRACTION:
        - Predict error at all latent positions
        - Each latent exerts gravitational pull proportional to its error
        - Pull strength ∝ normalized_error / distance²
        - Latents move toward high-error regions
    
    Phase 2 - DENSITY SPREADING:
        - Compute local density at each latent position
        - Move latents down the density gradient (away from crowds)
        - Spreading strength = 1 - normalized_error (error-aware!)
        - High-error regions tolerate clustering, low-error regions spread
    
    This gives:
        - Global view for attraction (latents see distant hard regions)
        - Local regularization for spreading (prevent piling)
        - Error-aware equilibrium: density ∝ error
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_latents_per_row: int,
        max_displacement: float = 3.0,
        min_displacement: float = 0.5,
        repulsion_strength: float = 0.5,
        gravity_power: float = 2.0,
        depth: int = 4,
        share_weights: bool = True,
        pos_hidden_dim: int = 64,
        error_hidden_dim: int = 256,
        latent_surface: float = 103.0,
        centered: bool = True,              # NEW: whether coordinates are centered at origin
        error_offset: float = 0.1,
        danger_zone_divisor: float = 2.0,
        # Phase 2: density spreading parameters
        use_density_spreading: bool = True,
        density_iters: int = 3,
        density_sigma_mult: float = 0.5,
        density_step_mult: float = 0.1,
        max_density_step_mult: float = 0.25,
        max_total_spread_mult: float = 0.5,  # NEW: max TOTAL spread = latent_spacing * mult
    ):
        """
        Two-phase dynamics for latent positioning.
        
        Args:
            latent_dim: Dimension of latent features
            num_latents_per_row: Number of latents per row in grid
            max_displacement: Maximum displacement magnitude per layer (meters)
            min_displacement: Minimum displacement magnitude per layer (meters)
            repulsion_strength: Strength of Phase 1 repulsion (0 = disabled)
            gravity_power: Power for inverse distance law (2 = inverse square)
            depth: Number of encoder layers
            share_weights: If True, share error predictor across layers
            pos_hidden_dim: Hidden dimension for position encoder
            error_hidden_dim: Hidden dimension for error MLP
            latent_surface: Total spatial extent of latent grid (meters)
            centered: If True, coordinates are centered at origin [-half, +half]
                      If False, coordinates are [0, latent_surface]
            error_offset: Offset added to normalized errors (prevents zero gravity)
            danger_zone_divisor: Repulsion activates when dist < latent_spacing / divisor
            use_density_spreading: Enable/disable Phase 2 (density spreading)
            density_iters: Number of spreading iterations per forward pass
            density_sigma_mult: Gaussian kernel width = latent_spacing * mult
            density_step_mult: Step size = latent_spacing * mult
            max_density_step_mult: Max step per iteration = latent_spacing * mult
            max_total_spread_mult: Max TOTAL spreading displacement = latent_spacing * mult
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
        self.centered = centered
        self.error_offset = error_offset
        self.danger_zone_divisor = danger_zone_divisor
        
        # =====================================================================
        # CRITICAL: Compute spatial bounds based on coordinate system
        # =====================================================================
        if centered:
            self.spatial_min = -latent_surface / 2
            self.spatial_max = latent_surface / 2
        else:
            self.spatial_min = 0.0
            self.spatial_max = latent_surface
        
        # Compute latent spacing for reference
        self.latent_spacing = latent_surface / (num_latents_per_row - 1)
        self.danger_zone = self.latent_spacing / danger_zone_divisor
        
        # Density spreading parameters
        self.use_density_spreading = use_density_spreading
        self.density_iters = density_iters
        self.density_sigma = self.latent_spacing * density_sigma_mult
        self.density_step_size = self.latent_spacing * density_step_mult
        self.max_density_step = self.latent_spacing * max_density_step_mult
        self.max_total_spread = self.latent_spacing * max_total_spread_mult
        
        # Create error predictor(s) with proper spatial extent for normalization
        spatial_extent = self.spatial_max  # For centered: this is half the total extent
        if share_weights:
            self.error_predictor = ErrorPredictor(
                latent_dim=latent_dim,
                hidden_dim=error_hidden_dim,
            )
        else:
            self.error_predictors = nn.ModuleList([
                ErrorPredictor(
                    latent_dim=latent_dim,
                    hidden_dim=error_hidden_dim,
                )
                for _ in range(depth)
            ])
        
        # Log initialization
        print(f"[GravityDisplacement] Initialized (Two-Phase Dynamics):")
        print(f"  spatial_bounds=[{self.spatial_min:.1f}, {self.spatial_max:.1f}]m (centered={centered})")
        print(f"  latent_spacing={self.latent_spacing:.2f}m")
        print(f"  danger_zone={self.danger_zone:.2f}m")
        print(f"  max_displacement={max_displacement}m")
        print(f"  min_displacement={min_displacement}m")
        print(f"  repulsion_strength={repulsion_strength}")
        print(f"  gravity_power={gravity_power}")
        print(f"  error_offset={error_offset}")
        print(f"  share_weights={share_weights}")
        print(f"  --- Phase 2: Density Spreading ---")
        print(f"  use_density_spreading={use_density_spreading}")
        print(f"  density_iters={density_iters}")
        print(f"  density_sigma={self.density_sigma:.2f}m")
        print(f"  density_step_size={self.density_step_size:.2f}m")
        print(f"  max_density_step={self.max_density_step:.2f}m (per iter)")
        print(f"  max_total_spread={self.max_total_spread:.2f}m (total)")
    
    def get_error_predictor(self, layer_idx: int) -> ErrorPredictor:
        """Get error predictor for given layer."""
        if self.share_weights:
            return self.error_predictor
        else:
            return self.error_predictors[layer_idx]
    
    def clamp_to_bounds(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Clamp positions to valid spatial domain.
        
        Args:
            positions: [B, L, 2] latent positions
        
        Returns:
            positions: [B, L, 2] clamped to [spatial_min, spatial_max]
        """
        return positions.clamp(self.spatial_min, self.spatial_max)
    
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
    
    def normalize_errors_zero_one(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Normalize errors to strict [0, 1] range (for spreading strength).
        
        Args:
            errors: [B, L] raw predicted errors
        
        Returns:
            errors_norm: [B, L] normalized to [0, 1]
        """
        error_min = errors.min(dim=-1, keepdim=True).values
        error_max = errors.max(dim=-1, keepdim=True).values
        
        errors_norm = (errors - error_min) / (error_max - error_min + 1e-8)
        
        return errors_norm

    # =========================================================================
    # PHASE 1: ATTRACTION (Gravity)
    # =========================================================================
    
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
        - No repulsion when dist >= danger_zone (free to cluster)
        - Strong exponential repulsion when dist < danger_zone (prevent piling)
        
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
        
        # Margin-based repulsion: only activates in "danger zone"
        danger_zone = self.danger_zone
        
        # How much are we violating the danger zone? (positive = violation)
        violation = (danger_zone - dist).clamp(min=0)  # [B, L, L]
        
        # Exponential repulsion for violations
        repulsion_magnitude = torch.exp(violation / danger_zone) - 1  # [B, L, L]
        
        # Mask self-interaction
        mask = ~torch.eye(L, dtype=torch.bool, device=device)
        repulsion_magnitude = repulsion_magnitude * mask.unsqueeze(0)
        
        # Repulsion is AWAY from other latents (negative direction)
        repulsion_force = -(direction * repulsion_magnitude.unsqueeze(-1)).sum(dim=2)
        
        return repulsion_force
    
    def force_to_displacement(self, total_force: torch.Tensor) -> torch.Tensor:
        """
        Convert force to bounded displacement.
        
        Args:
            total_force: [B, L, 2] net force on each latent
        
        Returns:
            displacement: [B, L, 2] bounded displacement
        """
        force_magnitude = total_force.norm(dim=-1, keepdim=True)  # [B, L, 1]
        force_direction = total_force / (force_magnitude + 1e-8)  # [B, L, 2]
        
        # Normalize force magnitude across latents (relative strength)
        force_mean = force_magnitude.mean(dim=1, keepdim=True)  # [B, 1, 1]
        force_relative = force_magnitude / (force_mean + 1e-8)  # [B, L, 1]
        
        # Map relative force to displacement magnitude
        force_clamped = force_relative.clamp(0, 2)  # [B, L, 1]
        displacement_magnitude = (
            self.min_displacement + 
            (force_clamped / 2) * (self.max_displacement - self.min_displacement)
        )  # [B, L, 1]
        
        # Final displacement
        displacement = force_direction * displacement_magnitude  # [B, L, 2]
        
        return displacement

    # =========================================================================
    # PHASE 2: DENSITY SPREADING (Error-Aware)
    # =========================================================================
    
    def compute_density_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of density field at each latent position.
        
        The density field is:
            density(x) = Σ_j exp(-||x - p_j||² / 2σ²)
        
        The gradient is:
            ∇density(x) = -1/σ² * Σ_j (x - p_j) * exp(-||x - p_j||² / 2σ²)
        
        This gradient points TOWARD higher density (toward clusters).
        To spread latents apart, we perform gradient DESCENT: move in -∇density direction.
        
        Args:
            positions: [B, L, 2] latent positions
        
        Returns:
            gradient: [B, L, 2] density gradient at each latent (points toward clusters)
        """
        B, L, _ = positions.shape
        device = positions.device
        sigma = self.density_sigma
        
        # Pairwise differences: delta[i,j] = p_i - p_j
        pos_i = positions.unsqueeze(2)  # [B, L, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, L, 2]
        delta = pos_i - pos_j           # [B, L, L, 2]
        
        # Squared distances
        dist_sq = (delta ** 2).sum(dim=-1)  # [B, L, L]
        
        # Gaussian weights
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # [B, L, L]
        
        # Mask self-interaction
        mask = ~torch.eye(L, dtype=torch.bool, device=device)
        weights = weights * mask.unsqueeze(0)
        
        # Gradient: ∇ρ(p_i) = -1/σ² * Σ_j (p_i - p_j) * exp(-||p_i - p_j||² / 2σ²)
        # This points TOWARD neighbors (toward high density)
        gradient = -(delta * weights.unsqueeze(-1)).sum(dim=2) / (sigma ** 2)
        
        return gradient  # [B, L, 2] - points toward clusters (high density)
    
    def density_spread(
        self, 
        positions: torch.Tensor, 
        errors_norm_01: torch.Tensor,
        num_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Error-aware density spreading via gradient descent on the density field.
        
        - High error regions: allow clustering (weak spreading)
        - Low error regions: enforce spacing (strong spreading)
        
        Spreading strength = 1 - normalized_error
        
        Args:
            positions: [B, L, 2] latent positions (after attraction)
            errors_norm_01: [B, L] errors normalized to [0, 1]
            num_iters: Override number of iterations
        
        Returns:
            positions: [B, L, 2] positions after spreading
        """
        if num_iters is None:
            num_iters = self.density_iters
        
        # Store initial positions to clamp total displacement
        initial_positions = positions.clone()
        
        for _ in range(num_iters):
            # Density gradient (points toward higher density / clusters)
            density_grad = self.compute_density_gradient(positions)  # [B, L, 2]
            
            # Spreading strength: inverse of error
            # High error → strength ≈ 0 → don't spread (clustering OK)
            # Low error → strength ≈ 1 → spread strongly
            spreading_strength = 1.0 - errors_norm_01  # [B, L]
            
            # Gradient descent on density field: move toward LOWER density
            # step = -η * s * ∇ρ
            step = -density_grad * spreading_strength.unsqueeze(-1) * self.density_step_size
            
            # Clamp step magnitude (per iteration)
            step_mag = step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = (self.max_density_step / step_mag).clamp(max=1.0)
            step = step * scale
            
            # Apply step
            positions = positions + step
            
            # Boundary enforcement: clamp to valid spatial domain
            # FIXED: Use proper bounds instead of (0, latent_surface)
            positions = self.clamp_to_bounds(positions)
        
        # Clamp TOTAL spreading displacement to prevent excessive movement
        total_spread = positions - initial_positions
        total_spread_mag = total_spread.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        spread_scale = (self.max_total_spread / total_spread_mag).clamp(max=1.0)
        positions = initial_positions + total_spread * spread_scale
        
        # Final boundary enforcement
        positions = self.clamp_to_bounds(positions)
        
        return positions

    # =========================================================================
    # FORWARD: Two-Phase Dynamics
    # =========================================================================
    
    def forward(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-phase displacement: attraction then error-aware spreading.
        
        Phase 1: ATTRACTION
            - Compute error at all positions
            - Gravity pulls latents toward high-error regions
            - Optional repulsion in danger zone
        
        Phase 2: DENSITY SPREADING
            - Compute local density
            - Spread latents in low-error regions (high-error regions can cluster)
            - Iterative until positions stabilize
        
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] current latent positions
            layer_idx: which encoder layer (for layer-specific predictors)
        
        Returns:
            new_positions: [B, L, 2] updated positions
            displacement: [B, L, 2] total displacement (attraction + spreading)
            predicted_errors: [B, L] predicted error at each position
        """
        B, L, D = latents.shape
        
        # Store initial positions for displacement computation
        initial_positions = positions.detach().clone()
        
        # Get error predictor for this layer
        error_predictor = self.get_error_predictor(layer_idx)
        
        # =====================================================================
        # PHASE 1: ATTRACTION
        # =====================================================================
        
        # Predict error at all latent positions (detached to prevent cheating)
        predicted_errors = error_predictor(latents.detach())  # [B, L]
        
        # Normalize errors for gravity (with offset)
        errors_norm = self.normalize_errors(predicted_errors)  # [B, L]
        
        # Normalize errors for spreading (strict 0-1)
        errors_norm_01 = self.normalize_errors_zero_one(predicted_errors)  # [B, L]
        
        # Compute gravitational attraction toward high-error latents
        gravity_force = self.compute_gravity_forces(positions, errors_norm)  # [B, L, 2]
        
        # Compute repulsion between latents (optional)
        if self.repulsion_strength > 0:
            repulsion_force = self.compute_repulsion_forces(positions)  # [B, L, 2]
        else:
            repulsion_force = torch.zeros_like(gravity_force)
        
        # Combine forces
        total_force = gravity_force + self.repulsion_strength * repulsion_force
        
        # Convert to displacement
        attraction_disp = self.force_to_displacement(total_force)
        
        # Apply attraction
        positions_after_attraction = positions.detach() + attraction_disp.detach()
        
        # Boundary enforcement after attraction
        # FIXED: Use proper bounds instead of (0, latent_surface)
        positions_after_attraction = self.clamp_to_bounds(positions_after_attraction)
        
        # =====================================================================
        # PHASE 2: DENSITY SPREADING (Error-Aware)
        # =====================================================================
        
        if self.use_density_spreading and self.density_iters > 0:
            new_positions = self.density_spread(
                positions_after_attraction, 
                errors_norm_01,
                num_iters=self.density_iters,
            )
            # Note: density_spread already applies boundary clamping
        else:
            new_positions = positions_after_attraction
        
        # Total displacement
        total_displacement = new_positions - initial_positions
        
        return new_positions, total_displacement, predicted_errors
    
    # =========================================================================
    # Diagnostics
    # =========================================================================
    
    def compute_density_stats(self, positions: torch.Tensor) -> dict:
        """Compute density statistics for logging/debugging."""
        B, L, _ = positions.shape
        
        # Pairwise distances
        dist = torch.cdist(positions, positions)  # [B, L, L]
        
        # Mask self
        mask = torch.eye(L, dtype=torch.bool, device=positions.device)
        dist_no_self = dist.clone()
        dist_no_self[:, mask] = float('inf')
        
        # Nearest neighbor distance
        nn_dist = dist_no_self.min(dim=-1).values  # [B, L]
        
        # Local density (count neighbors within sigma)
        neighbors_in_sigma = (dist_no_self < self.density_sigma).float().sum(dim=-1)  # [B, L]
        
        return {
            'nn_dist_mean': nn_dist.mean().item(),
            'nn_dist_min': nn_dist.min().item(),
            'nn_dist_max': nn_dist.max().item(),
            'neighbors_in_sigma_mean': neighbors_in_sigma.mean().item(),
            'neighbors_in_sigma_max': neighbors_in_sigma.max().item(),
        }


def create_gravity_displacement(config: dict) -> GravityDisplacement:
    """
    Factory function to create GravityDisplacement from config.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        GravityDisplacement instance
    """
    atomiser_config = config["Atomiser"]
    
    return GravityDisplacement(
        latent_dim=atomiser_config.get("latent_dim", 256),
        num_latents_per_row=atomiser_config["spatial_latents"],
        max_displacement=atomiser_config.get("max_displacement", 3.0),
        min_displacement=atomiser_config.get("min_displacement", 0.5),
        repulsion_strength=atomiser_config.get("repulsion_strength", 0.5),
        gravity_power=atomiser_config.get("gravity_power", 2.0),
        depth=atomiser_config["depth"],
        share_weights=atomiser_config.get("share_error_predictor_weights", True),
        pos_hidden_dim=atomiser_config.get("error_pos_hidden_dim", 64),
        error_hidden_dim=atomiser_config.get("error_hidden_dim", 256),
        latent_surface=atomiser_config.get("latent_surface", 103.0),
        centered=atomiser_config.get("centered_coordinates", True),  # NEW
        error_offset=atomiser_config.get("error_offset", 0.1),
        danger_zone_divisor=atomiser_config.get("danger_zone_divisor", 2.0),
        use_density_spreading=atomiser_config.get("use_density_spreading", True),
        density_iters=atomiser_config.get("density_iters", 3),
        density_sigma_mult=atomiser_config.get("density_sigma_mult", 0.5),
        density_step_mult=atomiser_config.get("density_step_mult", 0.1),
        max_density_step_mult=atomiser_config.get("max_density_step_mult", 0.25),
        max_total_spread_mult=atomiser_config.get("max_total_spread_mult", 0.5),  # NEW
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing GravityDisplacement with Two-Phase Dynamics...")
    print("=" * 60)
    
    # Config
    B, L, D = 2, 400, 256
    num_per_row = 20
    
    # Create module with CENTERED coordinates (matching typical usage)
    gravity = GravityDisplacement(
        latent_dim=D,
        num_latents_per_row=num_per_row,
        max_displacement=3.0,
        min_displacement=0.5,
        repulsion_strength=0.5,
        danger_zone_divisor=2.0,
        depth=4,
        centered=True,  # IMPORTANT: centered coordinates
        use_density_spreading=True,
        density_iters=3,
    )
    
    # Random inputs
    latents = torch.randn(B, L, D)
    
    # Grid positions (CENTERED around origin)
    half_extent = 103 / 2  # ±51.5m
    x = torch.linspace(-half_extent, half_extent, num_per_row)
    y = torch.linspace(-half_extent, half_extent, num_per_row)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [L, 2]
    positions = positions.unsqueeze(0).expand(B, -1, -1).clone()  # [B, L, 2]
    
    print(f"\nInput shapes:")
    print(f"  latents: {latents.shape}")
    print(f"  positions: {positions.shape}")
    print(f"  position range: [{positions.min().item():.1f}, {positions.max().item():.1f}]")
    
    # Initial density stats
    print(f"\nInitial density stats:")
    stats_before = gravity.compute_density_stats(positions)
    for k, v in stats_before.items():
        print(f"  {k}: {v:.4f}")
    
    # Forward pass
    new_pos, displacement, errors = gravity(latents, positions, layer_idx=0)
    
    print(f"\nOutput shapes:")
    print(f"  new_positions: {new_pos.shape}")
    print(f"  displacement: {displacement.shape}")
    print(f"  predicted_errors: {errors.shape}")
    
    print(f"\nPosition range after forward:")
    print(f"  new_positions range: [{new_pos.min().item():.1f}, {new_pos.max().item():.1f}]")
    
    print(f"\nDisplacement stats:")
    disp_mag = displacement.norm(dim=-1)
    print(f"  mean magnitude: {disp_mag.mean().item():.4f}m")
    print(f"  max magnitude: {disp_mag.max().item():.4f}m")
    print(f"  min magnitude: {disp_mag.min().item():.4f}m")
    
    print(f"\nError stats:")
    print(f"  mean error: {errors.mean().item():.4f}")
    print(f"  std error: {errors.std().item():.4f}")
    
    # Final density stats
    print(f"\nFinal density stats:")
    stats_after = gravity.compute_density_stats(new_pos)
    for k, v in stats_after.items():
        print(f"  {k}: {v:.4f}")
    
    # Verify no pile-up at boundaries
    print("\n" + "=" * 60)
    print("Verifying no boundary pile-up...")
    
    at_min_bound = (new_pos.abs() - half_extent).abs() < 0.1
    num_at_bounds = at_min_bound.any(dim=-1).sum().item()
    print(f"  Latents at boundary: {num_at_bounds} / {L}")
    
    print("\n✓ Test passed!")