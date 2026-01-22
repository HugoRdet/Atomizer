"""
Gravity Displacement with MLP Error Predictor

Two-phase gravity-based displacement for adaptive latent positioning.

Key features:
- ERROR ANOMALY mass: removes geometric center bias
- Simple MLP error predictor: latent features → scalar error
- Two-phase dynamics: attraction + density spreading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# =============================================================================
# ERROR PREDICTOR (MLP)
# =============================================================================

class ErrorPredictor(nn.Module):
    """
    Predicts reconstruction error from latent features.
    
    Simple MLP: latent features → hidden → error scalar
    """
    
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
        """
        Args:
            latents: [B, L, D] latent features
            
        Returns:
            errors: [B, L] predicted error per latent
        """
        return self.mlp(latents).squeeze(-1)


# =============================================================================
# GRAVITY DISPLACEMENT
# =============================================================================

class GravityDisplacement(nn.Module):
    """
    Two-phase gravity-based displacement with error anomaly mass.
    
    Key innovation: Uses ERROR ANOMALY (error - mean_error) as gravitational mass.
    This removes the geometric center bias where boundary latents are pulled
    inward simply because there's more "space" in the interior.
    
    With anomaly mass:
    - Uniform error → zero net force (no geometric bias)
    - Above-average error → positive mass (attracts)
    - Below-average error → negative mass (repels)
    
    Two phases:
    - Phase 1 (Attraction): Gravity pulls latents toward high-error regions
    - Phase 2 (Spreading): Density-based spreading prevents piling
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
        error_hidden_dim: int = 256,
        latent_surface: float = 103.0,
        centered: bool = True,
        error_offset: float = 0.1,
        danger_zone_divisor: float = 2.0,
        use_density_spreading: bool = True,
        density_iters: int = 3,
        density_sigma_mult: float = 0.5,
        density_step_mult: float = 0.1,
        max_density_step_mult: float = 0.25,
        max_total_spread_mult: float = 0.5,
        freeze_boundary: bool = False,
        use_error_anomaly: bool = True,
    ):
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
        self.freeze_boundary = freeze_boundary
        self.use_error_anomaly = use_error_anomaly
        
        # Spatial bounds
        if centered:
            self.spatial_min = -latent_surface / 2
            self.spatial_max = latent_surface / 2
        else:
            self.spatial_min = 0.0
            self.spatial_max = latent_surface
        
        # Spacing stats
        self.latent_spacing = latent_surface / (num_latents_per_row - 1)
        self.danger_zone = self.latent_spacing / danger_zone_divisor
        
        # Density parameters
        self.use_density_spreading = use_density_spreading
        self.density_iters = density_iters
        self.density_sigma = self.latent_spacing * density_sigma_mult
        self.density_step_size = self.latent_spacing * density_step_mult
        self.max_density_step = self.latent_spacing * max_density_step_mult
        self.max_total_spread = self.latent_spacing * max_total_spread_mult
        
        # Boundary mask
        if freeze_boundary:
            boundary_mask = self._create_boundary_mask()
            self.register_buffer('boundary_mask', boundary_mask)
        
        # Error Predictor(s)
        if share_weights:
            self.error_predictor = ErrorPredictor(latent_dim, error_hidden_dim)
        else:
            self.error_predictors = nn.ModuleList([
                ErrorPredictor(latent_dim, error_hidden_dim)
                for _ in range(depth)
            ])
        
        # Log
        print(f"[GravityDisplacement] Initialized with MLP error predictor")
        print(f"[GravityDisplacement] use_error_anomaly={use_error_anomaly}")
        print(f"[GravityDisplacement] latent_spacing={self.latent_spacing:.2f}m")
        print(f"[GravityDisplacement] max_displacement={max_displacement}m, min_displacement={min_displacement}m")
    
    def _create_boundary_mask(self) -> torch.Tensor:
        """Create mask for boundary latents."""
        n = self.num_latents_per_row
        L = n * n
        indices = torch.arange(L)
        rows, cols = indices // n, indices % n
        return (rows == 0) | (rows == n - 1) | (cols == 0) | (cols == n - 1)
    
    def get_error_predictor(self, layer_idx: int) -> ErrorPredictor:
        """Get error predictor for given layer."""
        return self.error_predictor if self.share_weights else self.error_predictors[layer_idx]
    
    def clamp_to_bounds(self, positions: torch.Tensor) -> torch.Tensor:
        """Clamp positions to spatial bounds."""
        return positions.clamp(self.spatial_min, self.spatial_max)
    
    def apply_boundary_freeze(
        self,
        new_pos: torch.Tensor,
        initial_pos: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freeze boundary latents at their initial positions."""
        if not self.freeze_boundary:
            return new_pos, disp
        mask = self.boundary_mask.view(1, -1, 1).expand_as(new_pos)
        new_pos = torch.where(mask, initial_pos, new_pos)
        disp = torch.where(mask, torch.zeros_like(disp), disp)
        return new_pos, disp

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _robust_normalize(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Log-robust normalization to [0, 1].
        
        Log compression prevents extreme errors from dominating.
        """
        errors_log = torch.log1p(errors)
        val_min = errors_log.min(dim=-1, keepdim=True).values
        val_max = errors_log.max(dim=-1, keepdim=True).values
        denom = torch.maximum(val_max - val_min, torch.tensor(1e-6, device=errors.device))
        return (errors_log - val_min) / denom
    
    def normalize_errors(self, errors: torch.Tensor) -> torch.Tensor:
        """Normalize for Gravity (Phase 1)."""
        norm = self._robust_normalize(errors)
        if self.use_error_anomaly:
            # With anomaly mass, no offset needed
            return norm
        else:
            # Legacy: add offset so 0.0 still exerts some pull
            return norm + self.error_offset
    
    def normalize_errors_zero_one(self, errors: torch.Tensor) -> torch.Tensor:
        """Normalize for Spreading (Phase 2). Strict [0, 1]."""
        return self._robust_normalize(errors)

    # =========================================================================
    # PHYSICS ENGINES
    # =========================================================================
    
    def compute_gravity_forces(
        self,
        positions: torch.Tensor,
        errors_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gravitational forces with error anomaly mass.
        
        Key insight: mass = error - mean_error removes geometric bias.
        - Uniform error → zero net force
        - Above-average error → attracts
        - Below-average error → repels
        """
        B, L, _ = positions.shape
        
        # Pairwise vectors
        pos_i = positions.unsqueeze(2)  # [B, L, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, L, 2]
        delta = pos_j - pos_i           # [B, L, L, 2]
        dist = delta.norm(dim=-1).clamp(min=1e-6)  # [B, L, L]
        direction = delta / dist.unsqueeze(-1)     # [B, L, L, 2]
        
        # Compute mass
        if self.use_error_anomaly:
            mean_error = errors_norm.mean(dim=-1, keepdim=True)  # [B, 1]
            error_anomaly = errors_norm - mean_error             # [B, L]
            mass_j = error_anomaly.unsqueeze(1)                  # [B, 1, L]
        else:
            mass_j = errors_norm.unsqueeze(1)  # [B, 1, L]
        
        # Gravity: mass / dist^power
        gravity_mag = mass_j / (dist ** self.gravity_power)  # [B, L, L]
        
        # Mask self-interaction
        mask = ~torch.eye(L, dtype=torch.bool, device=positions.device)
        gravity_mag = gravity_mag * mask.unsqueeze(0)
        
        # Sum forces
        forces = (direction * gravity_mag.unsqueeze(-1)).sum(dim=2)  # [B, L, 2]
        return forces
    
    def compute_repulsion_forces(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute short-range repulsion to prevent latent overlap."""
        B, L, _ = positions.shape
        
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        delta = pos_j - pos_i
        dist = delta.norm(dim=-1).clamp(min=1e-6)
        direction = delta / dist.unsqueeze(-1)
        
        # Exponential repulsion when closer than danger_zone
        violation = (self.danger_zone - dist).clamp(min=0)
        repulsion_mag = torch.exp(violation / self.danger_zone) - 1
        
        mask = ~torch.eye(L, dtype=torch.bool, device=positions.device)
        repulsion_mag = repulsion_mag * mask.unsqueeze(0)
        
        # Negative = repulsive (away from neighbors)
        return -(direction * repulsion_mag.unsqueeze(-1)).sum(dim=2)
    
    def force_to_displacement(self, total_force: torch.Tensor) -> torch.Tensor:
        """Convert force magnitude to bounded displacement."""
        force_mag = total_force.norm(dim=-1, keepdim=True)
        force_dir = total_force / (force_mag + 1e-8)
        
        # Relative to batch mean
        force_mean = force_mag.mean(dim=1, keepdim=True)
        force_rel = force_mag / (force_mean + 1e-8)
        
        # Map to [min_displacement, max_displacement]
        force_clamped = force_rel.clamp(0, 2)
        disp_mag = self.min_displacement + (force_clamped / 2) * (self.max_displacement - self.min_displacement)
        
        return force_dir * disp_mag

    def compute_density_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Gaussian density field."""
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        delta = pos_i - pos_j
        dist_sq = (delta ** 2).sum(dim=-1)
        weights = torch.exp(-dist_sq / (2 * self.density_sigma ** 2))
        
        mask = ~torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)
        weights = weights * mask.unsqueeze(0)
        
        # Gradient points toward clusters
        return -(delta * weights.unsqueeze(-1)).sum(dim=2) / (self.density_sigma ** 2)

    def density_spread(
        self,
        positions: torch.Tensor,
        errors_norm_01: torch.Tensor,
        initial_positions: torch.Tensor,
        num_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Phase 2: Error-aware density spreading.
        
        High error → weak spreading (allow clustering)
        Low error → strong spreading (disperse)
        """
        if num_iters is None:
            num_iters = self.density_iters
        pos_start = positions.clone()
        
        for _ in range(num_iters):
            grad = self.compute_density_gradient(positions)
            
            # Spreading strength: low error → strong spreading
            strength = 1.0 - errors_norm_01
            step = -grad * strength.unsqueeze(-1) * self.density_step_size
            
            # Clamp step magnitude
            step_mag = step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = (self.max_density_step / step_mag).clamp(max=1.0)
            
            positions = positions + step * scale
            positions = self.clamp_to_bounds(positions)
            
            if self.freeze_boundary:
                mask = self.boundary_mask.view(1, -1, 1).expand_as(positions)
                positions = torch.where(mask, initial_positions, positions)
        
        # Clamp total spread
        total_disp = positions - pos_start
        total_mag = total_disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.max_total_spread / total_mag).clamp(max=1.0)
        positions = pos_start + total_disp * scale
        
        positions = self.clamp_to_bounds(positions)
        
        if self.freeze_boundary:
            mask = self.boundary_mask.view(1, -1, 1).expand_as(positions)
            positions = torch.where(mask, initial_positions, positions)
        
        return positions

    def forward(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute displacement for latents.
        
        Args:
            latents: [B, L, D] latent features
            positions: [B, L, 2] current positions in meters
            layer_idx: current encoder layer index
            
        Returns:
            new_positions: [B, L, 2] updated positions
            displacement: [B, L, 2] total displacement applied
            raw_errors: [B, L] predicted reconstruction errors
        """
        initial_pos = positions.detach().clone()
        
        # 1. Predict errors
        raw_errors = self.get_error_predictor(layer_idx)(latents.detach())
        errors_norm = self.normalize_errors(raw_errors)
        errors_norm_01 = self.normalize_errors_zero_one(raw_errors)
        
        # 2. Phase 1: Gravitational attraction
        g_force = self.compute_gravity_forces(positions, errors_norm)
        r_force = self.compute_repulsion_forces(positions) if self.repulsion_strength > 0 else 0
        
        attr_disp = self.force_to_displacement(g_force + self.repulsion_strength * r_force)
        pos_after = self.clamp_to_bounds(positions.detach() + attr_disp.detach())
        pos_after, _ = self.apply_boundary_freeze(pos_after, initial_pos, attr_disp)
        
        # 3. Phase 2: Density spreading
        new_pos = pos_after
        if self.use_density_spreading and self.density_iters > 0:
            new_pos = self.density_spread(pos_after, errors_norm_01, initial_pos)
        
        total_disp = new_pos - initial_pos
        new_pos, total_disp = self.apply_boundary_freeze(new_pos, initial_pos, total_disp)
        
        return new_pos, total_disp, raw_errors


# =============================================================================
# FACTORY
# =============================================================================

def create_gravity_displacement(config: dict) -> GravityDisplacement:
    """Create GravityDisplacement from config dict."""
    c = config["Atomiser"]
    
    return GravityDisplacement(
        latent_dim=c.get("latent_dim", 256),
        num_latents_per_row=c["spatial_latents"],
        max_displacement=c.get("max_displacement", 3.0),
        min_displacement=c.get("min_displacement", 0.5),
        repulsion_strength=c.get("repulsion_strength", 0.5),
        gravity_power=c.get("gravity_power", 2.0),
        depth=c["depth"],
        share_weights=c.get("share_error_predictor_weights", True),
        error_hidden_dim=c.get("error_hidden_dim", 256),
        latent_surface=c.get("latent_surface", 103.0),
        centered=c.get("centered_coordinates", True),
        error_offset=c.get("error_offset", 0.1),
        danger_zone_divisor=c.get("danger_zone_divisor", 2.0),
        use_density_spreading=c.get("use_density_spreading", True),
        density_iters=c.get("density_iters", 3),
        density_sigma_mult=c.get("density_sigma_mult", 0.5),
        density_step_mult=c.get("density_step_mult", 0.1),
        max_density_step_mult=c.get("max_density_step_mult", 0.25),
        max_total_spread_mult=c.get("max_total_spread_mult", 0.5),
        freeze_boundary=c.get("freeze_boundary", False),
        use_error_anomaly=c.get("use_error_anomaly", True),
    )