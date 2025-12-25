"""
Latent Position Update Strategies

This module provides different strategies for updating latent positions
during encoding. Each strategy predicts displacements based on latent embeddings.

Strategies:
- NoPositionUpdate: Fixed positions (baseline)
- MLPDisplacementUpdate: Simple MLP predicts (Δx, Δy)
- DeformableOffsetUpdate: Multi-point sampling with attention weights

Usage:
    from displacement import create_position_updater
    
    updater = create_position_updater(config)
    new_coords, displacement = updater(latents, current_coords, layer_idx)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


# =============================================================================
# Base Class
# =============================================================================

class PositionUpdateStrategy(nn.Module):
    """Abstract base class for position update strategies."""
    
    def forward(
        self,
        latents: torch.Tensor,
        current_coords: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute position update for latents.
        
        Args:
            latents: [B, L, D] latent embeddings
            current_coords: [B, L, 2] current positions
            layer_idx: current encoder layer index
            
        Returns:
            new_coords: [B, L, 2] updated positions
            displacement: [B, L, 2] displacement applied
        """
        raise NotImplementedError


# =============================================================================
# Strategy: No Update (Baseline)
# =============================================================================

class NoPositionUpdate(PositionUpdateStrategy):
    """Fixed positions - no update (baseline)."""
    
    def forward(self, latents, current_coords, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return current_coords, torch.zeros_like(current_coords)


# =============================================================================
# Strategy: Simple MLP Displacement
# =============================================================================

class MLPDisplacementUpdate(PositionUpdateStrategy):
    """
    Simple MLP predicts (Δx, Δy) from latent embeddings.
    
    Architecture:
        latents [B, L, D] -> MLP -> raw [B, L, 2] -> tanh -> bounded displacement
    
    Initialized to output small random displacements to encourage exploration.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        hidden_dim: int = None,
        max_displacement: float = 50.0,  # Max pixels per layer
        share_weights: bool = True
    ):
        """
        Args:
            latent_dim: Dimension of latent embeddings
            num_layers: Number of encoder layers
            hidden_dim: Hidden dimension (default: latent_dim // 2)
            max_displacement: Maximum displacement magnitude per layer (in pixels)
            share_weights: If True, share MLP weights across layers (except first)
        """
        super().__init__()
        self.max_displacement = 10#max_displacement
        self.share_weights = share_weights
        self.num_encoder_layers = num_layers
        
        hidden_dim = hidden_dim or latent_dim // 2
        
        def make_predictor():
            return nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2)  # Output (Δx, Δy)
            )
        
        if share_weights:
            self.predictors = nn.ModuleList([make_predictor(), make_predictor()])
        else:
            self.predictors = nn.ModuleList([make_predictor() for _ in range(num_layers)])
        
        # Initialize with small random values (not zeros!)
        self._init_small_displacement()
    
    def _init_small_displacement(self):
        """Initialize final layer to output small but non-zero values."""
        for pred in self.predictors:
            # Small random weights so network starts with some movement
            nn.init.normal_(pred[-1].weight, mean=0.0, std=0.00)
            # Small random bias to break symmetry
            nn.init.normal_(pred[-1].bias, mean=0.0, std=0.00)
    
    def forward(self, latents, current_coords, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select predictor
        if self.share_weights:
            pred_idx = 0 if layer_idx == 0 else 1
        else:
            pred_idx = layer_idx
        
        # Predict raw displacement
        raw_displacement = self.predictors[pred_idx](latents)
        
        # Bound with tanh to [-max_displacement, +max_displacement]
        displacement = torch.tanh(raw_displacement) * 10#self.max_displacement
        
        new_coords = current_coords + displacement

    
        return new_coords, displacement


class ConvexDisplacementUpdate(PositionUpdateStrategy):
    """
    Convex combination using Q·K attention for stable position updates.
    
    M = α * W + (1 - α) * I
    new_pos = M @ current_pos
    
    where W = softmax(Q @ K^T / √d) with natural self-attention bias.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_latents: int = None,  # Not needed for Q·K
        init_alpha: float = -2.0,
        share_weights: bool = True,
        **kwargs
    ):
        super().__init__()
        self.share_weights = share_weights
        self.num_encoder_layers = num_layers
        self.scale = latent_dim ** -0.5
        
        n_predictors = 2 if share_weights else num_layers
        
        # Q and K projections
        self.to_q = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim, bias=False)
            for _ in range(n_predictors)
        ])
        self.to_k = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim, bias=False)
            for _ in range(n_predictors)
        ])
        
        # Learnable mixing coefficient α
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_alpha))
            for _ in range(n_predictors)
        ])
    
    def forward(self, latents, current_coords, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = latents.shape
        

        
    
        # Debug: Check inputs
        if torch.isnan(latents).any():
            print(f"NaN in latents at layer {layer_idx}")
        if torch.isnan(current_coords).any():
            print(f"NaN in current_coords at layer {layer_idx}")
        
        pred_idx = 0 if (self.share_weights and layer_idx == 0) else 1
        
        q = self.to_q[pred_idx](latents)
        k = self.to_k[pred_idx](latents)

        print(f"to_q weight: min={self.to_q[pred_idx].weight.min():.4f}, max={self.to_q[pred_idx].weight.max():.4f}")
        print(f"to_k weight: min={self.to_k[pred_idx].weight.min():.4f}, max={self.to_k[pred_idx].weight.max():.4f}")
        print(f"alpha: {self.alphas[pred_idx].item():.4f}")
        
        # Debug: Check Q, K
        if torch.isnan(q).any():
            print(f"NaN in Q at layer {layer_idx}")
        if torch.isnan(k).any():
            print(f"NaN in K at layer {layer_idx}")
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / 0.1
        
        # Debug: Check scores
        print(f"Layer {layer_idx}: scores min={scores.min():.2f}, max={scores.max():.2f}")
        if torch.isnan(scores).any():
            print(f"NaN in scores at layer {layer_idx}")
        
        scores = scores.clamp(-50, 50)
        W = F.softmax(scores, dim=-1)
        
        # Debug: Check W
        if torch.isnan(W).any():
            print(f"NaN in W at layer {layer_idx}")
        
        alpha = torch.sigmoid(self.alphas[pred_idx])
        print(f"Layer {layer_idx}: alpha={alpha.item():.4f}")
        
        identity = torch.eye(L, device=latents.device, dtype=latents.dtype)
        M = alpha * W + (1.0 - alpha) * identity
        
        new_coords = torch.matmul(M, current_coords)
        
        # Debug: Check output
        print(f"Layer {layer_idx}: new_coords min={new_coords.min():.2f}, max={new_coords.max():.2f}")
        if torch.isnan(new_coords).any():
            print(f"NaN in new_coords at layer {layer_idx}")
        
        displacement = new_coords - current_coords
        
        return new_coords, displacement
# =============================================================================
# Strategy: Deformable Offset (Multiple Sampling Points)
# =============================================================================

class DeformableOffsetUpdate(PositionUpdateStrategy):
    """
    Deformable DETR-style position update.
    
    Instead of predicting a single (Δx, Δy), predicts:
    - K sampling points per attention head (offsets from current position)
    - Attention weights for each point
    
    Final position = weighted average of all sampling points.
    
    This allows the model to "hedge" and attend to multiple promising regions.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_heads: int = 4,
        num_points: int = 4,
        max_offset: float = 0.2,
        share_weights: bool = True
    ):
        """
        Args:
            latent_dim: Dimension of latent embeddings
            num_layers: Number of encoder layers
            num_heads: Number of attention heads (each gets K points)
            num_points: Number of sampling points per head
            max_offset: Maximum offset magnitude
            share_weights: Share predictor weights across layers
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.max_offset = max_offset
        self.share_weights = share_weights
        
        hidden_dim = latent_dim // 2
        
        def make_predictor():
            return nn.ModuleDict({
                'offsets': nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_heads * num_points * 2)
                ),
                'weights': nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_heads * num_points)
                )
            })
        
        n_predictors = 2 if share_weights else num_layers
        self.predictors = nn.ModuleList([make_predictor() for _ in range(n_predictors)])
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize offsets to small grid pattern."""
        K = self.num_points
        H = self.num_heads
        
        for pred in self.predictors:
            nn.init.zeros_(pred['offsets'][-1].weight)
            
            # Initialize bias to small grid pattern
            if K == 4:
                init_offsets = torch.tensor([
                    [-0.05, -0.05], [-0.05, 0.05],
                    [0.05, -0.05], [0.05, 0.05]
                ])
            else:
                init_offsets = torch.randn(K, 2) * 0.02
            
            init_bias = init_offsets.unsqueeze(0).expand(H, -1, -1).reshape(-1)
            pred['offsets'][-1].bias.data = init_bias
            
            # Initialize weights to uniform
            nn.init.zeros_(pred['weights'][-1].weight)
            nn.init.zeros_(pred['weights'][-1].bias)
    
    def forward(self, latents, current_coords, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = latents.shape
        H, K = self.num_heads, self.num_points
        
        # Select predictor
        pred_idx = 0 if (layer_idx == 0 or not self.share_weights) else 1
        if not self.share_weights:
            pred_idx = layer_idx
        
        predictor = self.predictors[pred_idx]
        
        # Predict offsets: [B, L, H, K, 2]
        offsets = predictor['offsets'](latents).view(B, L, H, K, 2)
        offsets = torch.tanh(offsets) * self.max_offset
        
        # Compute sampling points: reference + offset
        reference = current_coords.view(B, L, 1, 1, 2)
        sampling_points = reference + offsets  # [B, L, H, K, 2]
        
        # Predict attention weights: [B, L, H, K]
        weights = predictor['weights'](latents).view(B, L, H, K)
        weights = F.softmax(weights, dim=-1)  # Softmax over K points
        
        # Weighted average: sum over K (weighted), then mean over H
        weighted_points = sampling_points * weights.unsqueeze(-1)
        new_coords = weighted_points.sum(dim=3).mean(dim=2)  # [B, L, 2]
        
        displacement = new_coords - current_coords
        
        return new_coords, displacement


# =============================================================================
# Factory Function
# =============================================================================


def create_position_updater(config: Dict[str, Any]) -> PositionUpdateStrategy:
    """
    Create position update strategy from config dictionary.
    
    Args:
        config: Dictionary containing:
            - use_displacement: bool
            - position_strategy: str ("none", "mlp", "convex", "deformable")
            - latent_dim: int
            - depth: int (number of encoder layers)
            - share_displacement_weights: bool
            
            For "mlp":
                - max_displacement: float (default: 15.0 meters)
                - displacement_temperature: float (default: 3.0)
            
            For "convex":
                - num_spatial_latents: int (required)
                - convex_hidden_dim: int (default: latent_dim // 2)
                - convex_init_alpha: float (default: 0.0 → sigmoid = 0.5)
            
            For "deformable":
                - max_displacement: float (default: 15.0 meters)
                - deformable_heads: int (default: 4)
                - deformable_points: int (default: 4)
    
    Returns:
        PositionUpdateStrategy instance
    """
    if not config.get("use_displacement", False):
        return NoPositionUpdate()
    
    strategy = config.get("position_strategy", "mlp")
    
    if strategy == "none":
        return NoPositionUpdate()
    
    elif strategy == "mlp":
        return MLPDisplacementUpdate(
            latent_dim=config["latent_dim"],
            num_layers=config["depth"],
            max_displacement=config.get("max_displacement", 15.0),
            share_weights=config.get("share_displacement_weights", True)
        )
    
   
    
    elif strategy == "convex":

        return ConvexDisplacementUpdate(
            latent_dim=config["latent_dim"],
            num_layers=config["depth"],
            num_latents=8,
            hidden_dim=config.get("convex_hidden_dim"),
            init_alpha=config.get("convex_init_alpha", 0.5),
            share_weights=config.get("share_displacement_weights", True)
        )
    
    elif strategy == "deformable":
        return DeformableOffsetUpdate(
            latent_dim=config["latent_dim"],
            num_layers=config["depth"],
            num_heads=config.get("deformable_heads", 4),
            num_points=config.get("deformable_points", 4),
            max_offset=config.get("max_displacement", 15.0),
            share_weights=config.get("share_displacement_weights", True)
        )
    
    else:
        raise ValueError(f"Unknown position strategy: {strategy}. "
                        f"Valid options: 'none', 'mlp', 'convex', 'deformable'")





# =============================================================================
# Utility Functions
# =============================================================================

def compute_displacement_stats(trajectory: list) -> Dict[str, Any]:
    """
    Compute statistics about latent movement from position trajectory.
    
    Args:
        trajectory: List of [B, L, 2] position tensors (one per layer)
        
    Returns:
        Dictionary with displacement statistics per layer and total
    """
    stats = {}
    
    for i in range(1, len(trajectory)):
        displacement = trajectory[i] - trajectory[i - 1]
        magnitude = torch.norm(displacement, dim=-1)
        
        stats[f'layer_{i}'] = {
            'mean': magnitude.mean().item(),
            'max': magnitude.max().item(),
            'min': magnitude.min().item(),
            'std': magnitude.std().item(),
        }
    
    # Total displacement (start to end)
    if len(trajectory) > 1:
        total_disp = trajectory[-1] - trajectory[0]
        total_mag = torch.norm(total_disp, dim=-1)
        stats['total'] = {
            'mean': total_mag.mean().item(),
            'max': total_mag.max().item(),
            'std': total_mag.std().item(),
        }
    
    return stats
