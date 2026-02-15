import torch
import torch.nn as nn
from math import pi
from typing import Dict, Any, Optional, Union

from .fourier_features import fourier_encode


class PolarRelativeEncoder(nn.Module):
    """
    Encodes relative positions using Polar Coordinates.
    
    Pipeline:
    Position indices -> Relative (dx, dy) -> Polar (r, θ) -> Fourier
    
    Compression formula (RoPE-style):
        compressed = pos / (scale + |pos|)
        Maps [0, ∞) -> [0, 1) for radius
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Fourier feature parameters - SEPARATE for radius and theta
        self.num_bands_r = config["Atomiser"].get("polar_num_bands_r", 8)
        self.num_bands_theta = config["Atomiser"].get("polar_num_bands_theta", 8)
        self.max_freq_r = config["Atomiser"].get("polar_max_freq_r", 128)
        self.max_freq_theta = config["Atomiser"].get("polar_max_freq_theta", 8)
        
        # Compression scale (in index units, not meters)
        self.compression_scale = config["Atomiser"].get("position_compression_scale", 10.0)
        
        # Output dimension calculation
        # radius: 1 (normalized) + 2*num_bands_r (sin/cos pairs)
        # theta: 1 (normalized) + 2*num_bands_theta (sin/cos pairs)
        self.dim_r = 1 + 2 * self.num_bands_r
        self.dim_theta = 1 + 2 * self.num_bands_theta
        self.out_dim = self.dim_r + self.dim_theta
        
        print(f"[PolarRelativeEncoder] out_dim={self.out_dim} "
              f"(r: {self.dim_r}, theta: {self.dim_theta})")

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        compression_scale: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Encode relative positions in polar coordinates.
        
        Args:
            delta_x: [...] Relative position in INDEX SPACE (centered coordinates)
            delta_y: [...] Relative position in INDEX SPACE (centered coordinates)
            compression_scale: Optional override for compression scale (in index units)
        
        Returns:
            encoding: [..., out_dim] Polar Fourier features
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        if compression_scale is None:
            compression_scale = self.compression_scale
        
        if not isinstance(compression_scale, torch.Tensor):
            compression_scale = torch.tensor(compression_scale, device=device, dtype=dtype)
        
        # ─────────────────────────────────────────────────────
        # A. Convert Cartesian → Polar
        # ─────────────────────────────────────────────────────
        r = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)  # Radius
        theta = torch.atan2(delta_y, delta_x)           # Angle [-π, π]
        
        # ─────────────────────────────────────────────────────
        # B. Compress & Normalize
        # ─────────────────────────────────────────────────────
        # Radius: [0, ∞) → [0, 1) via RoPE-style compression
        r_normalized = r / (compression_scale + r)
        
        # Theta: [-π, π] → [-1, 1] (preserves periodicity)
        theta_normalized = theta / pi
        
        # ─────────────────────────────────────────────────────
        # C. Fourier Encoding (DIFFERENT for radius vs theta)
        # ─────────────────────────────────────────────────────
        # Radius: log-spaced frequencies for multi-scale distance
        r_enc = fourier_encode(
            r_normalized, 
            max_freq=self.max_freq_r, 
            num_bands=self.num_bands_r,
            log_sampling=True,  # Important for radius!
        )
        
        # Theta: integer frequencies for rotational symmetry
        theta_enc = fourier_encode(
            theta_normalized, 
            max_freq=self.max_freq_theta, 
            num_bands=self.num_bands_theta,
            log_sampling=False,  # Linear for angles
        )
        
        return torch.cat([r_enc, theta_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


class CartesianRelativeEncoder(nn.Module):
    """
    Encodes relative positions using Cartesian Coordinates.
    
    Pipeline:
    Position indices -> Relative (dx, dy) -> Fourier
    
    Compression formula:
        compressed = pos / (scale + |pos|)
        Maps (-∞, ∞) -> (-1, 1), preserving sign
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        self.compression_scale = config["Atomiser"].get("position_compression_scale", 3.0)
        
        # Output: X + Y (each with raw + Fourier features)
        self.per_component_dim = 1 + 2 * self.num_bands
        self.out_dim = self.per_component_dim * 2
        
        print(f"[CartesianRelativeEncoder] out_dim={self.out_dim} "
              f"({self.per_component_dim} per axis)")

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        compression_scale: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Encode relative positions in Cartesian coordinates.
        
        Args:
            delta_x, delta_y: [...] Relative position in INDEX SPACE
            compression_scale: Optional override for compression scale
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        if compression_scale is None:
            compression_scale = self.compression_scale
        
        if not isinstance(compression_scale, torch.Tensor):
            compression_scale = torch.tensor(compression_scale, device=device, dtype=dtype)
        
        # ─────────────────────────────────────────────────────
        # A. Compress (RoPE-style, preserves sign)
        # ─────────────────────────────────────────────────────
        dx_normalized = delta_x / (compression_scale + torch.abs(delta_x))
        dy_normalized = delta_y / (compression_scale + torch.abs(delta_y))
        
        # ─────────────────────────────────────────────────────
        # B. Fourier Encoding
        # ─────────────────────────────────────────────────────
        x_enc = fourier_encode(
            dx_normalized, 
            max_freq=self.max_freq, 
            num_bands=self.num_bands,
        )
        y_enc = fourier_encode(
            dy_normalized, 
            max_freq=self.max_freq, 
            num_bands=self.num_bands,
        )
        
        return torch.cat([x_enc, y_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


def build_position_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function for position encoders.
    
    Config options:
        position_encoding_type: "POLAR" | "CARTESIAN"
        
    Polar (recommended):
        - Rotation-invariant
        - Aligns with radial attention bias
        - Separate radius/angle encoding
        
    Cartesian:
        - Simpler, baseline
        - Axis-aligned features
    """
    strategy = config["Atomiser"].get("position_encoding_type", "POLAR")
    
    if strategy == "POLAR":
        return PolarRelativeEncoder(config)
    elif strategy == "CARTESIAN":
        return CartesianRelativeEncoder(config)
    else:
        raise ValueError(
            f"Unknown position encoding strategy: {strategy}. "
            f"Valid options: POLAR, CARTESIAN"
        )