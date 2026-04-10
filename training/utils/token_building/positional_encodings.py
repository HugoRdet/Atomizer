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
        theta = torch.atan2(delta_y, delta_x)            # Angle [-π, π]
        
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
        r_enc = fourier_encode(
            r_normalized, 
            max_freq=self.max_freq_r, 
            num_bands=self.num_bands_r,
            log_sampling=False,
        )
        
        theta_enc = fourier_encode(
            theta_normalized, 
            max_freq=self.max_freq_theta, 
            num_bands=self.num_bands_theta,
            log_sampling=False,
        )
        
        return torch.cat([r_enc, theta_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


class CartesianRelativeEncoder(nn.Module):
    """
    Encodes relative positions using Cartesian Coordinates.
    
    Pipeline:
        (delta_x, delta_y) → compress → Fourier → [x_enc, y_enc]

    Compression formula (RoPE-style, preserves sign):
        compressed = pos / (scale + |pos|)  ∈ (-1, 1)

    Config keys (under Atomiser):
        pos_num_freq_bands: number of Fourier frequency bands per axis
        pos_max_freq:       maximum frequency
        compression_alpha:  compression scale passed at runtime from processor
                            (fallback: position_compression_scale)

    Output dim: 2 × (1 + 2 × pos_num_freq_bands)
        e.g. pos_num_freq_bands=128 → 2 × 257 = 514
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Read the correct YAML keys (pos_num_freq_bands / pos_max_freq)
        self.num_bands = config["Atomiser"].get("pos_num_freq_bands", 32)
        self.max_freq  = config["Atomiser"].get("pos_max_freq", 32)

        # Fallback compression scale (normally overridden at runtime by processor)
        self.compression_scale = config["Atomiser"].get("position_compression_scale", 10.0)
       
        # Output: X + Y, each encoded as (raw value + Fourier features)
        self.per_component_dim = 1 + 2 * self.num_bands
        self.out_dim = self.per_component_dim * 2
        
        print(f"[CartesianRelativeEncoder] num_bands={self.num_bands}, "
              f"max_freq={self.max_freq}, "
              f"out_dim={self.out_dim} ({self.per_component_dim} per axis)")

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        compression_scale: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Encode relative positions in Cartesian coordinates.
        
        Args:
            delta_x, delta_y: [...] Relative position in physical space (meters)
            compression_scale: Override for compression scale (passed by processor
                               as compression_alpha * gsd)
        
        Returns:
            encoding: [..., out_dim]
        """
        device = delta_x.device
        dtype  = delta_x.dtype
        
        if compression_scale is None:
            compression_scale = self.compression_scale
        
        if not isinstance(compression_scale, torch.Tensor):
            compression_scale = torch.tensor(compression_scale, device=device, dtype=dtype)
        
        # ─────────────────────────────────────────────────────
        # A. Compress: (-∞, ∞) → (-1, 1), preserving sign
        # ─────────────────────────────────────────────────────
        dx_normalized = delta_x / (compression_scale + torch.abs(delta_x))
        dy_normalized = delta_y / (compression_scale + torch.abs(delta_y))
        
        # ─────────────────────────────────────────────────────
        # B. Fourier Encoding (same params for both axes)
        # ─────────────────────────────────────────────────────
        x_enc = fourier_encode(dx_normalized, max_freq=self.max_freq, num_bands=self.num_bands)
        y_enc = fourier_encode(dy_normalized, max_freq=self.max_freq, num_bands=self.num_bands)
        
        return torch.cat([x_enc, y_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


def build_position_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function for position encoders.
    Always returns CartesianRelativeEncoder (active mode).
    PolarRelativeEncoder is available but not wired in.
    """
    return CartesianRelativeEncoder(config)