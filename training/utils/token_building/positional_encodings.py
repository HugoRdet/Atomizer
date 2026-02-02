import torch
import torch.nn as nn
from math import pi
from typing import Dict, Any, Optional, Union

from .fourier_features import fourier_encode


class PolarRelativeEncoder(nn.Module):
    """
    Encodes relative positions (dx, dy) using Polar Coordinates.
    
    Pipeline:
    (dx, dy) -> Polar(r, theta) -> Compressed -> Fourier Features
    
    Compression formula (same as RoPE):
        compressed = pos / (scale + |pos|)
    
    Maps [0, inf) -> [0, 1) for radius
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        
        # Compression scale (same as RoPE)
        self.compression_scale = config["Atomiser"].get("position_compression_scale", 10.0)
        
        # Output Dimension: r + theta
        self.per_component_dim = self.num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 2

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        compression_scale: Optional[Union[torch.Tensor, float]] = None, 
    ) -> torch.Tensor:
        """
        Args:
            delta_x, delta_y: [...] Relative position in meters
            compression_scale: [...] or scalar, compression scale (default: self.compression_scale)
        Returns:
            encoding: [..., out_dim]
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        if compression_scale is None:
            compression_scale = self.compression_scale
        
        if not isinstance(compression_scale, torch.Tensor):
            compression_scale = torch.tensor(compression_scale, device=device, dtype=dtype)
        
        # A. Polar Conversion
        r = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        theta = torch.atan2(delta_y, delta_x)
        
        # B. Compression (same as RoPE)
        r_comp = r / (compression_scale + r)
        theta_norm = theta / pi
        
        # C. Fourier Encoding
        r_enc = fourier_encode(r_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        theta_enc = fourier_encode(theta_norm, max_freq=self.max_freq, num_bands=self.num_bands)
        
        return torch.cat([r_enc, theta_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


class CartesianRelativeEncoder(nn.Module):
    """
    Encodes relative positions (dx, dy) using Cartesian Coordinates.
    
    Pipeline:
    (dx, dy) -> Compressed -> Fourier Features
    
    Compression formula (same as RoPE):
        compressed = pos / (scale + |pos|)
    
    Maps (-inf, inf) -> (-1, 1), preserving sign
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        self.compression_scale = config["Atomiser"].get("position_compression_scale", 3.0)
        
        # Output: X + Y
        self.per_component_dim = self.num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 2

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        compression_scale: Optional[Union[torch.Tensor, float]] = None, 
    ) -> torch.Tensor:
        """
        Args:
            delta_x, delta_y: [...] Relative position in meters
            compression_scale: [...] or scalar, compression scale (default: self.compression_scale)
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        if compression_scale is None:
            compression_scale = self.compression_scale
        
        if not isinstance(compression_scale, torch.Tensor):
            compression_scale = torch.tensor(compression_scale, device=device, dtype=dtype)
        
        # A. Compression (same as RoPE)
        dx_comp = delta_x / (compression_scale + torch.abs(delta_x))
        dy_comp = delta_y / (compression_scale + torch.abs(delta_y))
        
        # B. Fourier Encoding
        x_enc = fourier_encode(dx_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        y_enc = fourier_encode(dy_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        
        return torch.cat([x_enc, y_enc], dim=-1)

    def get_output_dim(self) -> int:
        return self.out_dim


def build_position_encoder(config: Dict[str, Any]) -> nn.Module:
    """Factory function for position encoders."""
    strategy = config["Atomiser"].get("position_encoding_type", "POLAR")
    
    if strategy == "POLAR":
        return PolarRelativeEncoder(config)
    elif strategy == "CARTESIAN":
        return CartesianRelativeEncoder(config)
    else:
        raise ValueError(f"Unknown position encoding strategy: {strategy}")