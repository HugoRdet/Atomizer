import torch
import torch.nn as nn
from math import pi
from typing import Dict, Any, Optional, Union

from .fourier_features import fourier_encode


class PolarRelativeEncoder(nn.Module):
    """
    Encodes relative positions (dx, dy) using Polar Coordinates.
    
    Pipeline:
    (dx, dy) -> Polar(r, theta) -> Normalized & Compressed -> Fourier Features
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Use cartesian config to match original behavior
        # (original had polar keys but used cartesian values)
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        
        # Reference GSD for log-space encoding
        self.G_ref = config["Atomiser"].get("G_ref", 0.2)
        
        # Output Dimension: r + theta + gsd (all same size)
        self.per_component_dim = self.num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 3  # With GSD
        self.out_dim_no_gsd = self.per_component_dim * 2  # Without GSD

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        physical_scale: Union[torch.Tensor, float], 
        gsd: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        """
        Args:
            delta_x, delta_y: [...] Relative position in meters
            physical_scale:   [...] or scalar, meters per latent
            gsd:              [...] or scalar or None, Ground Sampling Distance in meters
                              If None, GSD encoding is omitted.
        Returns:
            encoding: [..., out_dim] or [..., out_dim_no_gsd]
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        # Convert physical_scale to tensor if needed
        if not isinstance(physical_scale, torch.Tensor):
            physical_scale = torch.tensor(physical_scale, device=device, dtype=dtype)
        
        # A. Polar Conversion
        r = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        theta = torch.atan2(delta_y, delta_x)
        
        # B. Normalization & Compression
        r_norm = r / (physical_scale + 1e-8)
        r_comp = r_norm / (1.0 + r_norm)  # [0, inf) -> [0, 1)
        
        theta_norm = theta / pi  # [-pi, pi] -> [-1, 1]
        
        # C. Fourier Encoding
        r_enc = fourier_encode(r_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        theta_enc = fourier_encode(theta_norm, max_freq=self.max_freq, num_bands=self.num_bands)
        
        # D. Optional GSD Encoding
        if gsd is not None:
            # Convert gsd to tensor if it's a scalar
            if not isinstance(gsd, torch.Tensor):
                # Create tensor matching the shape of r for broadcasting
                gsd = torch.full_like(r, gsd)
            
            # Ensure gsd has same shape as r for element-wise operations
            # If gsd is a different shape, try to broadcast
            if gsd.shape != r.shape:
                try:
                    # Attempt broadcast
                    gsd = gsd.expand_as(r)
                except RuntimeError:
                    # Fall back to creating a full tensor
                    gsd = torch.full_like(r, gsd.mean().item())
            
            log_gsd = torch.log((gsd / self.G_ref) + 1e-8)
            gsd_enc = fourier_encode(log_gsd, max_freq=self.max_freq, num_bands=self.num_bands)
            return torch.cat([r_enc, theta_enc, gsd_enc], dim=-1)
        
        return torch.cat([r_enc, theta_enc], dim=-1)

    def get_output_dim(self, include_gsd: bool = True) -> int:
        """Get output dimension with or without GSD."""
        return self.out_dim if include_gsd else self.out_dim_no_gsd


class CartesianRelativeEncoder(nn.Module):
    """
    Encodes relative positions (dx, dy) using Cartesian Coordinates.
    
    Pipeline:
    (dx, dy) -> Normalized & Compressed -> Fourier Features
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        self.G_ref = config["Atomiser"].get("G_ref", 0.2)
        
        # Components: X, Y, GSD
        self.per_component_dim = self.num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 3
        self.out_dim_no_gsd = self.per_component_dim * 2

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        physical_scale: Union[torch.Tensor, float], 
        gsd: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        """
        Args:
            delta_x, delta_y: [...] Relative position in meters
            physical_scale:   [...] or scalar, normalization factor
            gsd:              [...] or scalar or None, Ground Sampling Distance
        """
        device = delta_x.device
        dtype = delta_x.dtype
        
        # Convert physical_scale to tensor if needed
        if not isinstance(physical_scale, torch.Tensor):
            physical_scale = torch.tensor(physical_scale, device=device, dtype=dtype)
        
        # A. Normalization
        dx = delta_x / (physical_scale + 1e-8)
        dy = delta_y / (physical_scale + 1e-8)
        
        # B. Signed Compression: (-inf, inf) -> (-1, 1)
        dx_comp = dx / (1.0 + torch.abs(dx))
        dy_comp = dy / (1.0 + torch.abs(dy))
        
        # C. Fourier Encoding
        x_enc = fourier_encode(dx_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        y_enc = fourier_encode(dy_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        
        # D. Optional GSD Encoding
        if gsd is not None:
            # Convert gsd to tensor if it's a scalar
            if not isinstance(gsd, torch.Tensor):
                gsd = torch.full_like(delta_x, gsd)
            
            # Ensure gsd has same shape
            if gsd.shape != delta_x.shape:
                try:
                    gsd = gsd.expand_as(delta_x)
                except RuntimeError:
                    gsd = torch.full_like(delta_x, gsd.mean().item())
            
            log_gsd = torch.log((gsd / self.G_ref) + 1e-8)
            gsd_enc = fourier_encode(log_gsd, max_freq=self.max_freq, num_bands=self.num_bands)
            return torch.cat([x_enc, y_enc, gsd_enc], dim=-1)
        
        return torch.cat([x_enc, y_enc], dim=-1)

    def get_output_dim(self, include_gsd: bool = True) -> int:
        """Get output dimension with or without GSD."""
        return self.out_dim if include_gsd else self.out_dim_no_gsd


def build_position_encoder(config: Dict[str, Any]) -> nn.Module:
    """Factory function for position encoders."""
    strategy = config["Atomiser"].get("position_encoding_type", "POLAR")
    
    if strategy == "POLAR":
        return PolarRelativeEncoder(config)
    elif strategy == "CARTESIAN":
        return CartesianRelativeEncoder(config)
    else:
        raise ValueError(f"Unknown position encoding strategy: {strategy}")

class CartesianRelativeEncoder(nn.Module):
    """
    Encodes relative positions (dx, dy) using Cartesian Coordinates.
    
    Pipeline:
    (dx, dy) -> Normalized & Compressed -> Fourier Features
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_bands = config["Atomiser"].get("cartesian_num_bands", 32)
        self.max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        self.G_ref = config["Atomiser"].get("G_ref", 0.2)
        
        # Components: X, Y, GSD
        self.per_component_dim = self.num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 3
        self.out_dim_no_gsd = self.per_component_dim * 2

    def forward(
        self, 
        delta_x: torch.Tensor, 
        delta_y: torch.Tensor, 
        physical_scale: torch.Tensor, 
        gsd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            delta_x, delta_y: [...] Relative position in meters
            physical_scale:   [...] or scalar, normalization factor
            gsd:              [...] or None, Ground Sampling Distance
        """
        # A. Normalization
        dx = delta_x / (physical_scale + 1e-8)
        dy = delta_y / (physical_scale + 1e-8)
        
        # B. Signed Compression: (-inf, inf) -> (-1, 1)
        dx_comp = dx / (1.0 + torch.abs(dx))
        dy_comp = dy / (1.0 + torch.abs(dy))
        
        # C. Fourier Encoding
        x_enc = fourier_encode(dx_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        y_enc = fourier_encode(dy_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        
        # D. Optional GSD Encoding
        if gsd is not None:
            log_gsd = torch.log((gsd / self.G_ref) + 1e-8)
            gsd_enc = fourier_encode(log_gsd, max_freq=self.max_freq, num_bands=self.num_bands)
            return torch.cat([x_enc, y_enc, gsd_enc], dim=-1)
        
        return torch.cat([x_enc, y_enc], dim=-1)

    def get_output_dim(self, include_gsd: bool = True) -> int:
        return self.out_dim if include_gsd else self.out_dim_no_gsd


def build_position_encoder(config: Dict[str, Any]) -> nn.Module:
    """Factory function for position encoders."""
    strategy = config["Atomiser"].get("position_encoding_type", "POLAR")
    
    if strategy == "POLAR":
        return PolarRelativeEncoder(config)
    elif strategy == "CARTESIAN":
        return CartesianRelativeEncoder(config)
    else:
        raise ValueError(f"Unknown position encoding strategy: {strategy}")