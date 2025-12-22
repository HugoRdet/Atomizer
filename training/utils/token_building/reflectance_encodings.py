import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from .fourier_features import fourier_encode



class ReflectanceEncoder(nn.Module):
    """
    Encodes the raw pixel values (B-values).
    Usually applies Fourier Features to map scalar intensity to a vector.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.mode = config["Atomiser"].get("bandvalue_encoding", "NATURAL")
        
        if self.mode == "FF":
            self.num_bands = config["Atomiser"]["bandvalue_num_freq_bands"]
            self.max_freq = config["Atomiser"]["bandvalue_max_freq"]
            self.out_dim = self.num_bands * 2 + 1
        else:
            self.out_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Raw B-values [Batch, ...]"""
        if self.mode == "FF":
            return fourier_encode(x, self.max_freq, self.num_bands)
        return x.unsqueeze(-1)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_reflectance_encoder(config: Dict[str, Any]) -> nn.Module:
    return ReflectanceEncoder(config)