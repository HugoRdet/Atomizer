import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SpectralEncoder(nn.Module):
    """
    Unified encoder for Wavelengths and Learned Tokens (e.g., Elevation).
    
    Architecture:
    - Fixed physics buffer for wavelength embeddings (no gradients)
    - Learnable parameters for abstract channels (elevation, etc.)
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table
        
        # 1. Determine Encoding Strategy
        self.mode = config["Atomiser"].get("wavelength_encoding", "GAUSSIANS")
        
        if self.mode == "GAUSSIANS":
            # Config has gaussian anchors defined separately
            self.anchors = config.get("wavelengths_encoding", {})
            self.out_dim = len(self.anchors)
        elif self.mode == "FF":
            self.num_bands = config["Atomiser"]["wavelength_num_freq_bands"]
            self.max_freq = config["Atomiser"]["wavelength_max_freq"]
            if self.num_bands == -1:
                self.out_dim = int(self.max_freq) * 2 + 1
            else:
                self.out_dim = int(self.num_bands) * 2 + 1
        elif self.mode == "NATURAL":
            self.out_dim = 1
        else:  # "NOPE"
            self.out_dim = 0
        
        if self.out_dim == 0:
            return
            
        # 2. Setup Gaussian Anchors (if needed)
        if self.mode == "GAUSSIANS" and self.anchors:
            means = [self.anchors[k]["mean"] for k in self.anchors]
            stds = [self.anchors[k]["std"] for k in self.anchors]
            self.register_buffer("means", torch.tensor(means).float().view(1, 1, -1))
            self.register_buffer("stds", torch.tensor(stds).float().view(1, 1, -1))
        
        # 3. Build Fixed Codebook (buffer, not parameter!)
        num_channels = len(lookup_table.table_wave)
        physics_codebook = torch.zeros(num_channels, self.out_dim)
        
        # Track which indices are learnable
        learnable_indices = []
        
        for (bandwidth, central_wave), idx in lookup_table.table_wave.items():
            if bandwidth == -1 or central_wave == -1:
                # Mark as learnable (will be replaced by Parameter lookup)
                learnable_indices.append(idx)
            else:
                # Compute physics embedding (no gradients)
                physics_codebook[idx] = self._compute_physics_vector(central_wave, bandwidth)
        
        # Register physics as buffer (frozen)
        self.register_buffer("physics_codebook", physics_codebook)
        
        # 4. Create Learnable Embeddings for abstract tokens
        self.learnable_indices = learnable_indices
        if learnable_indices:
            # One learnable vector per abstract channel
            self.learned_embeddings = nn.Parameter(
                torch.zeros(len(learnable_indices), self.out_dim)
            )
            nn.init.trunc_normal_(self.learned_embeddings, std=0.02, a=-2., b=2.)
            
            # Create index mapping: global_idx -> local_idx in learned_embeddings
            self.register_buffer(
                "learnable_idx_map",
                torch.tensor(learnable_indices, dtype=torch.long)
            )
        else:
            self.learned_embeddings = None

    def _compute_physics_vector(self, center: float, bandwidth: float) -> torch.Tensor:
        """
        Compute embedding for a physical wavelength band.
        Uses 150 sample points to match original precision.
        """
        if self.mode == "GAUSSIANS":
            num_points = 150  # Match original!
            
            center = torch.tensor(center, dtype=torch.float32)
            bandwidth = torch.tensor(bandwidth, dtype=torch.float32)
            
            lambda_min = center - bandwidth / 2
            lambda_max = center + bandwidth / 2
            
            # Sample wavelengths across the band
            t = torch.linspace(0, 1, num_points)
            sampled_lambdas = lambda_min + (lambda_max - lambda_min) * t  # [num_points]
            sampled_lambdas = sampled_lambdas.view(-1, 1, 1)  # [num_points, 1, 1]
            
            # means/stds: [1, 1, num_anchors]
            # Broadcasting: [num_points, 1, num_anchors]
            gaussians = torch.exp(
                -0.5 * ((sampled_lambdas - self.means) / self.stds) ** 2
            )
            
            # Max over sample points -> [num_anchors]
            encoding = gaussians.max(dim=0).values.squeeze()
            return encoding
            
        elif self.mode == "FF":
            from .fourier_features import fourier_encode
            norm_wave = torch.tensor([center / 1000.0])
            return fourier_encode(norm_wave, self.max_freq, self.num_bands).squeeze(0)
            
        elif self.mode == "NATURAL":
            return torch.tensor([center])
        
        return torch.zeros(self.out_dim)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_indices: [...] integer indices into table_wave
            
        Returns:
            embeddings: [..., out_dim]
        """
        if self.out_dim == 0:
            return torch.zeros(
                (*channel_indices.shape, 0), 
                device=channel_indices.device
            )
        
        # Start with physics lookup
        embeddings = self.physics_codebook[channel_indices]  # [..., out_dim]
        
        # Replace learnable indices with learned embeddings
        if self.learned_embeddings is not None:
            for local_idx, global_idx in enumerate(self.learnable_indices):
                mask = (channel_indices == global_idx)
                if mask.any():
                    # Expand learned embedding to match masked positions
                    learned_vec = self.learned_embeddings[local_idx]
                    embeddings = torch.where(
                        mask.unsqueeze(-1),
                        learned_vec.expand_as(embeddings),
                        embeddings
                    )
        
        return embeddings


def build_spectral_encoder(config: Dict[str, Any], lookup_table: Any) -> SpectralEncoder:
    """Factory function for spectral encoder."""
    return SpectralEncoder(config, lookup_table)