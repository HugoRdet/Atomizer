import torch
import torch.nn as nn
from typing import Dict, Any

from .fourier_features import fourier_encode


class ResolutionEncoder(nn.Module):
    """
    Encodes spatial resolution (GSD in m/px) using compression + Fourier features.
    
    Pipeline:
        resolution_idx → GSD lookup → compress → Fourier features
    
    Compression formula (same as positional encoding):
        compressed = gsd / (S + gsd)
    
    Maps [0, ∞) → [0, 1) with S = 10 m/px default.
    
    Sentinel convention:
        resolution_idx < 0   →  zero vector (non-optical: SAR, DEM, etc.)
        resolution_idx >= 0  →  Fourier(compress(GSD))
    
    This follows the project-wide convention:
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        
        # Fourier parameters (can share with position or use own)
        self.num_bands = config["Atomiser"].get("resolution_num_bands", 16)
        self.max_freq = config["Atomiser"].get("resolution_max_freq", 16)
        
        # Compression scale: 10 m/px means GSD=10 maps to 0.5, GSD=30 maps to 0.75
        self.compression_scale = config["Atomiser"].get(
            "resolution_compression_scale", 10.0
        )
        
        # Output dimension: num_bands * 2 (sin + cos) + 1 (raw compressed value)
        self.out_dim = self.num_bands * 2 + 1
        
        # ── Build GSD lookup buffer ─────────────────────────
        # Maps resolution_idx → GSD value in m/px
        # All indices ≥ 0 are valid physical GSDs
        num_resolutions = lookup_table.num_resolution_indices
        gsd_values = torch.zeros(num_resolutions, dtype=torch.float32)
        
        for res_key, idx in lookup_table.table_resolution.items():
            gsd_values[idx] = res_key / 1000.0  # int key back to m/px
        
        self.register_buffer("gsd_values", gsd_values)
        
        print(f"[ResolutionEncoder] num_bands={self.num_bands}, "
              f"max_freq={self.max_freq}, "
              f"compression_scale={self.compression_scale}, "
              f"out_dim={self.out_dim}, "
              f"num_resolutions={num_resolutions} "
              f"(idx<0 → zeros, idx>=0 → optical)")
    
    def forward(self, resolution_idx: torch.Tensor) -> torch.Tensor:
        """
        Encode resolution indices into feature vectors.
        
        Args:
            resolution_idx: [...] int/float tensor of resolution indices.
                            < 0  = non-optical (returns zeros)
                            >= 0 = valid optical GSD index
        
        Returns:
            encoding: [..., out_dim] resolution features.
        """
        original_shape = resolution_idx.shape
        idx_flat = resolution_idx.long().reshape(-1)
        
        # ── Identify N/A tokens (idx < 0) ──────────────────
        is_na = idx_flat < 0
        
        # ── Clamp for safe lookup (negatives → 0) ──────────
        idx_clamped = idx_flat.clamp(0, self.gsd_values.shape[0] - 1)
        gsd = self.gsd_values[idx_clamped]  # [N_flat]

        
        # ── Compress: gsd / (S + gsd) → [0, 1) ─────────────
        compressed = gsd / (self.compression_scale + gsd)
        
        # ── Fourier encode ──────────────────────────────────
        encoded = fourier_encode(
            compressed,
            max_freq=self.max_freq,
            num_bands=self.num_bands,
        )  # [N_flat, out_dim]
        
        # ── Zero out N/A tokens (idx < 0) ──────────────────
        if is_na.any():
            encoded[is_na] = 0.0
        
        # ── Reshape back ────────────────────────────────────
        return encoded.reshape(*original_shape, self.out_dim)
    
    def get_output_dim(self) -> int:
        return self.out_dim
    
    def extra_repr(self) -> str:
        return (
            f"num_bands={self.num_bands}, "
            f"max_freq={self.max_freq}, "
            f"compression_scale={self.compression_scale}, "
            f"out_dim={self.out_dim}, "
            f"idx<0=zeros"
        )


def build_resolution_encoder(config: Dict[str, Any], lookup_table: Any) -> ResolutionEncoder:
    """Factory function for resolution encoder."""
    return ResolutionEncoder(config, lookup_table)