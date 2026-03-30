"""
Fourier Positional Encoding
=============================

Encodes spatial positions into high-dimensional features using
sinusoidal functions at multiple frequencies, as in the original
Perceiver (Jaegle et al., 2021).

For a 2D image of size H×W, each pixel gets a position in [-1, 1]²
encoded as:

    pos_encoding(x) = [sin(f₁πx), cos(f₁πx), ..., sin(fₖπx), cos(fₖπx), x]

where frequencies f₁...fₖ are linearly spaced from 1 to max_freq.
The raw coordinate is appended for absolute position information.

Output dimension per axis: 2 * num_bands + 1
Total for 2D: 2 * (2 * num_bands + 1)
"""

import torch
import torch.nn as nn
from math import pi
from einops import rearrange, repeat


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for spatial grids.

    Generates position encodings for any H×W grid on the fly.
    No learnable parameters — purely analytical.

    Args:
        num_bands: Number of frequency bands per axis.
        max_freq: Maximum frequency. Higher = finer spatial detail.

    Output dim: 2 * (2 * num_bands + 1) for 2D input.
    """

    def __init__(self, num_bands=6, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.out_dim = 2 * (2 * num_bands + 1)  # 2 axes × (sin + cos + raw)

    def forward(self, H, W, device, dtype=torch.float32):
        """
        Generate positional encoding for an H×W grid.

        Args:
            H: Height of the grid.
            W: Width of the grid.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            pos: [H*W, out_dim] positional features.
        """
        # Grid positions in [-1, 1]
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [H, W] each

        # Stack: [H, W, 2]
        pos = torch.stack([grid_y, grid_x], dim=-1)

        # Fourier features
        pos_encoded = self._encode(pos)  # [H, W, out_dim]

        # Flatten spatial: [H*W, out_dim]
        return rearrange(pos_encoded, 'h w d -> (h w) d')

    def _encode(self, pos):
        """
        Apply Fourier encoding to position tensor.

        Args:
            pos: [..., n_axes] positions in [-1, 1].

        Returns:
            encoded: [..., n_axes * (2 * num_bands + 1)]
        """
        # Frequency bands linearly spaced from 1 to max_freq
        bands = torch.linspace(
            1.0, self.max_freq / 2, self.num_bands,
            device=pos.device, dtype=pos.dtype,
        )  # [num_bands]

        # Reshape for broadcasting: pos [..., n_axes, 1] × bands [num_bands]
        pos_expanded = pos.unsqueeze(-1)  # [..., n_axes, 1]
        scaled = pos_expanded * bands * pi  # [..., n_axes, num_bands]

        # Sin and cos
        sin_feat = scaled.sin()  # [..., n_axes, num_bands]
        cos_feat = scaled.cos()  # [..., n_axes, num_bands]

        # Concatenate: [sin, cos, raw_pos] per axis
        # raw_pos: [..., n_axes, 1]
        encoded = torch.cat([sin_feat, cos_feat, pos_expanded], dim=-1)
        # [..., n_axes, 2*num_bands + 1]

        # Flatten last two dims: [..., n_axes * (2*num_bands + 1)]
        return rearrange(encoded, '... a d -> ... (a d)')

    def get_output_dim(self):
        """Return the output dimensionality."""
        return self.out_dim