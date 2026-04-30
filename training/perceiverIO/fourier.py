"""
Fourier Positional & Time Encoding
====================================

FourierPositionalEncoding — encodes spatial (y, x) grid positions into
sinusoidal features, as in Perceiver (Jaegle et al., 2021).

FourierTimeEncoding — encodes scalar day-of-year values into the same
style of sinusoidal features, with a single time axis.

Both encoders are purely analytical (no learnable parameters), share the
same `(num_bands, max_freq)` configuration, and use the convention:

    encoding(x) = [sin(f_1 pi x), cos(f_1 pi x), ...,
                    sin(f_K pi x), cos(f_K pi x), x]

where the K frequencies are linearly spaced from 1 to max_freq / 2 and
the raw normalized coordinate is appended.

Output dimension per axis: 2 * num_bands + 1.
For 2D positions:                2 * (2 * num_bands + 1).
For 1D time:                     1 * (2 * num_bands + 1).
"""

import torch
import torch.nn as nn
from math import pi
from einops import rearrange


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for spatial grids.

    Generates position encodings for any H x W grid on the fly.
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
        self.out_dim = 2 * (2 * num_bands + 1)  # 2 axes x (sin + cos + raw)

    def forward(self, H, W, device, dtype=torch.float32):
        """
        Generate positional encoding for an H x W grid.

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
        # Frequency bands linearly spaced from 1 to max_freq / 2
        bands = torch.linspace(
            1.0, self.max_freq / 2, self.num_bands,
            device=pos.device, dtype=pos.dtype,
        )  # [num_bands]

        # Reshape for broadcasting: pos [..., n_axes, 1] x bands [num_bands]
        pos_expanded = pos.unsqueeze(-1)  # [..., n_axes, 1]
        scaled = pos_expanded * bands * pi  # [..., n_axes, num_bands]

        # Sin and cos
        sin_feat = scaled.sin()  # [..., n_axes, num_bands]
        cos_feat = scaled.cos()  # [..., n_axes, num_bands]

        # Concatenate: [sin, cos, raw_pos] per axis
        encoded = torch.cat([sin_feat, cos_feat, pos_expanded], dim=-1)
        # [..., n_axes, 2*num_bands + 1]

        # Flatten last two dims: [..., n_axes * (2*num_bands + 1)]
        return rearrange(encoded, '... a d -> ... (a d)')

    def get_output_dim(self):
        """Return the output dimensionality."""
        return self.out_dim


class FourierTimeEncoding(nn.Module):
    """
    Fourier time encoding for day-of-year (DOY) values.

    Mirrors FourierPositionalEncoding's API: same num_bands, max_freq,
    and the raw normalized coordinate appended at the end. The encoder
    operates element-wise on a tensor of any shape — output is the
    same shape with one extra trailing dimension of size out_dim.

    DOY values are normalized via:
        t_norm = (doy - 1) / 365.0 * 2 - 1     in [-1, 1]
    so that day 1 -> -1.0 and day 366 -> +1.0 (with day 0 used as a
    zero-pad sentinel landing at -1.003, which is harmless for sin/cos).

    Args:
        num_bands: Number of frequency bands.
        max_freq: Maximum frequency.

    Output dim: 2 * num_bands + 1 (one axis: time).
    """

    DAYS_PER_YEAR = 365.0

    def __init__(self, num_bands=6, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.out_dim = 2 * num_bands + 1

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        Encode DOY values into Fourier features.

        Args:
            doy: [...] long or float tensor of day-of-year values
                 (any shape).

        Returns:
            encoded: [..., out_dim] same shape as input plus trailing dim.
        """
        # Normalize to [-1, 1]
        t_norm = (doy.to(torch.float32) - 1.0) / self.DAYS_PER_YEAR * 2.0 - 1.0

        # Cast to position-encoding dtype/device for downstream consistency
        # (caller can re-cast if needed).
        return self._encode(t_norm)

    def _encode(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding to a time tensor in [-1, 1].

        Args:
            t_norm: [...] normalized scalar time values.

        Returns:
            encoded: [..., 2 * num_bands + 1]
        """
        bands = torch.linspace(
            1.0, self.max_freq / 2, self.num_bands,
            device=t_norm.device, dtype=t_norm.dtype,
        )  # [num_bands]

        # Broadcast: [...]   ->   [..., 1]   x   [num_bands]   ->   [..., num_bands]
        t_expanded = t_norm.unsqueeze(-1)            # [..., 1]
        scaled = t_expanded * bands * pi             # [..., num_bands]

        sin_feat = scaled.sin()                      # [..., num_bands]
        cos_feat = scaled.cos()                      # [..., num_bands]

        # [sin, cos, raw_t]
        return torch.cat([sin_feat, cos_feat, t_expanded], dim=-1)
        # shape: [..., 2*num_bands + 1]

    def get_output_dim(self):
        """Return the output dimensionality."""
        return self.out_dim