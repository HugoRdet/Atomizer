import torch
from math import pi

def fourier_encode(x: torch.Tensor, max_freq: float, num_bands: int = 4) -> torch.Tensor:
    """
    Standard Fourier Feature mapping.
    
    Returns:
        Tensor of shape [..., 2 * num_bands + 1] as [sin, cos, original]
    """
    if x.dim() == 0 or x.size(-1) != 1:
        x = x.unsqueeze(-1)
    
    device, dtype, orig_x = x.device, x.dtype, x
    
    if num_bands == -1:
        scales = torch.linspace(1., max_freq, int(max_freq), device=device, dtype=dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, int(num_bands), device=device, dtype=dtype)
    
    scales = scales.view(*([1] * (len(x.shape) - 1)), -1)
    
    x_scaled = x * scales * pi
    
    # Preserve original order: [sin, cos, original]
    return torch.cat([x_scaled.sin(), x_scaled.cos(), orig_x], dim=-1)