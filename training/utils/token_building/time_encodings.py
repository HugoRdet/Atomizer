import torch
import torch.nn as nn
import math
from typing import Dict, Any


class TimeEncoder(nn.Module):
    """
    Encodes temporal information using circular Radial Basis Functions.
    
    Day-of-year is cyclic (day 1 ≈ day 365), so we use Gaussians placed
    uniformly around the annual cycle with circular distance. This mirrors
    the spectral RBF encoding used for wavelength/bandwidth.
    
    Pipeline:
        time_idx → day-of-year lookup → circular distance to K centers → Gaussian activations → L2 norm
    
    Centers are placed at linspace(0, 365, K), so the first and last
    centers coincide on the circle (day 0 ≡ day 365). This guarantees
    perfect periodicity: the encoding for Dec 31 smoothly wraps to Jan 1.
    
    Circular distance:
        d(t, c) = min(|t - c|, P - |t - c|)    where P = 365
    
    Activation:
        ϕᵢ(t) = exp(-d(t, cᵢ)² / (2σ²))
    
    Sentinel convention:
        time_idx < 0   →  zero vector (datasets without temporal info)
        time_idx >= 0  →  circular RBF of day-of-year
    
    This follows the project-wide convention:
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        
        # Number of Gaussian centers (includes both endpoints 0 and 365)
        self.num_centers = config["Atomiser"].get("time_num_centers", 24)
        
        # Period of the cycle
        self.cycle_period = config["Atomiser"].get("time_cycle_period", 365.0)
        
        # Output dimension: one activation per center
        self.out_dim = self.num_centers
        
        # ── Gaussian centers: uniformly from 0 to 365 ──────
        # First center at 0, last at 365 (same point on circle → periodicity)
        centers = torch.linspace(0.0, self.cycle_period, self.num_centers)
        self.register_buffer("centers", centers)
        
        # ── Sigma: based on spacing between centers ─────────
        # Spacing = 365 / (K-1). Sigma ≈ spacing so adjacent Gaussians
        # overlap smoothly. Using spacing * 0.8 for moderate overlap.
        spacing = self.cycle_period / (self.num_centers - 1)
        sigma = config["Atomiser"].get("time_rbf_sigma", spacing * 0.8)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))
        
        # ── Store reference to lookup for time values ───────
        self._lookup_table = lookup_table
        
        print(f"[TimeEncoder] Circular RBF: {self.num_centers} centers, "
              f"period={self.cycle_period} days, "
              f"spacing={spacing:.1f} days, sigma={sigma:.1f} days, "
              f"out_dim={self.out_dim}")
        print(f"[TimeEncoder] Centers: [{centers[0]:.0f}, {centers[1]:.0f}, "
              f"..., {centers[-2]:.0f}, {centers[-1]:.0f}]")
        print(f"[TimeEncoder] idx < 0 → zero vector (no temporal info)")
    
    def build_time_buffer(self):
        """
        Build the time_idx → day-of-year buffer from the lookup table.
        
        Call this AFTER all datasets have registered their timestamps.
        
        Time keys that are numeric (int/float) are used directly as DOY.
        String keys (ISO dates) are converted to day-of-year.
        Tuple keys (year, doy) use the doy component.
        
        All valid indices are ≥ 0. Negative indices (-1) are handled
        in forward() and never reach this buffer.
        """
        num_times = self._lookup_table.num_time_indices
        device = self.centers.device  # match existing buffer's device
        time_values = torch.zeros(num_times, dtype=torch.float32, device=device)
        
        for time_key, idx in self._lookup_table.table_time.items():
            if isinstance(time_key, (int, float)):
                time_values[idx] = float(time_key)
            elif isinstance(time_key, str):
                time_values[idx] = self._date_to_doy(time_key)
            elif isinstance(time_key, tuple) and len(time_key) >= 2:
                time_values[idx] = float(time_key[1])  # (year, doy)
            else:
                print(f"[TimeEncoder] Warning: unrecognized time key "
                      f"{type(time_key)}: {time_key}, defaulting to 0.0")
                time_values[idx] = 0.0
        
        if hasattr(self, 'time_values'):
            delattr(self, 'time_values')
        self.register_buffer('time_values', time_values)
        
        
    
    @staticmethod
    def _date_to_doy(date_str: str) -> float:
        """Convert ISO date string (YYYY-MM-DD) to day-of-year."""
        from datetime import datetime
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return float(dt.timetuple().tm_yday)
        except ValueError:
            print(f"[TimeEncoder] Warning: could not parse '{date_str}', "
                  f"defaulting to 0.0")
            return 0.0
    
    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Encode time indices into feature vectors via circular RBFs.
        
        Args:
            time_idx: [...] int/float tensor of time indices.
                      < 0  = no-time (returns zeros)
                      >= 0 = valid registered timestamp index
        
        Returns:
            encoding: [..., num_centers] temporal features.
        """
        original_shape = time_idx.shape
        idx_flat = time_idx.long().reshape(-1)  # [N_flat]
        
        # ── Lazy buffer build ───────────────────────────────
        if not hasattr(self, 'time_values') or self.time_values is None:
            self.build_time_buffer()
        
        # ── Rebuild if new times were registered ────────────
        valid_mask = idx_flat >= 0
        if valid_mask.any() and idx_flat[valid_mask].max() >= self.time_values.shape[0]:
            self.build_time_buffer()
        
        # ── Identify N/A tokens (idx < 0) ──────────────────
        is_na = idx_flat < 0
        
        # ── Clamp for safe lookup (negatives → 0) ──────────
        idx_clamped = idx_flat.clamp(0, self.time_values.shape[0] - 1)
        doy = self.time_values[idx_clamped]  # [N_flat]
        
        # ── Circular distance to each center ────────────────
        # doy: [N_flat, 1]  centers: [1, K]
        diff = torch.abs(doy.unsqueeze(-1) - self.centers.unsqueeze(0))  # [N_flat, K]
        
        # Wrap-around: min(|t - c|, P - |t - c|)
        circular_dist = torch.min(diff, self.cycle_period - diff)  # [N_flat, K]
        
        # ── Gaussian activation ─────────────────────────────
        encoded = torch.exp(-circular_dist.pow(2) / (2.0 * self.sigma.pow(2)))  # [N_flat, K]
        
        # ── L2 normalize ────────────────────────────────────
        norm = encoded.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        encoded = encoded / norm
        
        # ── Zero out N/A tokens (idx < 0) ──────────────────
        if is_na.any():
            encoded[is_na] = 0.0
        
        # ── Reshape back ────────────────────────────────────
        return encoded.reshape(*original_shape, self.out_dim)
    
    def get_output_dim(self) -> int:
        return self.out_dim
    
    def extra_repr(self) -> str:
        n_times = 0
        if hasattr(self, 'time_values') and self.time_values is not None:
            n_times = self.time_values.shape[0]
        sigma_val = self.sigma.item() if self.sigma.dim() == 0 else self.sigma[0].item()
        return (
            f"centers={self.num_centers}, "
            f"period={self.cycle_period}, "
            f"sigma={sigma_val:.1f}, "
            f"out_dim={self.out_dim}, "
            f"registered_times={n_times}, "
            f"idx<0=zeros"
        )


def build_time_encoder(config: Dict[str, Any], lookup_table: Any) -> TimeEncoder:
    """Factory function for time encoder."""
    return TimeEncoder(config, lookup_table)