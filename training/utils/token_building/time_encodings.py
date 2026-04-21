import torch
import torch.nn as nn
import math
from typing import Dict, Any


class TimeEncoder(nn.Module):
    """
    Encodes temporal information using Fourier features of day-of-year.

    Day-of-year is inherently cyclic (day 1 ≈ day 365), and Fourier features
    encode this periodicity exactly — no wrap-around distance computation
    needed. The encoding is smooth, translation-equivariant in angle, and
    generalizes across any number of frequency bands.

    Pipeline:
        time_idx → DOY lookup → normalize to angle → Fourier features

    Encoding:
        θ(t) = 2π · t / P                          (angle, P = 365)

        features = [
            sin(1·θ), cos(1·θ),                     # fundamental (period=365d)
            sin(2·θ), cos(2·θ),                     # 2nd harmonic (period=182.5d)
            ...
            sin(K·θ), cos(K·θ),                     # K-th harmonic
        ]

    Output dim: 2K  (K frequency bands × 2 for sin/cos pair)

    The fundamental (k=1) captures the yearly cycle (summer ↔ winter).
    Higher harmonics capture finer seasonal structure (spring-greening,
    autumn-senescence, etc.). K=8 covers periods from 365 days down to
    ~45 days, which matches the time scale at which vegetation changes.

    Sentinel convention:
        time_idx < 0   →  zero vector (datasets without temporal info)
        time_idx >= 0  →  Fourier features of day-of-year
    """

    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()

        # Number of frequency bands (harmonics of the fundamental)
        # Each band contributes 2 dims (sin, cos)
        self.num_freq_bands = config["Atomiser"].get("time_num_freq_bands",
            config["Atomiser"].get("time_num_centers", 8))  # back-compat with old key

        # Period of the cycle
        self.cycle_period = config["Atomiser"].get("time_cycle_period", 365.0)

        # Output dim: 2 per frequency band
        self.out_dim = 2 * self.num_freq_bands

        # ── Frequencies: 1, 2, 3, ..., K cycles per period ──────────
        # These are harmonics of the fundamental (1 cycle per year).
        # Pre-multiplied by 2π for efficiency in forward.
        freqs = torch.arange(1, self.num_freq_bands + 1, dtype=torch.float32)
        angular_freqs = 2.0 * math.pi * freqs / self.cycle_period   # [K]
        self.register_buffer("angular_freqs", angular_freqs)

        # ── Lookup table reference ──────────────────────────────────
        self._lookup_table = lookup_table

        # Pre-register all possible day-of-year values (1-365)
        for doy in range(1, 366):
            self._lookup_table.get_or_register_time_idx(doy)

        print(f"[TimeEncoder] Fourier features: {self.num_freq_bands} bands "
              f"(periods: {self.cycle_period:.0f}d → "
              f"{self.cycle_period / self.num_freq_bands:.0f}d), "
              f"out_dim={self.out_dim}")
        print(f"[TimeEncoder] idx < 0 → zero vector (no temporal info)")

    def build_time_buffer(self):
        """
        Build the time_idx → day-of-year buffer from the lookup table.

        Call this AFTER all datasets have registered their timestamps.
        Numeric keys are used directly. String keys (ISO dates) are
        converted to DOY. Tuple keys (year, doy) use the DOY component.
        """
        num_times = self._lookup_table.num_time_indices
        device = self.angular_freqs.device
        time_values = torch.zeros(num_times, dtype=torch.float32, device=device)

        for time_key, idx in self._lookup_table.table_time.items():
            if isinstance(time_key, (int, float)):
                time_values[idx] = float(time_key)
            elif isinstance(time_key, str):
                time_values[idx] = self._date_to_doy(time_key)
            elif isinstance(time_key, tuple) and len(time_key) >= 2:
                time_values[idx] = float(time_key[1])
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
        Encode time indices into Fourier feature vectors.

        Args:
            time_idx: [...] int/float tensor of time indices.
                      < 0  = no-time (returns zeros)
                      >= 0 = valid registered timestamp index

        Returns:
            encoding: [..., 2 * num_freq_bands] Fourier features.
        """
        original_shape = time_idx.shape
        idx_flat = time_idx.long().reshape(-1)                           # [N]

        # ── Lazy buffer build ───────────────────────────────────────
        if not hasattr(self, 'time_values') or self.time_values is None:
            self.build_time_buffer()

        # ── Rebuild if new times were registered ────────────────────
        valid_mask = idx_flat >= 0
        if valid_mask.any() and idx_flat[valid_mask].max() >= self.time_values.shape[0]:
            self.build_time_buffer()

        # ── Identify N/A tokens (idx < 0) ───────────────────────────
        is_na = idx_flat < 0

        # ── Clamp for safe lookup (negatives → 0, will be zeroed later) ─
        idx_clamped = idx_flat.clamp(0, self.time_values.shape[0] - 1)
        doy = self.time_values[idx_clamped]                              # [N]

        # ── Fourier encoding ────────────────────────────────────────
        # doy: [N, 1]     angular_freqs: [1, K]
        # angles: [N, K]  (each DOY times each frequency)
        angles = doy.unsqueeze(-1) * self.angular_freqs.unsqueeze(0)     # [N, K]

        # Interleave sin/cos — output shape [N, 2K]
        sin_enc = torch.sin(angles)                                       # [N, K]
        cos_enc = torch.cos(angles)                                       # [N, K]
        encoded = torch.stack([sin_enc, cos_enc], dim=-1).reshape(-1, self.out_dim)
        # encoded: [N, 2K]  — layout: [sin_1, cos_1, sin_2, cos_2, ..., sin_K, cos_K]

        # ── Zero out N/A tokens (idx < 0) ───────────────────────────
        if is_na.any():
            encoded[is_na] = 0.0

        # ── Reshape back to original shape ──────────────────────────
        return encoded.reshape(*original_shape, self.out_dim)

    def get_output_dim(self) -> int:
        return self.out_dim

    def extra_repr(self) -> str:
        n_times = 0
        if hasattr(self, 'time_values') and self.time_values is not None:
            n_times = self.time_values.shape[0]
        return (
            f"num_freq_bands={self.num_freq_bands}, "
            f"period={self.cycle_period}, "
            f"out_dim={self.out_dim}, "
            f"registered_times={n_times}, "
            f"idx<0=zeros"
        )


def build_time_encoder(config: Dict[str, Any], lookup_table: Any) -> TimeEncoder:
    """Factory function for time encoder."""
    return TimeEncoder(config, lookup_table)