"""
Temporal Reflectance Profile
==============================

Parameter-free temporal compression for multi-temporal tokens.

Modes:
    "doy"      — Gaussian kernels, uniform, circular (phenology)
    "delta_t"  — Gaussian kernels, non-uniform, linear (change detection)
    "fourier"  — Fourier harmonics on time (dense, global)
    "hybrid"   — Fourier harmonics + Gaussian kernels
    "lifted"   — Fourier(reflectance) × Gaussian(time)  [NEW]

The "lifted" mode applies Fourier features to reflectance BEFORE
temporal Gaussian projection:

    Standard:  φₖ = Σ_t  r_t              × G_k(t)     → 1 scalar per center
    Lifted:    φₖ = Σ_t  Fourier(r_t)     × G_k(t)     → 2L values per center

    where Fourier(r) = [sin(πf₁r), cos(πf₁r), ..., sin(πf_Lr), cos(πf_Lr)]

Why this matters:
    sin is ODD:  sin(πr) + sin(-πr) = 0    → sign changes CANCEL
    cos is EVEN: cos(πr) + cos(-πr) = 2c   → magnitudes ADD

    Deforestation (r goes +0.7 → -0.7):  sin component ≈ 0 (cancelled)
    Stable forest (r stays +0.7):         sin component ≈ 1.4 (reinforced)

    Change detection is built into the basis function, not learned.

Output:
    profile:  [N, K] — temporal encoding (K = n_centers)
    support:  [N, K_s] — observation window (K_s = n_support_centers)
    Total packed dims: K + K_s

Note on normalization:
    Lifted mode does NOT L2-normalize the profile. Values are naturally
    bounded: sin/cos ∈ [-1,1], Gaussian ∈ [0,1], product ∈ [-1,1].
    L2 normalization would destroy the per-frequency scale information.
    Other modes still L2-normalize for backward compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalReflectanceProfile(nn.Module):
    """
    Project T irregular observations onto K temporal basis functions.

    Args:
        n_centers:      total output dimension for profile
        max_delta_t:    maximum time range (365 for DOY, or max Δt)
        mode:           "delta_t" | "doy" | "fourier" | "hybrid" | "lifted"
        dropout:        fraction of profile entries to zero during training
        n_freqs:        number of Fourier frequencies for lifted mode (default 2)
        n_support:      number of support centers (default = n_centers)
        sigma_factor:   multiplier for Gaussian sigma (default 0.8)
    """

    def __init__(self, n_centers: int = 16, max_delta_t: float = 365.0,
                 mode: str = "delta_t", dropout: float = 0.0,
                 n_freqs: int = 2, n_support: int = None,
                 sigma_factor: float = 0.8):
        super().__init__()
        self.n_centers = n_centers
        self.max_delta_t = max_delta_t
        self.mode = mode
        self.dropout = dropout
        self.n_freqs = n_freqs
        self.sigma_factor = sigma_factor

        # ── Mode-specific configuration ──────────────────────────────
        if mode == "lifted":
            # Fourier(reflectance) × Gaussian(time)
            # Each center produces 2*n_freqs values (sin+cos per frequency)
            self.n_lifted_per_center = 2 * n_freqs
            self.n_gaussian = n_centers // self.n_lifted_per_center
            if self.n_gaussian < 2:
                self.n_gaussian = 2
            # Actual profile dim may differ slightly from n_centers
            self._profile_dim = self.n_gaussian * self.n_lifted_per_center
            self.n_fourier = 0
            self.n_harmonics = 0
            self._pad = 0

            # Fourier frequencies for reflectance: [1, 2, ..., n_freqs]
            freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
            self.register_buffer("refl_freqs", freqs)

            print(f"[TemporalProfile] Lifted mode: {self.n_gaussian} centers × "
                  f"{self.n_lifted_per_center} Fourier dims = {self._profile_dim}")
            print(f"[TemporalProfile] Reflectance frequencies: "
                  f"{[f'{f:.0f}' for f in freqs.tolist()]}")

        elif mode == "fourier":
            self.n_harmonics = (n_centers - 1) // 2
            self.n_fourier = 1 + 2 * self.n_harmonics
            self.n_gaussian = 0
            self._pad = n_centers - self.n_fourier
            self._profile_dim = n_centers
            print(f"[TemporalProfile] Fourier mode: {self.n_harmonics} harmonics, "
                  f"{self.n_fourier} coefficients" +
                  (f" + {self._pad} padding" if self._pad > 0 else ""))

        elif mode == "hybrid":
            n_fourier_target = max(n_centers // 3, 3)
            self.n_harmonics = (n_fourier_target - 1) // 2
            self.n_fourier = 1 + 2 * self.n_harmonics
            self.n_gaussian = n_centers - self.n_fourier
            self._pad = 0
            self._profile_dim = n_centers
            print(f"[TemporalProfile] Hybrid mode: {self.n_fourier} Fourier + "
                  f"{self.n_gaussian} Gaussian = {n_centers}")

        else:
            # Pure Gaussian (delta_t or doy)
            self.n_harmonics = 0
            self.n_fourier = 0
            self.n_gaussian = n_centers
            self._pad = 0
            self._profile_dim = n_centers

        # ── Build Gaussian centers (if needed) ───────────────────────
        if self.n_gaussian > 0:
            if mode == "delta_t":
                centers = self._build_nonuniform_centers(self.n_gaussian, max_delta_t)
            else:
                # doy, hybrid, lifted: uniform spacing
                centers = torch.linspace(0, max_delta_t, self.n_gaussian + 1)[:-1]

            self.register_buffer("centers", centers)

            spacing = torch.diff(centers, prepend=centers[:1])
            spacing[0] = centers[1] - centers[0] if self.n_gaussian > 1 else max_delta_t / self.n_gaussian
            sigma = torch.clamp(spacing * sigma_factor, min=1.0)
            self.register_buffer("sigma", sigma)

            print(f"[TemporalProfile] Gaussian: {self.n_gaussian} centers, "
                  f"range=[{centers[0]:.0f}, {centers[-1]:.0f}], "
                  f"σ={sigma[0]:.1f}-{sigma[-1]:.1f}")
        else:
            self.register_buffer("centers", torch.zeros(0))
            self.register_buffer("sigma", torch.zeros(0))

        # ── Support centers (always uniform Gaussian) ────────────────
        n_sup = n_support if n_support is not None else self._profile_dim
        self._support_dim = n_sup
        support_centers = torch.linspace(0, max_delta_t, n_sup + 1)[:-1]
        support_spacing = torch.diff(support_centers, prepend=support_centers[:1])
        support_spacing[0] = support_centers[1] - support_centers[0] if n_sup > 1 else max_delta_t / n_sup
        support_sigma = torch.clamp(support_spacing * sigma_factor, min=1.0)
        self.register_buffer("support_centers", support_centers)
        self.register_buffer("support_sigma", support_sigma)

        if dropout > 0:
            print(f"[TemporalProfile] Dropout: {dropout:.0%} during training")

        print(f"[TemporalProfile] Output: profile={self._profile_dim} + "
              f"support={self._support_dim} = {self.get_output_dim()}")

    @staticmethod
    def _build_nonuniform_centers(n_centers: int, max_t: float) -> torch.Tensor:
        if n_centers <= 1:
            return torch.tensor([0.0])
        alpha = 1.8
        t = torch.linspace(0, 1, n_centers)
        return max_t * (t ** alpha)

    # ─────────────────────────────────────────────────────────────────
    # GAUSSIAN WEIGHTS (shared by gaussian, hybrid, lifted)
    # ─────────────────────────────────────────────────────────────────

    def _compute_gaussian_weights(self, time_values, time_mask=None):
        """
        Gaussian temporal weights: [N, T, K_g].
        """
        dt = time_values.unsqueeze(-1)               # [N, T, 1]
        mu = self.centers.unsqueeze(0).unsqueeze(0)   # [1, 1, K_g]
        sig = self.sigma.unsqueeze(0).unsqueeze(0)    # [1, 1, K_g]

        if self.mode in ("doy", "hybrid", "lifted"):
            diff = torch.abs(dt - mu)
            diff = torch.min(diff, self.max_delta_t - diff)
        else:
            diff = dt - mu

        weights = torch.exp(-0.5 * (diff / sig) ** 2)  # [N, T, K_g]

        if time_mask is not None:
            weights = weights * (~time_mask).unsqueeze(-1).float()

        return weights

    # ─────────────────────────────────────────────────────────────────
    # MODE: GAUSSIAN (raw reflectance × Gaussian weights)
    # ─────────────────────────────────────────────────────────────────

    def _compute_gaussian(self, refl_values, time_values, time_mask=None):
        weights = self._compute_gaussian_weights(time_values, time_mask)

        refl = refl_values.unsqueeze(-1)  # [N, T, 1]
        profile = (refl * weights).sum(dim=1)  # [N, K_g]

        weight_sum = weights.sum(dim=1)
        max_weight = weight_sum.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        profile = profile / max_weight

        return profile

    # ─────────────────────────────────────────────────────────────────
    # MODE: FOURIER (Fourier harmonics of time)
    # ─────────────────────────────────────────────────────────────────

    def _compute_fourier(self, refl_values, time_values, time_mask=None):
        N, T = refl_values.shape
        P = self.max_delta_t

        if time_mask is not None:
            valid = (~time_mask).float()
            refl_masked = refl_values * valid
            n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            refl_masked = refl_values
            n_valid = T

        dc = refl_masked.sum(dim=1, keepdim=True) / n_valid

        coeffs = [dc]
        for k in range(1, self.n_harmonics + 1):
            angle = 2 * math.pi * k * time_values / P
            sin_basis = torch.sin(angle)
            cos_basis = torch.cos(angle)
            if time_mask is not None:
                sin_basis = sin_basis * valid
                cos_basis = cos_basis * valid
            sin_k = (refl_masked * sin_basis).sum(dim=1, keepdim=True) / n_valid
            cos_k = (refl_masked * cos_basis).sum(dim=1, keepdim=True) / n_valid
            coeffs.append(sin_k)
            coeffs.append(cos_k)

        fourier = torch.cat(coeffs, dim=1)

        if self._pad > 0:
            fourier = F.pad(fourier, (0, self._pad))

        return fourier

    # ─────────────────────────────────────────────────────────────────
    # MODE: LIFTED (Fourier(reflectance) × Gaussian(time))
    # ─────────────────────────────────────────────────────────────────

    def _compute_lifted(self, refl_values, time_values, time_mask=None):
        """
        Fourier-lifted Gaussian projection: [N, T] → [N, n_gaussian × 2L].

        For each center k and frequency f:
            φ_{k,f,sin} = Σ_t sin(π·f·r_t) × G_k(t_t)
            φ_{k,f,cos} = Σ_t cos(π·f·r_t) × G_k(t_t)

        Properties:
            sin(πfr) is ODD  → sign changes cancel → detects temporal change
            cos(πfr) is EVEN → magnitudes add     → captures mean intensity
        """
        N, T = refl_values.shape
        K_g = self.n_gaussian
        L = self.n_freqs

        # Gaussian temporal weights: [N, T, K_g]
        weights = self._compute_gaussian_weights(time_values, time_mask)

        # Normalize weights by max sum (bound magnitude regardless of T)
        weight_sum = weights.sum(dim=1)  # [N, K_g]
        max_weight = weight_sum.max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [N, 1]
        weights = weights / max_weight.unsqueeze(1)  # [N, T, K_g]

        # Fourier features of reflectance: [N, T, 2L]
        # freqs: [L] = [1, 2, ..., n_freqs]
        r = refl_values.unsqueeze(-1)               # [N, T, 1]
        f = self.refl_freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        angles = math.pi * f * r                    # [N, T, L]
        fourier_r = torch.cat([
            torch.sin(angles),   # [N, T, L]
            torch.cos(angles),   # [N, T, L]
        ], dim=-1)               # [N, T, 2L]

        # Combine: [N, T, 2L, 1] × [N, T, 1, K_g] → [N, T, 2L, K_g]
        # Sum over T → [N, 2L, K_g] → reshape to [N, K_g × 2L]
        fourier_r_exp = fourier_r.unsqueeze(-1)      # [N, T, 2L, 1]
        weights_exp = weights.unsqueeze(2)            # [N, T, 1, K_g]
        combined = (fourier_r_exp * weights_exp).sum(dim=1)  # [N, 2L, K_g]

        # Reshape: interleave per center [sin1, cos1, sin2, cos2, ...] per center
        # → [N, K_g, 2L] → [N, K_g × 2L]
        profile = combined.permute(0, 2, 1).reshape(N, K_g * 2 * L)

        return profile

    # ─────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        refl_values: torch.Tensor,
        time_values: torch.Tensor,
        time_mask: torch.Tensor = None,
    ):
        """
        Compute temporal profile and support.

        Args:
            refl_values:  [N, T] reflectance (normalized, signed)
            time_values:  [N, T] time values (Δt or DOY)
            time_mask:    [N, T] bool, True = invalid/padded

        Returns:
            profile:  [N, profile_dim]
            support:  [N, support_dim]
        """
        N, T = refl_values.shape

        # ── Profile ──────────────────────────────────────────────────
        if self.mode == "lifted":
            profile = self._compute_lifted(refl_values, time_values, time_mask)
            # NO L2 normalization — values are naturally bounded
            # sin/cos ∈ [-1,1], weights ∈ [0,1], normalized by max_weight

        elif self.mode == "fourier":
            profile = self._compute_fourier(refl_values, time_values, time_mask)
            profile = F.normalize(profile, p=2, dim=-1)

        elif self.mode == "hybrid":
            fourier_part = self._compute_fourier(refl_values, time_values, time_mask)
            gaussian_part = self._compute_gaussian(refl_values, time_values, time_mask)
            profile = torch.cat([fourier_part, gaussian_part], dim=1)
            profile = F.normalize(profile, p=2, dim=-1)

        else:
            # Pure Gaussian (delta_t or doy)
            profile = self._compute_gaussian(refl_values, time_values, time_mask)
            profile = F.normalize(profile, p=2, dim=-1)

        # ── Temporal dropout (training only) ─────────────────────────
        if self.training and self.dropout > 0:
            drop_mask = torch.rand(N, profile.shape[1], device=profile.device) < self.dropout
            profile = profile.masked_fill(drop_mask, 0.0)

        # ── Support ──────────────────────────────────────────────────
        if time_mask is not None:
            large_val = torch.tensor(1e6, device=time_values.device)
            masked_times = time_values.clone()
            masked_times[time_mask] = large_val
            t_min = masked_times.min(dim=1).values
            masked_times[time_mask] = -large_val
            t_max = masked_times.max(dim=1).values
        else:
            t_min = time_values.min(dim=1).values
            t_max = time_values.max(dim=1).values

        sqrt2 = math.sqrt(2.0)
        t_min_k = t_min.unsqueeze(-1)
        t_max_k = t_max.unsqueeze(-1)

        min_half_width = self.support_sigma[0] * 0.5
        t_range = t_max_k - t_min_k
        needs_expand = (t_range < min_half_width * 2)
        t_center = (t_min_k + t_max_k) * 0.5
        t_min_k = torch.where(needs_expand, t_center - min_half_width, t_min_k)
        t_max_k = torch.where(needs_expand, t_center + min_half_width, t_max_k)

        sc = self.support_centers.unsqueeze(0)
        ss = self.support_sigma.unsqueeze(0)

        erf_max = torch.erf((t_max_k - sc) / (ss * sqrt2))
        erf_min = torch.erf((t_min_k - sc) / (ss * sqrt2))
        support = 0.5 * (erf_max - erf_min)
        support = F.normalize(support, p=2, dim=-1)

        return profile, support

    def get_output_dim(self) -> int:
        """Total output dimension: profile + support."""
        return self._profile_dim + self._support_dim