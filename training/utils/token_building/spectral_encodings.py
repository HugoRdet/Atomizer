"""
Spectral Encoder — Uniform Gaussians + Linear Projection
==========================================================

Encodes spectral band identity (wavelength + bandwidth) into a compact
learned representation.

Architecture:
    1. Fixed codebook: 256 uniformly spaced Gaussians over 350–2600 nm.
       Each band's spectral support [λ - Δλ/2, λ + Δλ/2] is integrated
       against each Gaussian → 256-dim physics vector. L2-normalized.

    2. Abstract channels (SAR, DEM, indices): learned 256-dim embeddings
       stored as named parameters, treated identically to physics vectors.

    3. Linear projection: 256 → 64. Single linear layer (no nonlinearity)
       preserves cosine similarity between RBF vectors, enabling
       generalization to unseen sensor bands.

    4. Forward-time deduplication: spectral indices are highly redundant
       (e.g., 368 unique bands × 4096 pixels = 1.5M tokens, but only
       368 unique spectral vectors). We compute the projection on unique
       indices only, then scatter back.

Key change from previous version:
    MLP (256→128→GELU→64) replaced with Linear(256→64).
    Reason: the MLP memorizes training sensor patterns and fails on
    unseen bands. A linear projection preserves the RBF geometry —
    if two bands are similar in 256-dim RBF space, they remain similar
    in 64-dim projected space. This is critical for cross-sensor transfer.
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================

ABSTRACT_CHANNELS = {
    # Sentinel-1 SAR polarizations
    "VV": {"bandwidth": -1, "central_wavelength": -1},
    "VH": {"bandwidth": -2, "central_wavelength": -2},

    # Elevation / DEM
    "ELEVATION": {"bandwidth": -10, "central_wavelength": -10},
    "DEM": {"bandwidth": -10, "central_wavelength": -10},

    # Slope / Aspect
    "SLOPE": {"bandwidth": -11, "central_wavelength": -11},
    "ASPECT": {"bandwidth": -12, "central_wavelength": -12},

    # Indices
    "NDVI": {"bandwidth": -20, "central_wavelength": -20},
    "NDWI": {"bandwidth": -21, "central_wavelength": -21},
    "MNDWI": {"bandwidth": -22, "central_wavelength": -22},

    # Generic placeholders
    "ABSTRACT_1": {"bandwidth": -100, "central_wavelength": -100},
    "ABSTRACT_2": {"bandwidth": -101, "central_wavelength": -101},
    "ABSTRACT_3": {"bandwidth": -102, "central_wavelength": -102},
}


# =============================================================================
# GAUSSIAN CODEBOOK COMPUTATION
# =============================================================================

def build_uniform_gaussian_codebook(
    n_gaussians: int = 512,
    wl_min: float = 350.0,
    wl_max: float = 2600.0,
    n_sample_points: int = 200,
) -> tuple:
    """
    Build uniformly spaced Gaussian anchors for spectral encoding.

    Args:
        n_gaussians: Number of Gaussian basis functions.
        wl_min: Lower wavelength bound (nm).
        wl_max: Upper wavelength bound (nm).
        n_sample_points: Integration sample count per band.

    Returns:
        means: [n_gaussians] Gaussian center positions (nm).
        sigma: float, standard deviation = spacing between centers.
    """
    means = torch.linspace(wl_min, wl_max, n_gaussians)
    sigma = (wl_max - wl_min) / (n_gaussians - 1)  # = spacing
    return means, sigma


def compute_band_encoding(
    center_wl: float,
    bandwidth: float,
    gaussian_means: torch.Tensor,
    gaussian_sigma: float,
    n_points: int = 200,
) -> torch.Tensor:
    """
    Compute the spectral encoding for one physical band.

    Integrates the band's spectral response (uniform over [λ-Δλ/2, λ+Δλ/2])
    against each Gaussian basis function using dense sampling.

    Args:
        center_wl: Central wavelength (nm).
        bandwidth: Full bandwidth (nm).
        gaussian_means: [K] centers of the Gaussians.
        gaussian_sigma: Shared standard deviation.
        n_points: Number of integration sample points.

    Returns:
        encoding: [K] response vector, L2-normalized.
    """
    half_bw = bandwidth / 2.0
    wl_lo = center_wl - half_bw
    wl_hi = center_wl + half_bw

    # Sample wavelengths across the band's support
    device = gaussian_means.device
    t = torch.linspace(0, 1, n_points, device=device)
    lambdas = wl_lo + (wl_hi - wl_lo) * t  # [n_points]
    lambdas = lambdas.unsqueeze(1)  # [n_points, 1]

    means = gaussian_means.unsqueeze(0)  # [1, K]

    # Gaussian response: exp(-0.5 * ((λ - μ) / σ)^2)
    responses = torch.exp(
        -0.5 * ((lambdas - means) / gaussian_sigma) ** 2
    )  # [n_points, K]

    # Integrate (mean over sample points ≈ integral / bandwidth)
    encoding = responses.mean(dim=0)  # [K]

    # L2-normalize
    norm = encoding.norm(p=2)
    if norm > 1e-8:
        encoding = encoding / norm

    return encoding


# =============================================================================
# SPECTRAL ENCODER
# =============================================================================

class SpectralEncoder(nn.Module):
    """
    Spectral encoder with uniform Gaussians + linear projection.

    The codebook is fixed (no gradients); the projection is learned.
    Abstract channels (SAR, DEM, etc.) use learned embeddings of the
    same raw dimension as the physics codebook, then pass through the
    same projection.

    Key design choice: single linear layer (no nonlinearity) preserves
    the cosine similarity structure of the RBF codebook. This enables
    generalization to unseen sensors whose RBF patterns were never
    seen during training.

    Forward-time deduplication: only unique spectral indices are
    processed through the projection; results are scattered back.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        lookup_table: Any,
    ):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table

        # ── Gaussian config ─────────────────────────────────────────
        atom_cfg = config.get("Atomiser", config.get("Atomizer", {}))
        self.n_gaussians = atom_cfg.get("spectral_n_gaussians", 256)
        self.wl_min = atom_cfg.get("spectral_wl_min", 350.0)
        self.wl_max = atom_cfg.get("spectral_wl_max", 2600.0)
        self.n_sample_points = atom_cfg.get("spectral_n_sample_points", 200)

        # ── Projection config ───────────────────────────────────────
        self.mlp_hidden = atom_cfg.get("spectral_mlp_hidden", 128)  # unused but kept for compat
        self.mlp_out = atom_cfg.get("spectral_mlp_out", 64)
        self.out_dim = self.mlp_out  # External interface

        # ── Raw Gaussian dimension ──────────────────────────────────
        self.raw_dim = self.n_gaussians

        # ── Build Gaussian anchors ──────────────────────────────────
        gaussian_means, gaussian_sigma = build_uniform_gaussian_codebook(
            n_gaussians=self.n_gaussians,
            wl_min=self.wl_min,
            wl_max=self.wl_max,
            n_sample_points=self.n_sample_points,
        )
        self.register_buffer("gaussian_means", gaussian_means)
        self.gaussian_sigma = gaussian_sigma

        # ── Build fixed physics codebook ────────────────────────────
        num_channels = len(lookup_table.table_wave)
        physics_codebook = torch.zeros(num_channels, self.raw_dim)

        self.abstract_channel_map = {}  # idx → channel_name
        abstract_indices = []

        for (bandwidth, central_wavelength), idx in lookup_table.table_wave.items():
            if bandwidth < 0 or central_wavelength < 0:
                channel_name = self._identify_abstract_channel(
                    bandwidth, central_wavelength
                )
                self.abstract_channel_map[idx] = channel_name
                abstract_indices.append(idx)
            else:
                physics_codebook[idx] = compute_band_encoding(
                    center_wl=float(central_wavelength),
                    bandwidth=float(bandwidth),
                    gaussian_means=gaussian_means,
                    gaussian_sigma=gaussian_sigma,
                    n_points=self.n_sample_points,
                )

        self.register_buffer("physics_codebook", physics_codebook)

        # ── Learnable embeddings for abstract channels ──────────────
        self._create_named_embeddings(abstract_indices)

        # ── MLP: raw_dim → mlp_hidden → mlp_out ────────────────────
        self.spectral_mlp = nn.Sequential(
            nn.Linear(self.raw_dim, self.mlp_out),
            #nn.GELU(),
            #nn.Linear(self.mlp_hidden, self.mlp_out),
        )

        # ── Logging ─────────────────────────────────────────────────
        n_phys = num_channels - len(abstract_indices)
        print(f"[SpectralEncoder] {self.n_gaussians} uniform Gaussians "
              f"over [{self.wl_min}, {self.wl_max}] nm, "
              f"σ = {gaussian_sigma:.1f} nm")
        print(f"[SpectralEncoder] MLP: {self.raw_dim} → {self.mlp_hidden} "
              f"→ {self.mlp_out}")
        print(f"[SpectralEncoder] Physics channels: {n_phys}, "
              f"Abstract channels: {len(abstract_indices)}")
        for idx, name in self.abstract_channel_map.items():
            print(f"[SpectralEncoder]   idx={idx} → {name}")

    # ─────────────────────────────────────────────────────────────────
    # Abstract channel helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _identify_abstract_channel(
        bandwidth: int, central_wavelength: int,
    ) -> str:
        for name, info in ABSTRACT_CHANNELS.items():
            if (info["bandwidth"] == bandwidth
                    and info["central_wavelength"] == central_wavelength):
                return name
        return f"ABSTRACT_bw{bandwidth}_wl{central_wavelength}"

    def _create_named_embeddings(self, abstract_indices: List[int]):
        unique_channels = set(self.abstract_channel_map.values())

        if not unique_channels:
            self.named_embeddings = nn.ModuleDict()
            self.name_to_safe_name = {}
            return

        self.named_embeddings = nn.ModuleDict()

        for channel_name in sorted(unique_channels):
            embedding = nn.Parameter(torch.zeros(self.raw_dim))
            nn.init.trunc_normal_(embedding, std=0.02, a=-2.0, b=2.0)

            safe_name = channel_name.replace(".", "_").replace("-", "_")
            mod = nn.Module()
            mod.register_parameter("embedding", embedding)
            self.named_embeddings[safe_name] = mod

        self.name_to_safe_name = {
            name: name.replace(".", "_").replace("-", "_")
            for name in unique_channels
        }

    # ─────────────────────────────────────────────────────────────────
    # Codebook expansion (for newly registered bands at inference time)
    # ─────────────────────────────────────────────────────────────────

    def _maybe_expand_codebook(self):
        """
        If new bands were registered in lookup_table since __init__,
        expand the physics codebook to cover them.
        """
        current_size = self.physics_codebook.shape[0]
        needed_size = len(self.lookup_table.table_wave)

        if needed_size <= current_size:
            return

        # Expand
        extra = torch.zeros(
            needed_size - current_size,
            self.raw_dim,
            device=self.physics_codebook.device,
        )

        for (bandwidth, central_wavelength), idx in self.lookup_table.table_wave.items():
            if idx < current_size:
                continue  # already in codebook

            if bandwidth < 0 or central_wavelength < 0:
                channel_name = self._identify_abstract_channel(
                    bandwidth, central_wavelength
                )
                self.abstract_channel_map[idx] = channel_name
                # Will be handled by named_embeddings lookup
            else:
                row = compute_band_encoding(
                    center_wl=float(central_wavelength),
                    bandwidth=float(bandwidth),
                    gaussian_means=self.gaussian_means,
                    gaussian_sigma=self.gaussian_sigma,
                    n_points=self.n_sample_points,
                )
                extra[idx - current_size] = row

        self.physics_codebook = torch.cat(
            [self.physics_codebook, extra.to(self.physics_codebook.device)],
            dim=0,
        )

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode spectral band indices into compact representations.

        Uses deduplication: only unique indices go through the projection.

        Args:
            channel_indices: [*] integer spectral indices (col 3 of tokens)

        Returns:
            embeddings: [*, out_dim] compressed spectral features
        """
        self._maybe_expand_codebook()

        original_shape = channel_indices.shape
        flat_indices = channel_indices.reshape(-1)  # [N]

        # ── Deduplicate ─────────────────────────────────────────────
        unique_indices, inverse_map = torch.unique(
            flat_indices, return_inverse=True
        )
        # unique_indices: [U], inverse_map: [N] → index into unique

        # ── Build raw embeddings for unique indices only ────────────
        raw = self.physics_codebook[unique_indices]  # [U, raw_dim]

        # ── Replace abstract channels with learned embeddings ───────
        if self.named_embeddings:
            for idx_val, channel_name in self.abstract_channel_map.items():
                mask = (unique_indices == idx_val)
                if mask.any():
                    safe_name = self.name_to_safe_name[channel_name]
                    learned = self.named_embeddings[safe_name].embedding
                    raw = torch.where(
                        mask.unsqueeze(-1),
                        learned.unsqueeze(0).expand_as(raw),
                        raw,
                    )

        # ── Linear projection (only on U unique vectors) ────────────
        compressed = self.spectral_mlp(raw)  # [U, out_dim]

        # ── Scatter back to all tokens ──────────────────────────────
        result = compressed[inverse_map]  # [N, out_dim]

        # ── Reshape to original batch dims ──────────────────────────
        return result.reshape(*original_shape, self.out_dim)

    # ─────────────────────────────────────────────────────────────────
    # Accessors (backward compat)
    # ─────────────────────────────────────────────────────────────────

    def get_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        """Get raw learned embedding for an abstract channel (pre-projection)."""
        if hasattr(self, "name_to_safe_name") and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            return self.named_embeddings[safe_name].embedding
        return None

    def set_embedding(self, channel_name: str, values: torch.Tensor):
        """Set raw learned embedding for an abstract channel."""
        if hasattr(self, "name_to_safe_name") and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            with torch.no_grad():
                self.named_embeddings[safe_name].embedding.copy_(values)

    def get_compressed_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        """Get the post-projection embedding for an abstract channel."""
        raw = self.get_embedding(channel_name)
        if raw is None:
            return None
        with torch.no_grad():
            return self.spectral_mlp(raw.unsqueeze(0)).squeeze(0)


# =============================================================================
# FACTORY
# =============================================================================

def build_spectral_encoder(
    config: Dict[str, Any], lookup_table: Any,
) -> SpectralEncoder:
    """Factory function — drop-in replacement."""
    return SpectralEncoder(config, lookup_table)


# =============================================================================
# LOOKUP TABLE HELPERS
# =============================================================================

def register_abstract_channel(lookup_table, channel_name: str) -> int:
    """
    Register an abstract channel in the lookup table.

    Args:
        lookup_table: Lookup object with table_wave dict.
        channel_name: Must be in ABSTRACT_CHANNELS.

    Returns:
        Assigned index.
    """
    if channel_name not in ABSTRACT_CHANNELS:
        raise ValueError(
            f"Unknown abstract channel: {channel_name}. "
            f"Known: {list(ABSTRACT_CHANNELS.keys())}"
        )

    info = ABSTRACT_CHANNELS[channel_name]
    key = (info["bandwidth"], info["central_wavelength"])

    if key in lookup_table.table_wave:
        return lookup_table.table_wave[key]

    new_idx = len(lookup_table.table_wave)
    lookup_table.table_wave[key] = new_idx
    return new_idx