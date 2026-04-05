import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


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


class SpectralEncoder(nn.Module):
    """
    Spectral encoder using uniform Gaussian basis functions.
    
    Physics-based encoding for optical bands:
        1. Place K Gaussians uniformly from wl_min to wl_max
        2. For each band (λ, μ), integrate each Gaussian over [λ-μ/2, λ+μ/2]
        3. L2-normalize the integral vector
        4. Project to output dimension via learned linear layer
    
    This naturally generalizes from narrow hyperspectral bands to unseen
    broadband configurations: a broad band's encoding is the weighted
    average of narrow bands' encodings, lying in the convex hull
    of training encodings.
    
    Non-optical modalities (SAR, DEM) use learned embeddings.
    
    The physics codebook is precomputed at init and frozen — only the
    linear projection and abstract embeddings are learned.
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table
        
        # ── Gaussian basis config ──────────────────────────────────
        atomiser_cfg = config.get("Atomiser", {})
        self.n_gaussians = atomiser_cfg.get("spectral_n_gaussians", 256)
        self.wl_min = atomiser_cfg.get("spectral_wl_min", 350.0)
        self.wl_max = atomiser_cfg.get("spectral_wl_max", 2600.0)
        self.proj_dim = atomiser_cfg.get("spectral_proj_dim", 32)
        
        # Raw dimension = n_gaussians (before projection)
        self.raw_dim = self.n_gaussians
        self.out_dim = self.proj_dim
        
        # ── Fixed Gaussian centers and sigma ───────────────────────
        centers = torch.linspace(self.wl_min, self.wl_max, self.n_gaussians)
        self.register_buffer("centers", centers)
        
        # Sigma = spacing between centers (smooth coverage, no gaps)
        sigma = (self.wl_max - self.wl_min) / self.n_gaussians
        self.register_buffer("sigma", torch.tensor(sigma))
        
        # ── Learned linear projection ──────────────────────────────
        # No bias: the L2-normalized input is already centered
        self.proj = nn.Linear(self.raw_dim, self.out_dim, bias=False)
        
        # ── Build physics codebook ─────────────────────────────────
        num_channels = len(lookup_table.table_wave)
        
        # Track abstract channels
        self.abstract_channel_map = {}  # idx -> channel_name
        abstract_indices = []
        
        # Precompute raw integral vectors for all optical channels
        raw_vectors = torch.zeros(num_channels, self.raw_dim)
        
        for (bandwidth, central_wavelength), idx in lookup_table.table_wave.items():
            if bandwidth < 0 or central_wavelength < 0:
                channel_name = self._identify_abstract_channel(
                    bandwidth, central_wavelength)
                self.abstract_channel_map[idx] = channel_name
                abstract_indices.append(idx)
            else:
                raw_vectors[idx] = self._compute_integral_vector(
                    float(central_wavelength), float(bandwidth))
        
        # Store raw vectors for projection in forward()
        self.register_buffer("raw_codebook", raw_vectors)
        
        # ── Abstract channel embeddings ────────────────────────────
        self._create_named_embeddings(abstract_indices)
        
        # ── Info ───────────────────────────────────────────────────
        n_optical = num_channels - len(abstract_indices)
        print(f"[SpectralEncoder] Uniform Gaussians: K={self.n_gaussians}, "
              f"range=[{self.wl_min}, {self.wl_max}]nm, "
              f"σ={sigma:.1f}nm")
        print(f"[SpectralEncoder] Projection: {self.raw_dim} → {self.out_dim}")
        print(f"[SpectralEncoder] Optical channels: {n_optical}, "
              f"Abstract: {len(abstract_indices)}")
        for idx, name in self.abstract_channel_map.items():
            print(f"[SpectralEncoder]   idx={idx} → {name}")
    
    # =========================================================================
    # GAUSSIAN INTEGRAL
    # =========================================================================
    
    def _compute_integral_vector(
        self, central_wavelength: float, bandwidth: float
    ) -> torch.Tensor:
        """
        Compute the integral of each Gaussian basis over the band's
        spectral support [λ - μ/2, λ + μ/2].
        
        Uses the error function for analytical integration:
            ∫ N(c_i, σ) dλ = 0.5 * [erf((hi - c_i)/(σ√2)) - erf((lo - c_i)/(σ√2))]
        
        Returns L2-normalized vector of shape [n_gaussians].
        """
        lo = central_wavelength - bandwidth / 2.0
        hi = central_wavelength + bandwidth / 2.0
        
        sigma_val = self.sigma.item()
        sqrt2_sigma = sigma_val * math.sqrt(2.0)
        
        z_lo = (lo - self.centers) / sqrt2_sigma
        z_hi = (hi - self.centers) / sqrt2_sigma
        
        integrals = 0.5 * (torch.erf(z_hi) - torch.erf(z_lo))
        
        # L2 normalize
        norm = integrals.norm(p=2)
        if norm > 1e-8:
            integrals = integrals / norm
        
        return integrals
    
    # =========================================================================
    # ABSTRACT CHANNELS
    # =========================================================================
    
    def _identify_abstract_channel(
        self, bandwidth: int, central_wavelength: int
    ) -> str:
        """Identify abstract channel type from its lookup key."""
        for name, info in ABSTRACT_CHANNELS.items():
            if (info["bandwidth"] == bandwidth and
                    info["central_wavelength"] == central_wavelength):
                return name
        return f"ABSTRACT_bw{bandwidth}_wl{central_wavelength}"
    
    def _create_named_embeddings(self, abstract_indices: List[int]):
        """Create one learned embedding per unique abstract channel type."""
        unique_channels = set(self.abstract_channel_map.values())
        
        if not unique_channels:
            self.named_embeddings = nn.ModuleDict()
            self.name_to_safe_name = {}
            return
        
        self.named_embeddings = nn.ModuleDict()
        
        for channel_name in unique_channels:
            # Learned embedding in output space (proj_dim)
            embedding = nn.Parameter(torch.zeros(self.out_dim))
            nn.init.trunc_normal_(embedding, std=0.02, a=-2., b=2.)
            
            safe_name = channel_name.replace(".", "_").replace("-", "_")
            self.named_embeddings[safe_name] = nn.Module()
            self.named_embeddings[safe_name].register_parameter(
                "embedding", embedding)
        
        self.name_to_safe_name = {
            name: name.replace(".", "_").replace("-", "_")
            for name in unique_channels
        }
        
        print(f"[SpectralEncoder] Learned embeddings: "
              f"{', '.join(sorted(unique_channels))}")
    
    # =========================================================================
    # FORWARD
    # =========================================================================
    
    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_indices: [...] integer indices into table_wave
            
        Returns:
            embeddings: [..., out_dim]
        """
        # Bounds check
        max_idx = channel_indices.max().item()
        min_idx = channel_indices.min().item()
        codebook_size = self.raw_codebook.shape[0]
        
        if max_idx >= codebook_size or min_idx < 0:
            raise IndexError(
                f"channel_indices [{min_idx}, {max_idx}] out of bounds "
                f"for codebook size {codebook_size}")
        
        # Look up raw integral vectors [..., n_gaussians]
        raw = self.raw_codebook[channel_indices]
        
        # Project: [..., n_gaussians] → [..., out_dim]
        orig_shape = raw.shape[:-1]
        flat = raw.reshape(-1, self.raw_dim)
        projected = self.proj(flat)
        embeddings = projected.reshape(*orig_shape, self.out_dim)
        
        # Replace abstract channel positions with learned embeddings
        if self.named_embeddings:
            for idx, channel_name in self.abstract_channel_map.items():
                mask = (channel_indices == idx)
                if mask.any():
                    safe_name = self.name_to_safe_name[channel_name]
                    learned_vec = self.named_embeddings[safe_name].embedding
                    embeddings = torch.where(
                        mask.unsqueeze(-1),
                        learned_vec.expand_as(embeddings),
                        embeddings,
                    )
        
        return embeddings
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def get_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        """Get learned embedding for an abstract channel."""
        if (hasattr(self, 'name_to_safe_name') and
                channel_name in self.name_to_safe_name):
            safe_name = self.name_to_safe_name[channel_name]
            return self.named_embeddings[safe_name].embedding
        return None
    
    def set_embedding(self, channel_name: str, values: torch.Tensor):
        """Set learned embedding for an abstract channel."""
        if (hasattr(self, 'name_to_safe_name') and
                channel_name in self.name_to_safe_name):
            safe_name = self.name_to_safe_name[channel_name]
            with torch.no_grad():
                self.named_embeddings[safe_name].embedding.copy_(values)
    
    def encode_band(self, wavelength: float, bandwidth: float) -> torch.Tensor:
        """
        Encode a single band on-the-fly (not in codebook).
        Useful for debugging or novel bands at inference.
        
        Returns: [out_dim] tensor on same device as projection weights.
        """
        raw = self._compute_integral_vector(wavelength, bandwidth)
        raw = raw.to(self.proj.weight.device).unsqueeze(0)
        return self.proj(raw).squeeze(0)
    
    def visualize_codebook(self, n_samples: int = 10):
        """Print a few entries for debugging."""
        print(f"\n[SpectralEncoder] Codebook samples:")
        for (bw, wl), idx in list(self.lookup_table.table_wave.items())[:n_samples]:
            if bw < 0:
                name = self.abstract_channel_map.get(idx, "?")
                print(f"  idx={idx:>4d}  ABSTRACT ({name})")
            else:
                raw = self.raw_codebook[idx]
                peak_gauss = raw.argmax().item()
                peak_wl = self.centers[peak_gauss].item()
                n_active = (raw > 0.01).sum().item()
                print(f"  idx={idx:>4d}  λ={wl:>6.0f}nm  Δλ={bw:>4.0f}nm  "
                      f"peak_basis={peak_gauss} (~{peak_wl:.0f}nm)  "
                      f"active={n_active}")


def build_spectral_encoder(
    config: Dict[str, Any], lookup_table: Any
) -> SpectralEncoder:
    """Factory function for spectral encoder."""
    return SpectralEncoder(config, lookup_table)


# =============================================================================
# LOOKUP TABLE HELPER
# =============================================================================

def register_abstract_channel(lookup_table, channel_name: str) -> int:
    """
    Register an abstract channel in the lookup table.
    
    Args:
        lookup_table: The lookup table object with table_wave dict
        channel_name: Name (must be in ABSTRACT_CHANNELS)
        
    Returns:
        The assigned index for this channel
    """
    if channel_name not in ABSTRACT_CHANNELS:
        raise ValueError(
            f"Unknown abstract channel: {channel_name}. "
            f"Known: {list(ABSTRACT_CHANNELS.keys())}")
    
    info = ABSTRACT_CHANNELS[channel_name]
    key = (info["bandwidth"], info["central_wavelength"])
    
    if key in lookup_table.table_wave:
        return lookup_table.table_wave[key]
    
    new_idx = len(lookup_table.table_wave)
    lookup_table.table_wave[key] = new_idx
    return new_idx