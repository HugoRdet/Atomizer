import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import erf
from typing import Dict, Any, List, Optional


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================

ABSTRACT_CHANNELS = {
    # Sentinel-1 SAR polarizations
    "VV":    {"bandwidth": -1,   "central_wavelength": -1},
    "VH":    {"bandwidth": -2,   "central_wavelength": -2},
    "VV_VH": {"bandwidth": -3,   "central_wavelength": -3},

    # Elevation / DEM
    "ELEVATION": {"bandwidth": -10, "central_wavelength": -10},
    "DEM":       {"bandwidth": -10, "central_wavelength": -10},  # Alias

    # Slope / Aspect
    "SLOPE":  {"bandwidth": -11, "central_wavelength": -11},
    "ASPECT": {"bandwidth": -12, "central_wavelength": -12},

    # Spectral indices
    "NDVI":  {"bandwidth": -20, "central_wavelength": -20},
    "NDWI":  {"bandwidth": -21, "central_wavelength": -21},
    "MNDWI": {"bandwidth": -22, "central_wavelength": -22},

    # Generic placeholders
    "ABSTRACT_1": {"bandwidth": -100, "central_wavelength": -100},
    "ABSTRACT_2": {"bandwidth": -101, "central_wavelength": -101},
    "ABSTRACT_3": {"bandwidth": -102, "central_wavelength": -102},
}


class SpectralEncoder(nn.Module):
    """
    Unified encoder for physical wavelength bands and abstract channels
    (SAR, DEM, slope, etc.).

    Architecture:
        - Fixed physics buffer: Gaussian integral encoding per optical band (no gradients)
        - Named learnable parameters: one per abstract channel type (VV, VH, etc.)

    Gaussian integral encoding (GAUSSIANS mode):
        For each band defined by (central_wavelength λ, bandwidth µ), compute
        the integral of each anchor Gaussian N(mean_i, std_i) over the band's
        spectral support [λ - µ/2, λ + µ/2]:

            ϕ_i(λ, µ) = 0.5 × [erf((λ+µ/2 - mean_i) / (std_i × √2))
                               - erf((λ-µ/2 - mean_i) / (std_i × √2))]

        This is the closed-form integral of the Gaussian, matching the paper.
        Output is NOT L2-normalized (values are in [0, 1] naturally).

    Out dim = number of Gaussian anchors in wavelengths_encoding YAML section.
    """

    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table

        # ── 1. Encoding mode ───────────────────────────────────────────
        self.mode = config["Atomiser"].get("wavelength_encoding", "GAUSSIANS")

        if self.mode == "GAUSSIANS":
            self.anchors = config.get("wavelengths_encoding", {})
            self.out_dim = len(self.anchors)
        elif self.mode == "FF":
            self.num_bands = config["Atomiser"]["wavelength_num_freq_bands"]
            self.max_freq  = config["Atomiser"]["wavelength_max_freq"]
            self.out_dim   = (int(self.max_freq) if self.num_bands == -1
                              else int(self.num_bands)) * 2 + 1
        elif self.mode == "NATURAL":
            self.out_dim = 1
        else:  # "NOPE"
            self.out_dim = 0

        if self.out_dim == 0:
            return

        # ── 2. Gaussian anchor buffers ─────────────────────────────────
        if self.mode == "GAUSSIANS" and self.anchors:
            means = [self.anchors[k]["mean"] for k in self.anchors]
            stds  = [self.anchors[k]["std"]  for k in self.anchors]
            # shapes [1, K] for broadcasting against scalar band values
            self.register_buffer("means", torch.tensor(means, dtype=torch.float32).view(1, -1))
            self.register_buffer("stds",  torch.tensor(stds,  dtype=torch.float32).view(1, -1))

        # ── 3. Build fixed physics codebook + identify abstract channels ─
        num_channels     = len(lookup_table.table_wave)
        physics_codebook = torch.zeros(num_channels, self.out_dim)

        self.abstract_channel_map = {}   # idx → channel_name
        abstract_indices = []

        for (bandwidth, central_wavelength), idx in lookup_table.table_wave.items():
            if bandwidth < 0 or central_wavelength < 0:
                channel_name = self._identify_abstract_channel(bandwidth, central_wavelength)
                self.abstract_channel_map[idx] = channel_name
                abstract_indices.append(idx)
            else:
                physics_codebook[idx] = self._compute_physics_vector(
                    central_wavelength, bandwidth)

        self.register_buffer("physics_codebook", physics_codebook)

        # ── 4. Learnable embeddings for abstract channels ──────────────
        self._create_named_embeddings(abstract_indices)

        print(f"[SpectralEncoder] Mode: {self.mode}, out_dim: {self.out_dim}")
        print(f"[SpectralEncoder] Physics channels: {num_channels - len(abstract_indices)}")
        print(f"[SpectralEncoder] Abstract channels: {len(abstract_indices)}")
        for idx, name in self.abstract_channel_map.items():
            print(f"[SpectralEncoder]   idx={idx} → {name} (learnable)")

    # =========================================================================
    # ABSTRACT CHANNEL HELPERS
    # =========================================================================

    def _identify_abstract_channel(self, bandwidth: int, central_wavelength: int) -> str:
        for name, info in ABSTRACT_CHANNELS.items():
            if (info["bandwidth"] == bandwidth
                    and info["central_wavelength"] == central_wavelength):
                return name
        return f"ABSTRACT_bw{bandwidth}_wl{central_wavelength}"

    def _create_named_embeddings(self, abstract_indices: List[int]):
        unique_channels = set(self.abstract_channel_map.values())

        if not unique_channels:
            self.named_embeddings   = nn.ModuleDict()
            self.name_to_safe_name  = {}
            return

        self.named_embeddings = nn.ModuleDict()
        for channel_name in unique_channels:
            emb = nn.Parameter(torch.zeros(self.out_dim))
            nn.init.trunc_normal_(emb, std=0.02, a=-2., b=2.)
            safe_name = channel_name.replace(".", "_").replace("-", "_")
            self.named_embeddings[safe_name] = nn.Module()
            self.named_embeddings[safe_name].register_parameter("embedding", emb)

        self.name_to_safe_name = {
            name: name.replace(".", "_").replace("-", "_")
            for name in unique_channels
        }

        print(f"[SpectralEncoder] Learnable embeddings created:")
        for name in unique_channels:
            print(f"[SpectralEncoder]   - {name}")

    # =========================================================================
    # PHYSICS VECTOR COMPUTATION
    # =========================================================================

    def _compute_physics_vector(self, center: float, bandwidth: float) -> torch.Tensor:
        """
        Compute the spectral encoding for an optical band via closed-form
        Gaussian integral over the band's spectral support.

        For each anchor Gaussian N(mean_i, std_i²), compute:

            ϕ_i = 0.5 × [erf((λ_max - mean_i) / (std_i × √2))
                        - erf((λ_min - mean_i) / (std_i × √2))]

        where λ_min = center - bandwidth/2,  λ_max = center + bandwidth/2.

        This is the exact integral of the Gaussian over the band, as specified
        in the paper. Values are in [0, 1] — no normalization needed.
        """
        if self.mode == "GAUSSIANS":
            sqrt2     = torch.tensor(2.0).sqrt()
            lam_min   = torch.tensor(center - bandwidth / 2.0, dtype=torch.float32)
            lam_max   = torch.tensor(center + bandwidth / 2.0, dtype=torch.float32)
            means     = self.means.squeeze(0)   # [K]
            stds      = self.stds.squeeze(0)    # [K]

            upper = (lam_max - means) / (stds * sqrt2)
            lower = (lam_min - means) / (stds * sqrt2)

            encoding = 0.5 * (erf(upper) - erf(lower))  # [K], values in [0, 1]
            return encoding

        elif self.mode == "FF":
            from .fourier_features import fourier_encode
            norm_wave = torch.tensor([center / 1000.0])
            return fourier_encode(norm_wave, self.max_freq, self.num_bands).squeeze(0)

        elif self.mode == "NATURAL":
            return torch.tensor([center])

        return torch.zeros(self.out_dim)

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
        if self.out_dim == 0:
            return torch.zeros(*channel_indices.shape, 0,
                               device=channel_indices.device)

        # Physics lookup (zeros at abstract channel positions)
        embeddings = self.physics_codebook[channel_indices]  # [..., out_dim]

        # Replace abstract positions with their learned embeddings
        if self.named_embeddings:
            for idx, channel_name in self.abstract_channel_map.items():
                mask = (channel_indices == idx)
                if mask.any():
                    safe_name   = self.name_to_safe_name[channel_name]
                    learned_vec = self.named_embeddings[safe_name].embedding
                    embeddings  = torch.where(
                        mask.unsqueeze(-1),
                        learned_vec.expand_as(embeddings),
                        embeddings,
                    )

        return embeddings

    # =========================================================================
    # EMBEDDING ACCESS HELPERS
    # =========================================================================

    def get_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        if hasattr(self, 'name_to_safe_name') and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            return self.named_embeddings[safe_name].embedding
        return None

    def set_embedding(self, channel_name: str, values: torch.Tensor):
        if hasattr(self, 'name_to_safe_name') and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            with torch.no_grad():
                self.named_embeddings[safe_name].embedding.copy_(values)


# =============================================================================
# FACTORY
# =============================================================================

def build_spectral_encoder(config: Dict[str, Any], lookup_table: Any) -> SpectralEncoder:
    return SpectralEncoder(config, lookup_table)


# =============================================================================
# LOOKUP TABLE HELPER
# =============================================================================

def register_abstract_channel(lookup_table, channel_name: str) -> int:
    if channel_name not in ABSTRACT_CHANNELS:
        raise ValueError(f"Unknown abstract channel: {channel_name}. "
                         f"Known: {list(ABSTRACT_CHANNELS.keys())}")
    info    = ABSTRACT_CHANNELS[channel_name]
    key     = (info["bandwidth"], info["central_wavelength"])
    if key in lookup_table.table_wave:
        return lookup_table.table_wave[key]
    new_idx = len(lookup_table.table_wave)
    lookup_table.table_wave[key] = new_idx
    return new_idx