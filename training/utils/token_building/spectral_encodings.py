import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================

# Abstract channels use NEGATIVE bandwidth AND central_wavelength values.
# These match what's defined in the YAML config file.
# Key format in table_wave: (int(bandwidth), int(central_wavelength))

ABSTRACT_CHANNELS = {
    # Sentinel-1 SAR polarizations (as defined in YAML)
    "VV": {"bandwidth": -1, "central_wavelength": -1},   # key = (-1, -1)
    "VH": {"bandwidth": -2, "central_wavelength": -2},   # key = (-2, -2)
    
    # Elevation / DEM
    "ELEVATION": {"bandwidth": -10, "central_wavelength": -10},
    "DEM": {"bandwidth": -10, "central_wavelength": -10},  # Alias
    
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
    Unified encoder for Wavelengths and Learned Tokens (e.g., Elevation, SAR).
    
    Architecture:
    - Fixed physics buffer for wavelength embeddings (no gradients)
    - Named learnable parameters for abstract channels (VV, VH, Elevation, etc.)
    
    Abstract channels are identified by bandwidth < 0 OR central_wavelength < 0.
    Each abstract channel type gets its own learnable embedding.
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table
        
        # 1. Determine Encoding Strategy
        self.mode = config["Atomiser"].get("wavelength_encoding", "GAUSSIANS")
        
        if self.mode == "GAUSSIANS":
            self.anchors = config.get("wavelengths_encoding", {})
            self.out_dim = len(self.anchors)
        elif self.mode == "FF":
            self.num_bands = config["Atomiser"]["wavelength_num_freq_bands"]
            self.max_freq = config["Atomiser"]["wavelength_max_freq"]
            if self.num_bands == -1:
                self.out_dim = int(self.max_freq) * 2 + 1
            else:
                self.out_dim = int(self.num_bands) * 2 + 1
        elif self.mode == "NATURAL":
            self.out_dim = 1
        else:  # "NOPE"
            self.out_dim = 0
        
        if self.out_dim == 0:
            return
            
        # 2. Setup Gaussian Anchors (if needed)
        if self.mode == "GAUSSIANS" and self.anchors:
            means = [self.anchors[k]["mean"] for k in self.anchors]
            stds = [self.anchors[k]["std"] for k in self.anchors]
            self.register_buffer("means", torch.tensor(means).float().view(1, 1, -1))
            self.register_buffer("stds", torch.tensor(stds).float().view(1, 1, -1))
        
        # 3. Build Fixed Physics Codebook + Identify Abstract Channels
        num_channels = len(lookup_table.table_wave)
        physics_codebook = torch.zeros(num_channels, self.out_dim)
        
        # Track abstract channel indices and their types
        self.abstract_channel_map = {}  # idx -> channel_name
        abstract_indices = []
        
        for (bandwidth, central_wavelength), idx in lookup_table.table_wave.items():
            # Abstract channels have negative bandwidth OR negative central_wavelength
            if bandwidth < 0 or central_wavelength < 0:
                # This is an abstract channel - identify its type
                channel_name = self._identify_abstract_channel(bandwidth, central_wavelength)
                self.abstract_channel_map[idx] = channel_name
                abstract_indices.append(idx)
            else:
                # Compute physics embedding (no gradients)
                physics_codebook[idx] = self._compute_physics_vector(central_wavelength, bandwidth)
        
        # Register physics codebook as buffer (frozen)
        self.register_buffer("physics_codebook", physics_codebook)
        
        # 4. Create Named Learnable Embeddings for Abstract Channels
        self._create_named_embeddings(abstract_indices)
        
        # Print info
        print(f"[SpectralEncoder] Mode: {self.mode}, out_dim: {self.out_dim}")
        print(f"[SpectralEncoder] Physics channels: {num_channels - len(abstract_indices)}")
        print(f"[SpectralEncoder] Abstract channels: {len(abstract_indices)}")
        for idx, name in self.abstract_channel_map.items():
            print(f"[SpectralEncoder]   idx={idx} -> {name}")

    def _identify_abstract_channel(self, bandwidth: int, central_wavelength: int) -> str:
        """
        Identify the type of abstract channel based on its lookup key.
        
        Args:
            bandwidth: Bandwidth value from lookup table (negative for abstract)
            central_wavelength: Central wavelength value (negative for abstract)
            
        Returns:
            Channel name string
        """
        # Check against known abstract channels
        for name, info in ABSTRACT_CHANNELS.items():
            if (info["bandwidth"] == bandwidth and 
                info["central_wavelength"] == central_wavelength):
                return name
        
        # Unknown abstract channel - create generic name
        return f"ABSTRACT_bw{bandwidth}_wl{central_wavelength}"

    def _create_named_embeddings(self, abstract_indices: List[int]):
        """
        Create named learnable embeddings for each unique abstract channel type.
        """
        # Get unique channel names
        unique_channels = set(self.abstract_channel_map.values())
        
        if not unique_channels:
            self.named_embeddings = nn.ModuleDict()
            self.name_to_safe_name = {}
            return
        
        # Create a learnable embedding for each unique channel type
        self.named_embeddings = nn.ModuleDict()
        
        for channel_name in unique_channels:
            # Create embedding as a small module containing a Parameter
            embedding = nn.Parameter(torch.zeros(self.out_dim))
            nn.init.trunc_normal_(embedding, std=0.02, a=-2., b=2.)
            
            # Use safe name (replace invalid characters)
            safe_name = channel_name.replace(".", "_").replace("-", "_")
            self.named_embeddings[safe_name] = nn.Module()
            self.named_embeddings[safe_name].register_parameter("embedding", embedding)
        
        # Build reverse mapping: channel_name -> safe_name
        self.name_to_safe_name = {
            name: name.replace(".", "_").replace("-", "_") 
            for name in unique_channels
        }
        
        print(f"[SpectralEncoder] Created learnable embeddings:")
        for name in unique_channels:
            print(f"[SpectralEncoder]   - {name}")

    def _compute_physics_vector(self, center: float, bandwidth: float) -> torch.Tensor:
        """
        Compute embedding for a physical wavelength band.
        Uses 150 sample points to match original precision.
        """
        if self.mode == "GAUSSIANS":
            num_points = 150
            
            center = torch.tensor(center, dtype=torch.float32)
            bandwidth = torch.tensor(bandwidth, dtype=torch.float32)
            
            lambda_min = center - bandwidth / 2
            lambda_max = center + bandwidth / 2
            
            # Sample wavelengths across the band
            t = torch.linspace(0, 1, num_points)
            sampled_lambdas = lambda_min + (lambda_max - lambda_min) * t
            sampled_lambdas = sampled_lambdas.view(-1, 1, 1)
            
            # Compute Gaussian responses
            gaussians = torch.exp(
                -0.5 * ((sampled_lambdas - self.means) / self.stds) ** 2
            )
            
            # Max over sample points
            encoding = gaussians.max(dim=0).values.squeeze()
            return encoding
            
        elif self.mode == "FF":
            from .fourier_features import fourier_encode
            norm_wave = torch.tensor([center / 1000.0])
            return fourier_encode(norm_wave, self.max_freq, self.num_bands).squeeze(0)
            
        elif self.mode == "NATURAL":
            return torch.tensor([center])
        
        return torch.zeros(self.out_dim)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_indices: [...] integer indices into table_wave
            
        Returns:
            embeddings: [..., out_dim]
        """
        if self.out_dim == 0:
            return torch.zeros(
                (*channel_indices.shape, 0), 
                device=channel_indices.device
            )
        
        # DEBUG: Check for out-of-bounds indices
        max_idx = channel_indices.max().item()
        min_idx = channel_indices.min().item()
        codebook_size = self.physics_codebook.shape[0]
        
        if max_idx >= codebook_size or min_idx < 0:
            print(f"[SpectralEncoder] ERROR: Index out of bounds!")
            print(f"  channel_indices range: [{min_idx}, {max_idx}]")
            print(f"  codebook size: {codebook_size}")
            print(f"  channel_indices shape: {channel_indices.shape}")
            print(f"  channel_indices dtype: {channel_indices.dtype}")
            # Force sync to get proper error location
            torch.cuda.synchronize()
            raise IndexError(f"channel_indices [{min_idx}, {max_idx}] out of bounds for codebook size {codebook_size}")
        
        
        # Start with physics lookup (includes zeros for abstract channels)
        embeddings = self.physics_codebook[channel_indices]  # [..., out_dim]
        
        # Replace abstract channel positions with learned embeddings
        if self.named_embeddings:
            for idx, channel_name in self.abstract_channel_map.items():
                mask = (channel_indices == idx)
                if mask.any():
                    # Get the learned embedding for this channel type
                    safe_name = self.name_to_safe_name[channel_name]
                    learned_vec = self.named_embeddings[safe_name].embedding
                    
                    # Expand and apply
                    embeddings = torch.where(
                        mask.unsqueeze(-1),
                        learned_vec.expand_as(embeddings),
                        embeddings
                    )
        
        return embeddings
    
    def get_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        """
        Get the learned embedding for a specific abstract channel.
        
        Args:
            channel_name: Name of the abstract channel (e.g., "VV", "ELEVATION")
            
        Returns:
            The learned embedding tensor, or None if not found
        """
        if hasattr(self, 'name_to_safe_name') and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            return self.named_embeddings[safe_name].embedding
        return None
    
    def set_embedding(self, channel_name: str, values: torch.Tensor):
        """
        Set the learned embedding for a specific abstract channel.
        
        Args:
            channel_name: Name of the abstract channel
            values: New embedding values
        """
        if hasattr(self, 'name_to_safe_name') and channel_name in self.name_to_safe_name:
            safe_name = self.name_to_safe_name[channel_name]
            with torch.no_grad():
                self.named_embeddings[safe_name].embedding.copy_(values)


def build_spectral_encoder(config: Dict[str, Any], lookup_table: Any) -> SpectralEncoder:
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
        channel_name: Name of the abstract channel (must be in ABSTRACT_CHANNELS)
        
    Returns:
        The assigned index for this channel
    """
    if channel_name not in ABSTRACT_CHANNELS:
        raise ValueError(f"Unknown abstract channel: {channel_name}. "
                        f"Known channels: {list(ABSTRACT_CHANNELS.keys())}")
    
    info = ABSTRACT_CHANNELS[channel_name]
    key = (info["bandwidth"], info["central_wavelength"])
    
    # Check if already registered
    if key in lookup_table.table_wave:
        return lookup_table.table_wave[key]
    
    # Register new
    new_idx = len(lookup_table.table_wave)
    lookup_table.table_wave[key] = new_idx
    
    return new_idx