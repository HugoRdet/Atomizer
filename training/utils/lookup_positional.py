import torch
import pytorch_lightning as pl


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================
# Abstract channels use NEGATIVE bandwidth AND central_wavelength values.
# These match what's defined in the YAML config file.
# Key format: (int(bandwidth), int(central_wavelength))

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


class Lookup_encoding(pl.LightningModule):
    def __init__(self, modalities_config, bands_info, config_model):
        super().__init__()
        self.config = modalities_config
        self.bands_info = bands_info
        self.modalities = None
        self.table = None
        self.pixel_coords_table = None
        self.table_wave = None
        self.table_queries = None
        self.nb_tokens_queries = config_model["Atomiser"]["spatial_latents"]
        
        # Track abstract channels for debugging/inspection
        self.abstract_channel_indices = {}  # name -> idx

        self.init_config()
        self.init_lookup_table()
        self.init_pixel_coords_table() 
        self.init_lookup_table_wave()
        self.init_queries_lookup_table()

    def init_config(self):
        modalities = []
        # Manual entry for 0.2m GSD and 512x512 size
        modalities.append((0.2, 512))
        modalities.append((10, 512))
        modalities.append((0.2, 28))
        self.modalities = modalities
    
    def init_lookup_table(self):
        table = dict()
        idx_torch_array = 0
        for couple in self.modalities:
            resolution, size = couple
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += size
        self.table = table

    def init_pixel_coords_table(self):
        """
        Initializes a lookup table that maps each modality to a 
        static global (x, y) coordinate grid.
        """
        coords_table = dict()
        
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            
            grid_y, grid_x = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing='ij'
            )
            
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            coords_table[(res_key, size)] = coords
            
        self.pixel_coords_table = coords_table

    def get_pixel_coords(self, resolution, size):
        """
        Returns the (x, y) global coordinates for a given resolution and image size.
        """
        res_key = int(resolution * 1000)
        key = (res_key, size)
        
        if key not in self.pixel_coords_table:
            raise ValueError(f"Coords for resolution {resolution} and size {size} not found.")
            
        return self.pixel_coords_table[key]

    def get_grid_pos(self, resolution, size):
        """Original 1D index getter (kept for backward compatibility)"""
        res_key = int(resolution * 1000)
        key = (res_key, size)
        if key not in self.table:
            raise ValueError(f"Resolution {res_key} and size {size} not found.")
        
        idx = self.table[key]
        return torch.arange(idx, idx + size, dtype=torch.float32)

    def init_queries_lookup_table(self):
        table = dict()
        idx_torch_array = 0
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += self.nb_tokens_queries
        self.table_queries = table

    def init_lookup_table_wave(self):
        """
        Initialize wavelength lookup table from bands_info.
        
        Handles both:
        - Physical bands (bandwidth > 0 AND central_wavelength > 0)
        - Abstract channels (bandwidth < 0 OR central_wavelength < 0)
        """
        table = dict()
        idx_torch_array = 0
        
        for sat in self.bands_info:
            sat_content = self.bands_info[sat]
            for band in sat_content:
                band_content = sat_content[band]
                
                if "bandwidth" not in band_content or "central_wavelength" not in band_content:
                    continue
                
                bandwidth = band_content["bandwidth"]
                central_wavelength = band_content["central_wavelength"]
                key = (int(bandwidth), int(central_wavelength))
                
                if key not in table:
                    table[key] = idx_torch_array
                    
                    # Track abstract channels (negative bandwidth OR central_wavelength)
                    if bandwidth < 0 or central_wavelength < 0:
                        # Find the name by matching both bandwidth and central_wavelength
                        for name, info in ABSTRACT_CHANNELS.items():
                            if (info["bandwidth"] == int(bandwidth) and 
                                info["central_wavelength"] == int(central_wavelength)):
                                self.abstract_channel_indices[name] = idx_torch_array
                                break
                    
                    idx_torch_array += 1
        
        self.table_wave = table
        
        # Print summary
        n_physical = sum(1 for k in table if k[1] >= 0)
        n_abstract = sum(1 for k in table if k[1] < 0)
        print(f"[Lookup] table_wave: {len(table)} entries ({n_physical} physical, {n_abstract} abstract)")
        if self.abstract_channel_indices:
            print(f"[Lookup] Abstract channels: {self.abstract_channel_indices}")

    # =========================================================================
    # ABSTRACT CHANNEL REGISTRATION
    # =========================================================================
    
    def register_abstract_channel(self, channel_name: str) -> int:
        """
        Register an abstract channel and return its index.
        
        Args:
            channel_name: Name of the abstract channel (e.g., "VV", "VH", "ELEVATION")
            
        Returns:
            Index in the wavelength lookup table
        """
        if channel_name not in ABSTRACT_CHANNELS:
            raise ValueError(
                f"Unknown abstract channel: {channel_name}. "
                f"Known channels: {list(ABSTRACT_CHANNELS.keys())}"
            )
        
        info = ABSTRACT_CHANNELS[channel_name]
        key = (int(info["bandwidth"]), int(info["central_wavelength"]))
        
        # Check if already registered
        if key in self.table_wave:
            return self.table_wave[key]
        
        # Register new
        new_idx = len(self.table_wave)
        self.table_wave[key] = new_idx
        self.abstract_channel_indices[channel_name] = new_idx
        
        print(f"[Lookup] Registered abstract channel '{channel_name}' at index {new_idx}")
        return new_idx
    
    def get_abstract_channel_idx(self, channel_name: str) -> int:
        """
        Get the index for a registered abstract channel.
        
        Args:
            channel_name: Name of the abstract channel
            
        Returns:
            Index in the wavelength lookup table
            
        Raises:
            KeyError if channel not registered
        """
        if channel_name not in self.abstract_channel_indices:
            raise KeyError(
                f"Abstract channel '{channel_name}' not registered. "
                f"Registered channels: {list(self.abstract_channel_indices.keys())}"
            )
        return self.abstract_channel_indices[channel_name]
    
    def get_wave_idx(self, bandwidth: int, central_wavelength: int) -> int:
        """
        Get index for a wavelength specification.
        
        Args:
            bandwidth: Band bandwidth in nm (int)
            central_wavelength: Central wavelength in nm (int), negative for abstract
            
        Returns:
            Index in the wavelength lookup table
        """
        key = (bandwidth, central_wavelength)
        if key not in self.table_wave:
            raise KeyError(f"Wavelength key {key} not found in table_wave")
        return self.table_wave[key]
    
    def is_abstract_channel(self, idx: int) -> bool:
        """Check if an index corresponds to an abstract channel."""
        for key, table_idx in self.table_wave.items():
            if table_idx == idx:
                # Abstract if either bandwidth or central_wavelength is negative
                return key[0] < 0 or key[1] < 0
        return False


# =============================================================================
# HELPER FUNCTION FOR BANDS_INFO CONFIGURATION
# =============================================================================

def create_sen1floods11_bands_info() -> dict:
    """
    Create bands_info dictionary for Sen1Floods11 dataset.
    
    Includes:
    - Sentinel-2: 13 optical bands
    - Sentinel-1: VV, VH (SAR) as abstract channels
    """
    bands_info = {
        "bands_sen2_info": {
            "B01": {"bandwidth": 20, "central_wavelength": 443},
            "B02": {"bandwidth": 65, "central_wavelength": 497},
            "B03": {"bandwidth": 35, "central_wavelength": 560},
            "B04": {"bandwidth": 30, "central_wavelength": 665},
            "B05": {"bandwidth": 15, "central_wavelength": 704},
            "B06": {"bandwidth": 15, "central_wavelength": 740},
            "B07": {"bandwidth": 20, "central_wavelength": 783},
            "B08": {"bandwidth": 115, "central_wavelength": 835},
            "B8A": {"bandwidth": 20, "central_wavelength": 865},
            "B09": {"bandwidth": 20, "central_wavelength": 945},
            "B10": {"bandwidth": 30, "central_wavelength": 1374},
            "B11": {"bandwidth": 90, "central_wavelength": 1614},
            "B12": {"bandwidth": 180, "central_wavelength": 2202},
        },
        "bands_sen1_info": {
            # SAR bands - use abstract channel definitions
            "VV": {
                "bandwidth": ABSTRACT_CHANNELS["VV"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV"]["central_wavelength"],
            },
            "VH": {
                "bandwidth": ABSTRACT_CHANNELS["VH"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH"]["central_wavelength"],
            },
        },
    }
    return bands_info


def create_multimodal_bands_info(include_elevation: bool = False) -> dict:
    """
    Create bands_info with S1 + S2 + optional elevation.
    """
    bands_info = create_sen1floods11_bands_info()
    
    if include_elevation:
        bands_info["bands_dem_info"] = {
            "ELEVATION": {
                "bandwidth": ABSTRACT_CHANNELS["ELEVATION"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["ELEVATION"]["central_wavelength"],
            },
        }
    
    return bands_info