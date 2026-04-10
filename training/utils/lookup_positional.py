import torch
import pytorch_lightning as pl


# =============================================================================
# TOKEN FORMAT
# =============================================================================
# Token layout: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
#                 0      1  2       3          4        5            6             7
TOKEN_VALUE_IDX      = 0
TOKEN_X_IDX          = 1
TOKEN_Y_IDX          = 2
TOKEN_SPECTRAL_IDX   = 3
TOKEN_LABEL_IDX      = 4
TOKEN_QUERY_IDX      = 5
TOKEN_RESOLUTION_IDX = 6
TOKEN_TIME_IDX       = 7
TOKEN_DIM            = 8


# =============================================================================
# SPECIAL INDICES (reserved, model uses learnable embeddings for these)
# =============================================================================
LEARNED_RESOLUTION_IDX = 0   # Non-optical modalities (SAR, DEM, slope, etc.)
LEARNED_TIME_IDX       = 0   # Datasets without temporal dimension


# =============================================================================
# SENTINEL CONVENTION
# =============================================================================
# Column 7 (time_idx) uses TWO distinct sentinel mechanisms:
#
#   -1 = "not applicable" → TimeEncoder outputs ZERO vector
#        Used by single-date datasets (SenFlood) that have no temporal info.
#
#    0 = LEARNED_TIME_IDX → TimeEncoder outputs a LEARNED embedding
#        Reserved for future use. NOT the same as -1.
#
#   ≥1 = registered timestamp → TimeEncoder looks up DOY → circular RBF
#        Used by multi-temporal datasets (PASTIS).
#
# Datasets MUST use the lookup table to get proper indices:
#   - No temporal info:  set column 7 to -1 directly
#   - Has temporal info:  call look_up.get_or_register_time_idx(doy)
#                         which returns indices ≥ 1


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================
# Abstract channels use NEGATIVE bandwidth AND central_wavelength values.
# Key format: (int(bandwidth), int(central_wavelength))

# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================
# Abstract channels use NEGATIVE bandwidth AND central_wavelength values.
# Key format: (int(bandwidth), int(central_wavelength))
#
# IMPORTANT: Each abstract channel MUST have a unique (bandwidth, central_wavelength) pair.
# Before adding new entries, check for collisions with existing ones.
#
# Current allocation:
#   -1, -2, -3        : SAR polarizations (VV, VH, VV/VH)
#   -10               : Elevation / DEM
#   -11               : Slope
#   -12               : Aspect
#   -13               : Canopy Height
#   -20, -21, -22     : Spectral indices (NDVI, NDWI, MNDWI)
#   -100, -101, -102  : Generic placeholders

ABSTRACT_CHANNELS = {
    # Sentinel-1 SAR polarizations
    "VV":    {"bandwidth": -1, "central_wavelength": -1},
    "VH":    {"bandwidth": -2, "central_wavelength": -2},
    "VV_VH": {"bandwidth": -3, "central_wavelength": -3},  # VV/VH ratio

    # Elevation / DEM
    "ELEVATION": {"bandwidth": -10, "central_wavelength": -10},
    "DEM":       {"bandwidth": -10, "central_wavelength": -10},  # Alias

    # Slope / Aspect
    "SLOPE":  {"bandwidth": -11, "central_wavelength": -11},
    "ASPECT": {"bandwidth": -12, "central_wavelength": -12},

    # Canopy Height
    "CANOPY_HEIGHT": {"bandwidth": -13, "central_wavelength": -13},

    # Indices
    "NDVI":  {"bandwidth": -20, "central_wavelength": -20},
    "NDWI":  {"bandwidth": -21, "central_wavelength": -21},
    "MNDWI": {"bandwidth": -22, "central_wavelength": -22},

    # Generic placeholders
    "ABSTRACT_1": {"bandwidth": -100, "central_wavelength": -100},
    "ABSTRACT_2": {"bandwidth": -101, "central_wavelength": -101},
    "ABSTRACT_3": {"bandwidth": -102, "central_wavelength": -102},
}


class Lookup_encoding(pl.LightningModule):
    """
    Lookup table for position, spectral, query, resolution, and time encoding.
    
    Uses a reference grid system: instead of registering every crop size,
    maintains reference grids (e.g., 512×512) from which crops extract windows.
    
    Time encoding:
        Datasets with temporal info register their timestamps (as DOY values)
        via get_or_register_time_idx(doy). This returns indices ≥ 1 that go
        into token column 7. The TimeEncoder maps these indices back to DOY
        values for circular RBF encoding.
        
        Datasets WITHOUT temporal info set column 7 to -1 directly.
        The TimeEncoder outputs zeros for -1 indices.
    """
    
    def __init__(self, modalities_config, bands_info, config_model):
        super().__init__()
        self.config = modalities_config
        self.bands_info = bands_info
        self.modalities = []
        self.table = {}
        self.table_wave = None
        self.table_queries = {}
        self.table_resolution = None
        self.table_time = None


        self.nb_tokens_queries=1
   
        
        # Track next available offset for dynamic registration
        self.next_position_offset = 0
        self.next_query_offset = 0

        # Track abstract channels
        self.abstract_channel_indices = {}

        self.init_config()
        self.init_lookup_table()
        self.init_lookup_table_wave()
        self.init_queries_lookup_table()
        self.init_resolution_lookup_table()
        self.init_time_lookup_table()

    # =========================================================================
    # MODALITY CONFIG (Reference Grid System)
    # =========================================================================

    def init_config(self):
        """
        Initialize with reference grid sizes.
        
        These are large grids (typically 512×512) from which crops
        of any size can extract coordinate windows.
        """
        # Reference grid sizes (matching TokenBuilder.REFERENCE_SIZES)
        reference_modalities = [
            (10.0, 512),  # Sentinel-2/Sentinel-1 at 10m
        ]

   
        
        self.modalities = reference_modalities
        print(f"[Lookup] Configured {len(reference_modalities)} reference grids")

    # =========================================================================
    # POSITION LOOKUP
    # =========================================================================

    def init_lookup_table(self):
        """Build initial lookup table from reference grids."""
        table = {}
        idx_torch_array = 0
        
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += size
        
        self.table = table
        self.next_position_offset = idx_torch_array
        
        print(f"[Lookup] Position table: {len(table)} reference grids")
        for (res_key, size), offset in sorted(table.items()):
            gsd = res_key / 1000
            print(f"  {gsd:>8.1f} m/px × {size:4d}px → offset {offset:6d}")

    def get_offset_for_resolution(self, resolution: float) -> tuple:
        """Find the pre-registered reference grid for a given resolution.
        
        Returns:
            (offset, reference_size)
        """
        res_key = int(resolution * 1000)
        for (rk, size), offset in self.table.items():
            if rk == res_key:
                return offset, size
        raise KeyError(f"No reference grid for resolution {resolution} m/px")

    def register_modality(self, resolution: float, size: int):
        """
        Register a new (resolution, size) modality.
        Idempotent: does nothing if already registered.
        
        Args:
            resolution: GSD in m/px
            size: Reference grid size in pixels
        
        Returns:
            offset: Position encoding offset for this modality
        """
        res_key = int(resolution * 1000)
        key = (res_key, size)

        
        
        if key in self.table:
            return self.table[key]
        
        # Add new position entry
        offset = self.next_position_offset
        self.table[key] = offset
        self.next_position_offset += size
        
        # Add to modalities list
        self.modalities.append((resolution, size))
        
        # Create query offset
        self.table_queries[key] = self.next_query_offset
        self.next_query_offset += self.nb_tokens_queries
        
        gsd = resolution
        print(f"[Lookup] Registered modality: {gsd:>8.1f} m/px × {size:4d}px → "
              f"position offset {offset:6d}, query offset {self.table_queries[key]:6d}")
        
        return offset

    def get_or_register_modality(self, resolution: float, size: int) -> int:
        """
        Get offset for (resolution, size), registering if needed.
        
        This is the main entry point for TokenBuilder.
        
        Args:
            resolution: GSD in m/px
            size: Reference grid size (e.g., 512)
        
        Returns:
            offset: Position encoding offset
        """
        return self.register_modality(resolution, size)

    # =========================================================================
    # QUERY LOOKUP
    # =========================================================================

    def init_queries_lookup_table(self):
        """Build initial query lookup table."""
        table = {}
        idx_torch_array = 0
        
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += self.nb_tokens_queries
        
        self.table_queries = table
        self.next_query_offset = idx_torch_array
        
        print(f"[Lookup] Query table: {len(table)} entries ({self.nb_tokens_queries} slots each)")

    def get_query_offset(self, resolution: float, size: int) -> int:
        """
        Get query offset for a modality (auto-registers if needed).
        
        Args:
            resolution: GSD in m/px
            size: Reference grid size
        
        Returns:
            offset: Query encoding offset
        """
        res_key = int(resolution * 1000)
        key = (res_key, size)
        
        if key not in self.table_queries:
            # Auto-register
            self.register_modality(resolution, size)
        
        return self.table_queries[key]

    # =========================================================================
    # WAVELENGTH / SPECTRAL LOOKUP
    # =========================================================================

    def init_lookup_table_wave(self):
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

                    # Track abstract channels
                    if bandwidth < 0 or central_wavelength < 0:
                        for name, info in ABSTRACT_CHANNELS.items():
                            if (info["bandwidth"] == int(bandwidth)
                                    and info["central_wavelength"] == int(central_wavelength)):
                                self.abstract_channel_indices[name] = idx_torch_array
                                break

                    idx_torch_array += 1

        self.table_wave = table

        n_physical = sum(1 for k in table if k[1] >= 0)
        n_abstract = sum(1 for k in table if k[1] < 0)
        print(f"[Lookup] Wavelength table: {len(table)} entries "
              f"({n_physical} physical, {n_abstract} abstract)")
        if self.abstract_channel_indices:
            print(f"[Lookup] Abstract channels: {self.abstract_channel_indices}")

    # =========================================================================
    # RESOLUTION LOOKUP
    # =========================================================================

    def init_resolution_lookup_table(self):
        """
        Build a lookup from GSD (m/px) → resolution index.

        Index 0 is reserved for LEARNED_RESOLUTION_IDX (non-optical modalities
        like SAR, DEM, etc.). Physical optical resolutions get indices ≥ 1.

        Key: int(resolution * 1000)  (e.g., 10m → 10000, 0.2m → 200)
        """
        table = dict()
        # Index 0 reserved for learned (non-optical)
        idx = 1

        # Collect unique resolutions from modalities
        seen_resolutions = set()
        for resolution, _ in self.modalities:
            res_key = int(resolution * 1000)
            if res_key not in seen_resolutions:
                seen_resolutions.add(res_key)
                table[res_key] = idx
                idx += 1

        self.table_resolution = table
        self.num_resolution_indices = idx  # total count (including index 0)

        print(f"[Lookup] Resolution table: {len(table)} optical entries + "
              f"1 learned (idx=0). Total = {self.num_resolution_indices}")
        for res_key, res_idx in sorted(table.items()):
            gsd = res_key / 1000
            print(f"  {gsd:>8.1f} m/px → idx {res_idx}")

    def get_resolution_idx(self, resolution: float) -> int:
        """
        Get the resolution index for a given GSD.

        Args:
            resolution: Ground sample distance in m/px.

        Returns:
            Integer index. Returns LEARNED_RESOLUTION_IDX (0) if the
            resolution is not registered (e.g. non-optical modality).
        """
        res_key = int(resolution * 1000)
        return self.table_resolution.get(res_key, LEARNED_RESOLUTION_IDX)

    def register_resolution(self, resolution: float) -> int:
        """
        Register a new resolution and return its index.
        Idempotent: returns existing index if already registered.
        """
        res_key = int(resolution * 1000)
        if res_key in self.table_resolution:
            return self.table_resolution[res_key]

        new_idx = self.num_resolution_indices
        self.table_resolution[res_key] = new_idx
        self.num_resolution_indices += 1

        gsd = res_key / 1000
        print(f"[Lookup] Registered resolution {gsd} m/px → idx {new_idx}")
        return new_idx

    # =========================================================================
    # TIME LOOKUP
    # =========================================================================

    def init_time_lookup_table(self):
        """
        Build a lookup from time identifiers → time index.

        Index 0 is reserved for LEARNED_TIME_IDX (datasets without temporal
        dimension, i.e. single-date observations).

        Time keys can be anything hashable — typically:
          - int day-of-year (1–365)
          - str ISO date "2021-06-15"
          - int sequential timestep (1, 2, 3, …)

        Starts empty; datasets register their timesteps at init.
        
        IMPORTANT: Datasets WITHOUT temporal info should set column 7 to -1
        (not 0). The TimeEncoder zeros out -1 indices. Index 0 is a valid
        learned embedding, not a sentinel.
        """
        self.table_time = dict()
        self.num_time_indices = 1  # index 0 reserved for learned

        print(f"[Lookup] Time table: initialized (idx=0 reserved for learned/no-time)")

    def get_time_idx(self, time_key) -> int:
        """
        Get the time index for a given time key.

        Args:
            time_key: Hashable time identifier (int, str, tuple, etc.)

        Returns:
            Integer index. Returns LEARNED_TIME_IDX (0) if not registered.
        """
        return self.table_time.get(time_key, LEARNED_TIME_IDX)

    def get_or_register_time_idx(self, time_key) -> int:
        """
        Get time index for a key, auto-registering if not yet known.
        
        This is the main entry point for datasets with temporal info.
        Returns indices ≥ 1 (index 0 is reserved for learned/no-time).
        
        Args:
            time_key: Hashable time identifier — typically day-of-year (int)
                      but can be any hashable (str date, tuple, etc.)
        
        Returns:
            Integer index (≥ 1).
        """
        if time_key in self.table_time:
            return self.table_time[time_key]
        return self.register_time(time_key)

    def register_time(self, time_key) -> int:
        """
        Register a new time key and return its index.
        Idempotent: returns existing index if already registered.

        Args:
            time_key: Hashable identifier — e.g. "2021-06-15", 172 (DOY),
                      or a sequential timestep number.

        Returns:
            Integer index (≥ 1).
        """
        if time_key in self.table_time:
            return self.table_time[time_key]

        new_idx = self.num_time_indices
        self.table_time[time_key] = new_idx
        self.num_time_indices += 1
        return new_idx

    def register_times(self, time_keys) -> list:
        """
        Batch-register multiple time keys. Returns list of indices.

        Args:
            time_keys: Iterable of hashable time identifiers.

        Returns:
            List of integer indices.
        """
        indices = []
        for key in time_keys:
            indices.append(self.register_time(key))

        print(f"[Lookup] Registered {len(indices)} time steps "
              f"(total = {self.num_time_indices})")
        return indices

    # =========================================================================
    # ABSTRACT CHANNEL HELPERS
    # =========================================================================

    def register_abstract_channel(self, channel_name: str) -> int:
        if channel_name not in ABSTRACT_CHANNELS:
            raise ValueError(
                f"Unknown abstract channel: {channel_name}. "
                f"Known: {list(ABSTRACT_CHANNELS.keys())}"
            )

        info = ABSTRACT_CHANNELS[channel_name]
        key = (int(info["bandwidth"]), int(info["central_wavelength"]))

        if key in self.table_wave:
            return self.table_wave[key]

        new_idx = len(self.table_wave)
        self.table_wave[key] = new_idx
        self.abstract_channel_indices[channel_name] = new_idx
        print(f"[Lookup] Registered abstract channel '{channel_name}' at index {new_idx}")
        return new_idx

    def get_abstract_channel_idx(self, channel_name: str) -> int:
        if channel_name not in self.abstract_channel_indices:
            raise KeyError(
                f"Abstract channel '{channel_name}' not registered. "
                f"Registered: {list(self.abstract_channel_indices.keys())}"
            )
        return self.abstract_channel_indices[channel_name]

    def get_wave_idx(self, bandwidth: int, central_wavelength: int) -> int:
        key = (bandwidth, central_wavelength)
        if key not in self.table_wave:
            raise KeyError(f"Wavelength key {key} not found in table_wave")
        return self.table_wave[key]

    def is_abstract_channel(self, idx: int) -> bool:
        for key, table_idx in self.table_wave.items():
            if table_idx == idx:
                return key[0] < 0 or key[1] < 0
        return False

    def is_non_optical_band(self, bandwidth: int, central_wavelength: int) -> bool:
        """Check whether a band should use LEARNED_RESOLUTION_IDX."""
        return bandwidth < 0 or central_wavelength < 0

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def print_summary(self):
        print("\n" + "=" * 70)
        print("LOOKUP TABLE SUMMARY")
        print("=" * 70)
        print(f"  Position modalities:  {len(self.modalities)} (reference grids)")
        print(f"  Spectral entries:     {len(self.table_wave)}")
        print(f"  Resolution entries:   {self.num_resolution_indices} "
              f"(1 learned + {self.num_resolution_indices - 1} optical)")
        print(f"  Time entries:         {self.num_time_indices} "
              f"(1 learned + {self.num_time_indices - 1} registered)")
        if self.abstract_channel_indices:
            print(f"  Abstract channels:    {list(self.abstract_channel_indices.keys())}")
        print(f"  Next position offset: {self.next_position_offset}")
        print(f"  Next query offset:    {self.next_query_offset}")
        print("=" * 70 + "\n")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_sen1floods11_bands_info() -> dict:
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


def create_pastis_bands_info() -> dict:
    """Band info for PASTIS-HD: S2 (10 bands) + S1A (3 channels: VV, VH, VV/VH)."""
    bands_info = {
        "bands_sen2_info": {
            "B02": {"bandwidth": 65, "central_wavelength": 490},
            "B03": {"bandwidth": 35, "central_wavelength": 560},
            "B04": {"bandwidth": 30, "central_wavelength": 665},
            "B05": {"bandwidth": 15, "central_wavelength": 705},
            "B06": {"bandwidth": 15, "central_wavelength": 740},
            "B07": {"bandwidth": 20, "central_wavelength": 783},
            "B08": {"bandwidth": 115, "central_wavelength": 842},
            "B8A": {"bandwidth": 20, "central_wavelength": 865},
            "B11": {"bandwidth": 90, "central_wavelength": 1610},
            "B12": {"bandwidth": 180, "central_wavelength": 2190},
        },
        "bands_sen1_info": {
            "VV": {
                "bandwidth": ABSTRACT_CHANNELS["VV"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV"]["central_wavelength"],
            },
            "VH": {
                "bandwidth": ABSTRACT_CHANNELS["VH"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH"]["central_wavelength"],
            },
            "VV_VH": {
                "bandwidth": ABSTRACT_CHANNELS["VV_VH"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV_VH"]["central_wavelength"],
            },
        },
    }
    return bands_info


def create_multimodal_bands_info(include_elevation: bool = False) -> dict:
    bands_info = create_sen1floods11_bands_info()
    if include_elevation:
        bands_info["bands_dem_info"] = {
            "ELEVATION": {
                "bandwidth": ABSTRACT_CHANNELS["ELEVATION"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["ELEVATION"]["central_wavelength"],
            },
        }
    return bands_info