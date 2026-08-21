"""
Lookup_encoding with ECHO lookup table + BioMassters abstract channels added.

This file is intended to REPLACE the existing Lookup_encoding module.
Changes vs the previous (echo-table) version:

  - New ABSTRACT_CHANNELS entries: CLP, VV_ASC, VH_ASC, VV_DESC, VH_DESC
    (BioMassters-specific pseudo-spectral channels: cloud probability and
    the 4 distinct SAR polarization/orbit-direction combinations). Codes
    -40..-44 were chosen to NOT collide with any existing entry (current
    range in use before this change: -1..-3, -10..-15, -20..-22, -100..-102).
  - New helper function create_biomassters_bands_info(), matching the
    existing create_pastis_bands_info() / create_sen1floods11_bands_info()
    pattern.

Everything else (position / spectral / query / resolution / time / echo /
abstract channel mechanics) is unchanged from the previous version.
"""

import torch
import pytorch_lightning as pl


# =============================================================================
# TOKEN FORMAT (unchanged)
# =============================================================================
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
# SPECIAL INDICES
# =============================================================================
LEARNED_RESOLUTION_IDX = 0   # Non-optical modalities (SAR, DEM, slope, etc.)
LEARNED_TIME_IDX       = 0   # Datasets without temporal dimension

# Echo encoding: index 0 reserved for non-LIDAR tokens (no echo info).
# All other indices map to specific (return_number, number_of_returns) pairs.
LEARNED_ECHO_IDX = 0

# Maximum number of returns per pulse to pre-register at init.
# Airborne LIDAR (including FRACTAL) typically has ≤5 returns; we go to 10
# for safety. Any (r, t) outside [1, MAX_ECHOS] falls back to LEARNED_ECHO_IDX
# at runtime, which yields a (0, 0) continuous encoding and a learned
# fallback embedding downstream.
MAX_ECHOS = 10


# =============================================================================
# SENTINEL CONVENTION (unchanged)
# =============================================================================
# Column 7 (time_idx) uses TWO distinct sentinel mechanisms:
#
#   -1 = "not applicable" → TimeEncoder outputs ZERO vector
#    0 = LEARNED_TIME_IDX → TimeEncoder outputs a LEARNED embedding
#   ≥1 = registered timestamp → TimeEncoder looks up DOY → circular RBF


# =============================================================================
# ABSTRACT CHANNEL DEFINITIONS
# =============================================================================
ABSTRACT_CHANNELS = {
    "VV":    {"bandwidth": -1, "central_wavelength": -1},
    "VH":    {"bandwidth": -2, "central_wavelength": -2},
    "VV_VH": {"bandwidth": -3, "central_wavelength": -3},
    "ELEVATION":     {"bandwidth": -10, "central_wavelength": -10},
    "DEM":           {"bandwidth": -10, "central_wavelength": -10},
    "SLOPE":         {"bandwidth": -11, "central_wavelength": -11},
    "ASPECT":        {"bandwidth": -12, "central_wavelength": -12},
    "CANOPY_HEIGHT": {"bandwidth": -13, "central_wavelength": -13},
    "DSM":           {"bandwidth": -14, "central_wavelength": -14},
    "DTM":           {"bandwidth": -15, "central_wavelength": -15},
    "NDVI":          {"bandwidth": -20, "central_wavelength": -20},
    "NDWI":          {"bandwidth": -21, "central_wavelength": -21},
    "MNDWI":         {"bandwidth": -22, "central_wavelength": -22},
    "ABSTRACT_1": {"bandwidth": -100, "central_wavelength": -100},
    "ABSTRACT_2": {"bandwidth": -101, "central_wavelength": -101},
    "ABSTRACT_3": {"bandwidth": -102, "central_wavelength": -102},
    # --- BioMassters-specific (NEW) ---
    # CLP: Sentinel-2 cloud probability layer, not a physical reflectance band.
    # VV_ASC/VH_ASC/VV_DESC/VH_DESC: BioMassters' Sentinel-1 tiles carry 4
    # distinct polarization/orbit-direction channels per month, unlike
    # PASTIS/Sen1Floods11's generic VV/VH (which don't distinguish orbit
    # direction) -- these get their OWN codes so the model can tell them
    # apart rather than aliasing onto the existing VV(-1)/VH(-2) entries.
    "CLP":     {"bandwidth": -40, "central_wavelength": -40},
    "VV_ASC":  {"bandwidth": -41, "central_wavelength": -41},
    "VH_ASC":  {"bandwidth": -42, "central_wavelength": -42},
    "VV_DESC": {"bandwidth": -43, "central_wavelength": -43},
    "VH_DESC": {"bandwidth": -44, "central_wavelength": -44},
}


class Lookup_encoding(pl.LightningModule):
    """
    Lookup table for position, spectral, query, resolution, TIME, and ECHO
    encoding.

    The echo table mirrors the resolution table pattern:
      - Index 0 (LEARNED_ECHO_IDX) is reserved for tokens that have no echo
        info (i.e., non-LIDAR tokens). The downstream encoder treats this
        as "use the learned default embedding" / "(a, b) = (0, 0)".
      - Indices ≥ 1 map to specific (return_number, number_of_returns)
        pairs. Each entry stores the precomputed continuous (a, b) encoding:
            a = (r - 1) / t   (echoes above)
            b = (t - r) / t   (echoes below)

    The table is pre-registered at init for all (r, t) up to MAX_ECHOS, so
    DDP ranks all build identical tables without any coordination.

    Downstream models should call `build_echo_continuous_lut()` once to
    obtain a [num_echo_indices, 2] tensor of (a, b) values. Token column
    7 (TIME_IDX) for LIDAR tokens holds the echo_idx; the model gathers
    the corresponding (a, b) via the LUT and Fourier-encodes it.
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

        # NEW: echo table
        self.table_echo = None        # {(r, t): int}
        self.num_echo_indices = 1     # idx 0 reserved for LEARNED_ECHO_IDX

        self.nb_tokens_queries = 1

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
        self.init_echo_lookup_table()       # NEW

    # =========================================================================
    # MODALITY CONFIG (unchanged)
    # =========================================================================

    def init_config(self):
        reference_modalities = [
            (10.0, 512),
            (15.0, 512),
            (0.1,  512),
            (0.5,  512),
            (30.0, 512),
            (0.2,  512),
            (1.6,  512),
        ]
        self.modalities = reference_modalities

    # =========================================================================
    # POSITION LOOKUP (unchanged)
    # =========================================================================

    def init_lookup_table(self):
        table = {}
        idx_torch_array = 0
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += size
        self.table = table
        self.next_position_offset = idx_torch_array


    def get_offset_for_resolution(self, resolution: float) -> tuple:
        res_key = int(resolution * 1000)
        for (rk, size), offset in self.table.items():
            if rk == res_key:
                return offset, size
        raise KeyError(f"No reference grid for resolution {resolution} m/px")

    def register_modality(self, resolution: float, size: int):
        res_key = int(resolution * 1000)
        key = (res_key, size)
        if key in self.table:
            return self.table[key]
        offset = self.next_position_offset
        self.table[key] = offset
        self.next_position_offset += size
        self.modalities.append((resolution, size))
        self.table_queries[key] = self.next_query_offset
        self.next_query_offset += self.nb_tokens_queries
        gsd = resolution

        return offset

    def get_or_register_modality(self, resolution: float, size: int) -> int:
        return self.register_modality(resolution, size)

    # =========================================================================
    # QUERY LOOKUP (unchanged)
    # =========================================================================

    def init_queries_lookup_table(self):
        table = {}
        idx_torch_array = 0
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += self.nb_tokens_queries
        self.table_queries = table
        self.next_query_offset = idx_torch_array

    def get_query_offset(self, resolution: float, size: int) -> int:
        res_key = int(resolution * 1000)
        key = (res_key, size)
        if key not in self.table_queries:
            self.register_modality(resolution, size)
        return self.table_queries[key]

    # =========================================================================
    # WAVELENGTH / SPECTRAL LOOKUP (unchanged)
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

    # =========================================================================
    # RESOLUTION LOOKUP (unchanged)
    # =========================================================================

    def init_resolution_lookup_table(self):
        table = dict()
        idx = 1
        seen_resolutions = set()
        for resolution, _ in self.modalities:
            res_key = int(resolution * 1000)
            if res_key not in seen_resolutions:
                seen_resolutions.add(res_key)
                table[res_key] = idx
                idx += 1
        self.table_resolution = table
        self.num_resolution_indices = idx

    def get_resolution_idx(self, resolution: float) -> int:
        res_key = int(resolution * 1000)
        return self.table_resolution.get(res_key, LEARNED_RESOLUTION_IDX)

    def register_resolution(self, resolution: float) -> int:
        res_key = int(resolution * 1000)
        if res_key in self.table_resolution:
            return self.table_resolution[res_key]
        new_idx = self.num_resolution_indices
        self.table_resolution[res_key] = new_idx
        self.num_resolution_indices += 1
        gsd = res_key / 1000

        return new_idx

    # =========================================================================
    # TIME LOOKUP (unchanged)
    # =========================================================================

    def init_time_lookup_table(self):
        self.table_time = dict()
        self.num_time_indices = 1

    def get_time_idx(self, time_key) -> int:
        return self.table_time.get(time_key, LEARNED_TIME_IDX)

    def get_or_register_time_idx(self, time_key) -> int:
        if time_key in self.table_time:
            return self.table_time[time_key]
        return self.register_time(time_key)

    def register_time(self, time_key) -> int:
        if time_key in self.table_time:
            return self.table_time[time_key]
        new_idx = self.num_time_indices
        self.table_time[time_key] = new_idx
        self.num_time_indices += 1
        return new_idx

    def register_times(self, time_keys) -> list:
        indices = []
        for key in time_keys:
            indices.append(self.register_time(key))

        return indices

    # =========================================================================
    # ECHO LOOKUP
    # =========================================================================

    def init_echo_lookup_table(self):
        """
        Pre-register all (r, t) combinations with 1 <= r <= t <= MAX_ECHOS.

        Index 0 is reserved for LEARNED_ECHO_IDX (non-LIDAR tokens with no
        echo info). Indices ≥ 1 map to specific (r, t) pairs in lexicographic
        order: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3), ...

        For MAX_ECHOS=10 this gives 55 (r, t) entries plus index 0 → 56 total.

        Pre-registering at init guarantees all DDP ranks build identical
        tables (no lazy registration → no inter-rank divergence).
        """
        self.table_echo = dict()        # (r, t) → int
        self.num_echo_indices = 1        # idx 0 reserved

        # Iterate in (t, r) order so neighboring indices share t — easier
        # to read in logs and gives some locality if downstream code uses
        # the indices directly.
        for t in range(1, MAX_ECHOS + 1):
            for r in range(1, t + 1):
                self.register_echo(r, t)




    @staticmethod
    def _continuous_for(r: int, t: int) -> tuple:
        """Return the continuous (a, b) encoding for a single (r, t) pair."""
        if t <= 0:
            return (0.0, 0.0)
        a = (r - 1) / t
        b = (t - r) / t
        return (a, b)

    def get_echo_idx(self, return_number: int, number_of_returns: int) -> int:
        """
        Look up the echo index for (return_number, number_of_returns).

        Returns LEARNED_ECHO_IDX (0) if the (r, t) pair is unknown — meaning
        we hit a sensor with more echoes than MAX_ECHOS. Downstream the model
        treats idx 0 as "no info / use learned default" via the (0, 0)
        continuous encoding entry.

        Use `get_or_register_echo` instead if you want to auto-register
        unknown combinations — but note that DDP-safe behavior requires
        pre-registration (see init_echo_lookup_table).
        """
        return self.table_echo.get((int(return_number), int(number_of_returns)),
                                   LEARNED_ECHO_IDX)

    def get_or_register_echo(self, return_number: int, number_of_returns: int) -> int:
        """
        Like get_echo_idx but auto-registers unknown (r, t) pairs.

        ⚠️ DDP WARNING: do not call this lazily during training. Different
        ranks may encounter different (r, t) combinations in different orders,
        leading to inconsistent table indices across ranks. Always register
        all expected combinations once at init via init_echo_lookup_table.
        """
        key = (int(return_number), int(number_of_returns))
        if key in self.table_echo:
            return self.table_echo[key]
        return self.register_echo(*key)

    def register_echo(self, return_number: int, number_of_returns: int) -> int:
        """Idempotent registration of a single (r, t) pair."""
        r = int(return_number)
        t = int(number_of_returns)
        # Defensive: silently clip nonsensical inputs rather than crash on
        # corrupt LAZ data. (r <= t and both >= 1.)
        if t < 1 or r < 1 or r > t:
            return LEARNED_ECHO_IDX
        key = (r, t)
        if key in self.table_echo:
            return self.table_echo[key]
        new_idx = self.num_echo_indices
        self.table_echo[key] = new_idx
        self.num_echo_indices += 1
        return new_idx

    def get_echo_continuous(self, idx: int) -> tuple:
        """
        Return the (a, b) continuous encoding for a given echo index.

        For idx=0 (LEARNED_ECHO_IDX), returns (0.0, 0.0) — the downstream
        encoder can treat this as "no echo info" or branch on the index
        explicitly to use a learned embedding instead.
        """
        if idx == LEARNED_ECHO_IDX:
            return (0.0, 0.0)
        # Reverse-lookup the (r, t) for this idx. Linear scan, only used at
        # LUT construction time, so the O(N) cost is fine (N ≤ 56).
        for (r, t), table_idx in self.table_echo.items():
            if table_idx == idx:
                return self._continuous_for(r, t)
        return (0.0, 0.0)  # unknown idx → safe default

    def build_echo_continuous_lut(self, dtype=torch.float32) -> torch.Tensor:
        """
        Build a [num_echo_indices, 2] tensor of (a, b) continuous encodings.

        Row 0 corresponds to LEARNED_ECHO_IDX and contains (0, 0).
        Row i (i ≥ 1) corresponds to the (r, t) pair registered at index i.

        The downstream encoder should:
          1. Call this once at model init.
          2. Register it as a buffer (so it moves to GPU with the model).
          3. At forward time, gather rows using token column TIME_IDX (col 7)
             for LIDAR tokens — that's where the dataset writes echo_idx.
          4. Fourier-encode the gathered (a, b) pairs and concatenate / sum
             into the rest of the token's metadata features.
        """
        lut = torch.zeros(self.num_echo_indices, 2, dtype=dtype)
        for (r, t), idx in self.table_echo.items():
            a, b = self._continuous_for(r, t)
            lut[idx, 0] = a
            lut[idx, 1] = b
        # Row 0 stays at (0, 0) — set by torch.zeros initialization.
        return lut

    # =========================================================================
    # ABSTRACT CHANNEL HELPERS (unchanged)
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
        return bandwidth < 0 or central_wavelength < 0


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


def create_flairhub_bands_info() -> dict:
    bands_info = {
        "bands_aerial_info": {
            "AERIAL_R":   {"bandwidth": 80,  "central_wavelength": 660, "idx": 0},
            "AERIAL_G":   {"bandwidth": 80,  "central_wavelength": 550, "idx": 1},
            "AERIAL_B":   {"bandwidth": 80,  "central_wavelength": 470, "idx": 2},
            "AERIAL_NIR": {"bandwidth": 100, "central_wavelength": 840, "idx": 3},
        },
        "bands_spot_info": {
            "SPOT_R":   {"bandwidth": 120, "central_wavelength": 660, "idx": 0},
            "SPOT_G":   {"bandwidth": 120, "central_wavelength": 560, "idx": 1},
            "SPOT_B":   {"bandwidth": 140, "central_wavelength": 490, "idx": 2},
            "SPOT_NIR": {"bandwidth": 120, "central_wavelength": 825, "idx": 3},
        },
        "bands_dem_info": {
            "DSM": {
                "bandwidth":         ABSTRACT_CHANNELS["DSM"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["DSM"]["central_wavelength"],
                "idx": 0,
            },
            "DTM": {
                "bandwidth":         ABSTRACT_CHANNELS["DTM"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["DTM"]["central_wavelength"],
                "idx": 1,
            },
        },
        "bands_sen2_info": {
            "B01": {"bandwidth": 20,  "central_wavelength": 443,  "idx": 0},
            "B02": {"bandwidth": 65,  "central_wavelength": 490,  "idx": 1},
            "B03": {"bandwidth": 35,  "central_wavelength": 560,  "idx": 2},
            "B04": {"bandwidth": 30,  "central_wavelength": 665,  "idx": 3},
            "B05": {"bandwidth": 15,  "central_wavelength": 705,  "idx": 4},
            "B06": {"bandwidth": 15,  "central_wavelength": 740,  "idx": 5},
            "B07": {"bandwidth": 20,  "central_wavelength": 783,  "idx": 6},
            "B08": {"bandwidth": 115, "central_wavelength": 842,  "idx": 7},
            "B8A": {"bandwidth": 20,  "central_wavelength": 865,  "idx": 8},
            "B09": {"bandwidth": 20,  "central_wavelength": 945,  "idx": 9},
            "B10": {"bandwidth": 30,  "central_wavelength": 1375, "idx": 10},
            "B11": {"bandwidth": 90,  "central_wavelength": 1610, "idx": 11},
            "B12": {"bandwidth": 180, "central_wavelength": 2190, "idx": 12},
        },
        "bands_sen1_info": {
            "VV": {
                "bandwidth":         ABSTRACT_CHANNELS["VV"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV"]["central_wavelength"],
                "idx": 0,
            },
            "VH": {
                "bandwidth":         ABSTRACT_CHANNELS["VH"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH"]["central_wavelength"],
                "idx": 1,
            },
        },
    }
    return bands_info


def create_biomassters_bands_info() -> dict:
    """
    Bands info for BioMassters: 10 Sentinel-2 physical bands (CLP excluded --
    matches PANGAEA's band set for this task) and 4 Sentinel-1 channels
    (VV/VH x ascending/descending orbit, each a DISTINCT pseudo-spectral
    entry -- unlike PASTIS/Sen1Floods11's generic VV/VH, BioMassters needs
    the model to tell orbit direction apart).

    Wavelengths for the 10 physical S2 bands match Atomizer-IO's S2_BANDS /
    create_pastis_bands_info's convention. The 4 SAR channels use the codes
    registered in ABSTRACT_CHANNELS (-41..-44). ABSTRACT_CHANNELS["CLP"]
    (-40) remains defined above but is unused here -- harmless if left in
    place, in case a future run wants it back.
    """
    bands_info = {
        "bands_sen2_info": {
            "B02": {"bandwidth": 65,  "central_wavelength": 490},
            "B03": {"bandwidth": 35,  "central_wavelength": 560},
            "B04": {"bandwidth": 30,  "central_wavelength": 665},
            "B05": {"bandwidth": 15,  "central_wavelength": 705},
            "B06": {"bandwidth": 15,  "central_wavelength": 740},
            "B07": {"bandwidth": 20,  "central_wavelength": 783},
            "B08": {"bandwidth": 115, "central_wavelength": 842},
            "B8A": {"bandwidth": 20,  "central_wavelength": 865},
            "B11": {"bandwidth": 90,  "central_wavelength": 1610},
            "B12": {"bandwidth": 180, "central_wavelength": 2190},
        },
        "bands_sen1_info": {
            "VV_ASC": {
                "bandwidth":         ABSTRACT_CHANNELS["VV_ASC"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV_ASC"]["central_wavelength"],
            },
            "VH_ASC": {
                "bandwidth":         ABSTRACT_CHANNELS["VH_ASC"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH_ASC"]["central_wavelength"],
            },
            "VV_DESC": {
                "bandwidth":         ABSTRACT_CHANNELS["VV_DESC"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VV_DESC"]["central_wavelength"],
            },
            "VH_DESC": {
                "bandwidth":         ABSTRACT_CHANNELS["VH_DESC"]["bandwidth"],
                "central_wavelength": ABSTRACT_CHANNELS["VH_DESC"]["central_wavelength"],
            },
        },
    }
    return bands_info
