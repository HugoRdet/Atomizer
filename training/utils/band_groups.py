"""
Band-group mapping for spectral attention analysis (Sen1Floods11).

Maps each token's spectral_idx (as assigned by Lookup_encoding.table_wave)
to one of three physically-motivated groups: SAR, SWIR, REST. Used to
aggregate pixel-skip cross-attention weights (Atomizer_skip_inter) into an
interpretable per-pixel spectral profile.
"""

import torch

SEN1FLOODS11_BAND_GROUPS = {
    # SAR
    "VV": "SAR", "VH": "SAR",
    # SWIR (B10 thermal cirrus is borderline; B11/B12 are the classic SWIR pair)
    "B10": "SWIR", "B11": "SWIR", "B12": "SWIR",
    # rest
    "B01": "REST", "B02": "REST", "B03": "REST", "B04": "REST",
    "B05": "REST", "B06": "REST", "B07": "REST", "B08": "REST", "B8A": "REST",
    "B09": "REST",
}
GROUP_TO_ID = {"SAR": 0, "SWIR": 1, "REST": 2}
ID_TO_GROUP = {v: k for k, v in GROUP_TO_ID.items()}


def build_band_group_lut(look_up, bands_info) -> torch.Tensor:
    """Returns a LongTensor indexed by spectral_idx -> group_id (0/1/2)."""
    max_idx = max(look_up.table_wave.values())
    lut = torch.full((max_idx + 1,), GROUP_TO_ID["REST"], dtype=torch.long)  # default
    for sat_dict in bands_info.values():
        for band_name, meta in sat_dict.items():
            key = (int(meta["bandwidth"]), int(meta["central_wavelength"]))
            if key not in look_up.table_wave:
                continue
            spectral_idx = look_up.table_wave[key]
            group = SEN1FLOODS11_BAND_GROUPS.get(band_name)
            if group is not None:
                lut[spectral_idx] = GROUP_TO_ID[group]
    return lut  # caller moves to correct device


# Sen1Floods11 spectral_idx -> group, read directly off the printed table_wave
# idx 0-9:  B01-B09  (all REST)
# idx 10:   B10  -> SWIR
# idx 11:   B11  -> SWIR
# idx 12:   B12  -> SWIR
# idx 13:   VV   -> SAR
# idx 14:   VH   -> SAR
SEN1FLOODS11_IDX_TO_GROUP = {
    0: "REST", 1: "REST", 2: "REST", 3: "REST", 4: "REST",
    5: "REST", 6: "REST", 7: "REST", 8: "REST", 9: "REST",
    10: "SWIR", 11: "SWIR", 12: "SWIR",
    13: "SAR", 14: "SAR",
}

def build_band_group_lut_by_index(max_idx=14):
    lut = torch.full((max_idx + 1,), GROUP_TO_ID["REST"], dtype=torch.long)
    for idx, group in SEN1FLOODS11_IDX_TO_GROUP.items():
        lut[idx] = GROUP_TO_ID[group]
    return lut
