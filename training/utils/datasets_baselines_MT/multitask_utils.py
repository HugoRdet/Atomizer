"""
Shared utilities for multi-task baseline datasets.

Defines the canonical 15-channel layout (13 S2 optical + 2 SAR) and the
helpers used by all task-specific datasets to canonicalize their input:

    1. `build_interpolation_matrix(source_wavelengths_nm) → [13, C_src]`
       Precomputed once per dataset. Linearly interpolates the source
       bands onto the canonical S2 wavelength grid; canonical wavelengths
       outside the source range stay zero.

    2. `apply_interpolation_matrix(image, M) → [13, H, W]`
       Per-sample. A flat matmul.

    3. `build_canonical_image(optical, sar=None) → [15, H, W]`
       Concatenates the 13 optical channels with 2 SAR channels (zero-filled
       if absent). Convention: channel 13 = VV, channel 14 = VH.

    4. `pad_to_canonical(image, target=None, size=512)`
       Pads spatial extent to size×size with zeros (top-left aligned).
       Segmentation targets are padded with IGNORE_INDEX. Returns also a
       `valid_mask` (1 = real pixel, 0 = pad) and `original_size`.

The image-key convention across all datasets is `"input"` — the modality
label served no purpose once spectral configurations are unified.
"""

import torch


# ────────────────────────────────────────────────────────────────────
# Canonical band layout
# ────────────────────────────────────────────────────────────────────

# 13 Sentinel-2 bands in fixed canonical order.
S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Central wavelength (nm) for each S2 band.
S2_WAVELENGTHS_NM = {
    "B01": 443,  "B02": 490,  "B03": 560,  "B04": 665,
    "B05": 705,  "B06": 740,  "B07": 783,  "B08": 842,
    "B8A": 865,  "B09": 945,  "B10": 1375, "B11": 1610,
    "B12": 2190,
}

# Map S2 band name → wavelength (nm), in canonical order.
S2_CANONICAL_WAVELENGTHS_NM = [S2_WAVELENGTHS_NM[name] for name in S2_BAND_NAMES]

NUM_OPTICAL = 13                              # B01 .. B12
NUM_SAR = 2                                   # VV, VH
NUM_CANONICAL_CHANNELS = NUM_OPTICAL + NUM_SAR  # = 15

SAR_VV_INDEX = 13
SAR_VH_INDEX = 14

CANONICAL_SIZE = 512
IGNORE_INDEX = 255


# ────────────────────────────────────────────────────────────────────
# Spectral canonicalization
# ────────────────────────────────────────────────────────────────────

def build_interpolation_matrix(source_wavelengths_nm) -> torch.Tensor:
    """
    Build the [13, C_src] interpolation matrix that maps a source
    band tensor onto the canonical S2 wavelength grid via linear
    interpolation, with zero-fill outside the source range.

    The source wavelengths must be sorted ascending.

    For each canonical wavelength λ_t:
        - If λ_t lies outside [src[0], src[-1]] → row of zeros (OOR).
        - Else find i such that src[i] ≤ λ_t ≤ src[i+1] and set
          w = (λ_t - src[i]) / (src[i+1] - src[i]),
          M[t, i] = 1 - w,
          M[t, i+1] = w.
        - Exact wavelength matches collapse to identity.

    Args:
        source_wavelengths_nm: list/tuple of length C_src, ascending.

    Returns:
        torch.FloatTensor of shape [13, C_src].
    """
    src_wl = [float(w) for w in source_wavelengths_nm]
    assert src_wl == sorted(src_wl), (
        f"source_wavelengths_nm must be ascending: got {src_wl}"
    )
    assert len(set(src_wl)) == len(src_wl), (
        f"source_wavelengths_nm must have distinct entries: got {src_wl}"
    )

    C_src = len(src_wl)
    M = torch.zeros(NUM_OPTICAL, C_src, dtype=torch.float32)

    for t_idx, lambda_t in enumerate(S2_CANONICAL_WAVELENGTHS_NM):
        # Out of source range → row stays zero (zero-fill policy)
        if lambda_t < src_wl[0] or lambda_t > src_wl[-1]:
            continue

        # Find bracketing source indices [i, i+1] s.t. src[i] ≤ λ_t ≤ src[i+1]
        for i in range(C_src - 1):
            lo, hi = src_wl[i], src_wl[i + 1]
            if lo <= lambda_t <= hi:
                if lambda_t == lo:
                    M[t_idx, i] = 1.0
                elif lambda_t == hi:
                    M[t_idx, i + 1] = 1.0
                else:
                    w = (lambda_t - lo) / (hi - lo)
                    M[t_idx, i] = 1.0 - w
                    M[t_idx, i + 1] = w
                break

    return M


def apply_interpolation_matrix(image: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Apply a precomputed [13, C_src] matrix to an image tensor.

    Args:
        image: [C_src, H, W] (single-frame) or [T, C_src, H, W] (multi-temporal),
               already normalized.
        M:     [13, C_src] interpolation matrix from build_interpolation_matrix.

    Returns:
        [13, H, W] or [T, 13, H, W] on the canonical S2 wavelength grid.
    """
    assert image.dim() in (3, 4), (
        f"Expected 3D or 4D, got shape {tuple(image.shape)}"
    )
    C_src = image.shape[-3]
    assert M.shape == (NUM_OPTICAL, C_src), (
        f"Matrix shape {tuple(M.shape)} doesn't match image channels ({C_src})"
    )

    M_dev = M.to(device=image.device, dtype=image.dtype)
    if image.dim() == 3:
        # [C, H, W] -> [13, H, W]
        return torch.einsum("kc,chw->khw", M_dev, image)
    # [T, C, H, W] -> [T, 13, H, W]
    return torch.einsum("kc,tchw->tkhw", M_dev, image)


def build_canonical_image(
    optical: torch.Tensor,
    sar: torch.Tensor = None,
) -> torch.Tensor:
    """
    Stack the 13 canonical optical channels with 2 SAR channels (VV, VH)
    to produce the full 15-channel canonical image. SAR is zero-filled
    when not provided.

    Args:
        optical: [13, H, W] (or [T, 13, H, W]) — canonical optical bands.
        sar:     [2, H, W]  (or [T, 2, H, W])  — VV, VH (in that order),
                 or None.

    Returns:
        [15, H, W] (or [T, 15, H, W]) canonical image.
    """
    if optical.dim() == 3:
        assert optical.shape[0] == NUM_OPTICAL, (
            f"Expected {NUM_OPTICAL} optical channels, got {optical.shape[0]}"
        )
        _, H, W = optical.shape
        if sar is None:
            sar = torch.zeros(NUM_SAR, H, W, dtype=optical.dtype, device=optical.device)
        else:
            assert sar.shape == (NUM_SAR, H, W), (
                f"Expected SAR shape ({NUM_SAR}, {H}, {W}), got {tuple(sar.shape)}"
            )
        return torch.cat([optical, sar], dim=0)

    if optical.dim() == 4:
        T, C, H, W = optical.shape
        assert C == NUM_OPTICAL, f"Expected {NUM_OPTICAL} optical channels, got {C}"
        if sar is None:
            sar = torch.zeros(T, NUM_SAR, H, W, dtype=optical.dtype, device=optical.device)
        else:
            assert sar.shape == (T, NUM_SAR, H, W), (
                f"Expected SAR shape ({T}, {NUM_SAR}, {H}, {W}), got {tuple(sar.shape)}"
            )
        return torch.cat([optical, sar], dim=1)

    raise ValueError(f"Optical must be 3D or 4D, got {optical.dim()}D")


# ────────────────────────────────────────────────────────────────────
# Spatial padding
# ────────────────────────────────────────────────────────────────────

def pad_to_canonical(
    image: torch.Tensor,
    target: torch.Tensor = None,
    size: int = CANONICAL_SIZE,
):
    """
    Pad image (and optional target) to size×size with zeros, top-left aligned.

    Args:
        image:  [C, H, W] or [T, C, H, W]. Float.
        target: [H, W] segmentation label (long), or None for classification.
                Padded region gets IGNORE_INDEX (255).
        size:   target spatial size (default 512).

    Returns:
        image_padded:  same rank as input, with last 2 dims = (size, size).
        target_padded: [size, size] long tensor, or None.
        valid_mask:    [size, size] uint8 — 1 = real, 0 = pad.
        original_size: torch.LongTensor of shape [2] — (H_orig, W_orig).
    """
    if image.dim() == 3:
        C, H, W = image.shape
        assert H <= size and W <= size, (
            f"Image larger than canonical size: ({H}, {W}) vs {size}"
        )
        image_padded = torch.zeros(C, size, size, dtype=image.dtype, device=image.device)
        image_padded[:, :H, :W] = image
    elif image.dim() == 4:
        T, C, H, W = image.shape
        assert H <= size and W <= size, (
            f"Image larger than canonical size: ({H}, {W}) vs {size}"
        )
        image_padded = torch.zeros(T, C, size, size, dtype=image.dtype, device=image.device)
        image_padded[:, :, :H, :W] = image
    else:
        raise ValueError(f"Image must be 3D or 4D, got {image.dim()}D")

    target_padded = None
    if target is not None:
        assert target.shape == (H, W), (
            f"Target shape {tuple(target.shape)} must match image spatial extent ({H}, {W})"
        )
        target_padded = torch.full(
            (size, size), IGNORE_INDEX,
            dtype=target.dtype, device=target.device,
        )
        target_padded[:H, :W] = target

    valid_mask = torch.zeros(size, size, dtype=torch.uint8)
    valid_mask[:H, :W] = 1

    original_size = torch.tensor([H, W], dtype=torch.long)

    return image_padded, target_padded, valid_mask, original_size