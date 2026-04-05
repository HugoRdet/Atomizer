"""
Baseline Collate Functions for C2Seg / MDAS
=============================================

PANGAEA-style collate with optional cross-sensor adaptation.

Two collate modes:
  1. Native collate: batch samples from a single sensor. Just stacks tensors.
  2. Cross-sensor collate: adapts test-time data to match the training sensor's
     format (spectral interpolation + spatial resize).

Augmentation collate (training):
  - Spectral merging with Gaussian weighting (simulates broadband sensors)
  - Resolution blur (downsample-upsample within fixed spatial shape)
  Both are applied in the collate so batch_size > 1 works without padding.
"""

import random
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


IGNORE_INDEX = 255
NUM_CLASSES = 6


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL PADDING HELPER
# ═══════════════════════════════════════════════════════════════════════

def _pad_temporal(batch: list, modalities: List[str]) -> None:
    """
    Pad time-series images to the same temporal length within a batch.

    Operates in-place on batch. For images with shape [C, T, H, W],
    pads the T dimension to the batch maximum. Images with shape
    [C, H, W] (no temporal dim) are left unchanged.
    """
    T_max = 0
    for modality in modalities:
        for x in batch:
            if modality in x["image"] and len(x["image"][modality].shape) == 4:
                T_max = max(T_max, x["image"][modality].shape[1])

    if T_max == 0:
        return

    for modality in modalities:
        for i, x in enumerate(batch):
            if modality in x["image"] and len(x["image"][modality].shape) == 4:
                T = x["image"][modality].shape[1]
                if T < T_max:
                    padding = (0, 0, 0, 0, 0, T_max - T)
                    batch[i]["image"][modality] = F.pad(
                        x["image"][modality], padding, "constant", 0
                    )


# ═══════════════════════════════════════════════════════════════════════
# NATIVE COLLATE (single sensor, no adaptation)
# ═══════════════════════════════════════════════════════════════════════

def get_collate_fn(modalities: List[str]) -> Callable:
    """
    Standard PANGAEA collate. All samples must have the same shape.

    Handles optional temporal padding for time-series data.
    Passes through dates if present (for temporal models like LTAE).
    """
    def collate_fn(batch):
        _pad_temporal(batch, modalities)

        result = {
            "image": {
                modality: torch.stack([x["image"][modality] for x in batch])
                for modality in modalities
            },
            "target": torch.stack([x["target"] for x in batch]),
            "metadata": [sample["metadata"] for sample in batch],
        }

        # Pass through dates if present (for LTAE temporal models)
        if "dates" in batch[0]:
            result["dates"] = {
                modality: torch.stack([x["dates"][modality] for x in batch])
                for modality in modalities
                if modality in batch[0]["dates"]
            }

        return result

    return collate_fn


# ═══════════════════════════════════════════════════════════════════════
# SPECTRAL INTERPOLATION (for cross-sensor eval)
# ═══════════════════════════════════════════════════════════════════════

def spectral_interpolate(
    image: torch.Tensor,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate spectral bands from source sensor to target sensor's wavelength grid.

    For each target wavelength, linearly interpolates between the two nearest
    source bands. Wavelengths outside the source range are clamped (nearest
    source band is used).

    Args:
        image: [C_source, H, W] — source sensor image (normalized).
        source_wavelengths: [C_source] — source central wavelengths in nm, sorted.
        target_wavelengths: [C_target] — target central wavelengths in nm, sorted.

    Returns:
        interpolated: [C_target, H, W] — image at target spectral grid.
    """
    C_src = source_wavelengths.shape[0]
    C_tgt = target_wavelengths.shape[0]
    H, W = image.shape[1], image.shape[2]

    # Use torch.searchsorted to find insertion points
    idx = torch.searchsorted(source_wavelengths, target_wavelengths)

    # Clamp to valid range for interpolation
    idx_hi = idx.clamp(1, C_src - 1)
    idx_lo = idx_hi - 1

    wl_lo = source_wavelengths[idx_lo]
    wl_hi = source_wavelengths[idx_hi]

    # Interpolation weight
    denom = (wl_hi - wl_lo).clamp(min=1e-6)
    alpha = ((target_wavelengths - wl_lo) / denom).clamp(0.0, 1.0)

    # Gather source bands
    val_lo = image[idx_lo]
    val_hi = image[idx_hi]

    # Linear interpolation
    interpolated = val_lo * (1.0 - alpha[:, None, None]) + val_hi * alpha[:, None, None]

    return interpolated


# ═══════════════════════════════════════════════════════════════════════
# CROSS-SENSOR COLLATE (for evaluation)
# ═══════════════════════════════════════════════════════════════════════

def get_cross_sensor_collate_fn(
    source_modality: str,
    target_modality: str,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
    target_spatial_size: int,
) -> Callable:
    """
    Collate that adapts source sensor data to target sensor format.

    Used at eval time: e.g. loading S2 data, adapting to HySpex format
    for a model trained on HySpex.

    Data stays normalized with source sensor stats — no re-normalization.
    Only spectral interpolation (channel count) and spatial resize are applied.
    Labels are upsampled (nearest-neighbor) to match the model's spatial size.
    """

    def collate_fn(batch):
        adapted_images = []
        targets = []
        metadata_list = []

        for sample in batch:
            image = sample["image"][source_modality]
            label = sample["target"]

            # Step 1: spectral interpolation
            adapted = spectral_interpolate(
                image, source_wavelengths, target_wavelengths
            )

            # Step 2: spatial resize to target size
            if adapted.shape[1] != target_spatial_size or adapted.shape[2] != target_spatial_size:
                adapted = F.interpolate(
                    adapted.unsqueeze(0),
                    size=(target_spatial_size, target_spatial_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                label = F.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(),
                    size=(target_spatial_size, target_spatial_size),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()

            adapted_images.append(adapted)
            targets.append(label)
            metadata_list.append(sample["metadata"])

        adapted_batch = [
            {"image": {target_modality: img}, "target": tgt, "metadata": meta}
            for img, tgt, meta in zip(adapted_images, targets, metadata_list)
        ]
        _pad_temporal(adapted_batch, [target_modality])

        return {
            "image": {
                target_modality: torch.stack(
                    [x["image"][target_modality] for x in adapted_batch]
                )
            },
            "target": torch.stack([x["target"] for x in adapted_batch]),
            "metadata": [x["metadata"] for x in adapted_batch],
        }

    return collate_fn


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATION COLLATE (for training)
# ═══════════════════════════════════════════════════════════════════════

def baseline_spectral_augmentation(
    image: torch.Tensor,
    wavelengths: list = None,
    spectral_aug_pool: list = None,
) -> torch.Tensor:
    """
    Physically correct spectral augmentation for baselines.

    Same approach as Atomizer-IO: picks 1-8 random sensor configs from
    a generative pool, then for each virtual band:
      1. Find overlapping source bands (±0.75× bandwidth)
      2. Gaussian-weighted average of those bands
      3. Replace ALL overlapping channels with the merged value

    Bands outside any virtual band's range stay untouched.
    Output shape is always identical to input shape.

    If no pool or wavelengths provided, falls back to uniform group merging.

    Args:
        image: [C, H, W]
        wavelengths: list[float] — central wavelength per band (nm)
        spectral_aug_pool: list of configs, each a list of (wl, bw) tuples

    Returns:
        [C, H, W] — same shape, some channels merged
    """
    C, H, W = image.shape

    # ── Generative sensor simulation (matches Atomizer-IO) ──────────
    if spectral_aug_pool is not None and wavelengths is not None:
        import numpy as np

        n_configs = random.choices([1, 2, 3, 5, 8], weights=[3, 3, 3, 2, 1], k=1)[0]
        selected_configs = random.choices(spectral_aug_pool, k=n_configs)

        wl_arr = np.array(wavelengths)
        claimed = set()

        for sensor_bands in selected_configs:
            if not sensor_bands or len(sensor_bands) < 2:
                continue

            for sim_wl, sim_bw in sensor_bands:
                lo = sim_wl - sim_bw * 0.75
                hi = sim_wl + sim_bw * 0.75
                band_mask = (wl_arr >= lo) & (wl_arr <= hi)
                band_indices = np.where(band_mask)[0]

                if len(band_indices) == 0:
                    continue

                # Skip if any band already claimed
                if any(idx in claimed for idx in band_indices):
                    continue

                # Gaussian-weighted average
                group_img = image[band_indices]  # [n_overlap, H, W]
                group_wl = wl_arr[band_indices]

                sigma = max(sim_bw / 4.0, 1.0)
                wl_t = torch.tensor(group_wl, dtype=torch.float32)
                weights = torch.exp(-0.5 * ((wl_t - sim_wl) / sigma) ** 2)
                weights = weights / weights.sum()

                merged = (weights[:, None, None] * group_img).sum(dim=0, keepdim=True)

                # Replace all overlapping channels with merged value
                for idx in band_indices:
                    image[idx] = merged.squeeze(0)
                    claimed.add(idx)

        return image

    # ── Fallback: uniform group merging ─────────────────────────────
    n_groups = random.choice([4, 8, 16, 32, 64, 128])
    if n_groups >= C:
        return image

    group_size = C // n_groups
    merged = torch.zeros_like(image)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else C
        group_img = image[start:end]

        if wavelengths is not None and len(wavelengths) >= end:
            group_wl = wavelengths[start:end]
            center_wl = (min(group_wl) + max(group_wl)) / 2.0
            half_width = (max(group_wl) - min(group_wl)) / 2.0
            sigma = max(half_width / 2.0, 1.0)

            wl_tensor = torch.tensor(group_wl, dtype=torch.float32)
            weights = torch.exp(-0.5 * ((wl_tensor - center_wl) / sigma) ** 2)
            weights = weights / weights.sum()

            group_mean = (weights[:, None, None] * group_img).sum(dim=0, keepdim=True)
        else:
            group_mean = group_img.mean(dim=0, keepdim=True)

        merged[start:end] = group_mean.expand(end - start, H, W)

    return merged


def baseline_resolution_augmentation(
    image: torch.Tensor,
    label: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate coarser resolution within fixed spatial size.
    Downsample then upsample back to original dimensions.

    Args:
        image: [C, H, W]
        label: [H, W]
        factor: blur factor (2, 3, 4, ...)

    Returns:
        image: [C, H, W] — same size, blurred
        label: [H, W] — same size, block-ified
    """
    if factor <= 1:
        return image, label

    C, H, W = image.shape

    # Downsample image
    small_img = F.avg_pool2d(image.unsqueeze(0), factor).squeeze(0)

    # Upsample back
    image_aug = F.interpolate(
        small_img.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Label: nearest-neighbor round trip
    small_label = F.interpolate(
        label.unsqueeze(0).unsqueeze(0).float(),
        size=(H // factor, W // factor),
        mode="nearest",
    )
    label_aug = F.interpolate(
        small_label,
        size=(H, W),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()

    return image_aug, label_aug


def get_augmented_collate_fn(
    modalities: List[str],
    spectral_aug_prob: float = 0.5,
    spectral_groups: Optional[List[int]] = None,
    spectral_aug_pool: Optional[list] = None,
    resolution_aug_prob: float = 0.5,
    resolution_factors: Optional[List[int]] = None,
) -> Callable:
    """
    Collate with generative spectral augmentation and resolution blur.

    Spectral augmentation uses the same generative sensor simulation pool
    as Atomizer-IO: picks 1-8 random configs per sample, Gaussian-averages
    overlapping bands, replaces them in-place. Ensures fair comparison.

    Args:
        modalities: list of sensor keys in sample["image"]
        spectral_aug_prob: probability of applying spectral augmentation
        spectral_groups: legacy uniform group counts (fallback if no pool)
        spectral_aug_pool: list of sensor configs from build_spectral_aug_pool()
        resolution_aug_prob: probability of applying resolution blur
        resolution_factors: list of possible blur factors (e.g. [2, 3, 4, 5])
    """

    def collate_fn(batch):
        for i, sample in enumerate(batch):
            for modality in modalities:
                image = sample["image"][modality]
                label = sample["target"]

                # Spectral augmentation (generative sensor simulation)
                if random.random() < spectral_aug_prob:
                    wavelengths = sample.get("metadata", {}).get("wavelengths", None)
                    image = baseline_spectral_augmentation(
                        image,
                        wavelengths=wavelengths,
                        spectral_aug_pool=spectral_aug_pool,
                    )

                # Resolution augmentation
                if resolution_factors and random.random() < resolution_aug_prob:
                    factor = random.choice(resolution_factors)
                    image, label = baseline_resolution_augmentation(image, label, factor)

                batch[i]["image"][modality] = image
                batch[i]["target"] = label

        # Temporal padding (for time-series datasets)
        _pad_temporal(batch, modalities)

        # Standard stacking (same as native collate)
        return {
            "image": {
                modality: torch.stack([x["image"][modality] for x in batch])
                for modality in modalities
            },
            "target": torch.stack([x["target"] for x in batch]),
            "metadata": [sample["metadata"] for sample in batch],
        }

    return collate_fn