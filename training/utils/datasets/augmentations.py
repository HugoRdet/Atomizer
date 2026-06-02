"""
D4 Dihedral Augmentation for FLAIR-HUB and FRACTAL
=====================================================

Applies rotation (0/90/180/270°) + optional horizontal/vertical flips
consistently across all modalities of a multi-modal sample.

Two flavors of `apply`:

  - `apply(array, transform)`                 — for RASTER tensors
                                                (VHR, SPOT, DEM, S2, S1, labels)
                                                operates on last 2 axes (H, W)

  - `apply_to_xy(xy, transform, patch_size)`  — for POINT CLOUD coordinates
                                                in patch-local pixel space
                                                (FRACTAL LIDAR)

  - `apply_jitter(xy, z, sigma_xy, sigma_z, seed)` — optional Gaussian
                                                noise on point positions
                                                (FRACTAL LIDAR)

D4 group = 8 transformations:
  4 rotations × {identity, horizontal flip} (vertical flip is implicit via rot+hflip)

Why D4 specifically:
  - 90° rotations of square rasters are pixel-exact (no interpolation)
  - Aerial imagery is rotationally invariant (no canonical "up")
  - Cheap to compute (just view manipulations / single copies)
  - All modalities share spatial geometry → consistent transformation
    keeps cross-modal correspondences exact

The augmentation is sampled ONCE per __getitem__ call and applied to
every modality, ensuring VHR/SPOT/DEM/S2/S1/label/LIDAR all stay aligned.

Usage (FLAIR-HUB raster modalities):
    from .augmentations import D4Augmentation
    self.augmenter = D4Augmentation(enabled=(self.split == "train"))
    transform = self.augmenter.sample(index=index)
    vhr = self.augmenter.apply(vhr, transform)           # [C, H, W]
    s2  = self.augmenter.apply(s2_stack, transform)      # [T, B, H, W]

Usage (FRACTAL raster + point cloud):
    transform = self.augmenter.sample(index=index)
    ortho = self.augmenter.apply(ortho, transform)                  # [C, H, W]
    xy    = self.augmenter.apply_to_xy(xy, transform, patch_size_px=250)
    xy, z = self.augmenter.apply_jitter(xy, z, sigma_xy=0.25, sigma_z=0.003,
                                        seed=index)
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


@dataclass(frozen=True)
class D4Transform:
    """Parameters of a single D4 transformation, immutable."""
    n_rot: int      # number of 90° rotations: 0, 1, 2, or 3
    flip_h: bool    # horizontal flip (along W axis)
    flip_v: bool    # vertical flip (along H axis)

    @property
    def is_identity(self) -> bool:
        return self.n_rot == 0 and not self.flip_h and not self.flip_v


class D4Augmentation:
    """
    Consistent D4 dihedral augmentation across multi-modal samples.

    All 8 elements of the D4 group are reachable:
      - 4 rotations (0°, 90°, 180°, 270°)
      - × identity or horizontal flip

    Vertical flip is technically redundant with rotation+hflip but we keep
    it as an independent option for an enlarged sampling surface. Empirical
    probability of identity = 1/16 (n_rot=0, no flips).

    Worker safety:
      Uses np.random.default_rng with a seed derived from index + worker_id.
      Each worker generates an independent stream, and each index within a
      worker generates a reproducible stream (good for debugging).
    """

    def __init__(
        self,
        enabled: bool = True,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        seed_offset: int = 0,
    ):
        """
        Args:
            enabled:      Master switch. False → always returns identity transform.
            p_flip_h:     Probability of horizontal flip.
            p_flip_v:     Probability of vertical flip.
            seed_offset:  Added to the per-sample seed. Useful for different
                          augmentation streams per epoch — pass `epoch` from
                          the training loop if you want epoch-varying augs
                          (otherwise leave at 0 and aug is deterministic per
                          sample index, which is fine — different samples
                          get different augs).
        """
        self.enabled = enabled
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.seed_offset = seed_offset

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample(self, index: int) -> D4Transform:
        """
        Sample a D4 transform for this sample index.

        Reproducible: same (index, seed_offset) → same transform.
        Worker-aware: incorporates worker_info if available.
        """
        if not self.enabled:
            return D4Transform(n_rot=0, flip_h=False, flip_v=False)

        # Build a per-(worker, index) seed so each sample gets its own
        # augmentation but it's reproducible for debugging.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # Mix worker_id, index, seed_offset into a single 64-bit seed.
        # The constants are arbitrary primes — they just spread the bits.
        seed = (
            (worker_id * 2654435761)
            ^ (index * 40503)
            ^ (self.seed_offset * 16777619)
        ) & 0xFFFFFFFF

        rng = np.random.default_rng(seed)

        n_rot = int(rng.integers(0, 4))                 # 0, 1, 2, or 3
        flip_h = bool(rng.random() < self.p_flip_h)
        flip_v = bool(rng.random() < self.p_flip_v)

        return D4Transform(n_rot=n_rot, flip_h=flip_h, flip_v=flip_v)

    # =========================================================================
    # Application
    # =========================================================================

    def apply(
        self,
        array: Union[torch.Tensor, np.ndarray],
        transform: D4Transform,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply transform to an array. Auto-detects spatial dimensions:

          - 2D [H, W]:        rotate/flip H,W axes
          - 3D [C, H, W]:     rotate/flip last 2 axes, keep C
          - 4D [T, B, H, W]:  rotate/flip last 2 axes, keep T and B

        Preserves dtype and device. For torch tensors uses torch ops;
        for numpy uses numpy ops. Both produce pixel-exact 90° rotations
        (no interpolation).

        Identity transform → return input unchanged (no copy).
        """
        if transform.is_identity:
            return array

        is_torch = isinstance(array, torch.Tensor)

        # Determine which axes are (H, W) — always the last two.
        if array.ndim < 2:
            raise ValueError(
                f"D4 augmentation requires at least 2D array, got shape {array.shape}"
            )

        h_axis = -2
        w_axis = -1

        if is_torch:
            return self._apply_torch(array, transform, h_axis, w_axis)
        else:
            return self._apply_numpy(array, transform, h_axis, w_axis)

    @staticmethod
    def _apply_torch(
        tensor: torch.Tensor,
        transform: D4Transform,
        h_axis: int,
        w_axis: int,
    ) -> torch.Tensor:
        """Apply transform using torch ops. Pixel-exact, no interpolation."""
        # Rotate first (k × 90° counter-clockwise in the (h_axis, w_axis) plane)
        if transform.n_rot != 0:
            tensor = torch.rot90(tensor, k=transform.n_rot, dims=(h_axis, w_axis))

        if transform.flip_h:
            tensor = torch.flip(tensor, dims=(w_axis,))

        if transform.flip_v:
            tensor = torch.flip(tensor, dims=(h_axis,))

        # Return contiguous so downstream ops (like .reshape in tokenization)
        # don't trip on non-contiguous strides from rotation+flip.
        return tensor.contiguous()

    @staticmethod
    def _apply_numpy(
        array: np.ndarray,
        transform: D4Transform,
        h_axis: int,
        w_axis: int,
    ) -> np.ndarray:
        """Apply transform using numpy ops. Pixel-exact, no interpolation."""
        if transform.n_rot != 0:
            array = np.rot90(array, k=transform.n_rot, axes=(h_axis, w_axis))

        if transform.flip_h:
            array = np.flip(array, axis=w_axis)

        if transform.flip_v:
            array = np.flip(array, axis=h_axis)

        # np.rot90 / np.flip return views with negative strides; downstream
        # torch.from_numpy will refuse these. Materialize with ascontiguousarray.
        return np.ascontiguousarray(array)

    # =========================================================================
    # Point-cloud transforms (for FRACTAL LIDAR)
    # =========================================================================

    @staticmethod
    def apply_to_xy(
        xy: Union[torch.Tensor, np.ndarray],
        transform: D4Transform,
        patch_size_px: float,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply a D4 transform to point-cloud (x, y) coordinates in patch-
        local pixel space.

        Args:
            xy:            [N, 2] array of (x, y) in pixel units in
                           [0, patch_size_px). x = column, y = row,
                           matching the raster convention used by
                           `apply()` on the corresponding raster.
            transform:     D4Transform sampled for this __getitem__.
            patch_size_px: Side length of the square patch in pixels
                           (e.g., 250 for FRACTAL VHR).

        Returns:
            Transformed xy of same shape and dtype, with values still
            inside [0, patch_size_px). For PIXEL-EXACT consistency with
            raster rotation (torch.rot90 / np.rot90):

              n_rot=1 (90° CCW):  (x, y) → (y, P-1 - x)
              n_rot=2 (180°):     (x, y) → (P-1 - x, P-1 - y)
              n_rot=3 (270° CCW): (x, y) → (P-1 - y, x)
              flip_h (W-axis):    (x, y) → (P-1 - x, y)
              flip_v (H-axis):    (x, y) → (x, P-1 - y)

            Composed in the same order as `apply()`:
                rotation → flip_h → flip_v.

        Identity transform returns the input unchanged.

        Numerical note: input xy is float (sub-pixel positions allowed).
        Output uses `P - 1 - x` rather than `P - x` so that an integer
        coordinate stays inside [0, P). For sub-pixel inputs this means
        the maximum representable value is `P - 1`, not `P`. Callers who
        rely on the exact upper bound should re-clip to
        [0, P - epsilon] after augmentation if needed.
        """
        if transform.is_identity:
            return xy

        is_torch = isinstance(xy, torch.Tensor)
        if not is_torch:
            xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[-1] != 2:
            raise ValueError(
                f"apply_to_xy expects [N, 2], got shape {tuple(xy.shape)}"
            )

        P = float(patch_size_px)
        # Work in a copy so we don't mutate the caller's array.
        if is_torch:
            x = xy[:, 0].clone()
            y = xy[:, 1].clone()
        else:
            x = xy[:, 0].copy()
            y = xy[:, 1].copy()

        # 1) Rotation (applied first, same as in apply()).
        if transform.n_rot == 1:
            # (x, y) -> (y, P-1 - x)
            new_x = y
            new_y = (P - 1.0) - x
            x, y = new_x, new_y
        elif transform.n_rot == 2:
            # (x, y) -> (P-1 - x, P-1 - y)
            x = (P - 1.0) - x
            y = (P - 1.0) - y
        elif transform.n_rot == 3:
            # (x, y) -> (P-1 - y, x)
            new_x = (P - 1.0) - y
            new_y = x
            x, y = new_x, new_y
        # n_rot == 0 → identity, skip.

        # 2) Horizontal flip (W-axis), then vertical flip (H-axis).
        if transform.flip_h:
            x = (P - 1.0) - x
        if transform.flip_v:
            y = (P - 1.0) - y

        # Reassemble
        if is_torch:
            xy_out = torch.stack([x, y], dim=-1).contiguous()
        else:
            xy_out = np.stack([x, y], axis=-1)
            xy_out = np.ascontiguousarray(xy_out)

        return xy_out

    @staticmethod
    def apply_jitter(
        xy: Union[torch.Tensor, np.ndarray],
        z: Union[torch.Tensor, np.ndarray, None] = None,
        sigma_xy: float = 0.25,
        sigma_z: float = 0.003,
        seed: Optional[int] = None,
    ):
        """
        Add small Gaussian noise to LIDAR point coordinates (and optional z).

        Args:
            xy:       [N, 2] point positions (any units; sigma_xy is in the
                      SAME units as xy — pixels for FRACTAL).
            z:        Optional [N] tensor of z values (same units as sigma_z).
                      Pass None to skip z jittering.
            sigma_xy: Std dev of XY noise. For FRACTAL pixel-space xy at
                      0.2m/px, sigma_xy=0.25 ≈ 5cm physical noise.
            sigma_z:  Std dev of Z noise in NORMALIZED units. For FRACTAL
                      z_norm scaled by 15m, sigma_z=0.003 ≈ 4.5cm physical.
            seed:     Per-sample integer seed for reproducibility. If None,
                      uses non-deterministic random state.

        Returns:
            (xy_jittered, z_jittered) if z is provided, else xy_jittered.

        Notes:
            - Caller is responsible for re-clipping xy back to valid
              pixel range if needed (clip to [0, P-epsilon]).
            - This adds independent noise per point — no spatial coherence.
              For sensor-noise simulation this is the standard model.
        """
        is_torch = isinstance(xy, torch.Tensor)
        if seed is not None:
            # Stable per-sample seed; worker_id mixing is the caller's job
            # if they want it (just XOR into seed before passing in).
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        else:
            rng = np.random.default_rng()

        # XY noise
        if is_torch:
            xy_np = xy.cpu().numpy()
        else:
            xy_np = np.asarray(xy)
        noise_xy = rng.normal(0.0, sigma_xy, size=xy_np.shape).astype(xy_np.dtype)
        xy_jit = xy_np + noise_xy

        if is_torch:
            xy_jit = torch.from_numpy(xy_jit).to(dtype=xy.dtype, device=xy.device)

        # Z noise (optional)
        if z is None:
            return xy_jit

        z_is_torch = isinstance(z, torch.Tensor)
        if z_is_torch:
            z_np = z.cpu().numpy()
        else:
            z_np = np.asarray(z)
        noise_z = rng.normal(0.0, sigma_z, size=z_np.shape).astype(z_np.dtype)
        z_jit = z_np + noise_z

        if z_is_torch:
            z_jit = torch.from_numpy(z_jit).to(dtype=z.dtype, device=z.device)

        return xy_jit, z_jit