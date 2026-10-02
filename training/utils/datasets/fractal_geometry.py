"""
fractal_geometry.py — shared, single-source-of-truth pixel-coordinate
helper used by both FractalDataset and precompute_fractal_latent_assignment.py.

NOTE: this module previously also carried a deterministic_lidar_subsample
helper, requiring context subsampling to be seeded by patch_id (not random
per epoch) so the offline Voronoi precompute would stay valid. That's no
longer necessary — following the DALES pattern, precompute now covers the
FULL point set (all n_points_raw, all 16 D4 variants) rather than a
subsample, so context subsampling can be genuinely random per epoch again;
whatever `sel` indices get drawn, `assignment[sel]` still gathers the
correct precomputed values. See precompute_fractal_latent_assignment.py's
module docstring.
"""

import numpy as np


def compute_patch_local_pixel_coords(las_x, las_y, x_min, y_max, resolution,
                                     patch_size_px):
    """
    Convert raw LAS x/y (Lambert-93) to patch-local pixel coordinates,
    matching the VHR raster's row/col frame exactly (row 0 = y_max / north
    edge, consistent with rasterio's default top-left-origin read).

    Args:
        las_x, las_y:  np.ndarray, raw coordinates from the LAZ file.
        x_min:         float, patch's minimum x (from las.x.min()).
        y_max:         float, patch's maximum y (from las.y.max()).
        resolution:    float, meters per pixel (FractalDataset.VHR_RESOLUTION).
        patch_size_px: int, patch side length in pixels.

    Returns:
        (pixel_x, pixel_y): np.ndarray float32, clipped to
        [0, patch_size_px - eps).
    """
    pixel_x = (np.asarray(las_x) - x_min) / resolution
    pixel_y = (y_max - np.asarray(las_y)) / resolution
    pixel_x = np.clip(pixel_x, 0.0, patch_size_px - 1e-3).astype(np.float32)
    pixel_y = np.clip(pixel_y, 0.0, patch_size_px - 1e-3).astype(np.float32)
    return pixel_x, pixel_y
