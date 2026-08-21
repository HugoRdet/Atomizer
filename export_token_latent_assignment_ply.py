"""
export_token_latent_assignment_ply.py
========================================

Visualizes the PRECOMPUTED token->latent (Voronoi) assignment for ONE
tiled patch as a colored PLY -- no trained model needed, reads directly
from the _latent_assign.npz sidecar produced by
precompute_dales_latent_assignment.py.

Colors each point by which latent it's assigned to, so you can see the
Voronoi partition's spatial structure in Blender: points sharing a color
were routed to the same latent's cross-attention.

Usage:
    python export_token_latent_assignment_ply.py \
        --laz_path .../DALES_140m/DALES/test/5175_54395_r003_c002.laz \
        --variant_idx 0 \
        --out_dir ./figures

--variant_idx selects which of the 16 precomputed D4 variants to show
(0 = identity/canonical, i.e. the UN-augmented assignment -- matches the
raw point positions in the .laz file directly).

--list_patches, given --tiled_dir instead of --laz_path, prints available
patch filenames and exits.
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import laspy


def build_random_color_lut(L_spatial: int, seed: int = 0) -> np.ndarray:
    """Independent random color per latent index (full hue/saturation/
    value randomness, not a structured scheme) — simplest fix for
    neighboring hexagons landing on similar colors, since there's no
    formula linking color to index that could create adjacency patterns.
    """
    rng = np.random.default_rng(seed)
    colors = np.empty((L_spatial, 3), dtype=np.uint8)
    for i in range(L_spatial):
        hue = rng.uniform(0.0, 1.0)
        sat = rng.uniform(0.5, 1.0)
        val = rng.uniform(0.75, 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def normalize_xyz_unit_cube(xyz: np.ndarray) -> np.ndarray:
    """Normalize coordinates into [0, 1] using a SINGLE uniform scale
    factor (the largest axis range), not independent per-axis min-max —
    this preserves the patch's true proportions (LIDAR patches are
    typically much wider in x/y than tall in z; independent normalization
    would stretch z to fill [0,1] too, distorting the shape).
    """
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    scale = float((maxs - mins).max())
    if scale <= 0:
        return np.zeros_like(xyz)
    return (xyz - mins) / scale


def write_ply_binary(path: Path, xyz: np.ndarray, colors: np.ndarray):
    """Minimal binary-little-endian PLY writer, no external deps."""
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    records = np.empty(n, dtype=dtype)
    records["x"] = xyz[:, 0].astype("<f4")
    records["y"] = xyz[:, 1].astype("<f4")
    records["z"] = xyz[:, 2].astype("<f4")
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(records.tobytes())
    print(f"[assign_ply] Wrote {n:,} points to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laz_path", type=str, default=None,
                         help="Path to ONE tiled patch .laz file")
    parser.add_argument("--tiled_dir", type=str, default=None,
                         help="Used only with --list_patches")
    parser.add_argument("--variant_idx", type=int, default=0,
                         help="Which of the 16 precomputed D4 variants to "
                              "show (0 = identity/canonical)")
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for the color shuffle — change "
                              "this to get a different (still well-spread) "
                              "color assignment if a particular shuffle "
                              "still looks unclear.")
    parser.add_argument("--list_patches", action="store_true",
                         help="List patches under --tiled_dir and exit")
    args = parser.parse_args()

    if args.list_patches:
        if args.tiled_dir is None:
            raise ValueError("--tiled_dir is required with --list_patches")
        patches = sorted(Path(args.tiled_dir).glob("*.laz"))
        print(f"[assign_ply] {len(patches)} patches under {args.tiled_dir}:")
        for p in patches[:50]:
            print(f"  {p.name}")
        if len(patches) > 50:
            print(f"  ... and {len(patches) - 50} more")
        return

    if args.laz_path is None:
        raise ValueError("--laz_path is required (or use --list_patches)")

    laz_path = Path(args.laz_path)
    assign_path = laz_path.parent / f"{laz_path.stem}_latent_assign.npz"
    if not assign_path.exists():
        raise FileNotFoundError(
            f"No matching assignment file found: {assign_path}\n"
            f"Did precompute_dales_latent_assignment.py run for this patch?"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(assign_path) as npz:
        assignment = npz["assignment"][args.variant_idx]  # [n_points]
        L_spatial = int(npz["L_spatial"])

    las = laspy.read(str(laz_path))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    n_points = x.shape[0]

    if assignment.shape[0] != n_points:
        raise RuntimeError(
            f"{laz_path.name} has {n_points} points but assignment has "
            f"{assignment.shape[0]} — tiling/precompute out of sync for "
            f"this patch. Re-run precompute_dales_latent_assignment.py."
        )

    print(f"[assign_ply] {laz_path.name}: {n_points:,} points, "
          f"L_spatial={L_spatial}, variant_idx={args.variant_idx}")

    color_lut = build_random_color_lut(L_spatial, seed=args.seed)  # [L_spatial, 3]
    colors = color_lut[assignment]  # [n_points, 3]

    xyz = np.stack([x, y, z], axis=-1)
    xyz = normalize_xyz_unit_cube(xyz)
    out_path = out_dir / f"{laz_path.stem}_variant{args.variant_idx}_latent_assignment.ply"
    write_ply_binary(out_path, xyz, colors)

    print(f"\n[assign_ply] DONE. Load into Blender:")
    print(f"  {out_path}")
    print(f"  ({L_spatial} distinct colors, one per latent's Voronoi cell)")


if __name__ == "__main__":
    main()
