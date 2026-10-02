"""
save_fractal_figs.py
======================

Step 2 of 2 for the FRACTAL figure export. Given a patch_id (picked from
browse_fractal_patches.py's contact sheet), exports:

  - VHR ortho channels as SVGs: R, G, B, NIR individually + RGB combined
    (ortho bands are stored as [NIR, R, G, B], per the dataset docstring)
  - The matching LIDAR point cloud as a .ply file, for opening in
    CloudCompare / MeshLab / Blender etc.

This bypasses `FractalPerceiverDataset.__getitem__` (no Fourier encoding,
no query building) and instead re-implements just the raw loading steps
(read .laz, read ortho, remap LAS classification -> FRACTAL 7 classes),
matching the dataset's own logic. Loaded from its .py file directly.

Point cloud coloring (--color_mode):
  "rgb"            (default) - each LIDAR point is colored by the ortho
                    pixel beneath it (nearest-neighbor), i.e. the same
                    VHR-colorization RandLA-Net uses in the paper.
  "classification" - each point is colored by its remapped FRACTAL class
                    (ground/vegetation/building/water/bridge/permanent
                    structure/other), using a fixed 7-color palette.

Usage:
    python save_fractal_figs.py --root ./data --split train \
        --patch_id <id_from_browse_step> --color_mode rgb
"""

import argparse
import importlib.machinery
import importlib.util
import os
import sys

import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def _load_dataset_module(module_path: str):
    """
    Load a dataset .py file directly, while still allowing its own
    `from .sibling_module import X` relative imports to resolve — see
    the matching function in browse_fractal_patches.py for the full
    explanation. We register a namespace package pointing at the file's
    own directory instead of importing the real "training" package
    (which would pull in pytorch_lightning/transformers unnecessarily).
    """
    module_path = os.path.abspath(module_path)
    pkg_dir = os.path.dirname(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    pkg_name = "_fractal_ds_pkg"

    if pkg_name not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
        pkg_spec.submodule_search_locations = [pkg_dir]
        pkg_module = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg_module

    full_name = f"{pkg_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def percentile_stretch(channel: np.ndarray, low=2, high=98):
    lo, hi = np.percentile(channel, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (channel.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def save_gray_svg(channel_01: np.ndarray, path: str, cmap="gray"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(channel_01, cmap=cmap, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


def save_rgb_svg(rgb_01: np.ndarray, path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb_01)
    ax.set_axis_off()
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


# FRACTAL 7-class palette (0..6): other, ground, vegetation, building,
# water, bridge, permanent_structure
CLASS_PALETTE = np.array([
    [128, 128, 128],  # 0 other        - gray
    [140,  90,  40],  # 1 ground       - brown
    [ 40, 160,  40],  # 2 vegetation   - green
    [220,  40,  40],  # 3 building     - red
    [ 40, 100, 220],  # 4 water        - blue
    [230, 200,  40],  # 5 bridge       - yellow
    [180,  40, 200],  # 6 permanent_structure - purple
], dtype=np.uint8)
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def normalize_xyz(xyz: np.ndarray, per_axis: bool = False) -> np.ndarray:
    """
    Normalize point coordinates into [0, 1].

    Default (per_axis=False): a single shared scale factor across x/y/z,
    so the point cloud keeps its true proportions (a tall building stays
    visually tall relative to its footprint) — only the axis with the
    largest extent will actually span the full [0, 1] range, the other
    two will be correspondingly narrower.

    per_axis=True: each axis independently stretched to fill [0, 1],
    which distorts relative proportions but maximizes use of the cube
    on every axis (useful if you mainly care about seeing detail on a
    thin axis, e.g. a very flat scene).
    """
    mins = xyz.min(axis=0)
    shifted = xyz - mins
    if per_axis:
        ranges = shifted.max(axis=0)
        ranges[ranges == 0] = 1.0
        return shifted / ranges
    else:
        scale = shifted.max()
        if scale == 0:
            scale = 1.0
        return shifted / scale


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    Minimal ASCII PLY writer: xyz [N,3] float, rgb [N,3] uint8.
    No external dependency (plyfile) required.
    """
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Vectorized formatting for speed on large point clouds
        lines = np.empty(n, dtype=object)
        xyz_r = np.round(xyz, 4)
        for i in range(n):
            x, y, z = xyz_r[i]
            r, g, b = rgb[i]
            lines[i] = f"{x} {y} {z} {r} {g} {b}"
        f.write("\n".join(lines))
        f.write("\n")
    print(f"saved {path}  ({n:,} points)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data", help="Root containing FRACTAL/ and FRACTAL-IRGB/")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "validation"])
    parser.add_argument("--patch_id", required=True, help="Patch id from browse_fractal_patches.py")
    parser.add_argument("--outdir", default="./figs")
    parser.add_argument("--color_mode", default="rgb", choices=["rgb", "classification"])
    parser.add_argument("--nir_cmap", default="inferno",
                         help="Matplotlib colormap for the NIR channel (perceptual, since NIR isn't visible light)")
    parser.add_argument("--per_axis_normalize", action="store_true",
                         help="Normalize the PLY's x/y/z independently to [0,1] each "
                              "(default: single shared scale across all axes, so the "
                              "scene keeps its true proportions instead of being "
                              "squashed/stretched per axis)")
    parser.add_argument(
        "--dataset_file",
        default="./training/utils/datasets_baselines/utils_dataset_fractal_perceiver.py",
        help="Path to the .py file defining FractalPerceiverDataset",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mod = _load_dataset_module(args.dataset_file)
    FractalPerceiverDataset = mod.FractalPerceiverDataset
    REMAP_LUT = mod.REMAP_LUT
    VHR_RESOLUTION = FractalPerceiverDataset.VHR_RESOLUTION
    PATCH_SIZE_PX = FractalPerceiverDataset.PATCH_SIZE_PX

    ds = FractalPerceiverDataset(
        root_path=args.root,
        mode=args.split,
        use_augmentation=False,
    )

    row = next((r for r in ds.patch_rows if r["patch_id"] == args.patch_id), None)
    if row is None:
        raise ValueError(f"patch_id '{args.patch_id}' not found in split='{args.split}'")

    # ── VHR ortho: NIR, R, G, B ────────────────────────────────────────
    with rasterio.open(row["ortho_path"]) as src:
        ortho = src.read().astype(np.float32)  # [4, H, W]

    nir = percentile_stretch(ortho[0])
    r = percentile_stretch(ortho[1])
    g = percentile_stretch(ortho[2])
    b = percentile_stretch(ortho[3])

    save_gray_svg(nir, os.path.join(args.outdir, "fractal_NIR.svg"), cmap=args.nir_cmap)
    save_gray_svg(r, os.path.join(args.outdir, "fractal_R.svg"), cmap="Reds")
    save_gray_svg(g, os.path.join(args.outdir, "fractal_G.svg"), cmap="Greens")
    save_gray_svg(b, os.path.join(args.outdir, "fractal_B.svg"), cmap="Blues")
    save_rgb_svg(np.stack([r, g, b], axis=-1), os.path.join(args.outdir, "fractal_RGB.svg"))

    # ── LIDAR point cloud ────────────────────────────────────────────
    las = laspy.read(row["laz_path"])
    x_raw = np.asarray(las.x, dtype=np.float64)
    y_raw = np.asarray(las.y, dtype=np.float64)
    z_raw = np.asarray(las.z, dtype=np.float64)

    if args.color_mode == "classification":
        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels = REMAP_LUT[las_cls]  # 0..6, or IGNORE_INDEX(255)
        colors = np.where(
            (labels[:, None] >= 0) & (labels[:, None] < 7),
            CLASS_PALETTE[np.clip(labels, 0, 6)],
            IGNORE_COLOR,
        )
    else:
        # Colorize each LIDAR point from the ortho pixel beneath it
        # (same patch-local pixel mapping the dataset itself uses).
        x_min = float(las.x.min())
        y_max = float(las.y.max())
        px = (x_raw - x_min) / VHR_RESOLUTION
        py = (y_max - y_raw) / VHR_RESOLUTION
        px = np.clip(px, 0, PATCH_SIZE_PX - 1).astype(np.int64)
        py = np.clip(py, 0, PATCH_SIZE_PX - 1).astype(np.int64)

        # ortho bands [NIR, R, G, B] -> uint8 stretch per-band (whole-image
        # percentile, same as the SVG export above) then index per point.
        r_img = (percentile_stretch(ortho[1]) * 255).astype(np.uint8)
        g_img = (percentile_stretch(ortho[2]) * 255).astype(np.uint8)
        b_img = (percentile_stretch(ortho[3]) * 255).astype(np.uint8)
        colors = np.stack([
            r_img[py, px],
            g_img[py, px],
            b_img[py, px],
        ], axis=-1)

    xyz = np.stack([x_raw, y_raw, z_raw], axis=-1)
    xyz = normalize_xyz(xyz, per_axis=args.per_axis_normalize)
    ply_path = os.path.join(args.outdir, f"fractal_{args.patch_id}_{args.color_mode}.ply")
    write_ply(ply_path, xyz, colors.astype(np.uint8))


if __name__ == "__main__":
    main()
