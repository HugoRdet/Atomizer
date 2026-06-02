"""
Export FRACTAL patches to Blender-friendly figures
====================================================

For each requested patch_id, writes to ./figures/:
  - <patch_id>.ply           — colored point cloud (one RGB per class)
  - <patch_id>_ortho.tif     — matching VHR IRGB ortho (copy from disk)
  - <patch_id>_legend.txt    — class→color legend + point counts

Usage:
    python export_patches_for_blender.py \\
        --fractal_root ./data/FRACTAL/data \\
        --irgb_root    ./data/FRACTAL-IRGB \\
        --output_dir   ./figures \\
        --patch_ids    TRAIN-0436_6414-002994543 TRAIN-0436_6417-002457323 \\
                       TRAIN-0436_6420-002760208

Blender import:
    File → Import → Stanford (.ply)
    Vertex colors will show up; enable "Vertex Color" in shader or use
    "Object Properties → Viewport Display → Color → Vertex".
    Drop the .tif as an image plane reference (Add → Image → Reference).

Class coloring uses the cleaned LAS-code LUT (codes 65/66/67 → IGNORE),
matching utils_dataset_fractal.py and precompute_fractal_weights.py.
Points with class IGNORE are kept in the PLY but colored neutral gray so
you can see them but they're distinguishable from real classes.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed. pip install laspy")
    sys.exit(1)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed — PNG conversion disabled, "
          "TIFFs will be copied verbatim. Install with: pip install rasterio")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[Warning] Pillow not installed — PNG conversion disabled. "
          "Install with: pip install Pillow")


# =============================================================================
# Class definitions (must match utils_dataset_fractal.py)
# =============================================================================

CLASS_NAMES = [
    "other",                # 0
    "ground",               # 1
    "vegetation",           # 2
    "building",             # 3
    "water",                # 4
    "bridge",               # 5
    "permanent_structure",  # 6
]
NUM_CLASSES = 7
IGNORE_INDEX = 255

LAS_CODE_TO_FRACTAL_CLASS = {
    1:  0,    # Unclassified → other
    2:  1,    # Ground
    3:  2,    # Low vegetation → vegetation
    4:  2,    # Medium vegetation → vegetation
    5:  2,    # High vegetation → vegetation
    6:  3,    # Building
    9:  4,    # Water
    17: 5,    # Bridge deck → bridge
    64: 6,    # Permanent structure
    # 65, 66, 67 → IGNORE
}
MAX_LAS_CODE = 255

# Per-class RGB colors (uint8). Chosen to be distinguishable in Blender:
#   - earthy tones for natural classes (ground, vegetation)
#   - bright/saturated for human-made (building, bridge, permanent)
#   - blue for water
#   - gray for ignored points
CLASS_COLORS = {
    0: (170, 170, 170),    # other — light gray
    1: (139, 105,  20),    # ground — earth brown
    2: ( 34, 139,  34),    # vegetation — forest green
    3: (220,  20,  60),    # building — crimson
    4: ( 30, 144, 255),    # water — dodger blue
    5: (255, 140,   0),    # bridge — dark orange
    6: (148,   0, 211),    # permanent_structure — dark violet
    IGNORE_INDEX: (60, 60, 60),  # ignored — dark gray
}


def build_remap_lut() -> np.ndarray:
    """LAS code → FRACTAL class (or IGNORE_INDEX). Same as precompute script."""
    lut = np.full(MAX_LAS_CODE + 1, IGNORE_INDEX, dtype=np.uint8)
    for las_code, fractal_class in LAS_CODE_TO_FRACTAL_CLASS.items():
        lut[las_code] = fractal_class
    return lut


# =============================================================================
# Path index for IRGB orthos
# =============================================================================

def build_ortho_index(irgb_root: Path) -> dict:
    """
    Build {patch_id → ortho .tif path}.

    FRACTAL-IRGB shards into subdirs that don't align with FRACTAL/'s
    shards, so the only reliable way to find the matching ortho is to
    glob the whole IRGB tree once.

    Recursively scans every subdirectory under irgb_root for .tif files —
    handles arbitrary nesting (e.g., FRACTAL-IRGB/, FRACTAL-IRGB/data/,
    FRACTAL-IRGB/train/, FRACTAL-IRGB/data/train/, etc.).
    """
    print(f"[index] Building IRGB ortho index from {irgb_root}...")

    if not irgb_root.is_dir():
        print(f"[index]   ERROR: {irgb_root} is not a directory")
        return {}

    # Recursive scan from the top — search both .tif and .tiff extensions
    # (FRACTAL-IRGB uses .tiff; other datasets sometimes use .tif).
    all_tifs = []
    for ext in ("*.tif", "*.tiff"):
        all_tifs.extend(irgb_root.rglob(ext))
    print(f"[index]   found {len(all_tifs)} .tif/.tiff files via recursive scan")

    if not all_tifs:
        # Diagnostic: show what IS in irgb_root so the user can locate
        # the right path manually
        print(f"[index]   diagnostic — top-level contents of {irgb_root}:")
        try:
            for p in sorted(irgb_root.iterdir())[:20]:
                kind = "DIR " if p.is_dir() else "file"
                print(f"[index]     {kind}  {p.name}")
        except Exception as e:
            print(f"[index]     could not list: {e}")
        print(f"[index]   try a different --irgb_root, or check that the "
              f"IRGB dataset is actually present.")
        return {}

    index = {}
    duplicates = 0
    for tif in all_tifs:
        if tif.stem in index:
            duplicates += 1
        index[tif.stem] = tif

    if duplicates > 0:
        print(f"[index]   WARN: {duplicates} duplicate patch_ids — "
              f"using last-found copy for each")

    print(f"[index] Indexed {len(index)} unique patch_ids")
    return index


# =============================================================================
# PLY writer (ASCII format, easiest to debug; switch to binary for size)
# =============================================================================

def write_ply(
    path: Path,
    xyz: np.ndarray,                # [N, 3] float32
    rgb: np.ndarray,                # [N, 3] uint8
    fractal_class: np.ndarray,      # [N] uint8 (for "class" scalar property)
):
    """
    Write a binary little-endian PLY with positions, vertex colors, and
    a per-point "class" scalar (so Blender users can color/filter by class
    later via Geometry Nodes or shader if they want).
    """
    assert xyz.shape[0] == rgb.shape[0] == fractal_class.shape[0]
    N = xyz.shape[0]

    # Cast everything to the right dtypes for the binary write
    xyz = xyz.astype(np.float32, copy=False)
    rgb = rgb.astype(np.uint8, copy=False)
    fractal_class = fractal_class.astype(np.uint8, copy=False)

    # Build a structured array — one row per vertex — and dump in one write
    vertex_dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
        ("class", np.uint8),
    ])
    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"]   = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"]  = rgb[:, 2]
    vertices["class"] = fractal_class

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar class\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())


# =============================================================================
# TIFF → PNG conversion (NIR-R-G-B → 8-bit RGB)
# =============================================================================

def convert_ortho_to_png(
    src_tif: Path,
    dst_png: Path,
    stretch: str = "percentile",
) -> bool:
    """
    Read a 4-band NIR-R-G-B FRACTAL ortho TIFF and save just the RGB
    bands as an 8-bit PNG. Image viewers handle PNG correctly out of the
    box and don't get confused by extra channels or non-standard band
    ordering.

    Band layout (verified for FRACTAL-IRGB):
      Band 1 = NIR  → dropped
      Band 2 = R    → red
      Band 3 = G    → green
      Band 4 = B    → blue

    Args:
        src_tif: Source .tif/.tiff path. Read via rasterio.
        dst_png: Destination .png path.
        stretch: How to map the source dtype to uint8 [0, 255]:
                 - 'percentile' (default): per-channel 2-98% percentile stretch.
                   Good for visualization — pushes the bulk of pixel values
                   across the full display range. Slightly recolors compared
                   to the original.
                 - 'uint8':                 assume input is already uint8 [0, 255]
                   and just copy it. Use this if FRACTAL-IRGB stores raw uint8.
                 - 'minmax':                per-channel min-max stretch.

    Returns True on success, False if conversion failed (e.g., missing libs).
    """
    if not (HAS_RASTERIO and HAS_PIL):
        return False

    with rasterio.open(src_tif) as src:
        if src.count < 4:
            print(f"[png]   WARN: {src_tif.name} has only {src.count} bands; "
                  f"expected ≥4. Using first 3 as RGB.")
            arr = src.read([1, 2, 3]).astype(np.float32)
        else:
            # FRACTAL-IRGB: band 1=NIR, 2=R, 3=G, 4=B — take R/G/B
            arr = src.read([2, 3, 4]).astype(np.float32)
    # arr shape: [3, H, W]

    if stretch == "uint8":
        # Source is uint8 [0, 255]; just clip and cast.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif stretch == "minmax":
        # Per-channel min-max.
        out = np.zeros_like(arr, dtype=np.uint8)
        for c in range(3):
            ch = arr[c]
            lo, hi = ch.min(), ch.max()
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            else:
                ch = np.zeros_like(ch)
            out[c] = np.clip(ch, 0, 255).astype(np.uint8)
        arr = out
    else:  # percentile (default)
        # Per-channel 2-98% stretch — bulk of histogram across [0, 255],
        # outliers clipped. Good visual default for aerial imagery.
        out = np.zeros_like(arr, dtype=np.uint8)
        for c in range(3):
            ch = arr[c]
            lo, hi = np.percentile(ch, [2, 98])
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            else:
                ch = np.zeros_like(ch)
            out[c] = np.clip(ch, 0, 255).astype(np.uint8)
        arr = out

    # [3, H, W] → [H, W, 3] for PIL
    rgb = np.transpose(arr, (1, 2, 0))

    img = Image.fromarray(rgb, mode="RGB")
    img.save(dst_png, format="PNG", optimize=True)
    return True


# =============================================================================
# Per-patch export
# =============================================================================

def export_patch(
    patch_id: str,
    fractal_root: Path,
    ortho_index: dict,
    output_dir: Path,
    lut: np.ndarray,
    center_xy: bool = True,
    ortho_stretch: str = "percentile",
):
    """
    Convert one patch to .ply + .png (RGB ortho) under output_dir.

    Args:
        center_xy:     If True, subtract the patch's median X/Y so the cloud
                       is centered around (0, 0) — easier to navigate in
                       Blender than absolute Lambert-93 coordinates.
        ortho_stretch: How to map ortho TIFF to 8-bit PNG. Options:
                       'percentile' (default), 'minmax', 'uint8'.
    """
    print(f"\n[export] Processing {patch_id}")

    # ── Locate the LAZ ────────────────────────────────────
    # FRACTAL shards into 80 numeric subdirs; we search all train splits.
    train_root = fractal_root / "train" / "train"
    laz_matches = list(train_root.rglob(f"{patch_id}.laz"))
    if not laz_matches:
        # Try val/test too in case it's not from train
        for alt in ("val/val", "test/test"):
            laz_matches = list((fractal_root / alt).rglob(f"{patch_id}.laz"))
            if laz_matches:
                break
    if not laz_matches:
        print(f"[export]   ERROR: no LAZ found for {patch_id}")
        return False
    if len(laz_matches) > 1:
        print(f"[export]   WARN: multiple LAZ matches, using first: {laz_matches[0]}")
    laz_path = laz_matches[0]
    print(f"[export]   LAZ:  {laz_path}")

    # ── Read points ───────────────────────────────────────
    las = laspy.read(str(laz_path))
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    classification = np.asarray(las.classification, dtype=np.int64)
    N = xyz.shape[0]
    print(f"[export]   {N:,} points")

    # ── Apply cleaned LUT ─────────────────────────────────
    clipped = np.clip(classification, 0, MAX_LAS_CODE)
    fractal_class = lut[clipped]  # [N] uint8

    # Per-class point counts (for the legend file)
    classes_unique, counts = np.unique(fractal_class, return_counts=True)
    counts_per_class = {int(c): int(n) for c, n in zip(classes_unique, counts)}

    # ── Build RGB per point ───────────────────────────────
    # Map each FRACTAL class index → its color tuple
    rgb = np.zeros((N, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        mask = fractal_class == c
        if mask.any():
            rgb[mask] = color

    # ── Center coordinates (optional, recommended for Blender) ──
    if center_xy:
        cx = float(np.median(xyz[:, 0]))
        cy = float(np.median(xyz[:, 1]))
        # Z is kept absolute — elevation is meaningful
        xyz_centered = xyz.copy()
        xyz_centered[:, 0] -= cx
        xyz_centered[:, 1] -= cy
        xyz_out = xyz_centered.astype(np.float32)
        print(f"[export]   centered XY around ({cx:.1f}, {cy:.1f}) Lambert-93")
    else:
        xyz_out = xyz.astype(np.float32)

    # ── Write PLY ─────────────────────────────────────────
    ply_path = output_dir / f"{patch_id}.ply"
    write_ply(ply_path, xyz_out, rgb, fractal_class)
    print(f"[export]   PLY:  {ply_path} ({N:,} verts)")

    # ── Convert IRGB ortho → PNG ──────────────────────────
    # Skip NIR, take RGB bands, stretch to 8-bit, save as PNG. Image
    # viewers (and Blender's Reference image loader) handle PNG correctly
    # out of the box, no fiddling with band ordering.
    if patch_id in ortho_index:
        src_tif = ortho_index[patch_id]
        dst_png = output_dir / f"{patch_id}_ortho.png"
        ok = convert_ortho_to_png(src_tif, dst_png, stretch=ortho_stretch)
        if ok:
            print(f"[export]   PNG:  {dst_png} (RGB, "
                  f"stretch={ortho_stretch})")
        else:
            # Fallback: copy raw TIFF if conversion failed (no PIL/rasterio)
            dst_tif = output_dir / f"{patch_id}_ortho{src_tif.suffix}"
            shutil.copy2(src_tif, dst_tif)
            print(f"[export]   TIF:  {dst_tif} (PNG conversion unavailable)")
    else:
        print(f"[export]   WARN: no matching IRGB ortho found for {patch_id}")

    # ── Write legend ──────────────────────────────────────
    legend_path = output_dir / f"{patch_id}_legend.txt"
    with open(legend_path, "w") as f:
        f.write(f"Patch ID: {patch_id}\n")
        f.write(f"Source LAZ: {laz_path}\n")
        f.write(f"Total points: {N:,}\n\n")
        f.write("Class → Color (RGB 0-255) → Point count\n")
        f.write("-" * 60 + "\n")
        for c in range(NUM_CLASSES):
            color = CLASS_COLORS.get(c, (0, 0, 0))
            count = counts_per_class.get(c, 0)
            pct = 100.0 * count / N if N > 0 else 0
            f.write(f"  {c}  {CLASS_NAMES[c]:20s}  "
                    f"RGB={color}  {count:7d} pts ({pct:5.2f}%)\n")
        ign_count = counts_per_class.get(IGNORE_INDEX, 0)
        if ign_count > 0:
            ign_pct = 100.0 * ign_count / N
            f.write(f"  -- ignore (codes 65/66/67) "
                    f"RGB={CLASS_COLORS[IGNORE_INDEX]}  "
                    f"{ign_count:7d} pts ({ign_pct:5.2f}%)\n")
        f.write("\nBlender notes:\n")
        f.write("  - Import via File → Import → Stanford (.ply)\n")
        f.write("  - Enable vertex colors: Object Properties → Viewport Display\n")
        f.write("    → Color → set to 'Vertex' (Blender 3.x: 'Attribute')\n")
        f.write("  - For rendering: shader → Attribute node with name 'Col'\n")
        if center_xy:
            f.write("  - Note: XY coordinates centered around 0,0 for navigation.\n")
            f.write("    Z is absolute elevation (Lambert-93 NGF-IGN69).\n")
    print(f"[export]   LEG:  {legend_path}")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fractal_root",
        type=str,
        required=True,
        help="Path to FRACTAL data root (containing train/train/{00..79}/*.laz).",
    )
    parser.add_argument(
        "--irgb_root",
        type=str,
        required=True,
        help="Path to FRACTAL-IRGB root (containing train/, val/, test/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures",
        help="Where to write .ply / .tif / legend files.",
    )
    parser.add_argument(
        "--patch_ids",
        nargs="+",
        required=True,
        help="One or more patch IDs (e.g., TRAIN-0436_6414-002994543).",
    )
    parser.add_argument(
        "--no_center",
        action="store_true",
        help="Skip centering XY around the median (keep absolute Lambert-93 coords).",
    )
    parser.add_argument(
        "--ortho_stretch",
        type=str,
        choices=("percentile", "minmax", "uint8"),
        default="percentile",
        help="How to convert ortho TIFF channels to 8-bit PNG. 'percentile' "
             "stretches the 2-98%% range across [0, 255] (best visualization). "
             "'minmax' is similar but uses absolute min/max (sensitive to "
             "outliers). 'uint8' assumes input is already uint8 and just "
             "copies the bytes (use if the source is unstretched 8-bit data).",
    )
    args = parser.parse_args()

    fractal_root = Path(args.fractal_root)
    irgb_root    = Path(args.irgb_root)
    output_dir   = Path(args.output_dir)

    if not fractal_root.is_dir():
        print(f"ERROR: --fractal_root not a directory: {fractal_root}")
        sys.exit(1)
    if not irgb_root.is_dir():
        print(f"WARN:  --irgb_root not a directory: {irgb_root}")
        print(f"       Will skip ortho copies.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] Output dir: {output_dir.resolve()}")

    # Build ortho index once for all requested patches
    ortho_index = build_ortho_index(irgb_root) if irgb_root.is_dir() else {}

    # Build cleaned LUT once
    lut = build_remap_lut()

    n_ok = 0
    for pid in args.patch_ids:
        ok = export_patch(
            patch_id=pid,
            fractal_root=fractal_root,
            ortho_index=ortho_index,
            output_dir=output_dir,
            lut=lut,
            center_xy=not args.no_center,
            ortho_stretch=args.ortho_stretch,
        )
        n_ok += int(ok)

    print(f"\n[main] Done. {n_ok}/{len(args.patch_ids)} patches exported "
          f"to {output_dir.resolve()}")


if __name__ == "__main__":
    main()