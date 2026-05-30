"""
FRACTAL z-value statistics diagnostic.

For a sample of train patches, compute:
  1. Global z distribution (raw Lambert-93 elevations across all points)
  2. Per-patch z range distribution (how variable is z within a 50m patch?)
  3. Per-class z statistics (what z range is typical for buildings vs ground?)
  4. Z relative to local ground median (proxy for "height above ground")

This informs the right normalization scheme for the decoder queries.
"""

import argparse
from pathlib import Path
import numpy as np
import laspy

# LAS -> FRACTAL 7-class remap (copy from utils_dataset_fractal.py to avoid imports)
LAS_TO_FRACTAL = {
    1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 9: 4, 17: 5,
    64: 6, 65: 6, 66: 6, 67: 6,
}
def _build_lut(num_codes=256, ignore=255):
    lut = np.full(num_codes, ignore, dtype=np.int64)
    for k, v in LAS_TO_FRACTAL.items():
        lut[k] = v
    return lut
REMAP = _build_lut()

FRACTAL_CLASSES = [
    "other", "ground", "vegetation", "building",
    "water", "bridge", "permanent_structure",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--n_patches", type=int, default=200,
                        help="Number of patches to sample from train.")
    args = parser.parse_args()

    laz_root = Path(args.root_path) / "FRACTAL" / "data" / "train" / "train"
    if not laz_root.exists():
        raise FileNotFoundError(f"LAZ root not found: {laz_root}")

    # Sample patches (one per subdirectory, up to n_patches)
    all_laz = sorted(laz_root.rglob("*.laz"))
    step = max(1, len(all_laz) // args.n_patches)
    sample = all_laz[::step][:args.n_patches]
    print(f"Sampling {len(sample)} patches out of {len(all_laz)} total.\n")

    # Accumulators
    all_z = []
    all_z_rel_ground = []   # z - per-patch ground median
    per_class_z = {i: [] for i in range(7)}
    per_class_z_rel = {i: [] for i in range(7)}
    patch_z_ranges = []
    patch_z_p1 = []
    patch_z_p99 = []

    for i, laz_path in enumerate(sample):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(sample)} patches")
        try:
            las = laspy.read(str(laz_path))
        except Exception as e:
            print(f"    skip {laz_path.name}: {e}")
            continue
        if las.x.shape[0] < 1000:
            continue

        z = np.asarray(las.z, dtype=np.float32)
        cls_raw = np.asarray(las.classification, dtype=np.int64)
        cls_raw = np.clip(cls_raw, 0, REMAP.shape[0] - 1)
        cls = REMAP[cls_raw]

        all_z.append(z)
        patch_z_ranges.append(float(z.max() - z.min()))
        patch_z_p1.append(float(np.percentile(z, 1)))
        patch_z_p99.append(float(np.percentile(z, 99)))

        # "Local ground" = median of ground points; if none, use z minimum
        ground_mask = (cls == 1)
        if ground_mask.sum() > 50:
            local_ground = float(np.median(z[ground_mask]))
        else:
            local_ground = float(np.percentile(z, 5))
        z_rel = z - local_ground
        all_z_rel_ground.append(z_rel)

        # Per-class stats (subsample to keep memory reasonable)
        for c in range(7):
            mask = (cls == c)
            if mask.sum() > 0:
                z_c = z[mask]
                z_rel_c = z_rel[mask]
                # Subsample if many
                if mask.sum() > 5000:
                    sel = np.random.choice(mask.sum(), 5000, replace=False)
                    z_c = z_c[sel]
                    z_rel_c = z_rel_c[sel]
                per_class_z[c].append(z_c)
                per_class_z_rel[c].append(z_rel_c)

    # Combine
    all_z = np.concatenate(all_z)
    all_z_rel_ground = np.concatenate(all_z_rel_ground)

    print(f"\n{'='*72}")
    print(f"  FRACTAL z-value statistics ({len(sample)} patches sampled)")
    print(f"{'='*72}")

    # ── (1) Global z distribution (raw Lambert-93 elevation) ─────────
    print(f"\n[1] GLOBAL Z (raw elevations, Lambert-93 meters):")
    print(f"    min={all_z.min():.1f}  max={all_z.max():.1f}")
    print(f"    mean={all_z.mean():.1f}  std={all_z.std():.1f}")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"    p{p}: {np.percentile(all_z, p):.1f}")

    # ── (2) Per-patch z range ─────────────────────────────────────────
    print(f"\n[2] PER-PATCH Z RANGE (z_max - z_min within a patch):")
    patch_z_ranges = np.array(patch_z_ranges)
    print(f"    min={patch_z_ranges.min():.2f}  max={patch_z_ranges.max():.2f}")
    print(f"    mean={patch_z_ranges.mean():.2f}  std={patch_z_ranges.std():.2f}")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"    p{p}: {np.percentile(patch_z_ranges, p):.2f}")

    print(f"\n[2b] PER-PATCH Z PERCENTILE 1 - PERCENTILE 99 (robust range):")
    robust_range = np.array(patch_z_p99) - np.array(patch_z_p1)
    print(f"    mean={robust_range.mean():.2f}  std={robust_range.std():.2f}")
    for p in (5, 25, 50, 75, 95):
        print(f"    p{p}: {np.percentile(robust_range, p):.2f}")

    # ── (3) Per-class raw z ───────────────────────────────────────────
    print(f"\n[3] PER-CLASS RAW Z (Lambert-93 elevations):")
    print(f"    {'class':<22}  {'count':>10}  {'p1':>7}  {'p50':>7}  {'p99':>7}  "
          f"{'mean':>7}")
    for c in range(7):
        if not per_class_z[c]:
            print(f"    {FRACTAL_CLASSES[c]:<22}  (no points)")
            continue
        z_c = np.concatenate(per_class_z[c])
        p1 = np.percentile(z_c, 1)
        p50 = np.percentile(z_c, 50)
        p99 = np.percentile(z_c, 99)
        print(f"    {FRACTAL_CLASSES[c]:<22}  {len(z_c):>10}  "
              f"{p1:>7.1f}  {p50:>7.1f}  {p99:>7.1f}  {z_c.mean():>7.1f}")

    # ── (4) Per-class z relative to local ground ─────────────────────
    print(f"\n[4] PER-CLASS Z RELATIVE TO LOCAL GROUND (z - ground_median_in_patch):")
    print(f"    This is the key statistic: 'height above local ground'.")
    print(f"    {'class':<22}  {'count':>10}  {'p1':>7}  {'p50':>7}  {'p99':>7}  "
          f"{'mean':>7}")
    for c in range(7):
        if not per_class_z_rel[c]:
            continue
        z_c = np.concatenate(per_class_z_rel[c])
        p1 = np.percentile(z_c, 1)
        p50 = np.percentile(z_c, 50)
        p99 = np.percentile(z_c, 99)
        print(f"    {FRACTAL_CLASSES[c]:<22}  {len(z_c):>10}  "
              f"{p1:>7.2f}  {p50:>7.2f}  {p99:>7.2f}  {z_c.mean():>7.2f}")

    # ── (5) Recommendation summary ───────────────────────────────────
    print(f"\n[5] NORMALIZATION RECOMMENDATIONS:")

    # Global z varies hugely (mountains vs coast). Per-patch percentile
    # clipping handles the global variation but loses absolute height info.
    abs_z_p99_p1 = np.percentile(all_z, 99) - np.percentile(all_z, 1)
    print(f"    - GLOBAL z range (p1-p99): {abs_z_p99_p1:.1f} m")
    print(f"      Way too much variation across patches (mountains -> sea level).")
    print(f"      Direct global normalization would lose all signal.")

    print(f"    - Per-patch percentile clipping (current approach): preserves")
    print(f"      WITHIN-patch z variation but loses 'absolute' meaning.")
    print(f"      A 5m bridge over 0m ground would have same normalized z as")
    print(f"      a 5m mound on flat ground. May be enough for classification.")

    all_z_rel = all_z_rel_ground
    print(f"    - 'Height above local ground' (z - ground_median):")
    print(f"      mean={all_z_rel.mean():.2f}  std={all_z_rel.std():.2f}")
    print(f"      p1={np.percentile(all_z_rel, 1):.2f}  "
          f"p99={np.percentile(all_z_rel, 99):.2f}")
    print(f"      Physically meaningful: ground=0, building=8-15m, bridge=3-10m.")
    print(f"      Suggested normalization: clip to [-2, 50] then scale to [-1, 1].")


if __name__ == "__main__":
    main()
