"""
FRACTAL LAS code 65/66 audit.

Counts how many points in FRACTAL train have LAS codes 65 (Artefact) or
66 (Synthetic). These are non-physical points (measurement noise + synthetic
gap-fill respectively) that should probably be excluded from training, not
merged into 'permanent_structure' as our current LUT does.

Reports:
  - Raw count and fraction of LAS codes 65, 66, 67 (if any)
  - Their fraction relative to true permanent_structure (code 64)
  - Whether the merge is significantly polluting permanent_structure
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import laspy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--n_patches", type=int, default=500)
    args = parser.parse_args()

    laz_root = Path(args.root_path) / "FRACTAL" / "data" / "train" / "train"
    all_laz = sorted(laz_root.rglob("*.laz"))
    step = max(1, len(all_laz) // args.n_patches)
    sample = all_laz[::step][:args.n_patches]

    print(f"Auditing {len(sample)} patches for LAS codes 64, 65, 66, 67...\n")

    code_counts = Counter()
    total_points = 0
    patches_with_64 = 0
    patches_with_65 = 0
    patches_with_66 = 0
    patches_with_67 = 0

    for i, laz_path in enumerate(sample):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample)}")
        try:
            las = laspy.read(str(laz_path))
        except Exception as e:
            print(f"    skip {laz_path.name}: {e}")
            continue
        cls = np.asarray(las.classification, dtype=np.int64)
        total_points += cls.shape[0]
        for c in (64, 65, 66, 67):
            n = int((cls == c).sum())
            code_counts[c] += n
        if (cls == 64).any(): patches_with_64 += 1
        if (cls == 65).any(): patches_with_65 += 1
        if (cls == 66).any(): patches_with_66 += 1
        if (cls == 67).any(): patches_with_67 += 1

    print(f"\n{'='*68}")
    print(f"  LAS code audit ({len(sample)} patches, {total_points:,} points)")
    print(f"{'='*68}\n")

    print(f"  {'code':<6}  {'name':<25}  {'count':>12}  {'frac %':>8}  {'patches':>8}")
    info = {
        64: ("Permanent structure",    patches_with_64),
        65: ("Artefact (noise)",       patches_with_65),
        66: ("Synthetic (gap-fill)",   patches_with_66),
        67: ("(spec: doesn't exist)",  patches_with_67),
    }
    for code in (64, 65, 66, 67):
        cnt = code_counts[code]
        frac = 100.0 * cnt / max(total_points, 1)
        name, n_patches = info[code]
        print(f"  {code:<6}  {name:<25}  {cnt:>12,}  {frac:>7.4f}%  {n_patches:>8}")

    # Compute pollution ratio
    real_permanent = code_counts[64]
    polluted = code_counts[65] + code_counts[66] + code_counts[67]
    if real_permanent > 0:
        ratio = 100.0 * polluted / real_permanent
        print(f"\n  Pollution ratio (codes 65/66/67 vs code 64): {ratio:.1f}%")
        print(f"  → Of points labeled 'permanent_structure' by current LUT,")
        print(f"    {100.0 * polluted / (real_permanent + polluted):.1f}% are non-physical "
              f"(noise/synthetic).")
    else:
        print(f"\n  No real permanent_structure points in this sample.")
        if polluted > 0:
            print(f"  But {polluted} points are being labeled as permanent_structure "
                  f"from codes 65/66/67!")

    print(f"\n  RECOMMENDATION:")
    if polluted > real_permanent * 0.1:
        print(f"  Significant pollution detected. Change the LUT to map")
        print(f"  codes 65, 66 to IGNORE_INDEX (255) instead of 6.")
        print(f"  This should improve permanent_structure IoU.")
    elif polluted > 0:
        print(f"  Small pollution detected. Worth fixing the LUT but probably")
        print(f"  won't dramatically change permanent_structure IoU.")
    else:
        print(f"  No pollution. Codes 65/66 don't appear in this sample.")
        print(f"  The current LUT mapping is functionally harmless but the")
        print(f"  semantic mapping is still incorrect — worth cleaning up.")


if __name__ == "__main__":
    main()
