"""
check_precompute_consistency.py
==================================

Scans every _latent_assign.npz sidecar in a tiled DALES directory and
reports the distribution of L_spatial values found. If more than one
distinct value shows up, your precomputed assignments are INCONSISTENT
across patches -- this is exactly what causes the
"expanded size of the tensor (X) must match existing size (Y)" crash in
GeographicPruningDales, since different patches end up feeding different
L_spatial into the same batched cell-building step.

Usage:
    python check_precompute_consistency.py --tiled_dir ./DALES_tiled/test
    python check_precompute_consistency.py --tiled_dir ./DALES_tiled/train
    python check_precompute_consistency.py --tiled_dir ./DALES_tiled/val
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiled_dir", type=str, required=True)
    args = parser.parse_args()

    tiled_dir = Path(args.tiled_dir)
    npz_files = sorted(tiled_dir.glob("*_latent_assign.npz"))
    if not npz_files:
        print(f"No _latent_assign.npz files found under {tiled_dir}")
        return

    print(f"Scanning {len(npz_files)} precomputed assignment files under "
          f"{tiled_dir}...")

    l_spatial_counter = Counter()
    lx_ly_counter = Counter()
    examples_per_l = {}

    for p in npz_files:
        with np.load(p) as npz:
            L = int(npz["L_spatial"])
            lx = int(npz["lx"])
            ly = int(npz["ly"])
        l_spatial_counter[L] += 1
        lx_ly_counter[(lx, ly)] += 1
        if L not in examples_per_l:
            examples_per_l[L] = []
        if len(examples_per_l[L]) < 3:
            examples_per_l[L].append(p.name)

    print(f"\nL_spatial value distribution across {len(npz_files)} patches:")
    for L, count in sorted(l_spatial_counter.items()):
        pct = 100.0 * count / len(npz_files)
        examples = ", ".join(examples_per_l[L])
        print(f"  L_spatial={L:>4d}: {count:>5d} patches ({pct:5.1f}%)  "
              f"e.g. {examples}")

    print(f"\n(lx, ly) grid dimension distribution:")
    for (lx, ly), count in sorted(lx_ly_counter.items()):
        print(f"  ({lx}, {ly}): {count} patches")

    if len(l_spatial_counter) > 1:
        print(f"\n*** INCONSISTENT: {len(l_spatial_counter)} distinct "
              f"L_spatial values found. This directory's precomputed "
              f"assignments were built at DIFFERENT times with different "
              f"tokens_per_latent/patch_size_m/max_lidar_points settings. "
              f"You need to re-run precompute_dales_latent_assignment.py "
              f"for ALL patches in this directory with a SINGLE consistent "
              f"set of parameters (matching your trained model's config) "
              f"before this crash goes away.")
    else:
        print(f"\nOK: all {len(npz_files)} patches share the same "
              f"L_spatial={list(l_spatial_counter.keys())[0]}. This "
              f"directory is internally consistent.")


if __name__ == "__main__":
    main()
