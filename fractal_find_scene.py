"""
FRACTAL — Top Scenes by Class
==============================

Scans LAZ files in the test split and finds:
  - Top-K scenes with most BUILDING points  (LAS class 6)
  - Top-K scenes with most BRIDGE points    (LAS class 17)

Reads full point classifications (not just headers) but needs no model,
no tokenization, no GPU.

Usage
-----
    python script_find_good_scene.py \
        --root_path ./data \
        --top_k 5 \
        --out_csv good_scenes.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import laspy
from tqdm import tqdm

# LAS classification codes for classes of interest
LAS_BUILDING = 6
LAS_BRIDGE   = 17

def main():
    parser = argparse.ArgumentParser(description="Find FRACTAL scenes by class")
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument("--out_csv",   type=str, default="good_scenes.csv")
    args = parser.parse_args()

    SPLIT_DIRS = {
        "train": "train/train",
        "val":   "val/val",
        "test":  "test/test",
    }
    laz_root = Path(args.root_path) / "FRACTAL" / "data" / SPLIT_DIRS[args.split]
    if not laz_root.exists():
        raise FileNotFoundError(f"LAZ root not found: {laz_root}")

    laz_files = sorted(laz_root.rglob("*.laz"))
    print(f"[Finder] {len(laz_files)} LAZ files — scanning classifications...")

    results = []
    for idx, path in enumerate(tqdm(laz_files)):
        las = laspy.read(path)
        cls = np.asarray(las.classification, dtype=np.int64)
        n_building = int((cls == LAS_BUILDING).sum())
        n_bridge   = int((cls == LAS_BRIDGE).sum())
        n_total    = len(cls)
        results.append((idx, path.stem, n_total, n_building, n_bridge))

    # ── Top-K buildings ───────────────────────────────────────────────
    top_buildings = sorted(results, key=lambda x: x[3], reverse=True)[:args.top_k]

    # ── Top-K bridges ─────────────────────────────────────────────────
    top_bridges   = sorted(results, key=lambda x: x[4], reverse=True)[:args.top_k]

    # ── Print ─────────────────────────────────────────────────────────
    def print_table(title, rows, sort_col, sort_label):
        print(f"\n{'─'*65}")
        print(f"  {title}")
        print(f"{'─'*65}")
        print(f"{'Rank':<5} {'Idx':<8} {'Total':>10} {sort_label:>12}  PatchID")
        print(f"{'─'*65}")
        for rank, (idx, patch_id, n_total, n_build, n_bridge) in enumerate(rows, 1):
            val = n_build if sort_col == "building" else n_bridge
            print(f"{rank:<5} {idx:<8} {n_total:>10,} {val:>12,}  {patch_id}")

    print_table("TOP BUILDING SCENES", top_buildings, "building", "Building pts")
    print_table("TOP BRIDGE SCENES",   top_bridges,   "bridge",   "Bridge pts")

    # ── Save CSV ──────────────────────────────────────────────────────
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "rank", "scene_idx", "patch_id",
                         "n_total", "n_building", "n_bridge"])
        for rank, (idx, pid, nt, nb, nbr) in enumerate(top_buildings, 1):
            writer.writerow(["building", rank, idx, pid, nt, nb, nbr])
        for rank, (idx, pid, nt, nb, nbr) in enumerate(top_bridges, 1):
            writer.writerow(["bridge",   rank, idx, pid, nt, nb, nbr])

    print(f"\n[Finder] Saved to {args.out_csv}")
    print(f"[Finder] Pass scene_idx values to script_pca_fractal.py "
          f"via --scene_indices")


if __name__ == "__main__":
    main()
