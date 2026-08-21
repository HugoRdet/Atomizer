"""
Per-class pixel-count comparison across MADOS train/val/test splits.

Mirrors the label-loading logic in MADOSDataset.__getitem__:
    label = rasterio.read(1) - 1          # shift to 0-indexed
    label[label == -1] = IGNORE_INDEX     # original 0 -> ignore
Assumes NUM_CLASSES = 15 (raw label values 1..15 in the source files,
post-shift 0..14). Reports per-class pixel share for each split, plus
per-split sample (scene) counts.
"""

import os
import numpy as np
import rasterio
from glob import glob

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ROOT = "./data/MADOS"
NUM_CLASSES = 15
IGNORE_INDEX = 255

SPLIT_FILES = {
    "train": "train_X.txt",
    "val":   "val_X.txt",
    "test":  "test_X.txt",
}


def discover_label_paths(split_key: str):
    split_file = os.path.join(ROOT, "splits", SPLIT_FILES[split_key])
    rois_split = np.genfromtxt(split_file, dtype="str")
    if rois_split.ndim == 0:
        rois_split = {str(rois_split)}
    else:
        rois_split = set(rois_split.tolist())

    label_paths = []
    tiles = sorted(glob(os.path.join(ROOT, "Scene_*")))
    for tile in tiles:
        tile_name = os.path.basename(tile)
        cl_files = glob(os.path.join(tile, "10", "*_cl_*"))
        for cl_file in cl_files:
            crop_suffix = os.path.basename(cl_file).split("_cl_")[-1]
            crop_name = tile_name + "_" + crop_suffix.split(".tif")[0]
            if crop_name in rois_split:
                label_paths.append(cl_file)
    return label_paths


def class_counts_for_split(split_key: str):
    paths = discover_label_paths(split_key)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignore_count = 0
    total = 0

    for p in paths:
        with rasterio.open(p) as src:
            lbl = src.read(1).astype(np.int64)
        lbl = lbl - 1
        ignore_mask = (lbl == -1)
        ignore_count += ignore_mask.sum()
        valid = lbl[~ignore_mask]
        for c in range(NUM_CLASSES):
            counts[c] += (valid == c).sum()
        total += lbl.size

    return paths, counts, ignore_count, total


if __name__ == "__main__":
    results = {}
    for split_key in ["train", "val", "test"]:
        paths, counts, ignore_count, total = class_counts_for_split(split_key)
        results[split_key] = (paths, counts, ignore_count, total)

    print(f"{'Split':<8}{'#Scenes':<10}{'#PixelsTotal':<15}{'%Ignore':<10}")
    for split_key, (paths, counts, ignore_count, total) in results.items():
        print(f"{split_key:<8}{len(paths):<10}{total:<15}{100*ignore_count/total:<10.3f}")

    print()
    valid_totals = {k: v[1].sum() for k, v in results.items()}
    print(f"{'Class':<8}" + "".join(f"{k+' %':<12}" for k in results))
    for c in range(NUM_CLASSES):
        row = f"{c:<8}"
        for split_key, (paths, counts, ignore_count, total) in results.items():
            vt = valid_totals[split_key]
            pct = 100 * counts[c] / vt if vt > 0 else 0.0
            row += f"{pct:<12.3f}"
        print(row)
