"""
FRACTAL Class-Balanced Sampling Weight Precompute
====================================================

Scans the FRACTAL training split and computes per-patch sampling weights
for use with torch.utils.data.WeightedRandomSampler. The weight scheme
matches Approach A from the design discussion:

    weight[patch] = max(class_boost[c] for c in classes_present_in_patch)

where:
    class_boost[c] = min(sqrt(N_total / N_class), clamp_max)

Rationale:
  - Patches containing rare classes (bridge, permanent_structure) receive
    high weight, increasing their sampling frequency.
  - Patches with only common classes (ground, vegetation) receive low
    weight (~1.6 for ground), keeping them in the training mix.
  - The clamp prevents extreme weights from one or two rarest-class
    patches dominating the sampling distribution.

Note on FRACTAL's pre-rebalancing:
  The official FRACTAL release is already class-rebalanced at construction
  time (raw water was 0.6%, raw bridge/permanent ~0.01% each, brought up
  to 0.5% / 0.13% / 0.04% in the curated split). This precompute adds a
  SECOND layer of rebalancing on top, oversampling rare-class patches
  during training. Expected gain: +2-5 macro mIoU from this intervention
  alone.

LAS code handling:
  Codes 65 (artefact noise) and 66 (synthetic gap-fill) are mapped to
  IGNORE_INDEX, NOT to permanent_structure. The previous LUT incorrectly
  mapped these to permanent (audited finding: ~16.5% of "permanent"
  labels were actually codes 65/66). This script uses the cleaned LUT.

Usage:
    python precompute_fractal_weights.py \\
        --fractal_root /path/to/FRACTAL/data \\
        --output /path/to/FRACTAL/data/fractal_class_weights.json \\
        --clamp_max 12 \\
        --num_workers 16

Output format (JSON):
    {
        "metadata": {
            "n_patches": 80123,
            "clamp_max": 12,
            "class_boosts": {"ground": 1.6, "bridge": 12.0, ...},
            "class_n_patches": {"ground": 80123, "bridge": 4012, ...},
            ...
        },
        "weights": {
            "2125_6243-12": 12.0,
            "2155_6243-34": 1.6,
            ...
        }
    }
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed. Install with: pip install laspy")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[Warning] tqdm not installed; falling back to printed progress. "
          "Install with: pip install tqdm")


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

# LAS code → FRACTAL class index. Indices 65, 66, 67 are NOT mapped to
# permanent_structure — they're artefacts/synthetic data that should be
# ignored entirely. This matches the cleaned LUT in utils_dataset_fractal.py.
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
    # 65, 66, 67 → IGNORE (not present in map = mapped to IGNORE_INDEX below)
}

MAX_LAS_CODE = 255


def build_remap_lut() -> np.ndarray:
    """Build an LUT mapping LAS code → FRACTAL class (or IGNORE_INDEX)."""
    lut = np.full(MAX_LAS_CODE + 1, IGNORE_INDEX, dtype=np.uint8)
    for las_code, fractal_class in LAS_CODE_TO_FRACTAL_CLASS.items():
        lut[las_code] = fractal_class
    return lut


# =============================================================================
# Per-patch worker
# =============================================================================

def scan_one_patch(args):
    """
    Read one LAZ file, return (patch_id, set of FRACTAL classes present).

    Returns None on read failure (so we can filter out bad files).
    Designed to be called via Pool.imap for parallel scanning.
    """
    laz_path, lut = args
    try:
        las = laspy.read(str(laz_path))
        classification = np.asarray(las.classification, dtype=np.int64)
        # Clip extreme values then map via LUT
        clipped = np.clip(classification, 0, MAX_LAS_CODE)
        fractal_classes = lut[clipped]
        # Drop ignore labels — they don't contribute to class presence
        valid = fractal_classes != IGNORE_INDEX
        present = set(int(c) for c in np.unique(fractal_classes[valid]))
        patch_id = laz_path.stem
        return (patch_id, present)
    except Exception as e:
        print(f"[WARNING] Failed to read {laz_path}: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-patch sampling weights for FRACTAL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fractal_root",
        type=str,
        required=True,
        help="Path to FRACTAL data root (containing data/train/train/{00..79}/*.laz).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to {fractal_root}/fractal_class_weights.json.",
    )
    parser.add_argument(
        "--clamp_max",
        type=float,
        default=12.0,
        help="Maximum class boost. Caps sqrt(N_total/N_class) at this value.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel processes for LAZ scanning.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: scan only the first N patches (useful for dry runs).",
    )
    args = parser.parse_args()

    fractal_root = Path(args.fractal_root)
    if not fractal_root.is_dir():
        print(f"ERROR: --fractal_root not a directory: {fractal_root}")
        sys.exit(1)

    output_path = (Path(args.output) if args.output
                   else fractal_root / "fractal_class_weights.json")

    if output_path.exists() and not args.force:
        print(f"Output already exists: {output_path}")
        print(f"Use --force to overwrite.")
        sys.exit(0)

    # ── Find all training LAZ files ─────────────────────────
    # FRACTAL layout: {fractal_root}/train/train/{00..79}/{patch_id}.laz
    train_root = fractal_root / "train" / "train"
    if not train_root.is_dir():
        # Fall back to common alternative layouts
        for alt in ("data/train/train", "train"):
            cand = fractal_root / alt
            if cand.is_dir():
                train_root = cand
                break
        else:
            print(f"ERROR: train subdirectory not found under {fractal_root}")
            print(f"  Expected: {fractal_root}/train/train/")
            sys.exit(1)

    print(f"[scan] Scanning LAZ files under: {train_root}")
    t_start = time.time()
    laz_files = sorted(train_root.rglob("*.laz"))
    if not laz_files:
        print(f"ERROR: No .laz files found under {train_root}")
        sys.exit(1)
    print(f"[scan] Found {len(laz_files)} LAZ files "
          f"({time.time() - t_start:.1f}s)")

    if args.limit is not None:
        laz_files = laz_files[:args.limit]
        print(f"[scan] Limited to first {len(laz_files)} files")

    # ── Build the cleaned LUT ───────────────────────────────
    lut = build_remap_lut()
    print(f"[scan] LAS code → FRACTAL class LUT:")
    for las_code, fractal_class in sorted(LAS_CODE_TO_FRACTAL_CLASS.items()):
        print(f"         LAS {las_code:3d} → {fractal_class} "
              f"({CLASS_NAMES[fractal_class]})")
    print(f"         All other LAS codes → IGNORE ({IGNORE_INDEX})")

    # ── Parallel scan ───────────────────────────────────────
    print(f"[scan] Reading {len(laz_files)} files with "
          f"{args.num_workers} workers...")
    t_start = time.time()

    pool_args = [(p, lut) for p in laz_files]
    patch_classes = {}     # patch_id -> set of FRACTAL classes present
    patch_paths   = {}     # patch_id -> absolute LAZ path (for top-K listing)
    n_failed = 0

    # Build a {patch_id: path} index from the file list so we can resolve
    # paths after the parallel scan finishes (workers only return patch_id).
    for p in laz_files:
        patch_paths[p.stem] = str(p)

    # n_patches_per_class: number of patches that contain at least one point
    # of class c (after applying the cleaned LUT). Used to compute boosts.
    n_patches_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)

    with Pool(args.num_workers) as pool:
        result_iter = pool.imap_unordered(scan_one_patch, pool_args, chunksize=64)

        # Wrap with tqdm if available, else use sparse manual printing
        if HAS_TQDM:
            result_iter = tqdm(
                result_iter,
                total=len(laz_files),
                desc="[scan]",
                unit="patches",
                smoothing=0.1,         # smoothed rate estimate
                mininterval=0.5,       # don't refresh more than 2/sec
                dynamic_ncols=True,    # adapt to terminal width
            )

        for i, result in enumerate(result_iter):
            if result is None:
                n_failed += 1
                continue
            patch_id, classes_present = result
            patch_classes[patch_id] = classes_present
            for c in classes_present:
                n_patches_per_class[c] += 1

            # Fallback progress when tqdm unavailable
            if not HAS_TQDM and (i + 1) % 5000 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(laz_files) - i - 1) / rate
                print(f"[scan]   {i + 1}/{len(laz_files)} "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t_start
    print(f"[scan] Done in {elapsed:.1f}s "
          f"({len(patch_classes)} succeeded, {n_failed} failed)")

    n_patches = len(patch_classes)
    if n_patches == 0:
        print("ERROR: No patches successfully scanned.")
        sys.exit(1)

    # ── Print per-class patch coverage ──────────────────────
    print(f"\n[stats] Patch coverage per class:")
    for c in range(NUM_CLASSES):
        pct = 100.0 * n_patches_per_class[c] / n_patches
        print(f"          {c} ({CLASS_NAMES[c]:20s}): "
              f"{n_patches_per_class[c]:7d} patches ({pct:5.1f}%)")

    # ── Compute per-class boosts ────────────────────────────
    # Boost = sqrt(N_total / N_class_patches), clamped at clamp_max.
    # A class present in only a few patches gets a high boost.
    print(f"\n[boost] Computing per-class boosts (clamp_max={args.clamp_max}):")
    class_boosts = {}
    for c in range(NUM_CLASSES):
        if n_patches_per_class[c] == 0:
            print(f"          {CLASS_NAMES[c]:20s}: NO PATCHES — assigning boost=1.0")
            class_boosts[c] = 1.0
            continue
        raw_boost = math.sqrt(n_patches / n_patches_per_class[c])
        clamped = min(raw_boost, args.clamp_max)
        class_boosts[c] = clamped
        print(f"          {CLASS_NAMES[c]:20s}: raw={raw_boost:6.2f}  "
              f"→ clamped={clamped:5.2f}")

    # ── Compute per-patch weight = max(boost) over present classes ──
    print(f"\n[weight] Assigning per-patch weights (Approach A: "
          f"max-of-present-class-boosts)...")
    weights = {}
    weight_distribution = np.zeros(NUM_CLASSES + 1)  # one bucket per "winning" class
    for patch_id, classes_present in patch_classes.items():
        if not classes_present:
            # No valid points (all IGNORE) — assign a small fallback weight
            # so the patch isn't selected but isn't NaN/None either.
            weights[patch_id] = 1.0
            weight_distribution[NUM_CLASSES] += 1
            continue
        # Pick the class with highest boost; tie-break by class index (rarest first)
        winning_class = max(classes_present, key=lambda c: class_boosts[c])
        weights[patch_id] = float(class_boosts[winning_class])
        weight_distribution[winning_class] += 1

    print(f"\n[weight] Distribution of weights (which class "
          f"determined each patch's weight):")
    for c in range(NUM_CLASSES):
        pct = 100.0 * weight_distribution[c] / n_patches
        print(f"           weight {class_boosts[c]:5.2f} (rarest class={CLASS_NAMES[c]:20s}): "
              f"{int(weight_distribution[c]):7d} patches ({pct:5.1f}%)")
    if weight_distribution[NUM_CLASSES] > 0:
        pct = 100.0 * weight_distribution[NUM_CLASSES] / n_patches
        print(f"           weight 1.00 (no valid points)              : "
              f"{int(weight_distribution[NUM_CLASSES]):7d} patches ({pct:5.1f}%)")

    # ── Effective sampling ratios (sanity check) ────────────
    total_weight = sum(weights.values())
    print(f"\n[sanity] Effective sampling distribution under WeightedRandomSampler:")
    for c in range(NUM_CLASSES):
        # Sum of weights of patches that would be sampled in proportion to
        # how often class c determined the weight
        n_c = int(weight_distribution[c])
        w_c = class_boosts[c] * n_c
        share = 100.0 * w_c / total_weight if total_weight > 0 else 0.0
        share_orig = 100.0 * n_patches_per_class[c] / n_patches
        # Note: this "share" is "fraction of sampling weight assigned to
        # patches whose rarest class is c", not "fraction of samples
        # containing class c" — those differ because the sampler picks
        # patches with replacement.
        print(f"           {CLASS_NAMES[c]:20s}: "
              f"weight share={share:5.1f}%   (vs raw patch share={share_orig:5.1f}%)")

    # ── Weight summary statistics ───────────────────────────
    print(f"\n[summary] Weight distribution statistics:")
    weight_values = np.array(list(weights.values()), dtype=np.float64)
    print(f"            count        : {len(weight_values)}")
    print(f"            min          : {weight_values.min():.4f}")
    print(f"            max          : {weight_values.max():.4f}")
    print(f"            mean         : {weight_values.mean():.4f}")
    print(f"            median       : {float(np.median(weight_values)):.4f}")
    print(f"            std          : {weight_values.std():.4f}")
    print(f"            sum          : {weight_values.sum():.4f}")
    # Percentiles useful for spotting heavy tails
    for q in (25, 50, 75, 90, 95, 99):
        v = float(np.percentile(weight_values, q))
        print(f"            p{q:02d}          : {v:.4f}")
    # Ratio between top and bottom weight tells us how aggressive the
    # rebalancing actually is.
    if weight_values.min() > 0:
        ratio = weight_values.max() / weight_values.min()
        print(f"            max / min    : {ratio:.2f}x   "
              f"(rarest-class patch is sampled this many times more than "
              f"common-only patch)")

    # ── Top-K patches by weight ─────────────────────────────
    TOP_K = 3
    print(f"\n[top {TOP_K}] Patches with the highest weight "
          f"(useful for inspecting what triggers the rare-class boost):")

    # Sort patches by weight, descending. Ties broken alphabetically by
    # patch_id for reproducibility.
    sorted_patches = sorted(
        weights.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    for rank, (patch_id, w) in enumerate(sorted_patches[:TOP_K], start=1):
        classes_present = patch_classes.get(patch_id, set())
        # The "winning" class is the rarest one in the patch (highest boost)
        if classes_present:
            winning_class = max(classes_present, key=lambda c: class_boosts[c])
            winning_name = CLASS_NAMES[winning_class]
        else:
            winning_name = "(none)"
        present_names = sorted(CLASS_NAMES[c] for c in classes_present)
        path = patch_paths.get(patch_id, "(path not found)")
        print(f"            #{rank}  weight={w:.3f}  "
              f"rarest_class={winning_name}")
        print(f"                  patch_id={patch_id}")
        print(f"                  path    ={path}")
        print(f"                  classes ={present_names}")

    # ── Write output ────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "n_patches":        n_patches,
            "n_failed_reads":   n_failed,
            "clamp_max":        args.clamp_max,
            "class_names":      CLASS_NAMES,
            "class_boosts":     {CLASS_NAMES[c]: class_boosts[c]
                                 for c in range(NUM_CLASSES)},
            "class_n_patches":  {CLASS_NAMES[c]: int(n_patches_per_class[c])
                                 for c in range(NUM_CLASSES)},
            "approach":         "A: weight=max(class_boost) over classes present in patch",
            "las_code_to_fractal_class": LAS_CODE_TO_FRACTAL_CLASS,
            "ignore_index":     IGNORE_INDEX,
            "weight_stats": {
                "min":    float(weight_values.min()),
                "max":    float(weight_values.max()),
                "mean":   float(weight_values.mean()),
                "median": float(np.median(weight_values)),
                "std":    float(weight_values.std()),
            },
        },
        "weights": weights,
    }

    print(f"\n[write] Writing {n_patches} weights to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[write] Done.")
    print(f"\n[done] Total time: {time.time() - t_start:.1f}s for "
          f"{n_patches} patches.")
    print(f"[done] Load in your training script with:")
    print(f"         with open('{output_path}') as f:")
    print(f"             data = json.load(f)")
    print(f"         patch_weights = data['weights']  # dict[patch_id, weight]")


if __name__ == "__main__":
    main()
