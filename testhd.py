"""
FLAIR-HUB Stratified Subset Selector
======================================

Selects a class-balanced subset of FLAIR-HUB training patches for compute-efficient
experiments while preserving the natural class distribution.

Selection strategy
------------------
Two-stage:
  1. RARE-CLASS GUARANTEE: hard-include patches that contain the rarest classes
     (e.g., ligneous, mixed, swimming_pool). This prevents random subsampling
     from erasing rare classes entirely.
  2. STRATIFIED REMAINDER: pick remaining patches via weighted sampling that
     gives each patch a weight proportional to how much it contributes to
     the natural class distribution. The result preserves train-set
     class proportions while keeping rare classes well-represented.

Output is reproducible (fixed seed) and is meant to be loaded by all model
training scripts (Atomizer + baselines) to ensure identical subsets.

Usage:
    python select_flair_subset.py \\
        --label_stats ./data/FLAIR-HUB/label_stats.json \\
        --output ./data/FLAIR-HUB/subset_indices.json \\
        --figure_dir ./figures \\
        --target_size 30000 \\
        --rare_class_quota 100 \\
        --seed 42

Outputs:
    - subset_indices.json: {"train_patch_ids": [list of patch_ids],
                            "val_patch_ids": [...], "test_patch_ids": [...]}
                           (val/test are full sets, included for convenience)
    - figures/flair_subset_distribution.png: comparison plot (full vs subset).
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Same constants as precompute script
NUM_CLASSES = 19
CLASS_NAMES = [
    "building", "greenhouse", "swimming_pool", "impervious", "pervious",
    "bare_soil", "water", "snow", "herbaceous", "agricultural",
    "plowed", "vineyard", "deciduous", "coniferous", "brushwood",
    "clear_cut", "ligneous", "mixed", "undefined",
]


def load_label_stats(path: str) -> dict:
    """Load the JSON output of precompute_flair_label_stats.py."""
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Selection logic
# ─────────────────────────────────────────────────────────────────────

def compute_class_rarity_scores(global_totals: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency weights per class. Used to score patches by their
    "rare class content" — patches with rare-class pixels get higher weight.

    Formula: weight_c = 1 / max(freq_c, eps)
    Then L1-normalized so sum to 1.
    """
    eps = 1e-9
    freq = global_totals / max(global_totals.sum(), 1.0)
    rarity = 1.0 / np.maximum(freq, eps)
    rarity /= rarity.sum()
    return rarity


def select_subset(
    counts_per_patch: dict,
    target_size: int,
    rare_class_indices: list,
    rare_class_quota: int,
    seed: int = 42,
) -> tuple:
    """
    Two-stage stratified selection.

    Stage 1: For each class in rare_class_indices, find the top
             `rare_class_quota` patches that contain the most pixels of that
             class. Hard-include them.
    Stage 2: For the remaining slots, sample patches by weighted probability.
             Weight = sum_c (rarity_c * pixel_count_c) / total_pixels_in_patch.
             This favors patches that are rich in rare classes while still
             reflecting the natural distribution.

    Args:
        counts_per_patch: {patch_id: [counts per class]}
        target_size:      total number of patches to select
        rare_class_indices: list of class indices to guarantee
        rare_class_quota:   how many patches per rare class to hard-include
        seed:               RNG seed for reproducibility

    Returns:
        selected_patch_ids: list of selected patch_ids (length = target_size)
        selection_metadata: dict with selection stats (per-class counts, etc.)
    """
    rng = np.random.default_rng(seed)

    patch_ids   = list(counts_per_patch.keys())
    n_total     = len(patch_ids)
    counts_arr  = np.asarray(
        [counts_per_patch[pid] for pid in patch_ids], dtype=np.int64
    )  # [n_total, NUM_CLASSES]

    if n_total <= target_size:
        print(f"[Select] Dataset has {n_total} patches ≤ target {target_size}. "
              f"Returning all.")
        return list(patch_ids), {
            "n_selected": n_total,
            "stage1_count": 0,
            "stage2_count": 0,
        }

    # Index lookup for fast set ops on selected patches
    pid_to_idx = {pid: i for i, pid in enumerate(patch_ids)}

    # ── Stage 1: rare-class guarantee ────────────────────────────────
    # For each rare class, find patches with the most pixels of that class
    # and add them to the guaranteed set.
    selected_idx_set = set()
    stage1_picks = {c: [] for c in rare_class_indices}

    for c in rare_class_indices:
        # Patches sorted by pixel-count of class c (descending).
        # Skip patches already selected.
        order = np.argsort(-counts_arr[:, c])
        added = 0
        for idx in order:
            if added >= rare_class_quota:
                break
            if counts_arr[idx, c] == 0:
                # No more patches contain this class at all
                break
            if idx not in selected_idx_set:
                selected_idx_set.add(int(idx))
                stage1_picks[c].append(patch_ids[idx])
                added += 1

    print(f"[Select] Stage 1: hard-included {len(selected_idx_set)} patches "
          f"for rare-class guarantee.")
    for c in rare_class_indices:
        n = len(stage1_picks[c])
        total_pixels_c = sum(counts_per_patch[pid][c] for pid in stage1_picks[c])
        print(f"   class {c:2d} {CLASS_NAMES[c]:<14}  +{n:>4} patches  "
              f"({total_pixels_c:,} px of class)")

    # ── Stage 2: weighted sample for remainder ───────────────────────
    remaining_target = target_size - len(selected_idx_set)
    if remaining_target <= 0:
        print(f"[Select] Stage 1 alone exceeds target. Truncating.")
        selected_indices = sorted(selected_idx_set)[:target_size]
        return [patch_ids[i] for i in selected_indices], {
            "n_selected": target_size,
            "stage1_count": target_size,
            "stage2_count": 0,
        }

    # Weight = patch's contribution to weighted-by-rarity total
    global_totals = counts_arr.sum(axis=0)
    rarity = compute_class_rarity_scores(global_totals)
    # Score each patch: sum_c rarity_c * count_c
    patch_scores = (counts_arr * rarity[None, :]).sum(axis=1)

    # Mask out already-selected patches
    available_mask = np.ones(n_total, dtype=bool)
    for idx in selected_idx_set:
        available_mask[idx] = False

    available_idx    = np.where(available_mask)[0]
    available_scores = patch_scores[available_idx]

    # Add small epsilon so patches with 0 score still have nonzero probability
    # (otherwise patches that are 100% "undefined" can never be sampled).
    eps = available_scores.max() * 1e-3 + 1e-9
    available_probs = available_scores + eps
    available_probs = available_probs / available_probs.sum()

    sampled_idx_in_available = rng.choice(
        len(available_idx), size=remaining_target,
        replace=False, p=available_probs,
    )
    sampled_indices = available_idx[sampled_idx_in_available]
    selected_idx_set.update(int(i) for i in sampled_indices)

    selected_indices = sorted(selected_idx_set)
    selected_patch_ids = [patch_ids[i] for i in selected_indices]

    print(f"[Select] Stage 2: sampled {remaining_target} additional patches "
          f"from {len(available_idx)} candidates.")
    print(f"[Select] Total selected: {len(selected_patch_ids)} "
          f"(target was {target_size}).")

    return selected_patch_ids, {
        "n_selected":   len(selected_patch_ids),
        "stage1_count": target_size - remaining_target,
        "stage2_count": remaining_target,
    }


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_subset_comparison_all_splits(
    full_totals_per_split: dict,
    subset_totals_per_split: dict,
    full_n_per_split: dict,
    subset_n_per_split: dict,
    output_path: str,
):
    """
    3×2 grid: rows = train/val/test, cols = full vs subset.
    Same y-axis (class names) and color scheme across all panels.
    """
    splits = ["train", "validation", "test"]
    splits = [s for s in splits if s in full_totals_per_split]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(NUM_CLASSES)]
    y_pos  = np.arange(NUM_CLASSES)

    fig, axes = plt.subplots(
        len(splits), 2,
        figsize=(14, 4 * len(splits) + 1),
        sharey=True,
    )
    if len(splits) == 1:
        axes = axes.reshape(1, 2)

    # Determine global x-axis max so all panels are comparable
    max_pct = 0.0
    for split in splits:
        for totals in (full_totals_per_split[split],
                       subset_totals_per_split[split]):
            tp = totals.sum()
            if tp > 0:
                max_pct = max(max_pct, (100.0 * totals / tp).max())
    max_pct = max(max_pct * 1.15, 1.0)

    for row_idx, split in enumerate(splits):
        for col_idx, (totals, n_patches, label) in enumerate([
            (full_totals_per_split[split],
             full_n_per_split[split],   f"{split} — full"),
            (subset_totals_per_split[split],
             subset_n_per_split[split], f"{split} — subset"),
        ]):
            ax = axes[row_idx, col_idx]
            total_pixels = totals.sum()
            fractions = (100.0 * totals / total_pixels
                         if total_pixels > 0 else np.zeros(NUM_CLASSES))

            ax.barh(y_pos, fractions, color=colors,
                    edgecolor="black", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Pixel %")
            ax.set_title(f"{label}\n(N={n_patches:,} patches, "
                         f"{total_pixels:,} pixels)", fontsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.set_xlim(0, max_pct)
            for i, frac in enumerate(fractions):
                label_str = f"{frac:.2f}%" if frac >= 0.01 else "<0.01%"
                ax.text(frac + 0.5, i, label_str, va="center", fontsize=6.5)

    fig.suptitle(
        "FLAIR-HUB class distributions: full vs stratified subset",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved comparison to: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select stratified FLAIR-HUB train/val/test subsets.")
    parser.add_argument("--label_stats", type=str,
                        default="./data/FLAIR-HUB/label_stats.json")
    parser.add_argument("--output",      type=str,
                        default="./data/FLAIR-HUB/subset_indices.json")
    parser.add_argument("--figure_dir",  type=str, default="./figures")
    parser.add_argument("--train_size",       type=int, default=30000,
                        help="Target number of train patches.")
    parser.add_argument("--val_size",         type=int, default=5000,
                        help="Target number of val patches. "
                             "Set 0 or -1 to keep full val.")
    parser.add_argument("--test_size",        type=int, default=10000,
                        help="Target number of test patches. "
                             "Set 0 or -1 to keep full test.")
    parser.add_argument("--rare_class_quota_train", type=int, default=100,
                        help="Min patches to guarantee per rare class in TRAIN.")
    parser.add_argument("--rare_class_quota_eval",  type=int, default=30,
                        help="Min patches to guarantee per rare class in VAL/TEST.")
    parser.add_argument("--rare_threshold",   type=float, default=1.0,
                        help="Classes with pixel-fraction below this %% are "
                             "considered rare and get the guarantee.")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    # Load precomputed stats
    print(f"Loading label stats from: {args.label_stats}")
    stats = load_label_stats(args.label_stats)
    print(f"  Loaded splits: {list(stats['splits'].keys())}")

    # ── Identify rare classes from the train split ───────────────────
    train_split = stats["splits"].get("train")
    if train_split is None:
        raise KeyError("No 'train' split in label_stats.json")

    train_totals     = np.asarray(train_split["totals"], dtype=np.int64)
    train_n_patches  = train_split["n_patches"]
    train_total_px   = train_totals.sum()

    train_pct = 100.0 * train_totals / train_total_px

    rare_class_indices = [
        i for i in range(NUM_CLASSES)
        if 0 < train_pct[i] < args.rare_threshold
    ]
    print(f"\n[Rare-class detection] Threshold: <{args.rare_threshold}%")
    print(f"  Rare classes ({len(rare_class_indices)}):")
    for i in rare_class_indices:
        print(f"   {i:2d} {CLASS_NAMES[i]:<14}  {train_pct[i]:.4f}%")

    # ── Per-split selection ──────────────────────────────────────────
    split_target_size = {
        "train":      args.train_size,
        "validation": args.val_size,
        "test":       args.test_size,
    }
    split_quota = {
        "train":      args.rare_class_quota_train,
        "validation": args.rare_class_quota_eval,
        "test":       args.rare_class_quota_eval,
    }

    selected_per_split = {}     # {split: [patch_ids]}
    totals_per_split   = {}     # {split: np.array([NUM_CLASSES])}
    full_totals_per_split    = {}
    full_n_patches_per_split = {}
    metadata_per_split = {}

    for split_name in ("train", "validation", "test"):
        split = stats["splits"].get(split_name)
        if split is None:
            print(f"\n[WARN] Split '{split_name}' missing — skipping.")
            continue

        full_totals_per_split[split_name]    = np.asarray(
            split["totals"], dtype=np.int64)
        full_n_patches_per_split[split_name] = split["n_patches"]

        target = split_target_size[split_name]
        if target <= 0 or target >= split["n_patches"]:
            print(f"\n[Select] {split_name}: keeping full split "
                  f"({split['n_patches']} patches).")
            selected_per_split[split_name] = list(
                split["counts_per_patch"].keys())
            totals_per_split[split_name]   = full_totals_per_split[split_name]
            metadata_per_split[split_name] = {
                "selected": split["n_patches"], "stage1": 0, "stage2": 0,
            }
            continue

        print(f"\n=== Selecting {split_name} subset ===")
        print(f"  Full size: {split['n_patches']:,}  →  Target: {target:,}")
        print(f"  Rare-class quota: {split_quota[split_name]} per class")

        selected_ids, meta = select_subset(
            split["counts_per_patch"],
            target_size=target,
            rare_class_indices=rare_class_indices,
            rare_class_quota=split_quota[split_name],
            seed=args.seed,
        )

        # Compute subset totals
        subset_totals = np.zeros(NUM_CLASSES, dtype=np.int64)
        for pid in selected_ids:
            subset_totals += np.asarray(
                split["counts_per_patch"][pid], dtype=np.int64)

        selected_per_split[split_name] = selected_ids
        totals_per_split[split_name]   = subset_totals
        metadata_per_split[split_name] = {
            "selected": len(selected_ids),
            "stage1":   meta["stage1_count"],
            "stage2":   meta["stage2_count"],
        }

        # Print per-split distribution comparison
        full_pct   = 100.0 * full_totals_per_split[split_name] / max(
            full_totals_per_split[split_name].sum(), 1)
        subset_pct = 100.0 * subset_totals / max(subset_totals.sum(), 1)
        print(f"\n  [{split_name}] full vs subset:")
        print(f"    {'class':<14}  {'full %':>8}  {'subset %':>9}  {'delta':>7}")
        for i in range(NUM_CLASSES):
            delta = subset_pct[i] - full_pct[i]
            print(f"    {CLASS_NAMES[i]:<14}  {full_pct[i]:>7.3f}%  "
                  f"{subset_pct[i]:>8.3f}%  {delta:>+6.3f}%")

    # ── Save subset indices JSON ─────────────────────────────────────
    output_payload = {
        "train_patch_ids": selected_per_split.get("train", []),
        "val_patch_ids":   selected_per_split.get("validation", []),
        "test_patch_ids":  selected_per_split.get("test", []),
        "metadata": {
            "train_size":           args.train_size,
            "val_size":             args.val_size,
            "test_size":            args.test_size,
            "rare_class_quota_train": args.rare_class_quota_train,
            "rare_class_quota_eval":  args.rare_class_quota_eval,
            "rare_threshold_pct":   args.rare_threshold,
            "rare_class_indices":   rare_class_indices,
            "seed":                 args.seed,
            "per_split":            metadata_per_split,
            "full_train_n_patches": full_n_patches_per_split.get("train", 0),
            "full_val_n_patches":   full_n_patches_per_split.get("validation", 0),
            "full_test_n_patches":  full_n_patches_per_split.get("test", 0),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[Saved] Subset indices to: {args.output}")
    print(f"  train: {len(output_payload['train_patch_ids']):,}  "
          f"val: {len(output_payload['val_patch_ids']):,}  "
          f"test: {len(output_payload['test_patch_ids']):,}")

    # ── Plot full vs subset for each split ───────────────────────────
    figure_path = os.path.join(args.figure_dir, "flair_subset_distribution.png")
    plot_subset_comparison_all_splits(
        full_totals_per_split, totals_per_split,
        full_n_patches_per_split,
        {k: len(v) for k, v in selected_per_split.items()},
        figure_path,
    )


if __name__ == "__main__":
    main()