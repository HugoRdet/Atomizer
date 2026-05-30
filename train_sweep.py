"""
FLAIR-HUB Subset Comparison Diagnostic
========================================

Compares two subset_indices.json files (or the current one against a freshly-
generated one) to determine whether subset selection is reproducible across
runs, and if not, which patches differ.

Usage:
    # Compare two existing JSON files:
    python compare_subset_indices.py \\
        --json1 ./data/FLAIR-HUB/subset_indices.json \\
        --json2 ./data/FLAIR-HUB/subset_indices_new.json

    # Or generate a fresh subset and compare against an existing one:
    python compare_subset_indices.py \\
        --json1 ./data/FLAIR-HUB/subset_indices.json \\
        --regenerate \\
        --label_stats ./data/FLAIR-HUB/label_stats.json \\
        --output_new ./data/FLAIR-HUB/subset_indices_check.json

Output:
    Overlap statistics per split (train / val / test).
    Tells you whether the subsets are identical, partially overlapping,
    or completely different.
"""

import os
import json
import argparse
import subprocess
import sys


def load_subset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_split(name: str, ids1: list, ids2: list):
    """Compare two lists of patch IDs and report overlap."""
    s1 = set(ids1)
    s2 = set(ids2)

    only_in_1 = s1 - s2
    only_in_2 = s2 - s1
    common    = s1 & s2

    n1 = len(s1)
    n2 = len(s2)
    nc = len(common)

    pct_overlap = (100.0 * nc / max(n1, 1)) if n1 > 0 else 0.0

    print(f"\n=== Split: {name} ===")
    print(f"  json1: {n1:,} patches")
    print(f"  json2: {n2:,} patches")
    print(f"  Common: {nc:,} patches ({pct_overlap:.2f}% of json1)")
    print(f"  Only in json1: {len(only_in_1):,}")
    print(f"  Only in json2: {len(only_in_2):,}")

    if n1 == n2 == nc:
        print(f"  ✓ IDENTICAL")
        return "identical"
    elif nc == 0:
        print(f"  ✗ COMPLETELY DIFFERENT")
        return "disjoint"
    elif pct_overlap > 95:
        print(f"  ~ NEARLY IDENTICAL ({100 - pct_overlap:.2f}% differ)")
        return "near_identical"
    elif pct_overlap > 50:
        print(f"  ⚠ PARTIALLY OVERLAPPING")
        return "partial"
    else:
        print(f"  ✗ MOSTLY DIFFERENT")
        return "mostly_different"


def maybe_regenerate(args):
    """Optionally call the original selection script with the same args
    as the current subset was produced with, saving to a new path."""
    if not args.regenerate:
        return

    selection_script = args.selection_script
    if not os.path.exists(selection_script):
        print(f"[ERROR] Selection script not found: {selection_script}")
        sys.exit(1)

    print(f"\n[Regenerate] Running selection script with seed={args.seed}...")
    print(f"[Regenerate] Output: {args.output_new}")

    # Read the current subset's metadata to use matching args.
    with open(args.json1) as f:
        current = json.load(f)
    meta = current.get("metadata", {})

    cmd = [
        sys.executable,
        selection_script,
        "--label_stats", args.label_stats,
        "--output", args.output_new,
        "--figure_dir", "./figures_compare",
        "--train_size", str(meta.get("train_size", 30000)),
        "--val_size",   str(meta.get("val_size",   5000)),
        "--test_size",  str(meta.get("test_size",  10000)),
        "--rare_class_quota_train",
            str(meta.get("rare_class_quota_train", 100)),
        "--rare_class_quota_eval",
            str(meta.get("rare_class_quota_eval", 30)),
        "--rare_threshold",
            str(meta.get("rare_threshold_pct", 1.0)),
        "--seed", str(args.seed),
    ]

    print(f"[Regenerate] Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[Regenerate] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", type=str, required=True,
                        help="First (canonical / existing) subset JSON.")
    parser.add_argument("--json2", type=str, default=None,
                        help="Second subset JSON to compare against. "
                             "If omitted with --regenerate, uses --output_new.")

    # Regeneration mode: optionally run the selection script first.
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-run select_flair_subset.py before comparing.")
    parser.add_argument("--selection_script", type=str,
                        default="./scripts/select_flair_subset.py",
                        help="Path to the subset selection script.")
    parser.add_argument("--label_stats", type=str,
                        default="./data/FLAIR-HUB/label_stats.json")
    parser.add_argument("--output_new", type=str,
                        default="./data/FLAIR-HUB/subset_indices_check.json",
                        help="Where to save the regenerated subset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for regeneration. Match the seed used to "
                             "produce json1 (usually 42).")
    args = parser.parse_args()

    if args.regenerate:
        maybe_regenerate(args)
        if args.json2 is None:
            args.json2 = args.output_new

    if args.json2 is None:
        print("[ERROR] Need either --json2 or --regenerate.")
        sys.exit(1)

    print(f"\nLoading json1: {args.json1}")
    sub1 = load_subset(args.json1)
    print(f"Loading json2: {args.json2}")
    sub2 = load_subset(args.json2)

    # Compare metadata
    meta1 = sub1.get("metadata", {})
    meta2 = sub2.get("metadata", {})

    print(f"\n=== Metadata comparison ===")
    keys = sorted(set(meta1.keys()) | set(meta2.keys()))
    differing = []
    for k in keys:
        v1 = meta1.get(k, "<missing>")
        v2 = meta2.get(k, "<missing>")
        if v1 != v2:
            differing.append((k, v1, v2))

    if not differing:
        print("  ✓ All metadata identical.")
    else:
        print(f"  ⚠ {len(differing)} differing metadata keys:")
        for k, v1, v2 in differing:
            # Don't print giant per-split dicts in full
            v1_str = str(v1)[:100] + "..." if len(str(v1)) > 100 else str(v1)
            v2_str = str(v2)[:100] + "..." if len(str(v2)) > 100 else str(v2)
            print(f"    {k}:")
            print(f"      json1: {v1_str}")
            print(f"      json2: {v2_str}")

    # Compare each split
    results = {}
    for split_key, split_name in [
        ("train_patch_ids", "train"),
        ("val_patch_ids", "validation"),
        ("test_patch_ids", "test"),
    ]:
        ids1 = sub1.get(split_key, [])
        ids2 = sub2.get(split_key, [])
        results[split_name] = compare_split(split_name, ids1, ids2)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for name, status in results.items():
        symbol = {
            "identical":       "✓",
            "near_identical":  "~",
            "partial":         "⚠",
            "mostly_different": "✗",
            "disjoint":        "✗",
        }.get(status, "?")
        print(f"  {symbol}  {name:<12} : {status}")
    print(f"{'=' * 70}\n")

    # ── Verdict ────────────────────────────────────────────────────
    if all(r == "identical" for r in results.values()):
        print("VERDICT: subsets are perfectly identical. Bookkeeping is fine.")
    elif all(r in ("identical", "near_identical") for r in results.values()):
        print("VERDICT: subsets are nearly identical. Likely tiny float / "
              "tie-breaking issues. Should be fine to use either.")
    elif any(r in ("disjoint", "mostly_different") for r in results.values()):
        print("VERDICT: subsets differ substantially. Models trained on one "
              "subset cannot be fairly compared against models trained on the "
              "other. Need to pick a canonical subset and re-test.")
    else:
        print("VERDICT: subsets are partially overlapping. Some comparison may "
              "be valid depending on how much overlap there is on test data.")


if __name__ == "__main__":
    main()
