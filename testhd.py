"""
Compute per-band, per-modality normalization statistics for FLAIR-HUB.

Iterates over the first N patches (default 10,000) and computes:
    - mean per band per modality
    - std per band per modality

Uses Welford's online algorithm to avoid loading everything into memory.

Saves results as a JSON file that can be loaded by the dataset class.

Usage:
    python compute_flairhub_stats.py \
        --root ./data/FLAIR-HUB/extracted \
        --n_samples 10000 \
        --output ./data/FLAIR-HUB/normalization_stats.json
"""

import os
import json
import argparse
import re
import numpy as np

try:
    import rasterio
except ImportError:
    raise ImportError("rasterio required: pip install rasterio")


# ─── FLAIR-HUB constants ────────────────────────────────────────────────────

MOD_SUFFIXES = {
    "aerial":  "AERIAL_RGBI",
    "spot":    "SPOT_RGBI",
    "s2":      "SENTINEL2_TS",
    "s1_asc":  "SENTINEL1-ASC_TS",
    "s1_des":  "SENTINEL1-DESC_TS",
}

# Number of spectral bands per modality (not counting time)
N_BANDS = {
    "aerial":  4,
    "spot":    4,
    "s2":      10,
    "s1_asc":  2,
    "s1_des":  2,
}

CSV_SPLIT_FILES = {
    "train": "FLAIR-HUB_TRAIN.csv",
}

BAND_NAMES = {
    "aerial":  ["RED", "GREEN", "BLUE", "NIR"],
    "spot":    ["RED", "GREEN", "BLUE", "NIR"],
    "s2":      ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "s1_asc":  ["VV", "VH"],
    "s1_des":  ["VV", "VH"],
}


# ─── Patch discovery ────────────────────────────────────────────────────────

def parse_patch_id(patch_id):
    match = re.match(r"^(D\d+-\w+?)_(.+)_(\d+-\d+)$", patch_id)
    if not match:
        return None
    return {
        "patch_id": patch_id,
        "domain": match.group(1),
        "roi": match.group(2),
        "coords": match.group(3),
    }


def load_patches_from_csv(root, max_n=10000):
    """Load patch list from CSV, searching root and parent dir."""
    csv_name = CSV_SPLIT_FILES["train"]
    csv_path = None
    for d in [root, os.path.dirname(root)]:
        candidate = os.path.join(d, csv_name)
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(f"Cannot find {csv_name} in {root} or parent")

    import csv
    patches = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pid = row.get("patch_id", "").strip()
            if not pid:
                continue
            info = parse_patch_id(pid)
            if info is not None:
                patches.append(info)
            if len(patches) >= max_n:
                break

    print(f"Loaded {len(patches)} patches from {csv_path}")
    return patches


def get_modality_path(root, domain, roi, coords, mod):
    suffix = MOD_SUFFIXES[mod]
    folder = f"{domain}_{suffix}"
    filename = f"{domain}_{suffix}_{roi}_{coords}.tif"
    return os.path.join(root, folder, roi, filename)


# ─── Welford online stats ───────────────────────────────────────────────────

class WelfordStats:
    """Online computation of mean and variance per band."""

    def __init__(self, n_bands):
        self.n_bands = n_bands
        self.count = np.zeros(n_bands, dtype=np.float64)
        self.mean = np.zeros(n_bands, dtype=np.float64)
        self.M2 = np.zeros(n_bands, dtype=np.float64)

    def update(self, data_per_band):
        """
        Update stats with data.

        Args:
            data_per_band: dict {band_idx: flat_array} or array [C, ...]
        """
        if isinstance(data_per_band, np.ndarray):
            # [C, H, W] or [C, ...] — iterate over first dim
            for c in range(min(data_per_band.shape[0], self.n_bands)):
                vals = data_per_band[c].ravel().astype(np.float64)
                for v in [vals]:  # batch update
                    n = len(v)
                    if n == 0:
                        continue
                    batch_mean = v.mean()
                    batch_var = v.var()
                    batch_count = n

                    delta = batch_mean - self.mean[c]
                    total_count = self.count[c] + batch_count

                    new_mean = self.mean[c] + delta * batch_count / max(total_count, 1)
                    m_a = self.M2[c]
                    m_b = batch_var * batch_count
                    self.M2[c] = m_a + m_b + delta ** 2 * self.count[c] * batch_count / max(total_count, 1)
                    self.mean[c] = new_mean
                    self.count[c] = total_count

    def get_mean(self):
        return self.mean.tolist()

    def get_std(self):
        var = np.where(self.count > 1, self.M2 / self.count, 0.0)
        return np.sqrt(var).tolist()

    def get_count(self):
        return self.count.tolist()


# ─── Main ────────────────────────────────────────────────────────────────────

def compute_stats(root, n_samples=10000):
    patches = load_patches_from_csv(root, max_n=n_samples)

    # Init per-modality stats
    stats = {}
    for mod, nb in N_BANDS.items():
        stats[mod] = WelfordStats(nb)

    n_processed = 0
    n_missing = {mod: 0 for mod in N_BANDS}

    for i, patch in enumerate(patches):
        if i % 500 == 0:
            print(f"  Processing patch {i}/{len(patches)}...")

        for mod, nb in N_BANDS.items():
            path = get_modality_path(root, patch["domain"], patch["roi"], patch["coords"], mod)

            if not os.path.exists(path):
                n_missing[mod] += 1
                continue

            try:
                with rasterio.open(path) as f:
                    data = f.read().astype(np.float64)
            except Exception as e:
                print(f"  Warning: failed to read {path}: {e}")
                n_missing[mod] += 1
                continue

            # For temporal: [T*C, H, W] → [T, C, H, W] → per-band across all T
            total_bands = data.shape[0]
            if total_bands > nb:
                # Temporal modality
                T = total_bands // nb
                data = data[:T * nb].reshape(T, nb, data.shape[1], data.shape[2])
                # Flatten time into spatial: [C, T*H*W]
                data = data.transpose(1, 0, 2, 3).reshape(nb, -1)

            stats[mod].update(data)

        n_processed += 1

    print(f"\nProcessed {n_processed} patches")
    for mod in N_BANDS:
        print(f"  {mod}: {n_missing[mod]} missing files")

    # Build output
    result = {}
    for mod, nb in N_BANDS.items():
        result[mod] = {
            "mean": stats[mod].get_mean(),
            "std": stats[mod].get_std(),
            "count": stats[mod].get_count(),
            "band_names": BAND_NAMES[mod],
            "n_patches": n_processed - n_missing[mod],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute FLAIR-HUB normalization stats")
    parser.add_argument("--root", type=str, default="./data/FLAIR-HUB/extracted",
                        help="Path to extracted FLAIR-HUB data")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of patches to use")
    parser.add_argument("--output", type=str, default="./data/FLAIR-HUB/normalization_stats.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"Computing normalization stats from {args.root}")
    print(f"Using first {args.n_samples} patches")
    print()

    result = compute_stats(args.root, args.n_samples)

    # Pretty print
    print("\n" + "=" * 60)
    print("NORMALIZATION STATISTICS")
    print("=" * 60)
    for mod, s in result.items():
        print(f"\n{mod} ({s['n_patches']} patches):")
        for i, name in enumerate(s["band_names"]):
            print(f"  {name:8s}: mean={s['mean'][i]:10.3f}  std={s['std'][i]:10.3f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()