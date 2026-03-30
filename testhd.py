"""
Compute per-band normalization stats for BigEarthNet S2.

Computes p2/p98 percentiles per band from a random subset of the training set,
then saves as JSON for use by the dataset class.

Usage:
    python compute_ben_stats.py

Output:
    data/Encoded-BigEarthNet/ben_norm_stats.json
"""

import json
import numpy as np
import torch
from configilm.extra.DataSets import BENv2_DataSet

DATA_DIRS = {
    "images_lmdb": "data/Encoded-BigEarthNet",
    "metadata_parquet": "data/Encoded-BigEarthNet/metadata.parquet",
    "metadata_snow_cloud_parquet": "data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

OUTPUT_PATH = "data/Encoded-BigEarthNet/ben_norm_stats.json"

# 10 S2 bands (no 60m bands B01/B09), matching reBEN paper
# configilm 14-ch layout: [S1_VV, S1_VH, B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
S2_CHANNEL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
S2_BAND_NAMES = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

N_SAMPLES = 10000  # random subset for stats computation
SEED = 42


def main():
    print(f"Loading BEN training set...")
    ds = BENv2_DataSet.BENv2DataSet(
        data_dirs=DATA_DIRS,
        split="train",
        img_size=(14, 120, 120),
        include_snowy=False,
        include_cloudy=False,
    )
    print(f"Training set: {len(ds)} samples")

    n_bands = len(S2_CHANNEL_INDICES)
    n_samples = min(N_SAMPLES, len(ds))

    # Collect per-band values from random subset
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(ds), size=n_samples, replace=False)

    # Accumulate per-band: we'll store all pixel values and compute percentiles
    # Each band has 120*120 = 14400 pixels per sample
    # 10k samples × 14400 = 144M values per band — too much to hold in memory
    # Instead: use reservoir sampling / streaming percentile via sorted subset
    # Practical approach: subsample pixels too
    PIXELS_PER_SAMPLE = 500  # random pixels per sample per band
    n_total = n_samples * PIXELS_PER_SAMPLE

    print(f"Sampling {n_samples} images × {PIXELS_PER_SAMPLE} pixels = "
          f"{n_total:,} values per band")

    band_values = {i: np.zeros(n_total, dtype=np.float32) for i in range(n_bands)}

    for count, idx in enumerate(indices):
        if (count + 1) % 1000 == 0:
            print(f"  {count+1}/{n_samples}")

        img, _ = ds[int(idx)]  # [14, 120, 120]

        for b_idx, ch_idx in enumerate(S2_CHANNEL_INDICES):
            band_data = img[ch_idx].numpy().flatten()  # [14400]
            # Random pixel subsample
            pix_idx = rng.choice(len(band_data), size=PIXELS_PER_SAMPLE, replace=False)
            start = count * PIXELS_PER_SAMPLE
            end = start + PIXELS_PER_SAMPLE
            band_values[b_idx][start:end] = band_data[pix_idx]

    # Compute percentiles
    stats = {}
    print(f"\nPer-band percentile stats:")
    print(f"  {'Band':<8s} {'p2':>10s} {'p98':>10s} {'median':>10s} {'mean':>10s} {'std':>10s}")
    print(f"  {'-'*52}")

    for b_idx, (ch_idx, name) in enumerate(zip(S2_CHANNEL_INDICES, S2_BAND_NAMES)):
        vals = band_values[b_idx]
        p2 = float(np.percentile(vals, 2))
        p98 = float(np.percentile(vals, 98))
        median = float(np.median(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))

        stats[name] = {
            "ch_idx": ch_idx,
            "p2": p2,
            "p98": p98,
            "median": median,
            "mean": mean,
            "std": std,
        }

        print(f"  {name:<8s} {p2:>10.2f} {p98:>10.2f} {median:>10.2f} {mean:>10.2f} {std:>10.2f}")

    # Save
    output = {
        "description": "Per-band normalization stats for BigEarthNet S2 (10 bands, no 60m)",
        "method": "percentile",
        "percentiles": [2, 98],
        "n_samples": n_samples,
        "n_pixels_per_sample": PIXELS_PER_SAMPLE,
        "bands": stats,
        # Also save as flat arrays for easy loading
        "band_names": S2_BAND_NAMES,
        "band_p2": [stats[n]["p2"] for n in S2_BAND_NAMES],
        "band_p98": [stats[n]["p98"] for n in S2_BAND_NAMES],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n→ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()