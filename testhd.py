"""
Normalization Diagnostic Script
================================

Checks the distribution of normalized reflectance values across the
Sen1Floods11 training set. Reports:

  - Fraction of values outside [-1, +1]   → Fourier aliasing risk
  - Fraction of values outside [-3σ, +3σ] → 3-sigma clipping threshold
  - Per-band statistics (mean, std, min, max, percentiles)
  - Recommendation: whether to apply 3-sigma clipping

Usage:
    python check_normalization.py \
        --data_path ./data/SENFLOOD \
        --bands_yaml ./data/bands_info/bands.yaml \
        --dataset_config ./data/Tiny_BigEarthNet/configs_dataset_senflood.yaml \
        --config_model ./training/configs/config_senflood_clean.yaml

The script loads the normalization stats, then iterates through the
training split computing normalized values WITHOUT the current clamp
to see the raw distribution before any clipping.
"""

import argparse
import numpy as np
import torch
import rasterio
import csv
import os
from tqdm import tqdm

from training.utils import read_yaml


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",      type=str, default="./data/SENFLOOD")
parser.add_argument("--bands_yaml",     type=str, default="./data/bands_info/bands.yaml")
parser.add_argument("--dataset_config", type=str, default="./data/Tiny_BigEarthNet/configs_dataset_senflood.yaml")
parser.add_argument("--config_model",   type=str, default="./training/configs/config_senflood_clean.yaml")
parser.add_argument("--max_samples",    type=int, default=None,
                    help="Limit number of samples (default: all)")
args = parser.parse_args()


# =============================================================================
# LOAD NORMALIZATION STATS
# =============================================================================

norm_file = os.path.join(args.data_path, "normalization_stats.pt")
assert os.path.exists(norm_file), (
    f"No normalization stats at {norm_file}. "
    f"Run training once to compute them."
)
stats = torch.load(norm_file, weights_only=True)

s2_mean = stats["s2_mean"].numpy()  # [13]
s2_std  = stats["s2_std"].numpy()   # [13]
s1_mean = stats["s1_mean"].numpy()  # [2]
s1_std  = stats["s1_std"].numpy()   # [2]

NUM_S2 = 13
NUM_S1 = 2
NUM_BANDS = NUM_S2 + NUM_S1

band_names = [
    "B01","B02","B03","B04","B05","B06","B07",
    "B08","B8A","B09","B10","B11","B12",
    "VV","VH",
]

print(f"\n{'='*70}")
print(f"NORMALIZATION STATS (from {norm_file})")
print(f"{'='*70}")
print(f"{'Band':<6} {'Mean':>10} {'Std':>10}")
print(f"{'-'*28}")
for i in range(NUM_S2):
    print(f"{band_names[i]:<6} {s2_mean[i]:>10.4f} {s2_std[i]:>10.4f}")
for i in range(NUM_S1):
    print(f"{band_names[NUM_S2+i]:<6} {s1_mean[i]:>10.4f} {s1_std[i]:>10.4f}")


# =============================================================================
# LOAD FILE LIST
# =============================================================================

split_file = os.path.join(
    args.data_path, "splits", "flood_handlabeled",
    "flood_train_data.csv"
)
data_root = os.path.join(
    args.data_path, "data", "flood_events", "HandLabeled"
)

s2_files, s1_files = [], []
with open(split_file) as f:
    for row in csv.reader(f):
        if len(row) < 2:
            continue
        s1_fn = row[0].replace("S1Hand/", "")
        s2_fn = s1_fn.replace("_S1Hand", "_S2Hand")
        s2_files.append(os.path.join(data_root, "S2Hand", s2_fn))
        s1_files.append(os.path.join(data_root, "S1Hand", s1_fn))

if args.max_samples:
    s2_files = s2_files[:args.max_samples]
    s1_files = s1_files[:args.max_samples]

print(f"\nAnalyzing {len(s2_files)} training samples...\n")


# =============================================================================
# ACCUMULATE STATISTICS
# =============================================================================

# Per-band accumulators
total_pixels   = np.zeros(NUM_BANDS, dtype=np.int64)
outside_1      = np.zeros(NUM_BANDS, dtype=np.int64)   # |x| > 1
outside_3sigma = np.zeros(NUM_BANDS, dtype=np.int64)   # |x| > 3
sum_vals       = np.zeros(NUM_BANDS, dtype=np.float64)
sum_sq         = np.zeros(NUM_BANDS, dtype=np.float64)
min_vals       = np.full(NUM_BANDS, np.inf)
max_vals       = np.full(NUM_BANDS, -np.inf)

# Percentile tracking — store a reservoir of values per band
RESERVOIR_SIZE = 100_000
reservoir      = [[] for _ in range(NUM_BANDS)]

for idx in tqdm(range(len(s2_files)), desc="Scanning"):
    try:
        with rasterio.open(s2_files[idx]) as src:
            s2 = src.read().astype(np.float32)
        with rasterio.open(s1_files[idx]) as src:
            s1 = src.read().astype(np.float32)
    except Exception as e:
        print(f"  [skip] {e}")
        continue

    s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize — WITHOUT any clipping
    s2_norm = (s2 - s2_mean[:, None, None]) / (s2_std[:, None, None] + 1e-8)
    s1_norm = (s1 - s1_mean[:, None, None]) / (s1_std[:, None, None] + 1e-8)

    image = np.concatenate([s2_norm, s1_norm], axis=0)  # [15, H, W]

    for b in range(NUM_BANDS):
        vals = image[b].flatten()
        n    = len(vals)

        total_pixels[b]   += n
        outside_1[b]      += np.sum(np.abs(vals) > 1.0)
        outside_3sigma[b] += np.sum(np.abs(vals) > 3.0)
        sum_vals[b]        += vals.sum()
        sum_sq[b]          += (vals ** 2).sum()
        min_vals[b]         = min(min_vals[b], vals.min())
        max_vals[b]         = max(max_vals[b], vals.max())

        # Reservoir sampling
        if len(reservoir[b]) < RESERVOIR_SIZE:
            reservoir[b].extend(vals.tolist())
        else:
            # Random replacement
            replace_idx = np.random.randint(0, RESERVOIR_SIZE,
                                            size=min(n, 1000))
            sample_idx  = np.random.randint(0, n, size=len(replace_idx))
            for ri, si in zip(replace_idx, sample_idx):
                reservoir[b][ri] = float(vals[si])


# =============================================================================
# REPORT
# =============================================================================

print(f"\n{'='*90}")
print(f"NORMALIZED VALUE DISTRIBUTION (no clipping)")
print(f"{'='*90}")
print(f"{'Band':<6} {'Mean':>7} {'Std':>7} {'Min':>8} {'Max':>8} "
      f"{'P1':>7} {'P99':>7} {'|x|>1':>9} {'|x|>3σ':>9} {'Verdict':>12}")
print(f"{'-'*90}")

needs_clipping = []

for b in range(NUM_BANDS):
    n     = total_pixels[b]
    mean  = sum_vals[b] / n
    std   = np.sqrt(sum_sq[b] / n - mean ** 2)
    pct1  = np.percentile(reservoir[b], 1)
    pct99 = np.percentile(reservoir[b], 99)
    frac1     = 100.0 * outside_1[b]      / n
    frac3sigma = 100.0 * outside_3sigma[b] / n

    if frac1 > 5.0:
        verdict = "⚠️  CLIP"
        needs_clipping.append(band_names[b])
    elif frac1 > 1.0:
        verdict = "△ borderline"
    else:
        verdict = "✓ ok"

    print(f"{band_names[b]:<6} {mean:>7.3f} {std:>7.3f} {min_vals[b]:>8.2f} "
          f"{max_vals[b]:>8.2f} {pct1:>7.3f} {pct99:>7.3f} "
          f"{frac1:>8.2f}% {frac3sigma:>8.2f}%  {verdict}")

print(f"{'='*90}")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print(f"\n{'='*70}")
print(f"RECOMMENDATION")
print(f"{'='*70}")

if needs_clipping:
    print(f"\n⚠️  The following bands have >5% of values outside [-1, +1]:")
    for name in needs_clipping:
        print(f"   - {name}")
    print(f"""
→ Apply 3-sigma clipping BEFORE Fourier encoding:

   In normalize_image(), replace the current clamp(-10, 10) with:

   image_s2 = torch.clamp(image_s2, -3.0, 3.0)
   image_s1 = torch.clamp(image_s1, -3.0, 3.0)

   This maps 99.7% of values to [-3, +3], then the Fourier encoder
   with bandvalue_max_freq=8 will see meaningful variation across
   most of the [-1, +1] range after further normalization.

   Alternatively, apply tanh squashing instead of hard clipping:
   image = torch.tanh(image / 3.0) * 3.0   →  keeps outliers but smooth

   Current clamp is (-10, 10) which is very loose — nearly all values
   pass through unmodified, and the Fourier encoder aliases on the
   concentrated region near 0.
""")
else:
    print(f"\n✓ All bands have <1% of values outside [-1, +1].")
    print(f"  Current normalization is adequate for Fourier encoding.")
    print(f"  The current clamp(-10, 10) can be tightened to clamp(-3, 3)")
    print(f"  for better Fourier frequency utilization without losing signal.")

print(f"\nFourier encoder config:")
print(f"  bandvalue_num_freq_bands: 8  →  frequencies [1..8]")
print(f"  Values in [-1, +1] map to full Fourier period")
print(f"  Values outside [-1, +1] produce aliased encodings")