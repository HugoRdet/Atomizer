#!/usr/bin/env python3
"""
viz_flair_dem.py -- inspect the FLAIR-HUB elevation (DEM_ELEV: ch0 = DSM, ch1 = DTM) before using it as a student input.

Where it goes: radio/viz_flair_dem.py
Reads flair_images_dem.csv (radio/add_dem_to_flair_index.py) and N random patches.

Outputs (in --out):
  dem_patch_stats.csv   per patch: median / p1 / p5 / p99 of nDSM = DSM - DTM, fraction < -0.5 m, DSM and DTM medians
  dem_summary.png       histograms of the per-patch nDSM median, p5 and negative fraction
  dem_worst.png         the --k patches with the largest negative fraction: RGB | DSM | DTM | nDSM | nDSM - p5 offset
  dem_random.png        --k random patches, same columns
Printed: how many patches look shifted as a whole (median nDSM < -1 m), how many have local negative areas only.

The last column applies the per-patch offset correction nDSM - min(p5, 0): if a patch is shifted as a whole, its ground
pixels come back to ~0; if negatives are local, it changes little. Comparing columns 4 and 5 shows which case dominates.

Usage (login node or a CPU job):
    python radio/viz_flair_dem.py --index ./data/flair_hub/idx/flair_images_dem.csv --root ./data/flair_hub/data \
        --n 500 --k 8 --out ./viz_pca/flair_dem
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read(root, z, off, cs, ct):
    import tifffile
    from training.utils.datasets.utils_dataset_distill import read_zip_member_raw
    a = tifffile.imread(io.BytesIO(read_zip_member_raw(os.path.join(root, z), int(off), int(cs), int(ct))))
    if a.ndim == 3 and a.shape[0] <= 8 and a.shape[-1] > 8:
        a = np.moveaxis(a, 0, -1)
    return a


def stretch(x, lo=None, hi=None):
    lo = np.percentile(x, 2) if lo is None else lo
    hi = np.percentile(x, 98) if hi is None else hi
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def figure(rows, root, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["RGB", "DSM (own range)", "DTM (own range)", "nDSM [-5, 30] m", "nDSM - min(p5, 0) [-5, 30] m"]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.6 * len(rows)), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        img = read(root, r["zip"], r["offset"], r["csize"], r["ctype"])
        dem = read(root, r["dem_zip"], r["dem_offset"], r["dem_csize"], r["dem_ctype"]).astype(np.float64)
        nd = dem[..., 0] - dem[..., 1]
        p5 = np.percentile(nd, 5)
        panels = [stretch(img[..., :3].astype(np.float64)), stretch(dem[..., 0]), stretch(dem[..., 1]),
                  stretch(nd, -5, 30), stretch(nd - min(p5, 0.0), -5, 30)]
        for j, p in enumerate(panels):
            ax[i, j].imshow(p, cmap=None if j == 0 else "viridis", interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(f"{r['patch']}\nneg {r['neg_frac']:.0%} | med {r['nd_med']:.1f} m", fontsize=7)
        ax[i, 1].set_xlabel(f"{dem[..., 0].min():.0f}..{dem[..., 0].max():.0f} m", fontsize=7)
        ax[i, 2].set_xlabel(f"{dem[..., 1].min():.0f}..{dem[..., 1].max():.0f} m", fontsize=7)
        ax[i, 3].set_xlabel(f"p1 {np.percentile(nd, 1):.1f} | p99 {np.percentile(nd, 99):.1f} m", fontsize=7)
        ax[i, 4].set_xlabel(f"offset {-min(p5, 0.0):+.1f} m", fontsize=7)
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=500, help="patches for the statistics")
    ap.add_argument("--k", type=int, default=8, help="patches per figure")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./viz_pca/flair_dem")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.index, dtype=str)
    df = df[(df["split"] == args.split) & df["dem_offset"].notna() & (df["dem_offset"] != "")]
    rng = np.random.default_rng(args.seed)
    rows = df.iloc[rng.choice(len(df), size=min(args.n, len(df)), replace=False)].copy()
    stats = []
    for t, (_, r) in enumerate(rows.iterrows()):
        dem = read(args.root, r["dem_zip"], r["dem_offset"], r["dem_csize"], r["dem_ctype"]).astype(np.float64)
        nd = dem[..., 0] - dem[..., 1]
        p = np.percentile(nd, [1, 5, 50, 99])
        stats.append(dict(nd_p1=p[0], nd_p5=p[1], nd_med=p[2], nd_p99=p[3], neg_frac=float((nd < -0.5).mean()),
                          dsm_med=float(np.median(dem[..., 0])), dtm_med=float(np.median(dem[..., 1])),
                          nodata=float((~np.isfinite(dem)).mean() + (dem < -1000).mean())))
        if (t + 1) % 100 == 0:
            print(f"[dem] {t + 1}/{len(rows)}")
    st = pd.DataFrame(stats, index=rows.index)
    rows = pd.concat([rows, st], axis=1)
    rows[["zip", "member", "dept", "patch"] + list(st.columns)].to_csv(os.path.join(args.out, "dem_patch_stats.csv"), index=False)

    shifted = rows["nd_med"] < -1.0
    local = (~shifted) & (rows["neg_frac"] > 0.05)
    clean = rows["neg_frac"] <= 0.05
    print(f"\n{len(rows)} {args.split} patches:")
    print(f"  shifted as a whole (median nDSM < -1 m): {int(shifted.sum())} ({shifted.mean():.1%})")
    print(f"  local negatives only (median ok, > 5 % of pixels < -0.5 m): {int(local.sum())} ({local.mean():.1%})")
    print(f"  clean (<= 5 % negative pixels): {int(clean.sum())} ({clean.mean():.1%})")
    print(f"  nodata-like values (non-finite or < -1000): {rows['nodata'].mean():.3%} of pixels")
    by_dept = rows.groupby("dept")["neg_frac"].mean().sort_values(ascending=False)
    print("  departments with the most negative pixels:", {k: round(v, 3) for k, v in by_dept.head(6).items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    ax[0].hist(rows["nd_med"].clip(-20, 20), bins=80); ax[0].set_title("per-patch median nDSM [m] (clipped +-20)")
    ax[1].hist(rows["nd_p5"].clip(-40, 10), bins=80); ax[1].set_title("per-patch p5 nDSM [m]")
    ax[2].hist(rows["neg_frac"], bins=50); ax[2].set_title("per-patch fraction nDSM < -0.5 m")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dem_summary.png"), dpi=100); plt.close(fig)

    figure(rows.sort_values("neg_frac", ascending=False).head(args.k), args.root,
           os.path.join(args.out, "dem_worst.png"), "largest fraction of nDSM < -0.5 m")
    figure(rows.sample(n=min(args.k, len(rows)), random_state=args.seed), args.root,
           os.path.join(args.out, "dem_random.png"), "random patches")
    print(f"\nfigures and dem_patch_stats.csv -> {args.out}")


if __name__ == "__main__":
    main()
