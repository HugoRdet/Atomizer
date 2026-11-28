#!/usr/bin/env python3
"""
inspect_olmo_gse.py -- the `gse` layer of the OlmoEarth pretraining h5 (Google Satellite Embedding = AlphaEarth, 64-d,
annual, 10 m): how many samples have it, how it is quantized, and which time windows those samples cover.

Where it goes: radio/inspect_olmo_gse.py
Quantization guess: OlmoEarth's predefined norm for gse is mean 8192 / std 4096 for all 64 bands, i.e. uint16
v = (x + 1) * 8192 with x in [-1, 1]. AlphaEarth embeddings are unit-length, so if the guess is right the per-pixel
L2 norm of x = v / 8192 - 1 is ~1. Also reports values equal to 0 (nodata?).

Usage (login node, CPU):
    python radio/inspect_olmo_gse.py --index ./data/olmo_uncompressed/idx/all.csv \
        --tar-dir ./data/olmo_uncompressed/tars --n 200 | tee olmo_gse.txt
"""
import argparse
import io
import os
from collections import Counter

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./data/olmo_uncompressed/idx/all.csv")
    ap.add_argument("--tar-dir", default="./data/olmo_uncompressed/tars")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401

    df = pd.read_csv(args.index, dtype=str).fillna("")
    df = df[df["error"] == ""]
    print("index examples (mods, t_first, t_last, lat, lon, sid):")
    print(df[["mods", "t_first", "t_last", "lat", "lon", "sid"]].head(3).to_string(index=False))
    has = df["mods"].str.contains("gse")
    print(f"\n== index: {int(has.sum())} / {len(df)} rows have gse ({has.mean():.1%})")
    mod_counts = Counter(m for s in df["mods"] for m in s.replace(";", ",").replace("|", ",").split(",") if m)
    print("  modality counts:", dict(mod_counts.most_common()))
    # t_first / t_last = start dates of the first / last 30-day slot, "day-month0-year" (month 0-indexed, as `timestamps`)
    def start(col):
        d = col.str.split("-", expand=True).astype(int)
        return pd.to_datetime(dict(year=d[2], month=d[1] + 1, day=d[0]), errors="coerce")
    for name, sub in (("all", df), ("gse", df[has])):
        t0, t1 = start(sub["t_first"]), start(sub["t_last"]) + pd.Timedelta(days=30)   # end of the last slot
        y0, y1 = t0.dt.year, t1.dt.year
        nxt = pd.to_datetime(dict(year=y0 + 1, month=1, day=1))
        days_y0 = (nxt.clip(upper=t1) - t0).dt.days                        # days of the window in its first year
        major = np.where(days_y0 >= (t1 - t0).dt.days / 2, y0, y1)
        print(f"  {name:4s}: window length days {dict(Counter((t1 - t0).dt.days).most_common(3))} | "
              f"start in January {(t0.dt.month == 1).mean():.1%}")
        print(f"        majority calendar year {dict(sorted(Counter(major).items()))}")
        print(f"        days in the majority year: p5 {np.percentile(np.maximum(days_y0, (t1 - t0).dt.days - days_y0), 5):.0f}"
              f" | median {np.median(np.maximum(days_y0, (t1 - t0).dt.days - days_y0)):.0f}")
    n_sid = df["sid"].nunique()
    print(f"  distinct sid: {n_sid} | rows per sid: {dict(Counter(df.groupby('sid').size()).most_common(5))}")

    rng = np.random.default_rng(args.seed)
    rows = df[has].iloc[rng.choice(int(has.sum()), size=min(args.n, int(has.sum())), replace=False)]
    vmin, vmax, norms, zeros, nodata_px, flat = [], [], [], 0, 0, []
    for r in rows.itertuples():
        with open(os.path.join(args.tar_dir, r.tar), "rb") as f:
            f.seek(int(r.offset))
            data = f.read(int(r.size))
        with h5py.File(io.BytesIO(data), "r") as h:
            v = h["gse"][()][:, :, 0, :].astype(np.float64)                  # (128, 128, 64)
        vmin.append(v.min()); vmax.append(v.max())
        bad = (v == 0).all(-1)
        nodata_px += int(bad.sum()); zeros += int((v == 0).sum())
        x = v[~bad] / 8192.0 - 1.0
        norms.append(np.linalg.norm(x, axis=-1))
        flat.append(x[rng.choice(len(x), size=min(200, len(x)), replace=False)] if len(x) else np.zeros((0, 64)))
    nr = np.concatenate(norms)
    fl = np.concatenate(flat)
    print(f"\n== gse values over {len(rows)} samples: raw min {min(vmin):.0f} max {max(vmax):.0f} "
          f"| all-zero pixels {nodata_px / (len(rows) * 128 * 128):.2%} | zero values {zeros / (len(rows) * 128 * 128 * 64):.2%}")
    print(f"  x = v / 8192 - 1: per-pixel L2 norm mean {nr.mean():.4f} std {nr.std():.4f} "
          f"p1 {np.percentile(nr, 1):.4f} p99 {np.percentile(nr, 99):.4f}   (unit-length AlphaEarth -> ~1)")
    print(f"  x range [{fl.min():.3f}, {fl.max():.3f}] | per-dim mean |avg| {np.abs(fl.mean(0)).mean():.3f} std avg {fl.std(0).mean():.3f}")


if __name__ == "__main__":
    main()
