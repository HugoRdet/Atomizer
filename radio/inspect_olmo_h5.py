#!/usr/bin/env python3
"""
inspect_olmo_h5.py -- how the OlmoEarth pretraining samples are laid out in time (before building a time-series loader).

Where it goes: radio/inspect_olmo_h5.py
Reads only the index and a few h5 samples (no model). Prints:
  1. index level (all rows): number of present months per sample for S2, S1 and both (from s2_mask / s1_mask)
  2. the full h5 tree of --dump samples: every dataset with shape / dtype, every attribute (location, CRS, bounds?)
  3. on --n samples: checks stored timesteps == mask.sum() per modality, decodes `timestamps` ([day, month, year],
     months 0-indexed per OlmoEarth), and reports: length of the timestamps array, whether the 12 slots are consecutive
     months, start month, year span, dates of missing slots
  4. which other modalities exist in the h5 (landsat, worldcover, srtm, latlon, ...) and how often

Usage (login node, CPU):
    python radio/inspect_olmo_h5.py --index ./data/olmo_uncompressed/idx/all.csv --tar-dir ./data/olmo_uncompressed/tars \
        --n 300 --dump 2 | tee olmo_h5_layout.txt
"""
import argparse
import io
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_h5(tar_dir, r):
    with open(os.path.join(tar_dir, r.tar), "rb") as f:
        f.seek(int(r.offset))
        return f.read(int(r.size))


def tree(h, out, prefix=""):
    import h5py
    for k, v in h.attrs.items():
        out.append(f"{prefix}@{k} = {np.array(v).tolist() if np.size(v) < 20 else np.shape(v)}")
    for k in h.keys():
        o = h[k]
        if isinstance(o, h5py.Dataset):
            extra = ""
            if o.size <= 24:
                extra = f" = {o[()].tolist()}"
            out.append(f"{prefix}{k}: {o.shape} {o.dtype}{extra}")
            for a, v in o.attrs.items():
                out.append(f"{prefix}  @{a} = {np.array(v).tolist() if np.size(v) < 20 else np.shape(v)}")
        else:
            out.append(f"{prefix}{k}/")
            tree(o, out, prefix + "  ")


def months_since0(ts):
    """[T, 3] (day, month 0-indexed, year) -> absolute month index."""
    return ts[:, 2].astype(int) * 12 + ts[:, 1].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./data/olmo_uncompressed/idx/all.csv")
    ap.add_argument("--tar-dir", default="./data/olmo_uncompressed/tars")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dump", type=int, default=2, help="samples whose whole h5 tree is printed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401

    df = pd.read_csv(args.index, dtype=str).fillna("")
    print(f"index: {len(df)} rows, columns {list(df.columns)}")
    ok = df[df["error"] == ""] if "error" in df else df
    m2 = np.array([[c == "1" for c in s.ljust(12, "0")[:12]] for s in ok["s2_mask"]])
    m1 = np.array([[c == "1" for c in s.ljust(12, "0")[:12]] for s in ok["s1_mask"]])
    lens = Counter(len(s) for s in ok["s2_mask"])
    print(f"\n== 1. index ({len(ok)} rows without error) | s2_mask string lengths {dict(lens)}")
    for name, m in (("S2", m2), ("S1", m1), ("S2 & S1", m2 & m1)):
        c = np.bincount(m.sum(1), minlength=13)
        print(f"  {name:8s} present months per sample: " + " ".join(f"{k}:{v}" for k, v in enumerate(c)))
        print(f"  {'':8s} mean {m.sum(1).mean():.2f} | >=6: {(m.sum(1) >= 6).mean():.1%} | 12: {(m.sum(1) == 12).mean():.1%}"
              f" | per slot presence {np.round(m.mean(0), 2).tolist()}")

    rng = np.random.default_rng(args.seed)
    rows = ok.iloc[rng.choice(len(ok), size=min(args.n, len(ok)), replace=False)]
    keys_seen, bad, t_len, consec, start_m, span, day = Counter(), [], Counter(), 0, Counter(), Counter(), []
    shapes = Counter()
    for j, r in enumerate(rows.itertuples()):
        try:
            data = read_h5(args.tar_dir, r)
        except Exception as e:  # noqa: BLE001
            bad.append((r.tar, repr(e)))
            continue
        with h5py.File(io.BytesIO(data), "r") as h:
            if j < args.dump:
                out = []
                tree(h, out)
                print(f"\n== 2. h5 tree of sample sid={getattr(r, 'sid', '?')} ({r.tar} @ {r.offset})")
                print("\n".join("  " + s for s in out))
            for k in h.keys():
                keys_seen[k] += 1
            ts = h["timestamps"][()] if "timestamps" in h else None
            if ts is not None:
                t_len[ts.shape[0]] += 1
                a = months_since0(ts)
                consec += int(np.all(np.diff(a) == 1))
                start_m[int(ts[0, 1])] += 1
                span[f"{int(ts[:, 2].min())}-{int(ts[:, 2].max())}"] += 1
                day.extend(ts[:, 0].tolist())
            if "missing_timesteps_masks" in h:
                for mod in h["missing_timesteps_masks"].keys():
                    mk = h[f"missing_timesteps_masks/{mod}"][()]
                    if mod in h:
                        shapes[f"{mod} {h[mod].shape[:2]} x T x {h[mod].shape[-1]}"] += 1
                        if h[mod].ndim >= 3 and int(mk.sum()) != h[mod].shape[2]:
                            bad.append((r.tar, f"{mod}: mask sum {int(mk.sum())} != stored {h[mod].shape[2]}"))
            if j < args.dump and ts is not None:
                print(f"  timestamps [day, month0, year]: {ts.tolist()}")
                for mod in h.get("missing_timesteps_masks", {}).keys():
                    print(f"  {mod:16s} present: {''.join('1' if v else '.' for v in h[f'missing_timesteps_masks/{mod}'][()])}")

    n = len(rows) - len(bad)
    print(f"\n== 3. {len(rows)} sampled h5 ({len(bad)} problems)")
    print(f"  timestamps length: {dict(t_len)} | 12 consecutive months: {consec}/{n}")
    print(f"  start month (0 = Jan): {dict(sorted(start_m.items()))}")
    print(f"  year spans: {dict(span.most_common(12))}")
    if day:
        print(f"  day-of-month of slots: min {min(day)} max {max(day)} | most common {Counter(day).most_common(5)}")
    print(f"  array layouts: {dict(shapes.most_common(10))}")
    for b in bad[:10]:
        print("  problem:", b)
    print(f"\n== 4. top-level keys over {len(rows)} samples: {dict(keys_seen.most_common())}")


if __name__ == "__main__":
    main()
