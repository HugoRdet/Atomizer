#!/usr/bin/env python3
"""
add_dem_to_flair_index.py -- add the FLAIR-HUB elevation (DEM_ELEV) of every aerial patch to the FLAIR index.

Where it goes: radio/add_dem_to_flair_index.py (next to build_flair_index.py)

Pairing: the DEM of <DEPT>-<YEAR>_AERIAL_RGBI.zip : <DEPT>-<YEAR>_AERIAL_RGBI/<ROI>/<DEPT>-<YEAR>_AERIAL_RGBI_<ROI>_<i-j>.tif
is            <DEPT>-<YEAR>_DEM_ELEV.zip    : the same path with AERIAL_RGBI replaced by DEM_ELEV.
Only the zip directories are read for the index (byte offset, compressed size, method of every DEM member, as
build_flair_index.py does for the aerial zips), so FlairAtoms can read a DEM patch with one seek.

Output: <out> (default: the input with _dem before .csv) = the input index + dem_zip, dem_member, dem_offset, dem_csize,
dem_ctype (rows without a DEM keep empty fields and are reported). The input file is not modified.

Then --check N decodes N random (aerial, DEM) pairs and prints:
  * shapes / dtypes (DEM expected 512 x 512 x 2 float32, same grid as the aerial patch),
  * which DEM channel is the surface model: fraction of pixels with ch0 >= ch1 (DSM >= DTM everywhere, up to noise),
  * nDSM = DSM - DTM statistics (height above ground, the elevation feature to give the student).

Usage (login node is fine: directories + a few hundred patches):
    python radio/add_dem_to_flair_index.py --index ./data/flair_hub/idx/flair_images.csv \
        --root ./data/flair_hub/data --check 200
"""
import argparse
import io
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC, DST = "AERIAL_RGBI", "DEM_ELEV"


def scan(path):
    """DEM zip -> {member: (offset, csize, ctype)}."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.is_dir() and info.filename.lower().endswith((".tif", ".tiff")):
                out[info.filename] = (info.header_offset, info.compress_size, info.compress_type)
    return os.path.basename(path), out


def read_member(root, zname, offset, csize, ctype):
    from training.utils.datasets.utils_dataset_distill import read_zip_member_raw
    import tifffile
    a = tifffile.imread(io.BytesIO(read_zip_member_raw(os.path.join(root, zname), int(offset), int(csize), int(ctype))))
    if a.ndim == 3 and a.shape[0] <= 8 and a.shape[-1] > 8:
        a = np.moveaxis(a, 0, -1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="flair_images.csv (build_flair_index.py)")
    ap.add_argument("--root", required=True, help="folder with the *_AERIAL_RGBI.zip and *_DEM_ELEV.zip files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=200, help="decode this many random pairs (0: skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or args.index[:-4] + "_dem.csv"

    df = pd.read_csv(args.index, dtype=str)
    df["dem_zip"] = df["zip"].str.replace(SRC, DST, regex=False)
    df["dem_member"] = df["member"].str.replace(SRC, DST, regex=False)
    zips = sorted(set(df["dem_zip"]))
    present = [z for z in zips if os.path.exists(os.path.join(args.root, z))]
    print(f"{len(df)} aerial patches in {df['zip'].nunique()} zips; DEM zips present: {len(present)}/{len(zips)}")
    for z in sorted(set(zips) - set(present))[:10]:
        print("   missing:", z)
    with Pool(args.workers) as pool:
        tables = dict(pool.map(scan, [os.path.join(args.root, z) for z in present]))

    off, cs, ct = [], [], []
    for z, m in zip(df["dem_zip"], df["dem_member"]):
        e = tables.get(z, {}).get(m)
        off.append("" if e is None else e[0]); cs.append("" if e is None else e[1]); ct.append("" if e is None else e[2])
    df["dem_offset"], df["dem_csize"], df["dem_ctype"] = off, cs, ct
    has = df["dem_offset"] != ""
    for sp in sorted(df["split"].unique()):
        s = df["split"] == sp
        print(f"  split {sp:5s}: DEM found for {int((has & s).sum())}/{int(s.sum())} patches")
    n_dem = sum(len(t) for t in tables.values())
    print(f"DEM members in the zips: {n_dem} ({n_dem - int(has.sum())} without an aerial patch in the index)")
    if not has.all():
        print("   e.g. no DEM for:", df.loc[~has, "member"].head(3).tolist())
    print("DEM compression methods:", df.loc[has, "dem_ctype"].value_counts().to_dict(), "(0 stored, 8 deflate)")
    df.to_csv(out, index=False)
    print("wrote", out)

    if args.check <= 0 or not has.any():
        return
    rng = np.random.default_rng(args.seed)
    rows = df[has].iloc[rng.choice(int(has.sum()), size=min(args.check, int(has.sum())), replace=False)]
    ge, nd, shapes, bad = [], [], set(), 0
    for _, r in rows.iterrows():
        img = read_member(args.root, r["zip"], r["offset"], r["csize"], r["ctype"])
        dem = read_member(args.root, r["dem_zip"], r["dem_offset"], r["dem_csize"], r["dem_ctype"])
        shapes.add((img.shape, str(img.dtype), dem.shape, str(dem.dtype)))
        if dem.ndim != 3 or dem.shape[-1] != 2 or dem.shape[:2] != img.shape[:2]:
            bad += 1
            continue
        a, b = dem[..., 0].astype(np.float64), dem[..., 1].astype(np.float64)
        ge.append(float((a >= b - 0.01).mean()))
        nd.append(a - b)
    print(f"\n{len(rows)} pairs decoded; (aerial shape, dtype, DEM shape, dtype): {sorted(shapes)}; unexpected: {bad}")
    if not ge:
        return
    ge = np.array(ge)
    print(f"fraction of pixels with ch0 >= ch1: mean {ge.mean():.4f}, min over patches {ge.min():.4f}")
    surf = 0 if ge.mean() > 0.5 else 1
    x = np.concatenate([v.ravel() for v in nd]) * (1 if surf == 0 else -1)
    q = np.percentile(x, [0.1, 1, 50, 99, 99.9])
    print(f"-> surface model (DSM) = channel {surf}, terrain (DTM) = channel {1 - surf}")
    print(f"nDSM = DSM - DTM [m]: mean {x.mean():.2f}, std {x.std():.2f}, min {x.min():.2f}, max {x.max():.2f}, "
          f"percentiles 0.1/1/50/99/99.9: {np.round(q, 2).tolist()}")
    print(f"pixels with nDSM < -0.5 m (inconsistent DSM/DTM): {(x < -0.5).mean():.4%}")


if __name__ == "__main__":
    main()
