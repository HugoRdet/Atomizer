"""
Inspect a BreizhCrops download (layout of download_breizhcrops.sh).

For each level (L1C / L2A) and region (FRH01-04):
  * index file: columns, number of parcels, class distribution, cloud score
  * HDF5: every index path resolves, array shape per parcel (dates x columns),
    columns as documented, value ranges per band, NaNs, padding, date range
  * "image size": a BreizhCrops sample is ONE spectrum per date, averaged over
    the parcel, so there is no image. What corresponds to an image size is
      - the time-series length (dates per parcel), always reported, and
      - the parcel footprint (area, and number of 10 m pixels averaged),
        reported when the shapefiles are present (--with_shp).

Usage:
    python inspect_breizhcrops.py --root ./data/BREIZHCROPS
    python inspect_breizhcrops.py --root ./data/BREIZHCROPS --with_shp --sample 5000
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd

COLUMNS = {   # column order of each per-parcel array (SELECTED_BANDS in the package)
    "L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
            "QA10", "QA20", "QA60", "doa"],
    "L2A": ["doa", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
            "CLD", "EDG", "SAT"],
}
REGIONS = ["frh01", "frh02", "frh03", "frh04"]
BENCHMARK_SPLIT = {"frh01": "train", "frh02": "train", "frh03": "val", "frh04": "test"}


def to_dates(v):
    """Epoch integers -> datetimes, whatever the unit (the package wrote them with
    pandas' astype(int), i.e. nanoseconds; newer pandas would give microseconds)."""
    v = np.asarray(v, dtype=np.float64)
    m = np.nanmedian(np.abs(v))
    unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
    return pd.to_datetime(v.astype(np.int64), unit=unit), unit


def q(x, qs=(0, 5, 50, 95, 100)):
    """Compact percentile string."""
    x = np.asarray(x, dtype=float)
    return " ".join(f"p{p}={np.percentile(x, p):.6g}" for p in qs)


def inspect_region(root, level, region, mapping, n_sample, with_shp, rng):
    d = os.path.join(root, "2017", level)
    idx_path, h5_path = os.path.join(d, f"{region}.csv"), os.path.join(d, f"{region}.h5")
    print(f"\n{'=' * 78}\n{level} {region.upper()}  ({BENCHMARK_SPLIT[region]} in the usual benchmark)")
    if not (os.path.exists(idx_path) and os.path.exists(h5_path)):
        print(f"  MISSING: {idx_path if not os.path.exists(idx_path) else h5_path}")
        return None

    # ── index ────────────────────────────────────────────────────────────
    idx = pd.read_csv(idx_path)
    print(f"  index columns: {list(idx.columns)}")
    n_all = len(idx)
    idx = idx[idx["CODE_CULTU"].isin(mapping.index)]
    idx = idx[idx["sequencelength"] > 0]
    print(f"  parcels: {n_all} in index, {len(idx)} after class mapping + non-empty series")
    cls = mapping.loc[idx["CODE_CULTU"], ["id", "classname"]].reset_index(drop=True)
    counts = cls.groupby(["id", "classname"]).size()
    print("  classes:")
    for (cid, name), c in counts.items():
        print(f"    {cid:2d} {name:22s} {c:7d}  ({c / len(idx):6.2%})")
    cld_col = "meanCLD" if "meanCLD" in idx.columns else "meanQA60"
    if cld_col in idx.columns:
        print(f"  {cld_col}: {q(idx[cld_col])}")
    print(f"  sequence length (index): {q(idx['sequencelength'])}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    cols = COLUMNS[level]
    sel = idx.sample(n=min(n_sample, len(idx)), random_state=0)
    shapes, missing, arrays, idx_len = [], 0, [], []
    with h5py.File(h5_path, "r") as f:
        for p, L in zip(sel["path"], sel["sequencelength"]):
            if p not in f:
                missing += 1
                continue
            a = np.asarray(f[p])
            shapes.append(a.shape)
            arrays.append(a)
            idx_len.append(L)
        example_dtype = arrays[0].dtype if arrays else None
    print(f"  h5 size: {os.path.getsize(h5_path) / 1e9:.2f} GB   sampled parcels: {len(sel)}  "
          f"missing paths: {missing}   dtype: {example_dtype}")
    if not arrays:
        return None

    ncols = {s[1] for s in shapes}
    lens = np.array([s[0] for s in shapes])
    print(f"  array shape per parcel: (T, {sorted(ncols)})   expected columns: {len(cols)} "
          f"{'OK' if ncols == {len(cols)} else 'MISMATCH'}")
    print(f"  T (dates per parcel, h5): {q(lens)}")
    match = (lens == np.asarray(idx_len)).mean()
    print(f"  T matches index sequencelength: {match:.1%}")

    X = np.concatenate(arrays)                                      # [sum T, C]
    print(f"  NaN: {int(np.isnan(X).sum())}   rows equal to padding (-1): "
          f"{int((X == -1).all(1).sum())}")

    # dates
    doa, unit = to_dates(X[:, cols.index("doa")])
    print(f"  dates ({unit} since epoch): {doa.min().date()} .. {doa.max().date()}   distinct dates in sample: "
          f"{doa.normalize().nunique()}   years: {sorted(doa.year.unique())}")
    dup = np.mean([len(np.unique(a[:, cols.index('doa')])) < len(a) for a in arrays])
    srt = np.mean([np.all(np.diff(a[:, cols.index('doa')]) > 0) for a in arrays])
    print(f"  per-parcel dates sorted: {srt:.1%}   with duplicate dates: {dup:.1%}")

    # values
    print("  per-column values (raw):")
    for j, c in enumerate(cols):
        if c == "doa":
            continue
        v = X[:, j]
        print(f"    {c:5s} min={v.min():9.1f} mean={v.mean():9.1f} max={v.max():9.1f} "
              f"neg={np.mean(v < 0):6.2%}")

    # ── parcel footprint ─────────────────────────────────────────────────
    area_stats = None
    if with_shp:
        shp = os.path.join(root, "2017", f"{region}.shp")
        if not os.path.exists(shp):
            print(f"  shapefile not found ({shp}): skip footprint")
        else:
            import geopandas as gpd
            gdf = gpd.read_file(shp).rename(columns={"ID_PARCEL": "ID"})
            if gdf.crs is not None and gdf.crs.is_geographic:
                gdf = gdf.to_crs(2154)                              # Lambert-93, metres
            gdf = gdf.set_index("ID")
            ids = idx["id"][idx["id"].isin(gdf.index)]
            area = gdf.loc[ids, "geometry"].area.values             # m^2
            px = area / 100.0                                       # 10 m pixels
            print(f"  parcels with geometry: {len(ids)}/{len(idx)}   CRS: {gdf.crs}")
            print(f"  area (ha):          {q(area / 1e4)}")
            print(f"  10 m pixels/parcel: {q(px)}")
            b = gdf.loc[ids, "geometry"].bounds
            ext = np.maximum(b.maxx - b.minx, b.maxy - b.miny).values
            print(f"  bbox side (m):      {q(ext)}   "
                  f"(~ {np.median(ext) / 10:.0f} px of 10 m for the median parcel)")
            area_stats = area
    return dict(level=level, region=region, n=len(idx), T=lens, area=area_stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/BREIZHCROPS")
    ap.add_argument("--levels", nargs="+", default=["L1C", "L2A"])
    ap.add_argument("--sample", type=int, default=2000, help="parcels read per region")
    ap.add_argument("--with_shp", action="store_true", help="also report parcel footprints")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cm = os.path.join(args.root, "classmapping.csv")
    mapping = pd.read_csv(cm, index_col=0).sort_values(by="id").set_index("code")
    print(f"class mapping: {mapping['id'].nunique()} classes, {len(mapping)} crop codes")

    rows = []
    for level in args.levels:
        for region in REGIONS:
            r = inspect_region(args.root, level, region, mapping, args.sample, args.with_shp, rng)
            if r:
                rows.append(r)

    print(f"\n{'=' * 78}\nSUMMARY")
    print(f"  {'level':5s} {'region':6s} {'split':5s} {'parcels':>8s} {'T median':>8s} "
          f"{'T max':>6s} {'area med (ha)':>14s} {'px med':>7s}")
    for r in rows:
        a = r["area"]
        am = f"{np.median(a) / 1e4:14.2f}" if a is not None else f"{'-':>14s}"
        pm = f"{np.median(a) / 100:7.0f}" if a is not None else f"{'-':>7s}"
        print(f"  {r['level']:5s} {r['region']:6s} {BENCHMARK_SPLIT[r['region']]:5s} {r['n']:8d} "
              f"{np.median(r['T']):8.0f} {r['T'].max():6d} {am} {pm}")


if __name__ == "__main__":
    main()
