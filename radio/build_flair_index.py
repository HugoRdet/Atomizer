#!/usr/bin/env python3
"""
build_flair_index.py -- index the FLAIR-HUB zips without extracting anything.

Reads only the zip directories (central directory) of
    <root>/<DEPT>-<YEAR>_AERIAL_RGBI.zip         (4-band 8-bit aerial patches, 512x512 @ 20 cm)
    <root>/<DEPT>-<YEAR>_AERIAL_LABEL-COSIA.zip  (land-cover labels, optional)
and writes one CSV row per .tif member. Labels are matched to images with a name heuristic (reported).

Outputs (in --out):
    flair_images.csv : zip, member, size, offset, csize, ctype, dept, year, roi, patch, split, label_zip, label_member
    flair_labels.csv : zip, member, size, dept, year, roi, patch, split

split = "test" for the official FLAIR-HUB test departments, else "train" (use "train" for distillation; keep
"test" for evaluation). Nothing is read from the image data itself.

Usage:
    python3 build_flair_index.py --root /path/to/flair_hub/data --out idx_flair
"""
import argparse
import csv
import os
import re
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool

TEST = {"D012", "D015", "D022", "D026", "D036", "D061", "D064", "D068", "D069", "D071", "D073", "D075", "D076",
        "D083", "D084", "D085"}
KINDS = {"AERIAL_RGBI": "image", "AERIAL_LABEL-COSIA": "label"}


def parse_zip_name(path):
    """'D004-2021_AERIAL_RGBI.zip' -> ('D004', '2021', 'AERIAL_RGBI')."""
    stem = os.path.basename(path)[:-4]
    left, _, mod = stem.partition("_")
    dept, _, year = left.partition("-")
    return dept, year, mod


def scan(path):
    dept, year, mod = parse_zip_name(path)
    rows = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".tif", ".tiff")):
                continue
            parts = info.filename.split("/")
            rows.append(dict(zip=os.path.basename(path), member=info.filename, size=info.file_size,
                             offset=info.header_offset, csize=info.compress_size, ctype=info.compress_type, dept=dept,
                             year=year, roi=parts[-2] if len(parts) > 1 else "", patch=os.path.splitext(parts[-1])[0],
                             split="test" if dept in TEST else "train"))
    return mod, rows


def patch_key(row):
    """Heuristic key shared by an image and its label: patch name with the modality tokens removed."""
    s = re.sub(r"AERIAL_LABEL-COSIA|AERIAL_RGBI|LABEL-COSIA|COSIA|RGBI|MSK|IMG|LABEL", "", row["patch"])
    return (row["dept"], row["year"], row["roi"], re.sub(r"[_\-]+", "_", s).strip("_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the *_AERIAL_RGBI.zip files")
    ap.add_argument("--out", default="idx_flair")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    zips = sorted(f for f in os.listdir(args.root) if f.endswith(".zip") and any(f.endswith(f"_{m}.zip") for m in KINDS))
    if not zips:
        sys.exit(f"no *_AERIAL_RGBI.zip / *_AERIAL_LABEL-COSIA.zip found in {args.root}")
    print(f"{len(zips)} zips found")
    with Pool(args.workers) as pool:
        results = pool.map(scan, [os.path.join(args.root, z) for z in zips])

    images = [r for mod, rows in results if KINDS.get(mod) == "image" for r in rows]
    labels = [r for mod, rows in results if KINDS.get(mod) == "label" for r in rows]
    print(f"{len(images)} image patches, {len(labels)} label patches")

    # join labels to images (heuristic; check the printed match rate)
    lab = {patch_key(r): r for r in labels}
    hit = 0
    for r in images:
        m = lab.get(patch_key(r))
        r["label_zip"], r["label_member"] = (m["zip"], m["member"]) if m else ("", "")
        hit += m is not None
    print(f"labels matched to images: {hit}/{len(images)} ({100 * hit / max(1, len(images)):.1f}%)")

    os.makedirs(args.out, exist_ok=True)
    for name, rows, cols in (("flair_images.csv", images, ["zip", "member", "size", "offset", "csize", "ctype", "dept", "year", "roi", "patch", "split", "label_zip", "label_member"]),
                             ("flair_labels.csv", labels, ["zip", "member", "size", "offset", "csize", "ctype", "dept", "year", "roi", "patch", "split"])):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print("wrote", os.path.join(args.out, "flair_images.csv"), "and flair_labels.csv")

    print("\nimage patches per split:", dict(Counter(r["split"] for r in images)))
    print("departments (train):", len({r["dept"] for r in images if r["split"] == "train"}),
          " (test):", len({r["dept"] for r in images if r["split"] == "test"}))
    print("\nexample image members:", [r["member"] for r in images[:3]])
    print("example label members:", [r["member"] for r in labels[:3]])
    if hit < 0.9 * len(images):
        print("\nWARNING: low label match rate -> the naming heuristic is wrong; paste the example members above.")


if __name__ == "__main__":
    main()
