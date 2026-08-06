"""
BioMassters prep script
========================

Your download directory has multi-part archives:
    train_features.zip + train_features.z01 ... z13
    test_features_splits.zip + test_features_splits.z01 ... z04
    train_agbm.zip
    test_agbm.tar
    The_BioMassters_-_features_metadata.csv.csv
    The_BioMassters_-_train_agbm_metadata.csv.csv

Multi-part zips (.zip + .z01, .z02, ...) must be concatenated/repaired with
`zip -F` (or `zip -FF`) before extraction -- a plain `unzip` on the .zip alone
will fail or silently truncate.

This script:
    1. Repairs + extracts train_features / test_features_splits / train_agbm / test_agbm
    2. Reads the features metadata CSV and groups files by chip_id
    3. Writes a chip-indexed manifest (one row per chip) that the Dataset class consumes,
       so __getitem__ never has to re-scan the filesystem.

Run once:
    python prepare_biomassters.py --root /path/to/download_dir --out ./data/biomassters
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def repair_and_extract_multipart_zip(root: Path, zip_stem: str, dest: Path):
    """
    zip_stem e.g. 'train_features' -> expects train_features.zip + train_features.z01..zNN
    """
    root = root.resolve()
    dest = dest.resolve()

    zip_path = root / f"{zip_stem}.zip"
    if not zip_path.exists():
        print(f"[skip] {zip_path} not found")
        return

    marker = dest / ".extraction_complete"
    if marker.exists():
        n_existing = sum(1 for _ in dest.rglob("*.tif"))
        print(f"[skip] {dest} extraction already complete ({n_existing} .tif files), skipping {zip_stem}")
        return

    dest.mkdir(parents=True, exist_ok=True)

    fixed_zip = root / f"{zip_stem}_fixed.zip"
    fixed_zip.unlink(missing_ok=True)  # zip -FF refuses to overwrite an existing --out file

    # `zip -FF` scans the directory for the .zNN parts automatically as long as
    # they share the same stem and live next to the .zip. Using fully-resolved
    # absolute paths for both zip_path and --out avoids relying on cwd, which
    # previously caused zip to look for its temp file under a doubled/nonexistent path.
    result = subprocess.run(
        ["zip", "-FF", str(zip_path), "--out", str(fixed_zip)],
        input=b"y\n",  # zip -FF sometimes prompts to confirm before proceeding
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout.decode(errors="replace"))
        print(result.stderr.decode(errors="replace"))
        raise RuntimeError(
            f"zip -FF failed on {zip_path} (exit {result.returncode}). "
            f"Check that all .z01..zNN parts are present alongside {zip_path.name} "
            f"and that {root} has write space for the repaired copy."
        )

    print(f"Extracting {zip_stem} (this may take a while, no per-file progress)...")
    with zipfile.ZipFile(fixed_zip) as zf:
        zf.extractall(path=str(dest))
    fixed_zip.unlink(missing_ok=True)
    marker.write_text("ok")
    print(f"[ok] extracted {zip_stem} -> {dest}")


def extract_single_zip(root: Path, zip_stem: str, dest: Path):
    root = root.resolve()
    dest = dest.resolve()
    zip_path = root / f"{zip_stem}.zip"
    if not zip_path.exists():
        print(f"[skip] {zip_path} not found")
        return

    marker = dest / ".extraction_complete"
    if marker.exists():
        n_existing = sum(1 for _ in dest.rglob("*.tif"))
        print(f"[skip] {dest} extraction already complete ({n_existing} .tif files), skipping {zip_stem}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_stem} (this may take a while, no per-file progress)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=str(dest))
    marker.write_text("ok")
    print(f"[ok] extracted {zip_stem} -> {dest}")


def extract_tar(root: Path, tar_stem: str, dest: Path):
    root = root.resolve()
    dest = dest.resolve()
    tar_path = root / f"{tar_stem}.tar"
    if not tar_path.exists():
        print(f"[skip] {tar_path} not found")
        return

    marker = dest / ".extraction_complete"
    if marker.exists():
        n_existing = sum(1 for _ in dest.rglob("*.tif"))
        print(f"[skip] {dest} extraction already complete ({n_existing} .tif files), skipping {tar_stem}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_stem} (this may take a while, no per-file progress)...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    marker.write_text("ok")
    print(f"[ok] extracted {tar_stem} -> {dest}")


MONTH_NAME_TO_IDX = {
    name: i for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    )
}


def parse_month(value) -> int:
    """
    The official BioMassters metadata 'month' column has been observed in the wild as
    either a 0-11 integer (chip-relative acquisition slot) or a calendar month name
    (e.g. 'September'). Handle both so this script doesn't break on either variant.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    if s in MONTH_NAME_TO_IDX:
        return MONTH_NAME_TO_IDX[s]
    raise ValueError(f"Could not parse month value: {value!r}")


def build_manifest(root: Path, out_dir: Path):
    """
    Reads the official features metadata CSV and groups rows by chip_id,
    producing one JSON manifest per split with the file paths for each
    available (chip_id, satellite, month) triplet.

    Expected columns in The_BioMassters_-_features_metadata.csv.csv:
        chip_id, filename, satellite, month, split
    where satellite in {'S1', 'S2'} and month in 0..11 (relative month index,
    NOT calendar month -- BioMassters months are chip-relative acquisition order).
    """
    meta_path = root / "The_BioMassters_-_features_metadata.csv.csv"

    existing = [out_dir / f"manifest_{s}.json" for s in ("train", "test")]
    if all(p.exists() for p in existing):
        print(f"[skip] manifests already exist ({[p.name for p in existing]}), skipping build_manifest")
        return

    df = pd.read_csv(meta_path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"chip_id", "filename", "satellite", "month", "split"}.issubset(df.columns), \
        f"Unexpected columns: {df.columns.tolist()}"

    agbm_meta_path = root / "The_BioMassters_-_train_agbm_metadata.csv.csv"
    agbm_df = pd.read_csv(agbm_meta_path)
    agbm_df.columns = [c.strip().lower() for c in agbm_df.columns]
    agbm_lookup = {row["chip_id"]: row["corresponding_agbm"] if "corresponding_agbm" in agbm_df.columns
                   else f"{row['chip_id']}_agbm.tif" for _, row in agbm_df.iterrows()}

    manifests = defaultdict(lambda: defaultdict(lambda: {"S1": {}, "S2": {}}))
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping metadata by chip"):
        chip_id = row["chip_id"]
        split = row["split"]
        sat = row["satellite"].upper()
        month = parse_month(row["month"])
        manifests[split][chip_id][sat][month] = row["filename"]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, chips in manifests.items():
        records = []
        for chip_id, sats in chips.items():
            records.append({
                "chip_id": chip_id,
                "s1_files": sats["S1"],   # dict: month(int) -> filename
                "s2_files": sats["S2"],   # dict: month(int) -> filename
                # Deterministic naming ({chip_id}_agbm.tif), confirmed against the
                # official PANGAEA BioMassters loader, applies to EVERY split --
                # not just train. Earlier this was gated to split=="train" only,
                # which silently discarded real test_agbm/ labels even when they
                # existed on disk (this dataset release ships test labels, unlike
                # the live DrivenData competition where the test set is withheld).
                "agbm_file": f"{chip_id}_agbm.tif",
            })
        out_path = out_dir / f"manifest_{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"[ok] wrote {out_path} ({len(records)} chips)")


def print_summary(out_dir: Path):
    """
    Prints a quick sanity-check summary per split: chip counts, how many months are
    actually present per chip for S1/S2 (BioMassters is notorious for missing S1 months),
    and basic AGB target stats for the train split.
    """
    for split in ["train", "test"]:
        manifest_path = out_dir / f"manifest_{split}.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            records = json.load(f)

        n_chips = len(records)
        if n_chips == 0:
            print(f"\n=== {split.upper()} split === (empty)")
            continue
        s1_counts = [len(r["s1_files"]) for r in records]
        s2_counts = [len(r["s2_files"]) for r in records]
        n_complete_s1 = sum(c == N_MONTHS_EXPECTED for c in s1_counts)
        n_complete_s2 = sum(c == N_MONTHS_EXPECTED for c in s2_counts)

        print(f"\n=== {split.upper()} split ===")
        print(f"  chips:                {n_chips}")
        print(f"  S1 months/chip:       min={min(s1_counts)}, max={max(s1_counts)}, "
              f"mean={sum(s1_counts)/n_chips:.2f}, complete(12/12)={n_complete_s1} "
              f"({100*n_complete_s1/n_chips:.1f}%)")
        print(f"  S2 months/chip:       min={min(s2_counts)}, max={max(s2_counts)}, "
              f"mean={sum(s2_counts)/n_chips:.2f}, complete(12/12)={n_complete_s2} "
              f"({100*n_complete_s2/n_chips:.1f}%)")

        n_missing_agbm = sum(1 for r in records if not r.get("agbm_file"))
        print(f"  chips missing AGBM:   {n_missing_agbm} ({100*n_missing_agbm/n_chips:.1f}%)")

        # extraction footprint, for a sense of disk usage
        feat_dir = out_dir / f"{split}_features"
        if feat_dir.exists():
            n_files = sum(1 for _ in feat_dir.rglob("*.tif"))
            size_gb = sum(f.stat().st_size for f in feat_dir.rglob("*.tif")) / 1e9
            print(f"  extracted tif files:  {n_files} ({size_gb:.1f} GB)")

    print()


N_MONTHS_EXPECTED = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing the raw downloads")
    ap.add_argument("--out", required=True, help="Output directory for extracted/organized data")
    ap.add_argument("--force", action="store_true",
                     help="Re-extract / rebuild manifests even if outputs already exist")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if args.force:
        for d in [out / "train_features", out / "test_features", out / "train_agbm", out / "test_agbm"]:
            if d.exists():
                shutil.rmtree(d)
        for f in [out / "manifest_train.json", out / "manifest_test.json"]:
            f.unlink(missing_ok=True)

    repair_and_extract_multipart_zip(root, "train_features", out / "train_features")
    repair_and_extract_multipart_zip(root, "test_features_splits", out / "test_features")
    extract_single_zip(root, "train_agbm", out / "train_agbm")
    extract_tar(root, "test_agbm", out / "test_agbm")

    # copy metadata CSVs alongside the manifests for provenance
    for fname in ["The_BioMassters_-_features_metadata.csv.csv",
                  "The_BioMassters_-_train_agbm_metadata.csv.csv"]:
        src = root / fname
        if src.exists():
            shutil.copy(src, out / fname)

    build_manifest(out, out)
    print_summary(out)


if __name__ == "__main__":
    main()
