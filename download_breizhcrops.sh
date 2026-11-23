#!/bin/bash
# Download BreizhCrops (2017, L1C and L2A, regions FRH01-04) into the folder layout
# expected by the official `breizhcrops` package:
#
#   <ROOT>/codes.csv, classmapping.csv
#   <ROOT>/2017/<LEVEL>/<region>.csv   (index: one row per parcel)
#   <ROOT>/2017/<LEVEL>/<region>.h5    (time series, one dataset per parcel)
#
# Usage:  bash download_breizhcrops.sh [ROOT] [LEVELS]
#   e.g.  bash download_breizhcrops.sh ./data/BREIZHCROPS "L1C L2A"
# Safe to rerun: finished files are skipped, partial downloads resume.

set -euo pipefail
ROOT=${1:-./data/BREIZHCROPS}
LEVELS=${2:-"L1C L2A"}
REGIONS="frh01 frh02 frh03 frh04"
BASE=https://breizhcrops.s3.eu-central-1.amazonaws.com

# expected .h5 sizes (bytes), from breizhcrops/datasets/urls.py
declare -A SIZE=(
  [L1C_frh01]=2559635960 [L1C_frh02]=2253658856 [L1C_frh03]=2493572704 [L1C_frh04]=1555075632
  [L2A_frh01]=987259904  [L2A_frh02]=803457960  [L2A_frh03]=890027448  [L2A_frh04]=639215848
)

mkdir -p "$ROOT"
for f in codes.csv classmapping.csv; do
  [ -s "$ROOT/$f" ] || curl -sSfL --retry 5 -o "$ROOT/$f" "$BASE/$f"
done

for LEVEL in $LEVELS; do
  DIR="$ROOT/2017/$LEVEL"; mkdir -p "$DIR"
  for R in $REGIONS; do
    H5="$DIR/$R.h5"
    # index file
    [ -s "$DIR/$R.csv" ] || curl -sSfL --retry 5 -o "$DIR/$R.csv" "$BASE/2017/$LEVEL/$R.csv"

    # time series database
    if [ -f "$H5" ] && [ "$(stat -c %s "$H5")" = "${SIZE[${LEVEL}_$R]}" ]; then
      echo "[ok]   $H5"; continue
    fi
    echo "[get]  $LEVEL $R"
    curl -fL --retry 5 -C - -o "$H5.tar.gz" "$BASE/2017/$LEVEL/$R.h5.tar.gz"
    # extract with paths stripped, as the official package does
    python - "$H5.tar.gz" "$DIR" <<'EOF'
import os, sys, tarfile
src, dst = sys.argv[1], sys.argv[2]
with tarfile.open(src, "r:gz") as tar:
    for m in tar.getmembers():
        if m.isreg():
            m.name = os.path.basename(m.name)
            tar.extract(m, dst)
EOF
    got=$(stat -c %s "$H5" 2>/dev/null || echo 0)
    if [ "$got" = "${SIZE[${LEVEL}_$R]}" ]; then
      rm -f "$H5.tar.gz"; echo "[done] $H5"
    else
      echo "[FAIL] $H5 has $got bytes, expected ${SIZE[${LEVEL}_$R]} (archive kept)"; exit 1
    fi
  done
done
echo "All done."
