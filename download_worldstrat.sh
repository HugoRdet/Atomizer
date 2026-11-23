#!/usr/bin/env bash
# Download WorldStrat (Zenodo record 6810792, v1) for DINOv3 + OlmoEarth distillation.
# Usage: bash download_worldstrat.sh /path/to/data/WorldStrat [--with-l1c] [--with-raw]
set -euo pipefail

DEST="${1:-./WorldStrat}"; shift || true
WITH_L1C=0; WITH_RAW=0
for a in "$@"; do
  case "$a" in
    --with-l1c) WITH_L1C=1 ;;
    --with-raw) WITH_RAW=1 ;;
  esac
done

BASE="https://zenodo.org/records/6810792/files"
mkdir -p "$DEST/archives"
cd "$DEST"

# name  md5
FILES=(
  "metadata.csv dfeb3348e79b719bf03c230d5d258839"
  "stratified_train_val_test_split.csv 745035835d835280aa0298a9dc1996d1"
  "WorldStrat_article_and_datasheet.pdf 58d8e87c52bfaec962ab1ca5c3bf48b1"
  "LICENSE.txt d97b8d86da83f7e51f2d3205509e4a7b"
  "hr_dataset.tar.gz ca7167334006f3c17f9071f14c435335"
  "lr_dataset_l2a.tar.gz 8cfc6a477cee9e9cd8b20ea27227de65"
)
[ "$WITH_L1C" = 1 ] && FILES+=("lr_dataset_l1c.tar.gz d2dcafa207b1e1bc6c754607f15e9ed6")
[ "$WITH_RAW" = 1 ] && FILES+=("hr_dataset_raw.tar.gz 40a4fd5241e91bffeddee0ed109778da")

for entry in "${FILES[@]}"; do
  name="${entry%% *}"; md5="${entry##* }"
  case "$name" in *.tar.gz) out="archives/$name" ;; *) out="$name" ;; esac

  if [ -f "$out" ] && echo "$md5  $out" | md5sum -c --status; then
    echo "[ok]   $name (already downloaded)"
    continue
  fi
  echo "[get]  $name"
  # -c resumes interrupted downloads; retries help with Zenodo throttling
  wget -c --tries=20 --waitretry=30 --read-timeout=120 \
       -O "$out" "$BASE/$name?download=1"
  echo "$md5  $out" | md5sum -c || { echo "md5 mismatch for $name"; exit 1; }
done

# Extract (skips archives already extracted, marked by a .done file)
for tgz in archives/*.tar.gz; do
  marker="${tgz%.tar.gz}.done"
  [ -f "$marker" ] && { echo "[skip] $(basename "$tgz") already extracted"; continue; }
  echo "[untar] $(basename "$tgz")"
  if command -v pigz >/dev/null; then
    tar -I pigz -xf "$tgz" -C "$DEST"
  else
    tar -xzf "$tgz" -C "$DEST"
  fi
  touch "$marker"
done

echo "Done. Contents of $DEST:"
du -sh "$DEST"/* 2>/dev/null
echo "Delete $DEST/archives/*.tar.gz once you've checked the extracted data."
