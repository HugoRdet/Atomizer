#!/bin/bash
# Download HLS Burn Scars dataset from HuggingFace
# Usage: bash download_hlsburnscars.sh [DATA_DIR]
# Default DATA_DIR is ./data/hls_burn_scars

DATA_DIR="${1:-./data/hls_burn_scars}"
mkdir -p "$DATA_DIR"

echo "=== Downloading HLS Burn Scars dataset ==="
echo "Target directory: $DATA_DIR"

# Method 1: Using huggingface_hub (preferred)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ibm-nasa-geospatial/hls_burn_scars',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
)
print('Download complete via huggingface_hub')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "huggingface_hub not available, trying wget..."
    
    # Method 2: Direct download of the tar.gz
    wget -c "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/resolve/main/hls_burn_scars.tar.gz" \
        -O "$DATA_DIR/hls_burn_scars.tar.gz"
    
    echo "Extracting..."
    tar -xzf "$DATA_DIR/hls_burn_scars.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/hls_burn_scars.tar.gz"
    
    echo "Download and extraction complete via wget"
fi

echo ""
echo "=== Verifying dataset structure ==="
echo "Directory contents:"
find "$DATA_DIR" -type d | head -20
echo ""
echo "File counts:"
echo "  Training scenes:    $(find "$DATA_DIR" -path '*/training/scenes/*' -name '*.tif' 2>/dev/null | wc -l)"
echo "  Training masks:     $(find "$DATA_DIR" -path '*/training/masks/*' -name '*.tif' 2>/dev/null | wc -l)"
echo "  Validation scenes:  $(find "$DATA_DIR" -path '*/validation/scenes/*' -name '*.tif' 2>/dev/null | wc -l)"
echo "  Validation masks:   $(find "$DATA_DIR" -path '*/validation/masks/*' -name '*.tif' 2>/dev/null | wc -l)"
echo ""
echo "Sample files:"
find "$DATA_DIR" -name '*.tif' | head -5