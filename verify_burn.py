"""
Verify HLS Burn Scars dataset structure and inspect data properties.
Run after downloading to confirm compatibility with Atomizer.
"""
import os
import glob
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    import tifffile

DATA_DIR = "./data/hls_burn_scars"

# --- Find files ---
# The tar.gz may extract with a nested directory
possible_roots = [
    DATA_DIR,
    os.path.join(DATA_DIR, "hls_burn_scars"),
]

root = None
for r in possible_roots:
    if os.path.exists(os.path.join(r, "training")):
        root = r
        break

if root is None:
    print("ERROR: Could not find 'training/' directory. Check extraction.")
    print(f"Contents of {DATA_DIR}:")
    for f in os.listdir(DATA_DIR):
        print(f"  {f}")
    exit(1)

print(f"Dataset root: {root}")

# --- Count files ---
train_scenes = sorted(glob.glob(os.path.join(root, "training", "scenes", "*.tif")))
train_masks = sorted(glob.glob(os.path.join(root, "training", "masks", "*.tif")))
val_scenes = sorted(glob.glob(os.path.join(root, "validation", "scenes", "*.tif")))
val_masks = sorted(glob.glob(os.path.join(root, "validation", "masks", "*.tif")))

print(f"\n=== File Counts ===")
print(f"Train scenes: {len(train_scenes)}")
print(f"Train masks:  {len(train_masks)}")
print(f"Val scenes:   {len(val_scenes)}")
print(f"Val masks:    {len(val_masks)}")

# --- Inspect a sample scene ---
if len(train_scenes) > 0:
    sample_scene = train_scenes[0]
    sample_mask = train_masks[0]
    
    print(f"\n=== Sample Scene: {os.path.basename(sample_scene)} ===")
    
    if HAS_RASTERIO:
        with rasterio.open(sample_scene) as src:
            data = src.read()  # (bands, H, W)
            print(f"Shape: {data.shape}  (bands, H, W)")
            print(f"Dtype: {data.dtype}")
            print(f"CRS: {src.crs}")
            print(f"Resolution: {src.res}")
            print(f"Bounds: {src.bounds}")
            for i in range(data.shape[0]):
                band = data[i]
                print(f"  Band {i}: min={band.min():.4f}, max={band.max():.4f}, "
                      f"mean={band.mean():.4f}, std={band.std():.4f}")
        
        with rasterio.open(sample_mask) as src:
            mask = src.read(1)  # single band
            print(f"\n=== Sample Mask: {os.path.basename(sample_mask)} ===")
            print(f"Shape: {mask.shape}")
            print(f"Dtype: {mask.dtype}")
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                label = {-1: "missing", 0: "not burned", 1: "burn scar"}.get(u, f"unknown({u})")
                print(f"  Value {u:3d} ({label}): {c} pixels ({100*c/mask.size:.1f}%)")
    else:
        data = tifffile.imread(sample_scene)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        if data.ndim == 3:
            # Could be (H, W, bands) or (bands, H, W)
            print(f"NOTE: Check axis order - shape is {data.shape}")
            if data.shape[0] == 6:
                print("  -> Likely (bands, H, W)")
                for i in range(6):
                    print(f"  Band {i}: min={data[i].min():.4f}, max={data[i].max():.4f}, "
                          f"mean={data[i].mean():.4f}")
            elif data.shape[-1] == 6:
                print("  -> Likely (H, W, bands)")
                for i in range(6):
                    print(f"  Band {i}: min={data[:,:,i].min():.4f}, max={data[:,:,i].max():.4f}, "
                          f"mean={data[:,:,i].mean():.4f}")
        
        mask = tifffile.imread(sample_mask)
        print(f"\n=== Sample Mask ===")
        print(f"Shape: {mask.shape}, Dtype: {mask.dtype}")
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            label = {-1: "missing", 0: "not burned", 1: "burn scar"}.get(u, f"unknown({u})")
            print(f"  Value {u:3d} ({label}): {c} pixels ({100*c/mask.size:.1f}%)")

    # --- Check value range (reflectance scaling) ---
    print(f"\n=== Reflectance Scaling Check ===")
    if HAS_RASTERIO:
        with rasterio.open(sample_scene) as src:
            data = src.read().astype(np.float32)
    else:
        data = tifffile.imread(sample_scene).astype(np.float32)
    
    global_max = data.max()
    if global_max > 100:
        print(f"Global max = {global_max:.1f} -> Values are likely scaled (e.g., x10000)")
        print(f"  Divide by 10000 to get [0, 1] reflectance")
    elif global_max > 1:
        print(f"Global max = {global_max:.4f} -> Check scaling convention")
    else:
        print(f"Global max = {global_max:.4f} -> Already in [0, 1] reflectance range")

    # --- Compute dataset-wide stats (sample a few images) ---
    print(f"\n=== Band Statistics (sampled from first 50 images) ===")
    band_names = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B8A (NIR)", "B11 (SWIR1)", "B12 (SWIR2)"]
    all_means = []
    all_stds = []
    n_sample = min(50, len(train_scenes))
    for scene_path in train_scenes[:n_sample]:
        if HAS_RASTERIO:
            with rasterio.open(scene_path) as src:
                d = src.read().astype(np.float32)
        else:
            d = tifffile.imread(scene_path).astype(np.float32)
        if d.shape[0] == 6:
            all_means.append(d.mean(axis=(1, 2)))
            all_stds.append(d.std(axis=(1, 2)))
    
    if all_means:
        means = np.array(all_means).mean(axis=0)
        stds = np.array(all_stds).mean(axis=0)
        print(f"{'Band':<15} {'Mean':>10} {'Std':>10}")
        print("-" * 37)
        for i, name in enumerate(band_names):
            print(f"{name:<15} {means[i]:>10.4f} {stds[i]:>10.4f}")

print("\n=== Done ===")