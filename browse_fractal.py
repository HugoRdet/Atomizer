"""
browse_fractal_patches.py
===========================

Step 1 of 2 for the FRACTAL figure export. Scans the FRACTAL dataset for
available patches (VHR ortho + matching LIDAR .laz) and saves a contact
sheet of small RGB thumbnails, labeled with patch_id, so you can pick a
good-looking patch before running the full export (save_fractal_figs.py).

This bypasses `FractalPerceiverDataset.__getitem__` entirely (no LIDAR
Fourier encoding, no query building, no torch tensors beyond a quick
uint8 preview) — it only reuses the dataset class's `_collect_patches`
method to get the list of valid (ortho, laz) pairs, then reads each
ortho directly with rasterio for a cheap thumbnail. Loaded from its .py
file directly, bypassing any package __init__.py.

Usage:
    python browse_fractal_patches.py --root ./data --split train --n 24
    # -> writes ./figs/browse/contact_sheet.png and prints the patch_id
    #    for each tile index so you can pick one.
"""

import argparse
import importlib.machinery
import importlib.util
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def _load_dataset_module(module_path: str):
    """
    Load a dataset .py file directly, while still allowing its own
    `from .sibling_module import X` relative imports to resolve.

    The file normally lives inside a real package (e.g.
    training/utils/datasets_baselines/), and importing it through that
    package would execute training/__init__.py, pulling in unrelated
    heavy deps (pytorch_lightning, transformers, ...) that are slow on a
    network filesystem and irrelevant here.

    Instead we register a *namespace* package in sys.modules pointing at
    the file's own directory (no __init__.py execution), then load the
    target module as a submodule of that fake package. This lets
    `from .augmentations import D4Augmentation`-style relative imports
    find "augmentations.py" next to the target file, without ever
    importing the real "training" package.
    """
    module_path = os.path.abspath(module_path)
    pkg_dir = os.path.dirname(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    pkg_name = "_fractal_ds_pkg"

    if pkg_name not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
        pkg_spec.submodule_search_locations = [pkg_dir]
        pkg_module = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg_module

    full_name = f"{pkg_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_class(module_path: str):
    return _load_dataset_module(module_path).FractalPerceiverDataset


def percentile_stretch(channel: np.ndarray, low=2, high=98):
    lo, hi = np.percentile(channel, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (channel.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def read_rgb_thumbnail(ortho_path: str) -> np.ndarray:
    """Ortho bands are [NIR, R, G, B] per the dataset docstring -> indices 1,2,3."""
    with rasterio.open(ortho_path) as src:
        ortho = src.read().astype(np.float32)  # [4, H, W]
    r = percentile_stretch(ortho[1])
    g = percentile_stretch(ortho[2])
    b = percentile_stretch(ortho[3])
    return np.stack([r, g, b], axis=-1)  # [H, W, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data", help="Root containing FRACTAL/ and FRACTAL-IRGB/")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "validation"])
    parser.add_argument("--n", type=int, default=24, help="Number of thumbnails to show")
    parser.add_argument("--seed", type=int, default=0, help="Random sample seed (use -1 for the first N patches in order instead of random)")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--outdir", default="./figs/browse")
    parser.add_argument(
        "--dataset_file",
        default="./training/utils/datasets_baselines/utils_dataset_fractal_perceiver.py",
        help="Path to the .py file defining FractalPerceiverDataset",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    FractalPerceiverDataset = _load_dataset_class(args.dataset_file)

    # Instantiate only to reuse _collect_patches; augmentation/LIDAR params
    # are irrelevant since we never call __getitem__.
    ds = FractalPerceiverDataset(
        root_path=args.root,
        mode=args.split,
        use_augmentation=False,
    )

    rows = ds.patch_rows
    print(f"[browse] {len(rows)} candidate patches found for split='{args.split}'")

    if args.seed == -1:
        chosen = rows[: args.n]
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(rows, k=min(args.n, len(rows)))

    n = len(chosen)
    cols = args.cols
    rows_n = math.ceil(n / cols)

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.2, rows_n * 2.4))
    axes = np.atleast_1d(axes).reshape(-1)

    print("\nindex : patch_id")
    for i, row in enumerate(chosen):
        try:
            thumb = read_rgb_thumbnail(row["ortho_path"])
        except Exception as e:
            print(f"[browse] failed to read {row['patch_id']}: {e}")
            thumb = np.zeros((10, 10, 3), dtype=np.float32)
        ax = axes[i]
        ax.imshow(thumb)
        ax.set_title(f"{i}", fontsize=8)
        ax.set_axis_off()
        print(f"{i:5d} : {row['patch_id']}")

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    fig.tight_layout()
    out_path = os.path.join(args.outdir, "contact_sheet.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}")
    print("Pick a tile index above, note its patch_id, then run save_fractal_figs.py --patch_id <id>")


if __name__ == "__main__":
    main()
