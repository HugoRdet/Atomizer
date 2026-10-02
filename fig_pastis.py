"""
save_pastis_figs.py
=====================

Exports Sentinel-2 (RGB) and Sentinel-1 (SAR) channels from a PASTIS-HD
patch as individual SVG figures — for the **first** and **last** acquisition
of the raw time series (before any temporal subsampling), for use in a
paper figure.

Like save_senflood_figs.py, this loads `PastisBaselineDataset` only to
reuse its metadata/file-loading logic (`_load_s2`, `_load_s1`, patch
listing), but bypasses its z-score normalization for display purposes:
each channel is percentile-stretched to [0, 1] independently instead.
The dataset class is loaded directly from its .py file, bypassing any
package __init__.py, so heavy unrelated imports (pytorch_lightning,
transformers, etc.) never get pulled in.

S2 band order (10 bands, indices 0..9), from S2_WAVELENGTHS in the
dataset module: 490,560,665,705,740,783,842,865,1610,2190 nm
    -> index 2 (665nm) = Red   (~B04)
       index 1 (560nm) = Green (~B03)
       index 0 (490nm) = Blue  (~B02)

S1 band order (3 bands): VV, VH, VV-VH (as stored by the dataset itself),
so the "combined SAR" figure is just those 3 channels stacked as RGB.

Outputs (in ./figs), one set per frame ("first"/"last"):
    s2_R_<frame>.svg, s2_G_<frame>.svg, s2_B_<frame>.svg
    s2_RGB_<frame>.svg
    s1_VV_<frame>.svg, s1_VH_<frame>.svg
    s1_SAR_<frame>.svg   (VV, VH, VV-VH stacked as RGB)

Usage:
    python save_pastis_figs.py --root ./data/PASTIS-HD --index 0
"""

import argparse
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_dataset_class(module_path: str):
    """Load PastisBaselineDataset directly from a file path, bypassing
    any package __init__.py."""
    spec = importlib.util.spec_from_file_location("_pastis_ds_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PastisBaselineDataset


# S2_WAVELENGTHS = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
S2_RGB_INDEX = {"R": 2, "G": 1, "B": 0}  # 665nm, 560nm, 490nm


def percentile_stretch(channel: np.ndarray, low=2, high=98):
    """Clip to [low, high] percentiles and rescale to [0, 1]."""
    lo, hi = np.percentile(channel, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (channel.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def save_gray_svg(channel_01: np.ndarray, path: str, cmap="gray"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(channel_01, cmap=cmap, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


def save_rgb_svg(rgb_01: np.ndarray, path: str):
    """rgb_01: [H, W, 3] in [0, 1]."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb_01)
    ax.set_axis_off()
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


def save_frame(s2_frame: np.ndarray, s1_frame: np.ndarray, outdir: str,
                tag: str, sar_cmap: str):
    """
    s2_frame: [10, H, W] raw (unnormalized) S2 bands for one date.
    s1_frame: [3, H, W]  raw (unnormalized) S1 bands (VV, VH, VV-VH) for one date.
    tag: "first" or "last" — used in output filenames.
    """
    r = percentile_stretch(s2_frame[S2_RGB_INDEX["R"]])
    g = percentile_stretch(s2_frame[S2_RGB_INDEX["G"]])
    b = percentile_stretch(s2_frame[S2_RGB_INDEX["B"]])

    save_gray_svg(r, os.path.join(outdir, f"s2_R_{tag}.svg"))
    save_gray_svg(g, os.path.join(outdir, f"s2_G_{tag}.svg"))
    save_gray_svg(b, os.path.join(outdir, f"s2_B_{tag}.svg"))
    save_rgb_svg(np.stack([r, g, b], axis=-1), os.path.join(outdir, f"s2_RGB_{tag}.svg"))

    vv = percentile_stretch(s1_frame[0])
    vh = percentile_stretch(s1_frame[1])
    vv_vh = percentile_stretch(s1_frame[2])

    save_gray_svg(vv, os.path.join(outdir, f"s1_VV_{tag}.svg"), cmap=sar_cmap)
    save_gray_svg(vh, os.path.join(outdir, f"s1_VH_{tag}.svg"), cmap=sar_cmap)
    save_rgb_svg(np.stack([vv, vh, vv_vh], axis=-1),
                 os.path.join(outdir, f"s1_SAR_{tag}.svg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/PASTIS-HD",
                         help="Path to PASTIS-HD dataset root")
    parser.add_argument("--split", default="test",
                         choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=0,
                         help="Patch index within the chosen split")
    parser.add_argument("--outdir", default="./figs")
    parser.add_argument("--sar_cmap", default="viridis",
                         help="Matplotlib colormap for the VV/VH grayscale figures")
    parser.add_argument(
        "--dataset_file",
        default="./training/utils/datasets_baselines/utils_dataset_pastis_baselines.py",
        help="Path to the .py file defining PastisBaselineDataset",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    PastisBaselineDataset = _load_dataset_class(args.dataset_file)

    # augment=False so nothing gets rotated/flipped; use_s1=True to get SAR too.
    # multi_temporal/max_temporal_samples are irrelevant here since we bypass
    # __getitem__ and call the raw loaders directly to get the untouched,
    # full-length time series.
    ds = PastisBaselineDataset(
        root_path=args.root,
        mode=args.split,
        use_s1=True,
        augment=False,
    )

    patch_row = ds.metadata.iloc[args.index]
    patch_id = patch_row["ID_PATCH"]
    print(f"Patch ID: {patch_id}")

    s2_data, s2_dates = ds._load_s2(patch_id, patch_row)  # [T, 10, H, W]
    s1_data, s1_dates = ds._load_s1(patch_id, patch_row)  # [T, 3, H, W]

    s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0).numpy()
    s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0).numpy()

    print(f"S2 time series length: {s2_data.shape[0]} (dates: {s2_dates[0]} .. {s2_dates[-1]})")
    print(f"S1 time series length: {s1_data.shape[0]} (dates: {s1_dates[0]} .. {s1_dates[-1]})")

    # S2 and S1 are acquired on different dates/cadences, so "first"/"last"
    # is taken independently per sensor rather than forcing a shared index.
    save_frame(s2_data[0], s1_data[0], args.outdir, "first", args.sar_cmap)
    save_frame(s2_data[-1], s1_data[-1], args.outdir, "last", args.sar_cmap)


if __name__ == "__main__":
    main()
