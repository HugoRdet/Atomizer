"""
save_senflood_figs.py
======================

Exports Sentinel-2 (RGB) and Sentinel-1 (SAR) channels from a Sen1Floods11
sample as individual SVG figures, for use in a paper figure.

Uses `Sen1Floods11BaselineDataset` to load the raw, aligned S1/S2 rasters
(reusing its file lists / splits), but does its own **display-oriented**
normalization instead of the dataset's z-score normalization. Z-scored
values are centered at 0 with unbounded range, which is fine for a network
but not for `imshow` — so here each channel is percentile-stretched to
[0, 1] independently before saving.

Outputs (in ./figs):
    s2_R.svg, s2_G.svg, s2_B.svg   - individual Sentinel-2 RGB bands, grayscale
    s2_RGB.svg                     - combined true-color composite
    s1_VV.svg, s1_VH.svg           - individual SAR channels, grayscale
    s1_SAR.svg                     - VV/VH combined into one image via colormap
                                      (VV -> R, VH -> G, VV/VH ratio -> B is a
                                      common false-color SAR composite; here we
                                      instead apply a perceptual colormap to
                                      each single-channel SAR image separately,
                                      see below for the "combined" version)

Usage:
    python save_senflood_figs.py --root ./data/SENFLOOD --index 0 --split test
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import Sen1Floods11BaselineDataset


# Sentinel-2 band order in this dataset's rasters (13 bands, standard L2A
# ordering): B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12
# True color = B04 (Red), B03 (Green), B02 (Blue)
S2_BAND_INDEX = {"B02": 1, "B03": 2, "B04": 3}
RGB_BAND_ORDER = ["B04", "B03", "B02"]  # R, G, B

# Sentinel-1: VV, VH (order used throughout the dataset code)
S1_BAND_INDEX = {"VV": 0, "VH": 1}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/SENFLOOD",
                         help="Path to Sen1Floods11 dataset root")
    parser.add_argument("--split", default="test",
                         choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=0,
                         help="Sample index within the chosen split")
    parser.add_argument("--outdir", default="./figures")
    parser.add_argument("--sar_cmap", default="viridis",
                         help="Matplotlib colormap for the combined SAR figure")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Instantiate the dataset only to reuse its split/file loading + filtering
    # logic; disable all training-time augmentation so we get the raw,
    # unmodified, full-size rasters.
    ds = Sen1Floods11BaselineDataset(
        root_path=args.root,
        mode=args.split,
        crop_size=None,
        augment=False,
        band_dropout=False,
    )

    s2_path = ds.s2_image_list[args.index]
    s1_path = ds.s1_image_list[args.index]
    print(f"S2 file: {s2_path}")
    print(f"S1 file: {s1_path}")

    with rasterio.open(s2_path) as src:
        s2 = src.read().astype(np.float32)  # [13, H, W]
    with rasterio.open(s1_path) as src:
        s1 = src.read().astype(np.float32)  # [2, H, W]

    s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Sentinel-2: individual RGB channels + combined composite ──────────
    rgb_stretched = {}
    for name in RGB_BAND_ORDER:
        band = s2[S2_BAND_INDEX[name]]
        stretched = percentile_stretch(band)
        rgb_stretched[name] = stretched
        save_gray_svg(stretched, os.path.join(args.outdir, f"s2_{name[1:]}.svg"))
        # also save under the plain R/G/B name expected in the request
    save_gray_svg(rgb_stretched["B04"], os.path.join(args.outdir, "s2_R.svg"))
    save_gray_svg(rgb_stretched["B03"], os.path.join(args.outdir, "s2_G.svg"))
    save_gray_svg(rgb_stretched["B02"], os.path.join(args.outdir, "s2_B.svg"))

    rgb_composite = np.stack(
        [rgb_stretched["B04"], rgb_stretched["B03"], rgb_stretched["B02"]],
        axis=-1,
    )  # [H, W, 3]
    save_rgb_svg(rgb_composite, os.path.join(args.outdir, "s2_RGB.svg"))

    # ── Sentinel-1: individual VV/VH + combined false-color composite ─────
    vv = percentile_stretch(s1[S1_BAND_INDEX["VV"]])
    vh = percentile_stretch(s1[S1_BAND_INDEX["VH"]])
    save_gray_svg(vv, os.path.join(args.outdir, "s1_VV.svg"), cmap=args.sar_cmap)
    save_gray_svg(vh, os.path.join(args.outdir, "s1_VH.svg"), cmap=args.sar_cmap)

    # Combined SAR figure: classic VV/VH/VV-VH false-color composite
    # (R=VV, G=VH, B=VV-VH) rather than colormapping a single channel,
    # since that's the standard way to show both channels at once.
    diff = percentile_stretch(s1[S1_BAND_INDEX["VV"]] - s1[S1_BAND_INDEX["VH"]])
    sar_composite = np.stack([vv, vh, diff], axis=-1)
    save_rgb_svg(sar_composite, os.path.join(args.outdir, "s1_SAR.svg"))


if __name__ == "__main__":
    main()
