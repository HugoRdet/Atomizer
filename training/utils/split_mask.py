"""
Split Mask Builder — Build spatial mask from crop index split column
====================================================================

Creates a binary mask [H, W] marking which pixels belong to each split.
Used by the eval script to restrict metrics to test-split pixels only.

Usage:
    from split_mask import build_split_mask, plot_split_overlay

    mask = build_split_mask(
        crop_index_path="c2seg_crop_index_split.csv",
        city="augsburg", subset="germany",
        split="test", label_shape=(886, 1360),
    )
    # mask[i,j] = True if pixel (i,j) is in the test split
"""

import csv
import numpy as np


def build_split_mask(
    crop_index_path: str,
    city: str,
    subset: str,
    split: str,
    label_shape: tuple,
) -> np.ndarray:
    """
    Build a boolean mask marking pixels belonging to a specific split.

    Parameters
    ----------
    crop_index_path : str
        Path to CSV with 'split' column.
    city : str
        City to filter (e.g., 'augsburg').
    subset : str
        Subset to filter (e.g., 'germany').
    split : str
        Split to select (e.g., 'test', 'val', 'train').
    label_shape : tuple
        (H, W) of the full label image at 10m.

    Returns
    -------
    np.ndarray [H, W] bool — True where pixel belongs to the split.
    """
    H, W = label_shape
    mask = np.zeros((H, W), dtype=bool)

    with open(crop_index_path) as f:
        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])

        if not has_split:
            print(f"[SplitMask] WARNING: no 'split' column in {crop_index_path}")
            print(f"[SplitMask] Returning all-True mask (no filtering)")
            return np.ones((H, W), dtype=bool)

        n_crops = 0
        for row in reader:
            if row["city"] != city or row["subset"] != subset:
                continue
            if row.get("split", "") != split:
                continue

            r0 = int(row["row_10m"])
            c0 = int(row["col_10m"])
            h = int(row["crop_h"])
            w = int(row["crop_w"])

            r_end = min(r0 + h, H)
            c_end = min(c0 + w, W)
            mask[r0:r_end, c0:c_end] = True
            n_crops += 1

    coverage = mask.sum() / (H * W) * 100
    print(f"[SplitMask] {split}: {n_crops} crops, "
          f"{mask.sum():,} pixels ({coverage:.1f}% of image)")

    return mask


def build_all_split_masks(
    crop_index_path: str,
    city: str,
    subset: str,
    label_shape: tuple,
) -> dict:
    """
    Build masks for all splits (train, val, test).

    Returns dict: {"train": mask, "val": mask, "test": mask}
    """
    masks = {}
    for split in ["train", "val", "test"]:
        masks[split] = build_split_mask(
            crop_index_path, city, subset, split, label_shape,
        )
    return masks


def plot_split_overlay(
    label: np.ndarray,
    masks: dict,
    output_path: str,
    class_colors: list = None,
    title: str = "Augsburg Train/Val/Test Split",
):
    """
    Plot the label map with colored overlays showing train/val/test regions.

    Parameters
    ----------
    label : np.ndarray [H, W]
        Full label map.
    masks : dict
        {"train": [H,W] bool, "val": [H,W] bool, "test": [H,W] bool}
    output_path : str
    class_colors : list of [R, G, B]
        Per-class colors for the label map background.
    title : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    H, W = label.shape

    # Default class colors
    if class_colors is None:
        class_colors = [
            [255, 255, 255],  # 0 Background
            [255,   0,   0],  # 1 Urban Fabric
            [204,   0, 230],  # 2 Industrial
            [  0,   0,   0],  # 3 Street
            [166,  77,   0],  # 4 Mine
            [255, 170, 255],  # 5 Artificially Veg
            [255, 255,   0],  # 6 Arable
            [255, 170,   0],  # 7 Permanent Crops
            [190, 255,   0],  # 8 Pastures
            [  0, 120,   0],  # 9 Forests
            [170, 210,  90],  # 10 Shrub
            [210, 200, 160],  # 11 Open Spaces
            [  0, 200, 200],  # 12 Wetlands
            [  0,   0, 255],  # 13 Water
        ]

    # Build RGB label map
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)  # gray background for ignore
    for cls_id in range(len(class_colors)):
        m = label == cls_id
        rgb[m] = class_colors[cls_id]

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: Label map
    axes[0].imshow(rgb)
    axes[0].set_title("Ground Truth Labels", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Label map with split overlay
    overlay = rgb.copy().astype(np.float32)

    split_colors = {
        "train": np.array([0, 100, 255], dtype=np.float32),   # blue
        "val":   np.array([255, 165, 0], dtype=np.float32),    # orange
        "test":  np.array([255, 0, 0], dtype=np.float32),      # red
    }

    alpha = 0.35
    for split_name, mask in masks.items():
        if split_name in split_colors:
            color = split_colors[split_name]
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("Train / Val / Test Regions", fontsize=13)
    axes[1].axis("off")

    # Draw boundaries between splits
    for split_name, mask in masks.items():
        if split_name not in split_colors:
            continue
        # Find contours by detecting edges of the mask
        edges = np.zeros_like(mask)
        edges[1:] |= mask[1:] != mask[:-1]
        edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        edge_coords = np.where(edges)
        if len(edge_coords[0]) > 0:
            color_norm = split_colors[split_name] / 255.0
            axes[1].scatter(edge_coords[1], edge_coords[0],
                           c=[color_norm], s=0.1, alpha=0.5)

    # Legend
    patches = [
        mpatches.Patch(color=split_colors["train"] / 255,
                       label=f"Train ({masks['train'].sum():,} px)"),
        mpatches.Patch(color=split_colors["val"] / 255,
                       label=f"Val ({masks['val'].sum():,} px)"),
        mpatches.Patch(color=split_colors["test"] / 255,
                       label=f"Test ({masks['test'].sum():,} px)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=11, frameon=True)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")