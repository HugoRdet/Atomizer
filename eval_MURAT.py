"""
MuRA-T Evaluation — Per-Sample Inference + Visualization
==========================================================

Loads a trained checkpoint, runs inference on all test samples for
a given sensor, computes metrics, and generates visualizations.

Usage:
    # Evaluate on held-out Landsat (cross-sensor transfer)
    python eval_murat.py \\
        --ckpt_path ./checkpoints/murat/best.ckpt \\
        --sensor landsat8 \\
        --output_dir ./results/murat/leave_landsat

    # Evaluate on training sensor (upper bound)
    python eval_murat.py \\
        --ckpt_path ./checkpoints/murat/best.ckpt \\
        --sensor planet \\
        --output_dir ./results/murat/planet_upper

    # Visualize N samples
    python eval_murat.py \\
        --ckpt_path ./checkpoints/murat/best.ckpt \\
        --sensor planet --n_viz 10 \\
        --output_dir ./results/murat/viz
"""

import os
import argparse
import json
import time

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")

from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_MURAT import (
    MuRATSegmentation, NUM_CLASSES, IGNORE_INDEX, DATASET_NAME,
    SENSOR_QUERY_RES, S2_RES_GROUPS,
    PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO,
    pad_to_canonical, CANONICAL_SIZES,
)
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# RESOLUTION & BAND REGISTRATION
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048,
    4.78: 2048,
    10.0: 2048,
    20.0: 2048,
    30.0: 2048,
    60.0: 2048,
}


def register_all_resolutions(lookup_table):
    """Pre-register all known resolutions before model construction."""
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.register_resolution(res)
        lookup_table.register_modality(res, ref_size)

    print(f"[Eval] Pre-registered resolutions: "
          f"{sorted(ALL_KNOWN_RESOLUTIONS.keys())}")


def register_all_bands(lookup_table):
    """Pre-register all MuRA-T sensor bands."""
    n_new = 0
    for bands_info in [PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO]:
        for band_name, data in bands_info.items():
            bw = int(data["bandwidth"])
            wl = int(data["central_wavelength"])
            key = (bw, wl)
            if key not in lookup_table.table_wave:
                lookup_table.table_wave[key] = len(lookup_table.table_wave)
                n_new += 1

    print(f"[Eval] Pre-registered {n_new} bands "
          f"(total: {len(lookup_table.table_wave)})")


def register_mdas_bands_if_available(lookup_table):
    """Pre-register MDAS bands if checkpoint was trained on MDAS."""
    try:
        from training.utils.datasets.utils_dataset_MDAS import (
            create_mdas_bands_info, register_mdas_bands,
        )
        spectral_meta_path = "./data/MDAS/Augsburg_data_4_publication/mdas_spectral_meta.json"
        if os.path.exists(spectral_meta_path):
            bands_yaml = read_yaml("./data/bands_info/bands.yaml")
            mdas_bands = create_mdas_bands_info(spectral_meta_path)
            bands_yaml.update(mdas_bands)
            register_mdas_bands(lookup_table, bands_yaml)
    except Exception:
        pass


# =============================================================================
# CHECKPOINT COMPATIBILITY
# =============================================================================

def align_model_to_checkpoint(model, ckpt_state):
    """Resize model buffers/parameters to match checkpoint sizes."""
    model_state = model.state_dict()
    resized = []

    for key, ckpt_tensor in ckpt_state.items():
        if key not in model_state:
            continue

        model_tensor = model_state[key]
        if ckpt_tensor.shape == model_tensor.shape:
            continue

        if ckpt_tensor.dim() == 1 and model_tensor.dim() == 1:
            ckpt_size = ckpt_tensor.shape[0]
            model_size = model_tensor.shape[0]

            if ckpt_size > model_size:
                new_tensor = torch.zeros(ckpt_size, dtype=model_tensor.dtype)
                new_tensor[:model_size] = model_tensor
            else:
                new_tensor = model_tensor[:ckpt_size].clone()

            _set_nested_attr(model, key, new_tensor)
            resized.append(f"  {key}: {model_size} → {ckpt_size}")

        elif ckpt_tensor.dim() == 2 and model_tensor.dim() == 2:
            ckpt_rows, ckpt_cols = ckpt_tensor.shape
            model_rows, model_cols = model_tensor.shape

            if ckpt_cols == model_cols:
                if ckpt_rows > model_rows:
                    new_tensor = torch.zeros(ckpt_rows, ckpt_cols, dtype=model_tensor.dtype)
                    new_tensor[:model_rows] = model_tensor
                else:
                    new_tensor = model_tensor[:ckpt_rows].clone()

                _set_nested_attr(model, key, new_tensor)
                resized.append(f"  {key}: [{model_rows},{model_cols}] → [{ckpt_rows},{ckpt_cols}]")

    if resized:
        print(f"[Eval] Resized {len(resized)} buffers:")
        for msg in resized:
            print(msg)

    return model


def _set_nested_attr(model, key, tensor):
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)

    attr_name = parts[-1]
    if attr_name in dict(obj.named_buffers(recurse=False)):
        obj.register_buffer(attr_name, tensor)
    elif attr_name in dict(obj.named_parameters(recurse=False)):
        obj.register_parameter(attr_name, torch.nn.Parameter(tensor))
    else:
        setattr(obj, attr_name, tensor)


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = ["Background", "Building"]
CLASS_COLORS = [
    [200, 200, 200],  # Background — light grey
    [255, 50, 50],    # Building — red
]
NODATA_COLOR = [255, 255, 255]


# =============================================================================
# HELPERS
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


def _round_gsd(gsd):
    return round(gsd, 2)


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(model, dataset, device, max_samples=None):
    """
    Run inference on all samples in the dataset.
    Returns list of dicts with prediction, label, metadata per sample.
    """
    model.eval()
    results = []
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(n):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Sample {i+1}/{n}...", flush=True)

        sample = dataset[i]
        batch = collate_multitask([sample])
        batch = _batch_to_device(batch, device)

        with torch.no_grad():
            preds = model.forward_multitask(batch, training=False)

        task_key = dataset.TASK_NAME
        logits = preds[task_key][0].cpu()
        queries = batch["tasks"][task_key]["queries"][0].cpu()

        labels = queries[:, 4].long()
        pred_classes = logits.argmax(dim=-1)

        meta = dataset.samples[i]
        sensor = meta["sensor"]
        gsd = _round_gsd(SENSOR_QUERY_RES[sensor])

        results.append({
            "index": i,
            "aoi": meta["aoi"],
            "sensor": sensor,
            "month": meta["month"],
            "labels": labels.numpy(),
            "predictions": pred_classes.numpy(),
            "logits": logits.numpy(),
            "queries": queries.numpy(),
            "gsd": gsd,
        })

    return results


def run_inference_with_spatial(model, dataset, device, sample_idx):
    """
    Run inference on a single sample and reconstruct spatial prediction.

    Returns:
        prediction_map: [H, W] int array (or None for S2)
        label_map: [H, W] int array (or None for S2)
        image_rgb: [H, W, 3] float array (for visualization)
        labels_flat: flat labels (for S2 fallback, else None)
        pred_flat: flat preds (for S2 fallback, else None)
    """
    model.eval()
    sample_meta = dataset.samples[sample_idx]
    sensor = sample_meta["sensor"]
    gsd = _round_gsd(SENSOR_QUERY_RES[sensor])

    # ── Load raw image for RGB visualization ────────────────────────
    if sensor == "planet":
        image, geo_info = dataset._load_planet(sample_meta["band_files"])
        rgb_bands = [0, 1, 2]  # R, G, B
    elif sensor == "sentinel2":
        res_images, geo_info, _ = dataset._load_sentinel2(sample_meta["band_files"])
        image = res_images[10.0]  # B02(blue), B03(green), B04(red), B08(nir)
        rgb_bands = [2, 1, 0]  # B04(red), B03(green), B02(blue)
    elif sensor == "landsat8":
        image, geo_info = dataset._load_landsat(sample_meta["band_files"])
        rgb_bands = [3, 2, 1]  # B4(red), B3(green), B2(blue)

    # Build RGB for viz
    C, H, W = image.shape
    rgb = image[rgb_bands].permute(1, 2, 0).numpy()
    p2, p98 = np.percentile(rgb[np.isfinite(rgb)], [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    # ── Time index ──────────────────────────────────────────────────
    from datetime import datetime
    year, month = sample_meta["month"].split("_")
    doy = datetime(int(year), int(month), 15).timetuple().tm_yday
    time_idx = dataset.look_up.get_or_register_time_idx(doy)

    # ── Build tokens and queries per sensor ─────────────────────────
    if sensor == "planet":
        image_full, _ = dataset._load_planet(sample_meta["band_files"])

        # Label at Planet resolution
        label = dataset._get_planet_label(
            sample_meta["label_path"],
            sample_meta["aoi"],
            sample_meta["month"],
        )

        # Align
        _, H_img, W_img = image_full.shape
        H_lab, W_lab = label.shape
        if H_img != H_lab or W_img != W_lab:
            H_min, W_min = min(H_img, H_lab), min(W_img, W_lab)
            image_full = image_full[:, :H_min, :W_min]
            label = label[:H_min, :W_min]

        image_full, label, valid_mask = pad_to_canonical(image_full, label, gsd)
        C, H_pad, W_pad = image_full.shape

        res_idx = dataset.resolution_indices[gsd]
        tokens = dataset.token_builder.build_tokens(
            image=image_full, label=label, resolution=gsd,
            spectral_indices=dataset.spectral_indices["planet"],
            resolution_idx=res_idx, time_idx=time_idx,
        )
        token_mask = dataset._build_token_mask(valid_mask, C)
        query_label = label
        query_gsd = gsd
        query_res_idx = res_idx
        first_spectral = dataset.spectral_indices["planet"][0].item()

    elif sensor == "landsat8":
        image_full, geo_info = dataset._load_landsat(sample_meta["band_files"])
        _, H_img, W_img = image_full.shape

        # Label: rasterize at Planet, downsample to Landsat
        label = dataset._get_label_for_sensor(
            sample_meta["label_path"],
            sample_meta["aoi"],
            sample_meta["month"],
            target_h=H_img,
            target_w=W_img,
            sensor_key="landsat8",
        )

        H_lab, W_lab = label.shape
        if H_img != H_lab or W_img != W_lab:
            H_min, W_min = min(H_img, H_lab), min(W_img, W_lab)
            image_full = image_full[:, :H_min, :W_min]
            label = label[:H_min, :W_min]

        image_full, label, valid_mask = pad_to_canonical(image_full, label, gsd)
        C, H_pad, W_pad = image_full.shape

        res_idx = dataset.resolution_indices[gsd]
        tokens = dataset.token_builder.build_tokens(
            image=image_full, label=label, resolution=gsd,
            spectral_indices=dataset.spectral_indices["landsat8"],
            resolution_idx=res_idx, time_idx=time_idx,
        )
        token_mask = dataset._build_token_mask(valid_mask, C)
        query_label = label
        query_gsd = gsd
        query_res_idx = res_idx
        first_spectral = dataset.spectral_indices["landsat8"][0].item()

    elif sensor == "sentinel2":
        # S2: use 10m group for spatial viz
        res_images, geo_info, res_geo_info = dataset._load_sentinel2(
            sample_meta["band_files"]
        )

        query_gsd = _round_gsd(10.0)
        image_10m = res_images[query_gsd]
        _, H_10m, W_10m = image_10m.shape

        # Label at 10m
        label = dataset._get_label_for_sensor(
            sample_meta["label_path"],
            sample_meta["aoi"],
            sample_meta["month"],
            target_h=H_10m,
            target_w=W_10m,
            sensor_key="sentinel2_10.0m",
        )

        H_lab, W_lab = label.shape
        if H_10m != H_lab or W_10m != W_lab:
            H_min = min(H_10m, H_lab)
            W_min = min(W_10m, image_10m.shape[2])
            image_10m = image_10m[:, :H_min, :W_min]
            label = label[:H_min, :W_min]

        image_10m, label, valid_mask = pad_to_canonical(image_10m, label, query_gsd)
        C_10m, H_pad, W_pad = image_10m.shape

        # Build tokens for all resolution groups
        groups = {}
        for gsd_g, group_image in res_images.items():
            gsd_g = _round_gsd(gsd_g)
            _, H_g, W_g = group_image.shape

            label_g = dataset._get_label_for_sensor(
                sample_meta["label_path"],
                sample_meta["aoi"],
                sample_meta["month"],
                target_h=H_g,
                target_w=W_g,
                sensor_key=f"sentinel2_{gsd_g}m",
            )

            # Align
            if label_g.shape[0] != H_g or label_g.shape[1] != group_image.shape[2]:
                H_min = min(label_g.shape[0], H_g)
                W_min = min(label_g.shape[1], group_image.shape[2])
                group_image = group_image[:, :H_min, :W_min]
                label_g = label_g[:H_min, :W_min]

            group_image, label_g, valid_mask_g = pad_to_canonical(
                group_image, label_g, gsd_g
            )
            C_g, H_g_pad, W_g_pad = group_image.shape

            res_idx_g = dataset.resolution_indices[gsd_g]
            tokens_g = dataset.token_builder.build_tokens(
                image=group_image, label=label_g, resolution=gsd_g,
                spectral_indices=dataset.s2_group_spectral_indices[gsd_g],
                resolution_idx=res_idx_g, time_idx=time_idx,
            )
            token_mask_g = dataset._build_token_mask(valid_mask_g, C_g)

            groups[gsd_g] = {
                "tokens": tokens_g,
                "mask": token_mask_g,
                "shape": (C_g, H_g_pad, W_g_pad),
            }

        # Queries at 10m
        query_res_idx = dataset.resolution_indices[query_gsd]
        query_label = label

        queries = dataset.token_builder.build_queries(
            label=query_label, resolution=query_gsd,
            first_spectral_idx=dataset.s2_group_spectral_indices[query_gsd][0].item(),
            resolution_idx=query_res_idx, time_idx=time_idx,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        batch_sample = {
            "groups": groups,
            "tasks": {
                dataset.TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                }
            },
            "target_resolution": query_gsd,
            "dataset_name": DATASET_NAME,
        }

        batch = collate_multitask([batch_sample])
        batch = _batch_to_device(batch, device)

        with torch.no_grad():
            preds = model.forward_multitask(batch, training=False)

        logits = preds[dataset.TASK_NAME][0].cpu()
        pred_flat = logits.argmax(dim=-1).numpy()
        label_flat = query_label.reshape(-1).numpy()

        prediction_map = pred_flat.reshape(H_pad, W_pad)
        label_map = label_flat.reshape(H_pad, W_pad)

        # Crop padding
        valid_np = valid_mask.numpy()
        h_valid = valid_np.any(axis=1).sum()
        w_valid = valid_np.any(axis=0).sum()
        prediction_map = prediction_map[:h_valid, :w_valid]
        label_map = label_map[:h_valid, :w_valid]
        rgb = rgb[:h_valid, :w_valid]

        return prediction_map, label_map, rgb, None, None

    # ── Common path for Planet / Landsat ────────────────────────────
    # Build full queries (no subsampling)
    queries = dataset.token_builder.build_queries(
        label=query_label, resolution=query_gsd,
        first_spectral_idx=first_spectral,
        resolution_idx=query_res_idx, time_idx=time_idx,
    )
    queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

    # Build batch
    groups = {
        query_gsd: {
            "tokens": tokens,
            "mask": token_mask,
            "shape": (C, H_pad, W_pad),
        }
    }

    batch_sample = {
        "groups": groups,
        "tasks": {
            dataset.TASK_NAME: {
                "queries": queries,
                "queries_mask": queries_mask,
            }
        },
        "target_resolution": query_gsd,
        "dataset_name": DATASET_NAME,
    }

    batch = collate_multitask([batch_sample])
    batch = _batch_to_device(batch, device)

    with torch.no_grad():
        preds = model.forward_multitask(batch, training=False)

    logits = preds[dataset.TASK_NAME][0].cpu()
    pred_flat = logits.argmax(dim=-1).numpy()
    label_flat = query_label.reshape(-1).numpy()

    prediction_map = pred_flat.reshape(H_pad, W_pad)
    label_map = label_flat.reshape(H_pad, W_pad)

    # Crop padding
    valid_np = valid_mask.numpy()
    h_valid = valid_np.any(axis=1).sum()
    w_valid = valid_np.any(axis=0).sum()
    prediction_map = prediction_map[:h_valid, :w_valid]
    label_map = label_map[:h_valid, :w_valid]
    rgb = rgb[:h_valid, :w_valid]

    return prediction_map, label_map, rgb, None, None


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(all_labels, all_preds):
    """Compute mIoU and per-class metrics from flat arrays."""
    valid = all_labels != IGNORE_INDEX
    pred_valid = all_preds[valid]
    label_valid = all_labels[valid]

    overall_acc = (pred_valid == label_valid).sum() / max(len(label_valid), 1)

    per_class = {}
    ious = []

    for cls_id in range(NUM_CLASSES):
        pred_cls = pred_valid == cls_id
        label_cls = label_valid == cls_id

        intersection = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()

        iou = float(intersection / union) if union > 0 else float("nan")

        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "iou": iou,
            "precision": float(intersection / max(pred_cls.sum(), 1)),
            "recall": float(intersection / max(label_cls.sum(), 1)),
            "support": int(label_cls.sum()),
        }

        if not np.isnan(iou):
            ious.append(iou)

    miou = np.mean(ious) if ious else 0.0

    return {
        "mIoU": float(miou),
        "overall_accuracy": float(overall_acc),
        "per_class": per_class,
        "n_valid_pixels": int(valid.sum()),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_to_rgb(label_map):
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        mask = label_map == cls_id
        rgb[mask] = CLASS_COLORS[cls_id]
    return rgb


def plot_sample(prediction_map, label_map, rgb_image, output_path,
                title="", metrics_text=""):
    """Plot RGB | Ground Truth | Prediction side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB", fontsize=12)
    axes[0].axis("off")

    gt_rgb = label_to_rgb(label_map)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    pred_rgb = label_to_rgb(prediction_map)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    patches.append(mpatches.Patch(color="white", edgecolor="black", label="Ignore/Pad"))
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=10, frameon=True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    if metrics_text:
        fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {output_path}")


def plot_metrics_summary(metrics, output_path, title=""):
    """Bar chart of per-class IoU."""
    names = [metrics["per_class"][i]["name"] for i in range(NUM_CLASSES)]
    ious = [
        metrics["per_class"][i]["iou"]
        if not np.isnan(metrics["per_class"][i]["iou"]) else 0.0
        for i in range(NUM_CLASSES)
    ]
    colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, ious, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title(f"{title}  (mIoU = {metrics['mIoU']:.4f})", fontsize=14)
    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f}")
    ax.legend(fontsize=10)

    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{iou:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {output_path}")


# =============================================================================
# DEBUG: Check for label leakage
# =============================================================================

def check_label_leakage(dataset, device, model):
    """Quick sanity check: does the model output correlate with query labels?"""
    print("\n" + "=" * 60)
    print("  DEBUG: Checking for label leakage")
    print("=" * 60)

    model.eval()

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        batch = collate_multitask([sample])
        batch = _batch_to_device(batch, device)

        task_key = dataset.TASK_NAME
        queries = batch["tasks"][task_key]["queries"][0]

        query_labels = queries[:, 4]
        unique_labels = query_labels.unique().cpu().numpy()

        with torch.no_grad():
            preds = model.forward_multitask(batch, training=False)

        logits = preds[task_key][0].cpu()
        pred_classes = logits.argmax(dim=-1)
        true_labels = queries[:, 4].long().cpu()

        valid = true_labels != IGNORE_INDEX
        if valid.sum() == 0:
            print(f"  Sample {i}: no valid labels")
            continue

        match = (pred_classes[valid] == true_labels[valid]).float().mean().item()
        print(f"  Sample {i}: "
              f"query_col4 unique={unique_labels}, "
              f"valid_pixels={valid.sum().item()}, "
              f"accuracy={match:.4f}, "
              f"pred_unique={pred_classes[valid].unique().cpu().numpy()}")

        if match > 0.99:
            print(f"    ⚠️  SUSPICIOUS: near-perfect accuracy — possible label leakage!")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MuRA-T Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_model", type=str,
                        default="config_test-Atomiser_Atos_One.yaml")
    parser.add_argument("--sensor", type=str, required=True,
                        choices=["planet", "sentinel2", "landsat8"])
    parser.add_argument("--output_dir", type=str, default="./results/murat")

    parser.add_argument("--data_root", type=str, default="./data/MURAT")
    parser.add_argument("--index_csv", type=str, default=None)
    parser.add_argument("--stats_json", type=str, default=None)

    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (None=all)")
    parser.add_argument("--n_viz", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--debug_leakage", action="store_true",
                        help="Run label leakage diagnostic")

    args = parser.parse_args()

    if args.index_csv is None:
        args.index_csv = os.path.join(args.data_root, "murat_index.csv")
    if args.stats_json is None:
        args.stats_json = os.path.join(args.data_root, "murat_norm_stats.json")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ──────────────────────────────────────────────────────
    config_model = read_yaml("./training/configs/" + args.config_model)
    bands_yaml_path = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    # ── Lookup table ────────────────────────────────────────────────
    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )

    register_all_bands(lookup_table)
    register_all_resolutions(lookup_table)
    register_mdas_bands_if_available(lookup_table)

    # ── Load checkpoint ─────────────────────────────────────────────
    print(f"\n[Eval] Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("state_dict", ckpt)

    # ── Build model ─────────────────────────────────────────────────
    model = Model_Pretrain(
        config=config_model,
        wand=False,
        name="eval_murat",
        transform=None,
        lookup_table=lookup_table,
    )

    align_model_to_checkpoint(model, ckpt_state)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"[Eval] Model on {device}")

    # ── Build dataset ───────────────────────────────────────────────
    dataset = MuRATSegmentation(
        index_csv=args.index_csv,
        stats_json=args.stats_json,
        look_up=lookup_table,
        mode=args.split,
        sensors=[args.sensor],
        config_model=config_model,
        max_queries=999_999,  # No subsampling for eval
        augment=False,
        data_root=args.data_root,
    )

    print(f"[Eval] Dataset: {len(dataset)} samples, "
          f"sensor={args.sensor}, split={args.split}")

    if len(dataset) == 0:
        print("[Eval] No samples found. Check sensor/split.")
        return

    # ── Debug leakage check ─────────────────────────────────────────
    if args.debug_leakage:
        check_label_leakage(dataset, device, model)

    # ── Run inference ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running inference on {args.sensor} ({args.split})")
    print(f"{'='*60}")

    results = run_inference(model, dataset, device, max_samples=args.max_samples)

    # ── Aggregate metrics ───────────────────────────────────────────
    all_labels = np.concatenate([r["labels"] for r in results])
    all_preds = np.concatenate([r["predictions"] for r in results])

    metrics = compute_metrics(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  Results: {args.sensor} ({args.split})")
    print(f"{'='*60}")
    print(f"  mIoU:             {metrics['mIoU']:.4f}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Valid Pixels:     {metrics['n_valid_pixels']:,}")
    print()
    print(f"  {'Class':<20s} {'IoU':>8s} {'Precision':>10s} "
          f"{'Recall':>8s} {'Support':>10s}")
    print(f"  {'-'*56}")
    for cls_id in range(NUM_CLASSES):
        info = metrics["per_class"][cls_id]
        iou_str = f"{info['iou']:.4f}" if not np.isnan(info["iou"]) else "  N/A"
        print(f"  {info['name']:<20s} {iou_str:>8s} "
              f"{info['precision']:>10.4f} {info['recall']:>8.4f} "
              f"{info['support']:>10,d}")

    # ── Per-AOI breakdown ───────────────────────────────────────────
    aoi_results = {}
    for r in results:
        aoi = r["aoi"]
        if aoi not in aoi_results:
            aoi_results[aoi] = {"labels": [], "preds": []}
        aoi_results[aoi]["labels"].append(r["labels"])
        aoi_results[aoi]["preds"].append(r["predictions"])

    print(f"\n  Per-AOI mIoU:")
    aoi_metrics = {}
    for aoi in sorted(aoi_results.keys()):
        aoi_labels = np.concatenate(aoi_results[aoi]["labels"])
        aoi_preds = np.concatenate(aoi_results[aoi]["preds"])
        aoi_m = compute_metrics(aoi_labels, aoi_preds)
        aoi_metrics[aoi] = aoi_m
        print(f"    {aoi}: mIoU={aoi_m['mIoU']:.4f}, "
              f"OA={aoi_m['overall_accuracy']:.4f}")

    # ── Save metrics ────────────────────────────────────────────────
    metrics["per_aoi"] = {aoi: m for aoi, m in aoi_metrics.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  → Saved {metrics_path}")

    # ── Visualizations ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Generating visualizations ({args.n_viz} samples)")
    print(f"{'='*60}")

    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    n_viz = min(args.n_viz, len(dataset))
    for i in range(n_viz):
        print(f"\n  Visualizing sample {i+1}/{n_viz}...")
        meta = dataset.samples[i]

        try:
            pred_map, label_map, rgb, labels_flat, pred_flat = \
                run_inference_with_spatial(model, dataset, device, i)

            if pred_map is not None and label_map is not None:
                sample_metrics = compute_metrics(
                    label_map.flatten(), pred_map.flatten()
                )

                title = f"{meta['sensor']} / {meta['aoi']} / {meta['month']}"
                metrics_text = (f"mIoU: {sample_metrics['mIoU']:.4f}  |  "
                               f"OA: {sample_metrics['overall_accuracy']:.4f}")

                fname = f"{meta['sensor']}_{meta['aoi']}_{meta['month']}.png"
                plot_sample(
                    pred_map, label_map, rgb,
                    os.path.join(viz_dir, fname),
                    title=title, metrics_text=metrics_text,
                )
            else:
                if labels_flat is not None:
                    sample_metrics = compute_metrics(labels_flat, pred_flat)
                    print(f"    {meta['sensor']}/{meta['aoi']}/{meta['month']}: "
                          f"mIoU={sample_metrics['mIoU']:.4f}")
        except Exception as e:
            print(f"    Error visualizing sample {i}: {e}")

    # ── Summary bar chart ───────────────────────────────────────────
    plot_metrics_summary(
        metrics,
        os.path.join(args.output_dir, "per_class_iou.png"),
        title=f"MuRA-T — {args.sensor} ({args.split})",
    )

    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()