#!/usr/bin/env python3
"""
PASTIS-HD Evaluation — Patch-Based Inference + Visualization
==============================================================

Loads a trained checkpoint, runs inference over all test patches
(fold 5), and produces:

  1. Per-class IoU bar chart (PNG)
  2. Confusion matrix (PNG)
  3. Sample prediction grids (PNG)
  4. Metrics JSON (mIoU, OA, mF1, per-class IoU)

mIoU is computed over classes present in the test set, excluding
background (class 0).

Usage:
    # S2-only evaluation
    python eval_pastis.py \
        --ckpt_path ./checkpoints/pastis/best.ckpt \
        --output_dir ./results/pastis/s2_only

    # S2 + S1, last 5 timesteps
    python eval_pastis.py \
        --ckpt_path ./checkpoints/pastis/best.ckpt \
        --use_s1 --temporal_last --multi_temporal 5 \
        --output_dir ./results/pastis/s2_s1_last5

    # S2 + S1 + SPOT
    python eval_pastis.py \
        --ckpt_path ./checkpoints/pastis/best.ckpt \
        --use_s1 --use_spot \
        --output_dir ./results/pastis/full
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
from training.trainer_PASTIS import PASTISTrainer
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES = 20
IGNORE_INDEX = 19   # Class 19 = "Void Label" (PANGAEA convention)
TASK_NAME = "pastis_segmentation"

CLASS_NAMES = {
    0: "Background",
    1: "Meadow",
    2: "Soft Winter Wheat",
    3: "Corn",
    4: "Winter Barley",
    5: "Winter Rapeseed",
    6: "Spring Barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter Triticale",
    11: "Winter Durum Wheat",
    12: "Fruits, Vegetables, Flowers",
    13: "Potatoes",
    14: "Leguminous Fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed Cereal",
    18: "Sorghum",
    19: "Void Label",
}

# Colors for 19 classes (0..18)
CLASS_COLORS = [
    [255, 255, 255],  #  0 Background
    [124, 252,   0],  #  1 Meadow
    [255, 215,   0],  #  2 Soft Winter Wheat
    [255, 165,   0],  #  3 Corn
    [218, 165,  32],  #  4 Winter Barley
    [255, 255,   0],  #  5 Winter Rapeseed
    [189, 183, 107],  #  6 Spring Barley
    [255, 140,   0],  #  7 Sunflower
    [128,   0, 128],  #  8 Grapevine
    [220,  20,  60],  #  9 Beet
    [ 50, 205,  50],  # 10 Soy
    [210, 180, 140],  # 11 Sorghum
    [  0, 191, 255],  # 12 Flax
    [ 60, 179, 113],  # 13 Protein Crops
    [245, 222, 179],  # 14 Other Cereals
    [255, 105, 180],  # 15 Fruits/Veg/Flowers
    [192, 192, 192],  # 16 Other Crops
    [ 34, 139,  34],  # 17 Grassland
    [  0, 100,   0],  # 18 Shrub/Forest
    [128, 128, 128],  # 19 (unused, mapped to ignore)
]

ALL_KNOWN_RESOLUTIONS = {
    1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


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
            new_size = ckpt_tensor.shape[0]
            new_tensor = torch.zeros(new_size, dtype=model_tensor.dtype)
            copy_size = min(new_size, model_tensor.shape[0])
            new_tensor[:copy_size] = model_tensor[:copy_size]
            _set_nested_attr(model, key, new_tensor)
            resized.append(f"  {key}: {model_tensor.shape[0]} → {new_size}")

        elif ckpt_tensor.dim() == 2 and model_tensor.dim() == 2:
            if ckpt_tensor.shape[1] == model_tensor.shape[1]:
                new_rows = ckpt_tensor.shape[0]
                new_tensor = torch.zeros(new_rows, ckpt_tensor.shape[1],
                                         dtype=model_tensor.dtype)
                copy_rows = min(new_rows, model_tensor.shape[0])
                new_tensor[:copy_rows] = model_tensor[:copy_rows]
                _set_nested_attr(model, key, new_tensor)
                resized.append(f"  {key}: {model_tensor.shape} → {ckpt_tensor.shape}")

    if resized:
        print(f"[Eval] Resized {len(resized)} buffers:")
        for msg in resized:
            print(msg)


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
# PATCH-BASED INFERENCE
# =============================================================================

def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


def run_inference(model, dataset, device, max_viz=8):
    """
    Run inference on all patches. Returns per-patch predictions, labels,
    and a selection of viz samples.

    Returns dict with:
        all_preds:  list of [H*W] int arrays
        all_labels: list of [H*W] int arrays
        viz_samples: list of (image, prediction_map, label_map, shape) tuples
    """
    model.eval()
    n_patches = len(dataset)

    all_preds = []
    all_labels = []
    viz_samples = []

    # Select patches to visualize (evenly spaced)
    viz_indices = set(
        np.linspace(0, n_patches - 1, min(max_viz, n_patches), dtype=int).tolist()
    )

    t_start = time.perf_counter()

    for i in range(n_patches):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (n_patches - i - 1)
            print(f"  Patch {i+1}/{n_patches} "
                  f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

        try:
            sample = dataset.get_viz_sample(i)
        except Exception as e:
            print(f"  [WARNING] Patch {i} failed: {e}")
            continue

        label = sample["label"]  # [H, W]
        H, W = sample["image_shape"]

        batch = collate_multitask([sample])
        batch = _batch_to_device(batch, device)

        with torch.no_grad():
            result = model(batch, training=False)

        if isinstance(result, dict):
            y_hat = result["predictions"]
        else:
            y_hat = result

        logits = y_hat[0].cpu()  # [H*W, num_classes]
        pred_classes = logits.argmax(dim=-1).numpy()  # [H*W]

        label_flat = label.numpy().flatten()

        all_preds.append(pred_classes)
        all_labels.append(label_flat)

        if i in viz_indices:
            pred_map = pred_classes.reshape(H, W)
            label_map = label.numpy()
            image = sample["image"]  # [C, H, W]
            viz_samples.append((image, pred_map, label_map, (H, W)))

    elapsed = time.perf_counter() - t_start
    print(f"[Eval] Done: {n_patches} patches in {elapsed:.1f}s "
          f"({n_patches / elapsed:.1f} patches/s)")

    return {
        "all_preds": all_preds,
        "all_labels": all_labels,
        "viz_samples": viz_samples,
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(all_preds, all_labels):
    """
    Compute mIoU, mF1, OA, per-class IoU/F1 across all patches.

    PANGAEA-compatible:
      - All classes (0-18) included in mIoU average
      - Absent classes get IoU = 0 (drags mean down, matching torchmetrics macro)
      - Only ignore_index (255) pixels are skipped
      - Also reports "present-only" mIoU for reference
    """
    pred_all = np.concatenate(all_preds)
    label_all = np.concatenate(all_labels)

    valid = (label_all != IGNORE_INDEX) & (pred_all != IGNORE_INDEX)

    pred_valid = pred_all[valid]
    label_valid = label_all[valid]

    overall_acc = float((pred_valid == label_valid).sum() / max(len(label_valid), 1))

    per_class = {}
    all_ious = []        # all classes (PANGAEA convention)
    present_ious = []    # only present classes (for reference)
    all_f1s = []
    present_f1s = []

    for cls_id in range(NUM_CLASSES):
        pred_cls = pred_valid == cls_id
        label_cls = label_valid == cls_id

        tp = int((pred_cls & label_cls).sum())
        fp = int((pred_cls & ~label_cls).sum())
        fn = int((~pred_cls & label_cls).sum())

        union = tp + fp + fn
        support = tp + fn

        iou = tp / union if union > 0 else 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        per_class[cls_id] = {
            "name": name,
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": support,
            "in_test": support > 0,
        }

        # PANGAEA: all classes contribute (absent → IoU=0)
        all_ious.append(iou)
        all_f1s.append(f1)

        # Present-only: for reference
        if support > 0:
            present_ious.append(iou)
            present_f1s.append(f1)

    return {
        "mIoU": float(np.mean(all_ious)) if all_ious else 0.0,
        "mF1": float(np.mean(all_f1s)) if all_f1s else 0.0,
        "mIoU_present": float(np.mean(present_ious)) if present_ious else 0.0,
        "mF1_present": float(np.mean(present_f1s)) if present_f1s else 0.0,
        "overall_accuracy": overall_acc,
        "n_classes_evaluated": len(all_ious),
        "n_classes_present": len(present_ious),
        "n_classes_total": NUM_CLASSES,
        "per_class": per_class,
        "n_valid_pixels": int(valid.sum()),
    }


def compute_confusion_matrix(all_preds, all_labels):
    pred_all = np.concatenate(all_preds)
    label_all = np.concatenate(all_labels)

    valid = (label_all != IGNORE_INDEX) & (pred_all != IGNORE_INDEX)

    pred_valid = pred_all[valid]
    label_valid = label_all[valid]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            cm[t, p] = ((label_valid == t) & (pred_valid == p)).sum()
    return cm


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_to_rgb(label_map):
    h, w = label_map.shape
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    for cls_id in range(min(NUM_CLASSES, len(CLASS_COLORS))):
        rgb[label_map == cls_id] = CLASS_COLORS[cls_id]
    return rgb


def plot_sample_grid(viz_samples, output_path, metrics=None, max_cols=4):
    """Grid of (RGB | GT | Prediction) for a selection of patches."""
    n = len(viz_samples)
    if n == 0:
        return

    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(5 * n_cols * 3, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols * 3 == 3:
        axes = axes[:, np.newaxis]
        # Actually this won't happen since n_cols >= 1 and we have *3

    for idx, (image, pred_map, label_map, shape) in enumerate(viz_samples):
        row = idx // n_cols
        col = idx % n_cols

        # RGB composite (bands 2,1,0 = R,G,B for S2)
        if image.shape[0] >= 3:
            rgb = image[[2, 1, 0]].numpy()  # R, G, B
            rgb = np.clip(rgb, 0, None)
            p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
            rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1)
            rgb = np.transpose(rgb, (1, 2, 0))
        else:
            rgb = np.zeros((*shape, 3))

        gt_rgb = label_to_rgb(label_map)
        pred_rgb = label_to_rgb(pred_map)

        c0 = col * 3
        axes[row, c0].imshow(rgb)
        axes[row, c0].set_title("RGB", fontsize=9)
        axes[row, c0].axis("off")

        axes[row, c0 + 1].imshow(gt_rgb)
        axes[row, c0 + 1].set_title("GT", fontsize=9)
        axes[row, c0 + 1].axis("off")

        axes[row, c0 + 2].imshow(pred_rgb)
        axes[row, c0 + 2].set_title("Pred", fontsize=9)
        axes[row, c0 + 2].axis("off")

    # Hide empty subplots
    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        for k in range(3):
            axes[row, col * 3 + k].axis("off")

    title = "PASTIS-HD — Sample Predictions"
    if metrics:
        title += f"  (mIoU={metrics['mIoU']:.4f})"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_per_class_iou(metrics, output_path, title="Per-Class IoU"):
    names, ious, colors, hatches = [], [], [], []

    for cls_id in range(NUM_CLASSES):
        if cls_id not in metrics["per_class"]:
            continue
        info = metrics["per_class"][cls_id]
        names.append(info["name"])
        iou = info["iou"] if not np.isnan(info["iou"]) else 0.0
        ious.append(iou)
        c = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else [128, 128, 128]
        colors.append(np.array(c) / 255.0)
        hatches.append("//" if (not info["in_test"] or cls_id == 0) else "")

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 6))
    bars = ax.bar(range(n), ious, color=colors, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{title}  (mIoU={metrics['mIoU']:.4f}, mF1={metrics['mF1']:.4f}, "
                 f"{metrics['n_classes_evaluated']} classes)", fontsize=13)
    ax.axhline(y=metrics["mIoU"], color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {metrics['mIoU']:.4f}")
    ax.legend(fontsize=10)

    for bar, iou, cls_id in zip(bars, ious, range(n)):
        info = metrics["per_class"][cls_id]
        if info["in_test"] and cls_id > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{iou:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


def plot_confusion_matrix(cm, output_path, title="Confusion Matrix",
                          classes_present=None):
    if classes_present is not None:
        idx = sorted(classes_present)
        cm_sub = cm[np.ix_(idx, idx)]
        names_sub = [CLASS_NAMES.get(i, f"Class {i}") for i in idx]
    else:
        cm_sub = cm
        names_sub = [CLASS_NAMES.get(i, f"Class {i}") for i in range(NUM_CLASSES)]

    n = cm_sub.shape[0]
    cm_norm = cm_sub.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names_sub, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names_sub, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.005:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=6)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PASTIS-HD Evaluation")
    parser.add_argument("--ckpt_path",     type=str, required=True)
    parser.add_argument("--config_model",  type=str,
                        default="config_test-Atomiser_Atos_One.yaml")
    parser.add_argument("--data_dir",      type=str, default="./data/PASTIS-HD")
    parser.add_argument("--output_dir",    type=str, default="./results/pastis")
    parser.add_argument("--split",         type=str, default="test",
                        choices=["train", "validation", "test"])

    # Modality toggles (must match training)
    parser.add_argument("--use_s1",        action="store_true")
    parser.add_argument("--use_spot",      action="store_true")

    # Temporal config (must match training)
    parser.add_argument("--multi_temporal", type=int, default=None)
    parser.add_argument("--temporal_last", action="store_true")

    # Viz
    parser.add_argument("--max_viz",       type=int, default=8,
                        help="Max number of sample patches to visualize")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build modality description ──────────────────────────────────
    modalities = ["S2"]
    if args.use_s1:
        modalities.append("S1")
    if args.use_spot:
        modalities.append("SPOT")
    modality_str = "+".join(modalities)

    print(f"\n{'='*60}")
    print(f"  PASTIS-HD Evaluation")
    print(f"  Modalities: {modality_str}")
    print(f"  Split:      {args.split}")
    print(f"  Ckpt:       {args.ckpt_path}")
    print(f"{'='*60}\n")

    # ── Config & lookup ─────────────────────────────────────────────
    config_model = read_yaml("./training/configs/" + args.config_model)
    bands_yaml_path = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    if args.multi_temporal is not None:
        if "dataset" not in config_model:
            config_model["dataset"] = {}
        config_model["dataset"]["multi_temporal"] = args.multi_temporal

    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )
    register_all_resolutions(lookup_table)

    # Register VV-VH SAR channel (not in bands.yaml, needed for S1 3rd band)
    if args.use_s1:
        lookup_table.register_abstract_channel("VV_VH")

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = PastisHDDataset(
        root_path=args.data_dir,
        mode=args.split,
        config_model=config_model,
        look_up=lookup_table,
        use_s1=args.use_s1,
        use_spot=args.use_spot,
        temporal_last=args.temporal_last,
    )

    multi_temporal = config_model.get("dataset", {}).get("multi_temporal", 10)
    temporal_str = f"{multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"
    print(f"[Eval] Temporal: {temporal_str}")
    print(f"[Eval] {len(dataset)} patches to evaluate")

    # ── Load model ──────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("state_dict", ckpt)

    model = PASTISTrainer(
        config=config_model, wand=False, name="eval",
        transform=None, lookup_table=lookup_table,
    )
    align_model_to_checkpoint(model, ckpt_state)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"[Eval] Model loaded on {device}")

    # ── Inference ───────────────────────────────────────────────────
    result = run_inference(model, dataset, device, max_viz=args.max_viz)

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = compute_metrics(result["all_preds"], result["all_labels"])

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  mIoU:          {metrics['mIoU']:>7.4f}  "
          f"({metrics['n_classes_evaluated']} classes, PANGAEA)  │")
    print(f"  │  mIoU (present): {metrics['mIoU_present']:>7.4f}  "
          f"({metrics['n_classes_present']} classes)         │")
    print(f"  │  mF1:           {metrics['mF1']:>7.4f}                          │")
    print(f"  │  OA:            {metrics['overall_accuracy']:>7.4f}                          │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Valid pixels: {metrics['n_valid_pixels']:,}")

    print()
    print(f"  {'Class':<25s} {'IoU':>8s} {'F1':>8s} {'Support':>10s}")
    print(f"  {'-'*51}")
    for cls_id in range(NUM_CLASSES):
        if cls_id not in metrics["per_class"]:
            continue
        info = metrics["per_class"][cls_id]
        if info["in_test"]:
            print(f"  {info['name']:<25s} {info['iou']:>8.4f} {info['f1']:>8.4f} "
                  f"{info['support']:>10,d}")
        else:
            print(f"  {info['name']:<25s} {'N/A':>8s} {'N/A':>8s} "
                  f"{info['support']:>10,d}  (absent)")

    # ── Save metrics ────────────────────────────────────────────────
    metrics["config"] = {
        "modalities": modalities,
        "split": args.split,
        "temporal_last": args.temporal_last,
        "multi_temporal": multi_temporal,
        "ckpt_path": args.ckpt_path,
        "n_patches": len(dataset),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → {metrics_path}")

    # ── Visualizations ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Visualizations")
    print(f"{'='*60}\n")

    title = f"PASTIS-HD — {modality_str} ({args.split})"

    # Sample prediction grid
    if result["viz_samples"]:
        plot_sample_grid(
            result["viz_samples"],
            os.path.join(args.output_dir, "sample_predictions.png"),
            metrics=metrics,
        )

    # Per-class IoU bar chart
    plot_per_class_iou(
        metrics,
        os.path.join(args.output_dir, "per_class_iou.png"),
        title=title,
    )

    # Confusion matrix
    cm = compute_confusion_matrix(result["all_preds"], result["all_labels"])
    present = [c for c in range(NUM_CLASSES)
               if c in metrics["per_class"] and metrics["per_class"][c]["in_test"]]
    plot_confusion_matrix(
        cm, os.path.join(args.output_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — {modality_str} ({args.split})",
        classes_present=present if len(present) < NUM_CLASSES else None,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()