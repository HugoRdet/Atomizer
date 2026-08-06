"""
plot_confusion_matrix_dales.py
================================

Runs a trained DALES checkpoint over the full test set and saves a
confusion matrix plot to ./figures/confusion_matrix_dales.png.

Usage:
    python plot_confusion_matrix_dales.py \
        --root_path ./data \
        --config_model config_test-DALES.yaml \
        --ckpt_path ./checkpoints/dales/precision32_dales_v1-epoch=XX-val_mIoU=X.ckpt \
        --max_lidar_points 100000

Notes:
    - Ignores IGNORE_INDEX (255) points (unknown/padding), same convention
      as the training metrics.
    - Row-normalized (each row sums to 1) so the plot shows RECALL per true
      class -- easier to read for imbalanced classes than raw counts, where
      the ground/vegetation cells would visually dominate everything else.
    - Saves both the normalized plot AND the raw counts as a .npy sidecar,
      in case you want to recompute a differently-normalized view later
      without re-running the whole test set.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless — no X server on compute nodes
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.utils_dataset_dales import DalesDataset
from training.utils.datasets.token_grouping import collate_grouped
from training.atomiser.Atomiser_dales import Atomiser_Dales


IGNORE_INDEX = 255
NUM_CLASSES = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--config_model", type=str,
                         default="config_test-DALES.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--max_lidar_points", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Setup (mirrors script_train_dales.py) ────────────────────────
    config_model = read_yaml(f"./training/configs/{args.config_model}")
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
    configs_dataset = read_yaml(configs_dataset_path)
    bands = {}

    lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
    TokenBuilder.REFERENCE_SIZES[0.2] = 2048
    lookup_table.get_or_register_modality(0.2, 2048)
    lookup_table.get_resolution_idx(0.2)
    lookup_table.register_abstract_channel("ELEVATION")

    print(f"[confmat] Loading test dataset...")
    test_ds = DalesDataset(
        root_path=args.root_path,
        mode="test",
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_grouped,
    )
    print(f"[confmat] {len(test_ds)} test patches, "
          f"{len(test_loader)} batches @ batch_size={args.batch_size}")

    # ── Model ─────────────────────────────────────────────────────────
    print(f"[confmat] Loading checkpoint: {args.ckpt_path}")
    model = Atomiser_Dales(config=config_model, lookup_table=lookup_table)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Lightning checkpoints prefix keys with "model." (self.model = Atomiser_Dales
    # inside Model_Dales) — strip that prefix to load directly into the bare model.
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    result = model.load_state_dict(state, strict=False)
    print(f"[confmat] missing keys: {len(result.missing_keys)}, "
          f"unexpected keys: {len(result.unexpected_keys)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def move_batch(b, device):
        out = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, dict):
                out[k] = {
                    res: {gk: (gv.to(device) if isinstance(gv, torch.Tensor) else gv)
                          for gk, gv in g.items()}
                    for res, g in v.items()
                }
            else:
                out[k] = v
        return out

    # ── Accumulate confusion matrix ──────────────────────────────────
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # [true, pred]

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = move_batch(batch, device)
            logits = model(batch, training=False)          # [B, M, 8]
            preds  = logits.argmax(dim=-1)                  # [B, M]
            labels = batch["queries"][:, :, 4].long()       # [B, M]

            preds_flat  = preds.reshape(-1).cpu().numpy()
            labels_flat = labels.reshape(-1).cpu().numpy()

            valid = labels_flat != IGNORE_INDEX
            p = preds_flat[valid]
            t = labels_flat[valid]

            np.add.at(cm, (t, p), 1)

            if (i + 1) % 20 == 0 or i == len(test_loader) - 1:
                print(f"  [{i+1}/{len(test_loader)}] batches processed, "
                      f"{cm.sum():,} points accumulated so far")

    print(f"[confmat] Done. Total valid points: {cm.sum():,}")
    DALES_CLASSES = test_ds.DALES_CLASSES

    # ── Save raw counts ───────────────────────────────────────────────
    raw_path = os.path.join(args.out_dir, "confusion_matrix_dales_raw.npy")
    np.save(raw_path, cm)
    print(f"[confmat] Saved raw counts to {raw_path}")

    # ── Row-normalize (recall per true class) ────────────────────────
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # avoid divide-by-zero for empty rows
    cm_norm = cm / row_sums

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(DALES_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(DALES_CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("DALES confusion matrix (row-normalized, i.e. recall)")

    # Annotate each cell with its value
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Recall (row-normalized)")
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, "confusion_matrix_dales.png")
    fig.savefig(out_path, dpi=200)
    print(f"[confmat] Saved plot to {out_path}")

    # ── Print per-class summary to console too ───────────────────────
    print("\nPer-class recall (diagonal of normalized matrix):")
    for i, name in enumerate(DALES_CLASSES):
        print(f"  {name:<14s}: {cm_norm[i, i]:.4f}  "
              f"(n={cm[i].sum():,} points)")


if __name__ == "__main__":
    main()
