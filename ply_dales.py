"""
export_scene_predictions_ply.py
==================================

Runs a trained DALES checkpoint over ALL tiled patches belonging to one
raw test scene, stitches the predictions back together, and writes TWO
binary PLY files (one colored by predicted class, one by ground-truth
class) so you can load both into Blender and compare visually.

Uses DalesDataset's eval_full_scene=True mode: for each patch, EVERY
point gets a query (not just the subsampled context), so the output
covers 100% of the scene's points, not just a random subsample.

Usage:
    python export_scene_predictions_ply.py \
        --root_path ./data \
        --config_model config_test-DALES.yaml \
        --ckpt_path ./checkpoints/dales/precision32_dales_v1-epoch=XX-val_mIoU=X.ckpt \
        --scene_stem 5175_54395 \
        --max_lidar_points 100000 \
        --out_dir ./figures

--scene_stem should match the raw DALES scene filename stem (before
tile_dales.py's "_r###_c###" suffix) -- e.g. for a raw scene
"5175_54395.las", pass --scene_stem 5175_54395. All tiled patches whose
filename starts with that stem + "_r" are gathered and stitched.

If you don't know which scene stems exist in your tiled test dir, run
with --list_scenes to print them and exit.
"""

import argparse
import os
import re
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import laspy

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.utils_dataset_dales import DalesDataset
from training.utils.datasets.token_grouping import collate_grouped
from training.atomiser.Atomiser_dales import Atomiser_Dales


IGNORE_INDEX = 255
NUM_CLASSES = 8

# RGB palette (0-255), one per DALES class, chosen for visual distinctness
# in Blender's default viewport.
CLASS_COLORS = np.array([
    [139, 90,  43],   # ground        - brown
    [34,  139, 34],   # vegetation    - green
    [220, 20,  60],   # cars          - crimson
    [255, 140, 0],    # trucks        - orange
    [255, 255, 0],    # power_lines   - yellow
    [148, 0,   211],  # fences        - purple
    [0,   255, 255],  # poles         - cyan
    [70,  130, 180],  # buildings     - steel blue
], dtype=np.uint8)

UNKNOWN_COLOR = np.array([80, 80, 80], dtype=np.uint8)   # gray, for IGNORE_INDEX
WRONG_OVERLAY_COLOR = np.array([255, 0, 255], dtype=np.uint8)  # magenta, optional


def write_ply_binary(path: Path, xyz: np.ndarray, colors: np.ndarray):
    """Minimal binary-little-endian PLY writer, no external deps.

    Args:
        xyz:    [N, 3] float32
        colors: [N, 3] uint8
    """
    n = xyz.shape[0]
    assert colors.shape[0] == n

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    xyz32 = xyz.astype("<f4")
    rgb8  = colors.astype(np.uint8)

    # Interleave x,y,z,r,g,b per vertex via a structured array — much
    # faster than a Python per-point loop for scenes with millions of pts.
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    records = np.empty(n, dtype=dtype)
    records["x"] = xyz32[:, 0]
    records["y"] = xyz32[:, 1]
    records["z"] = xyz32[:, 2]
    records["r"] = rgb8[:, 0]
    records["g"] = rgb8[:, 1]
    records["b"] = rgb8[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(records.tobytes())

    print(f"[export_ply] Wrote {n:,} points to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--config_model", type=str,
                         default="config_test-DALES.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None,
                         help="Required unless --list_scenes is given")
    parser.add_argument("--scene_stem", type=str, default=None,
                         help="Raw scene filename stem, e.g. 5175_54395")
    parser.add_argument("--max_lidar_points", type=int, default=100_000)
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--list_scenes", action="store_true",
                         help="Print available scene stems in the tiled "
                              "test dir and exit (no checkpoint needed)")
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

    print(f"[export_ply] Loading test dataset (eval_full_scene=True)...")
    test_ds = DalesDataset(
        root_path=args.root_path,
        mode="test",
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
        eval_full_scene=True,   # every point becomes a query, not just ctx
    )

    # ── List scenes and exit, if requested ───────────────────────────
    all_stems = sorted(set(
        re.sub(r"_r\d+_c\d+$", "", pid) for pid in
        [row["patch_id"] for row in test_ds.patch_rows]
    ))
    if args.list_scenes:
        print(f"\n[export_ply] {len(all_stems)} scene(s) available in "
              f"tiled test dir:")
        for s in all_stems:
            n_patches = sum(1 for row in test_ds.patch_rows
                             if row["patch_id"].startswith(s + "_r"))
            print(f"  {s}  ({n_patches} patches)")
        return

    if args.scene_stem is None or args.ckpt_path is None:
        raise ValueError("--scene_stem and --ckpt_path are required "
                          "(unless using --list_scenes)")

    if args.scene_stem not in all_stems:
        raise ValueError(
            f"--scene_stem '{args.scene_stem}' not found among tiled test "
            f"scenes. Run with --list_scenes to see what's available."
        )

    # ── Gather all patch indices belonging to this scene ─────────────
    patch_indices = [
        i for i, row in enumerate(test_ds.patch_rows)
        if row["patch_id"].startswith(args.scene_stem + "_r")
    ]
    print(f"[export_ply] Scene '{args.scene_stem}': {len(patch_indices)} "
          f"patches to process.")

    # ── Model ─────────────────────────────────────────────────────────
    print(f"[export_ply] Loading checkpoint: {args.ckpt_path}")
    model = Atomiser_Dales(config=config_model, lookup_table=lookup_table)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    result = model.load_state_dict(state, strict=False)
    print(f"[export_ply] missing keys: {len(result.missing_keys)}, "
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

    # ── Run inference per patch, collecting xyz + pred + gt ──────────
    # NOTE: eval_full_scene=True REQUIRES batch_size=1 (per DalesDataset's
    # own docstring) — the per-scene query counts vary and can't be
    # collated together. We loop patch-by-patch instead of using a
    # DataLoader with batch_size>1.
    all_xyz   = []
    all_pred  = []
    all_gt    = []

    with torch.no_grad():
        for j, idx in enumerate(patch_indices):
            sample = test_ds[idx]
            batch = collate_grouped([sample])
            batch = move_batch(batch, device)

            logits = model(batch, training=False)             # [1, M, 8]
            preds  = logits.argmax(dim=-1)[0]                  # [M]
            labels = batch["queries"][0, :, 4].long()          # [M]
            queries_mask = batch["queries_mask"][0]             # [M] True=padding

            # Recover x,y in the model's internal (reference-grid) pixel
            # frame from the query tokens, then convert back to raw world
            # coordinates using the patch's own origin — simplest robust
            # path: re-read the original patch geometry directly instead
            # of inverting the token encoding math.
            row = test_ds.patch_rows[idx]
            las = laspy.read(row["laz_path"])
            x_world = np.asarray(las.x)
            y_world = np.asarray(las.y)
            z_world = np.asarray(las.z)
            n_points = x_world.shape[0]

            # eval_full_scene=True builds queries in original point order
            # (see DalesDataset: full_lidar_x = lidar_x.copy() BEFORE any
            # subsampling, preserving point order) — so query row i
            # corresponds to raw point i, up to the valid (non-padding)
            # prefix of length n_points.
            valid = (~queries_mask.cpu().numpy())[:n_points]
            preds_np  = preds.cpu().numpy()[:n_points]
            labels_np = labels.cpu().numpy()[:n_points]

            valid = valid & (labels_np != IGNORE_INDEX)

            all_xyz.append(np.stack([x_world, y_world, z_world], axis=-1)[valid])
            all_pred.append(preds_np[valid])
            all_gt.append(labels_np[valid])

            print(f"  [{j+1}/{len(patch_indices)}] {row['patch_id']}: "
                  f"{valid.sum():,}/{n_points:,} points kept")

    xyz  = np.concatenate(all_xyz, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    gt   = np.concatenate(all_gt, axis=0)
    print(f"[export_ply] Scene total: {xyz.shape[0]:,} points across "
          f"{len(patch_indices)} patches.")

    # ── Colorize and write ────────────────────────────────────────────
    pred_colors = CLASS_COLORS[pred]
    gt_colors   = CLASS_COLORS[gt]

    pred_path = Path(args.out_dir) / f"{args.scene_stem}_predictions.ply"
    gt_path   = Path(args.out_dir) / f"{args.scene_stem}_groundtruth.ply"
    write_ply_binary(pred_path, xyz, pred_colors)
    write_ply_binary(gt_path,   xyz, gt_colors)

    # ── Bonus: an "errors" PLY highlighting misclassified points ─────
    # Correct points keep their ground-truth color; wrong points are
    # highlighted magenta, making mistakes trivially visible in Blender.
    err_colors = gt_colors.copy()
    wrong = (pred != gt)
    err_colors[wrong] = WRONG_OVERLAY_COLOR
    err_path = Path(args.out_dir) / f"{args.scene_stem}_errors.ply"
    write_ply_binary(err_path, xyz, err_colors)
    print(f"[export_ply] {wrong.sum():,}/{xyz.shape[0]:,} points "
          f"misclassified ({100*wrong.mean():.2f}%)")

    print(f"\n[export_ply] DONE. Load these into Blender:")
    print(f"  {pred_path}")
    print(f"  {gt_path}")
    print(f"  {err_path}  (magenta = misclassified)")


if __name__ == "__main__":
    main()
