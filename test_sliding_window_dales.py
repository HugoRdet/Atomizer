"""
test_sliding_window_dales.py
================================

Computes REAL test-set mIoU / per-class IoU using overlapping sliding-window
inference (softmax-probability averaging per point across windows), instead
of the tiled-patch evaluation DalesDataset normally does.

WHY this differs from a normal --test_only run: DalesDataset's tiled-patch
evaluation scores each 50m patch independently -- points near a patch
boundary get truncated context (their nearest latents/neighbors are cut off
by the tile edge). Sliding windows give every point several overlapping
"views" and average softmax probabilities across them, so boundary points
benefit from whichever window(s) placed them away from an edge. This is the
metric worth reporting as your final test number.

Since sliding windows are ad hoc (not the fixed training tiling), there's no
precomputed token->latent assignment for them -- this uses
GeographicPruningDales's FALLBACK on-the-fly path (patch_ids passed, not
token_latent_assignment). Slower per-window than the precompute-consuming
path, but there's no way around that for windows outside the training grid.

Usage:
    python test_sliding_window_dales.py \
        --root_path ./data \
        --config_model config_test-DALES.yaml \
        --ckpt_path ./checkpoints/dales/precision32_dales_v1-epoch=XX-val_mIoU=X.ckpt \
        --scene_dir ./data/DALES/test_raw \
        --overlap 0.5 \
        --max_lidar_points 256000

--scene_dir should point at the RAW (untiled) test scene .las files, e.g.
the original data/DALES/test/ before tile_dales.py ran on it -- NOT the
tiled patches directory (DalesDataset's root_path/DALES/test).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import laspy
from tqdm import tqdm

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.utils_dataset_dales import (
    DalesDataset, REMAP_LUT, _normalize_intensity,
    _resolve_elevation_spectral_idx,
)
from training.atomiser.Atomiser_dales import Atomiser_Dales


IGNORE_INDEX = 255
NUM_CLASSES = 8
DALES_CLASS_NAMES = [
    "ground", "vegetation", "cars", "trucks",
    "power_lines", "fences", "poles", "buildings",
]


def compute_window_size_m(scene_area_m2: float, n_points: int,
                           target_points: int) -> float:
    density = n_points / max(scene_area_m2, 1e-6)
    return float(np.sqrt(target_points / max(density, 1e-6)))


def generate_sliding_windows(x_min, y_min, x_max, y_max,
                              window_size_m: float, overlap: float):
    stride = window_size_m * (1.0 - overlap)
    assert stride > 0, "overlap must be < 1.0"
    n_cols = max(1, int(np.ceil((x_max - x_min - window_size_m) / stride)) + 1)
    n_rows = max(1, int(np.ceil((y_max - y_min - window_size_m) / stride)) + 1)
    for row in range(n_rows):
        wy_min = y_min + row * stride
        for col in range(n_cols):
            wx_min = x_min + col * stride
            yield (wx_min, wy_min, wx_min + window_size_m, wy_min + window_size_m)


class SlidingWindowInferencer:
    """Builds tokens/queries for one window using the SAME conventions as
    DalesDataset.__getitem__ (echo, intensity, elevation normalization,
    fixed pixel frame), but for an arbitrary window rather than a
    pre-tiled patch, and with NO augmentation (test-time). Single-atom
    (A=1) decoder-skip mapping, matching the current (non-k-NN)
    DalesDataset contract.
    """

    def __init__(self, model, lookup_table, max_lidar_points: int,
                 device: torch.device):
        self.model = model
        self.lookup = lookup_table
        self.max_lidar_points = max_lidar_points
        self.device = device

        self.PIXEL_RESOLUTION = DalesDataset.PIXEL_RESOLUTION
        self.PATCH_SIZE_M     = DalesDataset.PATCH_SIZE_M
        self.PATCH_SIZE_PX    = DalesDataset.PATCH_SIZE_PX
        self.Z_GROUND_REL_LO  = DalesDataset.Z_GROUND_REL_LO
        self.Z_GROUND_REL_HI  = DalesDataset.Z_GROUND_REL_HI
        self.Z_GROUND_REL_SCALE = DalesDataset.Z_GROUND_REL_SCALE
        self.GROUND_MEDIAN_MIN_PTS = DalesDataset.GROUND_MEDIAN_MIN_PTS
        self.IGNORE_INDEX = DalesDataset.IGNORE_INDEX

        self.lidar_spectral_idx = _resolve_elevation_spectral_idx(lookup_table)
        self.resolution_idx = lookup_table.get_resolution_idx(self.PIXEL_RESOLUTION)
        self.token_builder = TokenBuilder(lookup_table)

    def predict_window(self, x, y, z, classification, intensity_raw,
                        return_number, number_of_returns,
                        wx_min, wy_min, window_id: str):
        n_win = x.shape[0]
        if n_win == 0:
            return None, None

        las_cls = np.clip(classification, 0, REMAP_LUT.shape[0] - 1)
        labels = REMAP_LUT[las_cls]

        ground_mask = (labels == 0)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z[ground_mask]))
        else:
            local_ground = float(np.percentile(z, 5.0))
        z_rel  = z - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        intensity_norm = _normalize_intensity(intensity_raw)

        px = (x - wx_min) / self.PIXEL_RESOLUTION
        py = ((wy_min + self.PATCH_SIZE_M) - y) / self.PIXEL_RESOLUTION
        px = np.clip(px, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)
        py = np.clip(py, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)

        if n_win > self.max_lidar_points:
            rng = np.random.default_rng(hash(window_id) & 0xFFFFFFFF)
            sel = rng.choice(n_win, size=self.max_lidar_points, replace=False)
        else:
            sel = None

        ctx_px = px if sel is None else px[sel]
        ctx_py = py if sel is None else py[sel]
        ctx_z  = z_norm if sel is None else z_norm[sel]
        ctx_int = intensity_norm if sel is None else intensity_norm[sel]
        ctx_labels = labels if sel is None else labels[sel]
        ctx_rn = return_number if sel is None else return_number[sel]
        ctx_nr = number_of_returns if sel is None else number_of_returns[sel]

        positions_ctx = torch.from_numpy(np.stack([ctx_px, ctx_py], axis=1)).float()
        values_ctx    = torch.from_numpy(ctx_z).float()
        labels_ctx    = torch.from_numpy(ctx_labels.astype(np.int64))
        intensity_ctx = torch.from_numpy(ctx_int).float()

        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=values_ctx, positions=positions_ctx, labels=labels_ctx,
            resolution=self.PIXEL_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=-1,
            return_number=ctx_rn, number_of_returns=ctx_nr,
            intensity_override=intensity_ctx,
        )

        n_ctx = lidar_tokens.shape[0]
        if n_ctx < self.max_lidar_points:
            n_pad = self.max_lidar_points - n_ctx
            pad = torch.zeros(n_pad, 8)
            pad[:, 4] = self.IGNORE_INDEX
            lidar_tokens = torch.cat([lidar_tokens, pad], dim=0)
            lidar_mask = torch.cat([
                torch.zeros(n_ctx, dtype=torch.bool),
                torch.ones(n_pad, dtype=torch.bool),
            ])
        else:
            lidar_mask = torch.zeros(n_ctx, dtype=torch.bool)

        groups = {
            self.PIXEL_RESOLUTION: {
                "tokens": lidar_tokens.unsqueeze(0),
                "mask":   lidar_mask.unsqueeze(0),
                "shape":  (1, self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            }
        }

        positions_q = torch.from_numpy(np.stack([px, py], axis=1)).float()
        values_q    = torch.from_numpy(z_norm).float()
        labels_q    = torch.from_numpy(labels.astype(np.int64))

        queries = self.token_builder.build_sparse_queries(
            positions=positions_q, labels=labels_q,
            resolution=self.PIXEL_RESOLUTION,
            first_spectral_idx=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX, time_idx=-1,
        )
        queries[:, 0] = values_q
        queries_mask = torch.zeros(n_win, dtype=torch.bool)

        # -- context_map for decoder-skip query_token_idx (single-atom,
        # matching current non-k-NN DalesDataset contract) --------------
        if sel is not None:
            context_map = np.full(n_win, -1, dtype=np.int64)
            context_map[sel] = np.arange(sel.shape[0], dtype=np.int64)
        else:
            context_map = np.arange(n_win, dtype=np.int64)
        query_token_valid = torch.from_numpy(context_map >= 0).bool()
        query_token_idx = torch.from_numpy(
            np.clip(context_map, 0, None)
        ).long().unsqueeze(-1)

        batch = {
            "groups": groups,
            "queries": queries.unsqueeze(0).to(self.device),
            "queries_mask": queries_mask.unsqueeze(0).to(self.device),
            "target_resolution": self.PIXEL_RESOLUTION,
            "patch_id": [window_id],
            "query_token_idx": query_token_idx.unsqueeze(0).to(self.device),
            "query_token_valid": query_token_valid.unsqueeze(0).to(self.device),
        }
        batch["groups"][self.PIXEL_RESOLUTION]["tokens"] = \
            batch["groups"][self.PIXEL_RESOLUTION]["tokens"].to(self.device)
        batch["groups"][self.PIXEL_RESOLUTION]["mask"] = \
            batch["groups"][self.PIXEL_RESOLUTION]["mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(batch, training=False)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

        return probs, labels


def evaluate_scene(inferencer, las_path: Path, overlap: float, target_points: int):
    """Returns (pred_labels, gt_labels, coverage) for one raw scene, all
    points, sliding-window softmax-averaged."""
    las = laspy.read(str(las_path))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.int64)
    intensity_raw = np.asarray(las.intensity, dtype=np.float32)
    return_number = np.asarray(las.return_number, dtype=np.int64)
    number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)
    n_total = x.shape[0]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    window_size_m = DalesDataset.PATCH_SIZE_M

    windows = list(generate_sliding_windows(x_min, y_min, x_max, y_max,
                                             window_size_m, overlap))

    prob_sum  = np.zeros((n_total, NUM_CLASSES), dtype=np.float64)
    coverage  = np.zeros(n_total, dtype=np.int32)
    gt_labels_global = np.full(n_total, IGNORE_INDEX, dtype=np.int64)

    for wi, (wx_min, wy_min, wx_max, wy_max) in enumerate(
        tqdm(windows, desc=f"  {las_path.stem} windows", leave=False,
             disable=(len(windows) < 5))
    ):
        in_window = ((x >= wx_min) & (x < wx_max) &
                     (y >= wy_min) & (y < wy_max))
        win_idx = np.nonzero(in_window)[0]
        if win_idx.shape[0] == 0:
            continue

        probs, win_labels = inferencer.predict_window(
            x[win_idx], y[win_idx], z[win_idx], classification[win_idx],
            intensity_raw[win_idx], return_number[win_idx],
            number_of_returns[win_idx],
            wx_min, wy_min, window_id=f"{las_path.stem}_win{wi}",
        )
        if probs is None:
            continue

        prob_sum[win_idx] += probs
        coverage[win_idx] += 1
        gt_labels_global[win_idx] = win_labels

    safe_coverage = np.maximum(coverage, 1)[:, None]
    mean_probs = prob_sum / safe_coverage
    pred = np.argmax(mean_probs, axis=-1)
    pred[coverage == 0] = IGNORE_INDEX

    return pred, gt_labels_global, coverage, len(windows)


def _run_worker(rank: int, world_size: int, args, las_files_shard: list,
                 result_queue):
    """
    Runs on ONE GPU: loads its own copy of the model, processes its shard
    of scenes, pushes (cm, per_scene_miou_list) back to the parent via
    result_queue. No cross-process synchronization needed DURING
    inference -- every scene is fully independent, so this is a simple
    "shard the work, reduce at the end" pattern, not real DDP.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    config_model = read_yaml(f"./training/configs/{args.config_model}")
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
    configs_dataset = read_yaml(configs_dataset_path)
    bands = {}

    lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
    TokenBuilder.REFERENCE_SIZES[0.2] = 2048
    lookup_table.get_or_register_modality(0.2, 2048)
    lookup_table.get_resolution_idx(0.2)
    lookup_table.register_abstract_channel("ELEVATION")

    model = Atomiser_Dales(config=config_model, lookup_table=lookup_table)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    inferencer = SlidingWindowInferencer(
        model, lookup_table, args.max_lidar_points, device
    )

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_scene_miou = []

    desc = f"[GPU {rank}]" if world_size > 1 else "[sw_test]"
    for las_path in tqdm(las_files_shard, desc=desc, position=rank):
        pred, gt, coverage, n_windows = evaluate_scene(
            inferencer, las_path, args.overlap, args.max_lidar_points
        )

        uncovered = int((coverage == 0).sum())
        if uncovered > 0:
            print(f"  {desc} WARNING: {las_path.name} has {uncovered} "
                  f"points with NO window coverage -- increase --overlap.")

        valid = (gt != IGNORE_INDEX) & (pred != IGNORE_INDEX)
        t = gt[valid]
        p = pred[valid]
        scene_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(scene_cm, (t, p), 1)
        cm += scene_cm

        scene_ious = []
        for c in range(NUM_CLASSES):
            tp = scene_cm[c, c]
            fp = scene_cm[:, c].sum() - tp
            fn = scene_cm[c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                scene_ious.append(tp / denom)
        scene_miou = float(np.mean(scene_ious)) if scene_ious else float("nan")
        per_scene_miou.append((las_path.stem, scene_miou, int(valid.sum())))

    result_queue.put((rank, cm, per_scene_miou))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--config_model", type=str,
                         default="config_test-DALES.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, required=True,
                         help="Directory of RAW (untiled) test scene .las files")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max_lidar_points", type=int, default=256_000)
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--num_gpus", type=int, default=None,
                         help="Number of GPUs to shard scenes across "
                              "(default: all visible GPUs, or 1 if none). "
                              "Every scene is independent, so this is "
                              "simple work-sharding, not DDP.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scene_dir = Path(args.scene_dir)
    las_files = sorted(scene_dir.glob("*.las")) + sorted(scene_dir.glob("*.laz"))
    if not las_files:
        raise FileNotFoundError(f"No .las/.laz files found under {scene_dir}")

    n_gpus_available = torch.cuda.device_count()
    world_size = args.num_gpus if args.num_gpus is not None else max(1, n_gpus_available)
    world_size = min(world_size, max(1, n_gpus_available), len(las_files))

    print(f"[sw_test] {len(las_files)} test scenes, overlap={args.overlap}, "
          f"window_size={DalesDataset.PATCH_SIZE_M}m, "
          f"world_size={world_size} GPU(s)")

    # Shard scenes round-robin across workers (keeps shards balanced even
    # if scene sizes vary — no strong reason to prefer contiguous chunks).
    shards = [las_files[i::world_size] for i in range(world_size)]

    if world_size == 1:
        result_queue = mp.SimpleQueue()
        _run_worker(0, 1, args, shards[0], result_queue)
        _, cm, per_scene_miou = result_queue.get()
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        procs = []
        for rank in range(world_size):
            p = ctx.Process(target=_run_worker,
                             args=(rank, world_size, args, shards[rank], result_queue))
            p.start()
            procs.append(p)

        results = [result_queue.get() for _ in range(world_size)]
        for p in procs:
            p.join()

        # Reduce: sum confusion matrices, concatenate per-scene lists.
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        per_scene_miou = []
        for _, worker_cm, worker_per_scene in sorted(results, key=lambda r: r[0]):
            cm += worker_cm
            per_scene_miou.extend(worker_per_scene)

    # -- Final aggregate metrics --------------------------------------
    print(f"\n{'='*70}")
    print(f"  SLIDING-WINDOW TEST RESULTS (overlap={args.overlap})")
    print(f"{'='*70}")

    print(f"\nPer-scene mIoU:")
    for stem, miou, n_pts in per_scene_miou:
        print(f"  {stem:<20s} mIoU={miou:.4f}  ({n_pts:,} pts)")

    print(f"\nPer-class IoU (aggregated across all {len(las_files)} scenes):")
    ious = []
    for c, name in enumerate(DALES_CLASS_NAMES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        ious.append(iou)
        print(f"  {name:<14s}: {iou*100:6.2f}%" if denom > 0
              else f"  {name:<14s}:    n/a")

    valid_ious = [i for i in ious if not np.isnan(i)]
    miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    print(f"\n  mean IoU (sliding-window): {miou*100:.2f}%")

    total_correct = np.trace(cm)
    total_valid = cm.sum()
    acc = total_correct / max(total_valid, 1)
    print(f"  overall accuracy: {acc*100:.2f}%")

    out_path = Path(args.out_dir) / "sliding_window_test_results.npz"
    np.savez_compressed(
        out_path, confusion_matrix=cm, per_class_iou=np.array(ious),
        mean_iou=miou, overlap=args.overlap,
    )
    print(f"\n[sw_test] Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
