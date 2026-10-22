"""
test_sliding_window_dales_precomputed.py
===========================================

Sliding-window test evaluation using the AUTHORITATIVE (precomputed
token_latent_assignment) geo-pruning path exclusively -- NO fallback path
anywhere in this script, unlike the earlier ad hoc sliding-window
approach.

How: overlapping windows are pre-materialized as REAL tiled .laz patches
via tile_dales.py's --overlap_m (already supported, no new tiling code
needed), then precompute_dales_latent_assignment.py runs on them exactly
like any other tiled patch. This script then just uses DalesDataset
directly with eval_full_scene=True over that overlap-tiled directory --
every patch gets a full per-point query set with the CORRECT precomputed
assignment, and predictions get accumulated per PHYSICAL point (keyed by
exact raw LAS integer coordinates X,Y,Z -- not the scaled float x,y,z, to
guarantee exact-match accumulation across separately-read overlapping
tile files) across every overlapping patch covering it, softmax-averaged,
then argmax'd -- same aggregation logic as the earlier sliding-window
script, just now built entirely on the code path the model was actually
trained with.

REQUIRES: tile_dales.py + precompute_dales_latent_assignment.py already
run on the overlap-tiled test directory (see module docstring commands).

Usage:
    python test_sliding_window_dales_precomputed.py \
        --root_path ./data_sw \
        --config_model config_test-DALES.yaml \
        --ckpt_path <ckpt> \
        --max_lidar_points 256000
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import laspy
from tqdm import tqdm

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.utils_dataset_dales import DalesDataset, REMAP_LUT
from training.utils.datasets.token_grouping import collate_grouped
from training.atomiser.Atomiser_dales import Atomiser_Dales


IGNORE_INDEX = 255
NUM_CLASSES = 8
DALES_CLASS_NAMES = [
    "ground", "vegetation", "cars", "trucks",
    "power_lines", "fences", "poles", "buildings",
]


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


def scene_stem_from_patch_id(patch_id: str) -> str:
    """'5080_54400_r003_c002' -> '5080_54400' (strip tile_dales.py's
    _r###_c### suffix)."""
    return re.sub(r"_r\d+_c\d+$", "", patch_id)


def _run_worker(rank: int, world_size: int, args, scene_stems_shard: list,
                 result_queue):
    """
    Runs on ONE GPU: builds its own model/dataset, processes its shard of
    SCENES (each scene's set of overlapping patches, in full), pushes
    ("ok", rank, cm, per_scene_miou_list) back via result_queue, or
    ("error", rank, None, traceback_str) if anything crashes -- same
    crash-visibility pattern as the earlier sliding-window script's
    multi-GPU support.
    """
    import traceback as _traceback

    try:
        print(f"[GPU {rank}] starting, {len(scene_stems_shard)} scenes assigned",
              flush=True)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        config_model = read_yaml(f"./training/configs/{args.config_model}")
        configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
        configs_dataset = read_yaml(configs_dataset_path)
        bands = {}

        lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
        TokenBuilder.REFERENCE_SIZES[0.2] = 2048
        lookup_table.get_or_register_modality(0.2, 2048)
        lookup_table.get_resolution_idx(0.2)
        lookup_table.register_abstract_channel("ELEVATION")

        test_ds = DalesDataset(
            root_path=args.root_path, mode="test",
            dataset_config=bands, config_model=config_model,
            look_up=lookup_table, max_lidar_points=args.max_lidar_points,
            eval_full_scene=True,
        )

        print(f"[GPU {rank}] loading checkpoint...", flush=True)
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
        print(f"[GPU {rank}] model loaded, starting inference", flush=True)

        scene_to_indices = defaultdict(list)
        for idx, row in enumerate(test_ds.patch_rows):
            stem = scene_stem_from_patch_id(row["patch_id"])
            if stem in scene_stems_shard:
                scene_to_indices[stem].append(idx)

        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        per_scene_miou = []

        desc = f"[GPU {rank}]" if world_size > 1 else "[sw_precomp]"
        for scene_stem in tqdm(scene_stems_shard, desc=desc, position=rank,
                                mininterval=5.0):
            patch_indices = scene_to_indices[scene_stem]

            prob_sum = {}
            count    = {}
            gt_label = {}

            for idx in patch_indices:
                row = test_ds.patch_rows[idx]
                sample = test_ds[idx]
                batch = collate_grouped([sample])
                batch = move_batch(batch, device)

                with torch.no_grad():
                    logits = model(batch, training=False)
                    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
                labels = batch["queries"][0, :, 4].long().cpu().numpy()
                queries_mask = batch["queries_mask"][0].cpu().numpy()

                las = laspy.read(row["laz_path"])
                X = np.asarray(las.X, dtype=np.int64)
                Y = np.asarray(las.Y, dtype=np.int64)
                Z = np.asarray(las.Z, dtype=np.int64)
                n_points = X.shape[0]

                valid = (~queries_mask.astype(bool))[:n_points]
                probs_valid = probs[:n_points][valid]
                labels_valid = labels[:n_points][valid]
                X_v, Y_v, Z_v = X[valid], Y[valid], Z[valid]

                for i in range(X_v.shape[0]):
                    key = (int(X_v[i]), int(Y_v[i]), int(Z_v[i]))
                    if key not in prob_sum:
                        prob_sum[key] = np.zeros(NUM_CLASSES, dtype=np.float64)
                        count[key] = 0
                        gt_label[key] = int(labels_valid[i])
                    prob_sum[key] += probs_valid[i]
                    count[key] += 1

            scene_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for key, gt in gt_label.items():
                if gt == IGNORE_INDEX:
                    continue
                mean_probs = prob_sum[key] / max(count[key], 1)
                pred = int(np.argmax(mean_probs))
                scene_cm[gt, pred] += 1

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
            n_unique_points = len(prob_sum)
            per_scene_miou.append((scene_stem, scene_miou, n_unique_points))
            print(f"[GPU {rank}] {scene_stem}: {len(patch_indices)} patches, "
                  f"{n_unique_points:,} unique physical points, "
                  f"scene_mIoU={scene_miou:.4f}", flush=True)

            del prob_sum, count, gt_label

        print(f"[GPU {rank}] DONE, {len(scene_stems_shard)} scenes processed",
              flush=True)
        result_queue.put(("ok", rank, cm, per_scene_miou))

    except Exception:
        tb = _traceback.format_exc()
        print(f"[GPU {rank}] CRASHED:\n{tb}", flush=True)
        result_queue.put(("error", rank, None, tb))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=str, required=True,
                         help="Parent of the OVERLAP-tiled DALES/test dir "
                              "(e.g. ./data_sw, containing "
                              "./data_sw/DALES/test/*.laz + precomputed npz)")
    parser.add_argument("--config_model", type=str,
                         default="config_test-DALES.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--max_lidar_points", type=int, default=256_000)
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--num_gpus", type=int, default=None,
                         help="Number of GPUs to shard SCENES across "
                              "(default: all visible GPUs, or 1 if none). "
                              "Every scene is independent, so this is "
                              "simple work-sharding, not DDP.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Just list scene stems from filenames -- avoids constructing a full
    # DalesDataset (+ lookup table) in the parent when each worker will
    # build its own anyway.
    test_dir = Path(args.root_path) / "DALES" / "test"
    laz_files = sorted(test_dir.glob("*.laz"))
    if not laz_files:
        raise FileNotFoundError(f"No .laz files found under {test_dir}")
    all_scene_stems = sorted(set(
        scene_stem_from_patch_id(p.stem) for p in laz_files
    ))

    n_gpus_available = torch.cuda.device_count()
    world_size = args.num_gpus if args.num_gpus is not None else max(1, n_gpus_available)
    world_size = min(world_size, max(1, n_gpus_available), len(all_scene_stems))

    print(f"[sw_precomp] {len(all_scene_stems)} source scenes, "
          f"{len(laz_files)} overlapping patches total, "
          f"world_size={world_size} GPU(s)")

    shards = [set(all_scene_stems[i::world_size]) for i in range(world_size)]

    if world_size == 1:
        result_queue = mp.SimpleQueue()
        _run_worker(0, 1, args, sorted(shards[0]), result_queue)
        status, _, global_cm, per_scene_miou = result_queue.get()
        if status == "error":
            raise RuntimeError(f"Worker crashed:\n{per_scene_miou}")
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        procs = []
        for rank in range(world_size):
            p = ctx.Process(target=_run_worker,
                             args=(rank, world_size, args,
                                   sorted(shards[rank]), result_queue))
            p.start()
            procs.append(p)

        results = {}
        import time as _time
        while len(results) < world_size:
            got_one = False
            if not result_queue.empty():
                status, rank, cm_r, payload = result_queue.get()
                if status == "error":
                    raise RuntimeError(
                        f"[sw_precomp] Worker on GPU {rank} crashed:\n{payload}"
                    )
                results[rank] = (cm_r, payload)
                got_one = True

            if not got_one:
                for rank, p in enumerate(procs):
                    if rank not in results and not p.is_alive() and p.exitcode is not None:
                        raise RuntimeError(
                            f"[sw_precomp] Worker on GPU {rank} exited "
                            f"(exit code {p.exitcode}) without reporting a "
                            f"result -- check its stdout above for a crash "
                            f"traceback."
                        )
                _time.sleep(1.0)

        for p in procs:
            p.join()

        global_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        per_scene_miou = []
        for rank in sorted(results.keys()):
            worker_cm, worker_per_scene = results[rank]
            global_cm += worker_cm
            per_scene_miou.extend(worker_per_scene)

    # -- Final aggregate metrics --------------------------------------
    print(f"\n{'='*70}")
    print(f"  SLIDING-WINDOW TEST RESULTS (precomputed path, overlap-tiled)")
    print(f"{'='*70}")

    print(f"\nPer-scene mIoU:")
    for stem, miou, n_pts in per_scene_miou:
        print(f"  {stem:<20s} mIoU={miou:.4f}  ({n_pts:,} unique pts)")

    print(f"\nPer-class IoU (aggregated across all {len(all_scene_stems)} scenes):")
    ious = []
    for c, name in enumerate(DALES_CLASS_NAMES):
        tp = global_cm[c, c]
        fp = global_cm[:, c].sum() - tp
        fn = global_cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        ious.append(iou)
        print(f"  {name:<14s}: {iou*100:6.2f}%" if denom > 0
              else f"  {name:<14s}:    n/a")

    valid_ious = [i for i in ious if not np.isnan(i)]
    miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    print(f"\n  mean IoU (sliding-window, precomputed path): {miou*100:.2f}%")

    total_correct = np.trace(global_cm)
    total_valid = global_cm.sum()
    acc = total_correct / max(total_valid, 1)
    print(f"  overall accuracy: {acc*100:.2f}%")

    out_path = Path(args.out_dir) / "sliding_window_precomputed_results.npz"
    np.savez_compressed(
        out_path, confusion_matrix=global_cm, per_class_iou=np.array(ious),
        mean_iou=miou,
    )
    print(f"\n[sw_precomp] Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
