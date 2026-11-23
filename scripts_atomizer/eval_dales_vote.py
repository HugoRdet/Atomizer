"""
eval_dales_vote.py -- DALES test score with one prediction per physical point
============================================================================

Location: scripts_atomizer/eval_dales_vote.py

The test tiles from tile_dales.py are the evaluation windows (50 m, 25 m
stride), so most points appear in several tiles. Following the paper's
protocol, this script:
    1. runs the model on every test tile, with every point as a query;
    2. identifies each physical point by (scene, integer LAS X, Y, Z), the
       coordinates stored in the files (tiling keeps them unchanged);
    3. averages the softmax probabilities over all windows covering a point and
       takes ONE argmax per point;
    4. accumulates every scene into one confusion matrix -> per-class IoU, mIoU.

Also reported:
    * majority vote of per-window argmaxes (how the previous Atomizer-IO
      number was computed), for comparison;
    * the tile-level score (every occurrence counted), as logged by training;
    * per scene: windows, occurrences, unique points, points per window;
    * label conflicts: coordinate keys whose occurrences carry different labels
      (two distinct points with identical coordinates). Expected ~0.
      Such keys are scored once, with their most frequent label.

Points that appear in no test tile (tiles dropped at tiling time, points on a
scene's max edge) cannot be detected without the raw scenes.

    python scripts_atomizer/eval_dales_vote.py \
        --ckpt_path ./checkpoints/dales_unet/<best>.ckpt \
        --config_model config_test-DALES-unet-vhv.yaml --data_root ./data/DALES \
        --out ./results/dales_vhv_v1_test.json

Run on ONE GPU (no srun with several tasks).
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.utils import read_yaml
from training.trainer_DALES_unet import Model_DalesUnet
from training.utils.datasets.dataset_dales_points import (
    DalesPointDataset, collate_dales, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX)


def collate_with_xyz(batch):
    out = collate_dales(batch)
    B, nq = len(batch), out["query_points"].shape[1]
    xyz = np.zeros((B, nq, 3), np.int64)
    for b, s in enumerate(batch):
        xyz[b, :len(s["query_xyz_int"])] = s["query_xyz_int"]
    out["query_xyz_int"] = xyz
    out["las_scale_offset"] = [s["las_scale_offset"] for s in batch]
    return out


def iou_from_confmat(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return iou, float(np.nanmean(iou))


class SceneAccumulator:
    """Collects (xyz, probs, label) occurrences of one scene, then merges them."""

    def __init__(self, name):
        self.name, self.xyz, self.prob, self.lab = name, [], [], []
        self.n_windows, self.scale_offset = 0, None

    def add(self, xyz, prob, lab, scale_offset):
        if self.scale_offset is None:
            self.scale_offset = scale_offset
        elif not np.allclose(self.scale_offset, scale_offset):
            raise RuntimeError(f"[eval] scene {self.name}: tiles with different LAS scale/offset, "
                               "integer coordinates are not comparable")
        self.xyz.append(xyz); self.prob.append(prob.astype(np.float16)); self.lab.append(lab)
        self.n_windows += 1

    def finalize(self, C):
        xyz = np.concatenate(self.xyz)
        prob = np.concatenate(self.prob).astype(np.float32)
        lab = np.concatenate(self.lab)
        n_occ = len(lab)
        mn = xyz.min(0)
        rng = xyz.max(0) - mn + 1
        if float(rng[0]) * float(rng[1]) * float(rng[2]) < 2 ** 62:
            k = ((xyz[:, 0] - mn[0]) * rng[1] + (xyz[:, 1] - mn[1])) * rng[2] + (xyz[:, 2] - mn[2])
            _, inv = np.unique(k, return_inverse=True)
        else:
            _, inv = np.unique(xyz, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        U = int(inv.max()) + 1
        # softmax averaging (sum is enough for the argmax)
        psum = np.zeros((U, C), np.float64)
        np.add.at(psum, inv, prob)
        pred_avg = psum.argmax(1)
        # majority vote of per-window argmaxes (ties -> lowest class index)
        votes = np.bincount(inv * C + prob.argmax(1), minlength=U * C).reshape(U, C)
        pred_vote = votes.argmax(1)
        # one label per point: most frequent label among its occurrences (ignore excluded)
        valid = lab != IGNORE_INDEX
        lab_counts = np.bincount(inv[valid] * C + lab[valid], minlength=U * C).reshape(U, C)
        has_lab = lab_counts.sum(1) > 0
        n_ign = np.bincount(inv[~valid], minlength=U)
        label = np.where(has_lab, lab_counts.argmax(1), IGNORE_INDEX)
        conflicts = int(((lab_counts > 0).sum(1) > 1).sum() + ((n_ign > 0) & has_lab).sum())
        # tile-level reference: every occurrence counted
        cm_tiles = np.bincount(lab[valid] * C + prob[valid].argmax(1), minlength=C * C).reshape(C, C)
        keep = label != IGNORE_INDEX
        cm_avg = np.bincount(label[keep] * C + pred_avg[keep], minlength=C * C).reshape(C, C)
        cm_vote = np.bincount(label[keep] * C + pred_vote[keep], minlength=C * C).reshape(C, C)
        stats = {"windows": self.n_windows, "occurrences": n_occ, "unique_points": U,
                 "scored_points": int(keep.sum()), "occurrences_per_point": n_occ / max(U, 1),
                 "label_conflicts": conflicts,
                 "avg_vs_vote_disagree": int((pred_avg[keep] != pred_vote[keep]).sum())}
        return cm_avg, cm_vote, cm_tiles, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--config_model", default="config_test-DALES-unet-vhv.yaml")
    ap.add_argument("--data_root", default="./data/DALES")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_context_points", type=int, default=350_000,
                    help="context cap; 350k keeps every DALES test tile whole")
    ap.add_argument("--limit_scenes", type=int, default=0, help="> 0: only the first N scenes (quick check)")
    ap.add_argument("--out", default=None, help="JSON with metrics and confusion matrices")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = read_yaml(str(_REPO_ROOT / "training" / "configs" / args.config_model))

    # ---- model ----
    lit = Model_DalesUnet(config, class_weighting="none")
    state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    res = lit.load_state_dict(state, strict=False)
    loss_only = lambda k: k.endswith("_class_weights") or k.startswith("loss_fn.")   # training-only buffers
    missing = [k for k in res.missing_keys if not loss_only(k)]
    unexpected = [k for k in res.unexpected_keys if not loss_only(k)]
    print(f"[eval] checkpoint {args.ckpt_path}: missing {len(missing)}, unexpected {len(unexpected)} keys")
    if missing or unexpected:
        raise RuntimeError(f"[eval] checkpoint does not match the config (first missing: {missing[:5]}, "
                           f"first unexpected: {unexpected[:5]}): use the training config")
    model = lit.model.to(dev).eval()

    # ---- data (tiles sorted by name -> scenes contiguous) ----
    ds = DalesPointDataset(str(Path(args.data_root) / args.split), args.split,
                           max_context_points=args.max_context_points, eval_all_queries=True,
                           augment=False, return_raw_xyz=True,
                           load_partition=config.get("Atomiser", {}).get("unet_hier_nodes") == "ezsp_pre")
    if args.limit_scenes > 0:
        keep_sc = list(OrderedDict.fromkeys(r["scene_id"] for r in ds.rows))[:args.limit_scenes]
        ds.rows = [r for r in ds.rows if r["scene_id"] in keep_sc]
    n_scenes = len(set(r["scene_id"] for r in ds.rows))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_with_xyz, pin_memory=dev.type == "cuda")
    print(f"[eval] {len(ds)} tiles, {n_scenes} scenes, device {dev}")

    C = NUM_CLASSES
    cm = {k: np.zeros((C, C), np.int64) for k in ("avg", "vote", "tiles")}
    per_scene, acc = {}, None
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    t0, done = time.time(), 0

    def close(acc):
        a, v, t, st = acc.finalize(C)
        cm["avg"] += a; cm["vote"] += v; cm["tiles"] += t
        st["mIoU_avg"] = iou_from_confmat(a)[1]
        per_scene[acc.name] = st
        print(f"[eval] scene {acc.name}: {st['windows']} windows, {st['occurrences']:,} occurrences -> "
              f"{st['unique_points']:,} points ({st['occurrences_per_point']:.2f} per point), "
              f"label conflicts {st['label_conflicts']}, scene mIoU {st['mIoU_avg']:.4f}", flush=True)

    with torch.no_grad():
        for batch in dl:
            tb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with amp:
                prob = model(tb)["predictions"].float().softmax(-1).cpu().numpy()
            nq = batch["n_queries"].numpy()
            for b in range(prob.shape[0]):
                sc = batch["scene_id"][b]
                if acc is None or acc.name != sc:
                    if acc is not None:
                        close(acc)
                    acc = SceneAccumulator(sc)
                n = nq[b]
                acc.add(batch["query_xyz_int"][b, :n], prob[b, :n],
                        batch["query_labels"][b, :n].numpy(), batch["las_scale_offset"][b])
            done += prob.shape[0]
            if done % (25 * args.batch_size) < args.batch_size:
                el = time.time() - t0
                print(f"[eval] {done}/{len(ds)} tiles, {el / 60:.1f} min, "
                      f"ETA {el / done * (len(ds) - done) / 60:.1f} min", flush=True)
    if acc is not None:
        close(acc)

    # ---- results ----
    out = {"ckpt": args.ckpt_path, "config": args.config_model, "split": args.split,
           "classes": list(CLASS_NAMES), "per_scene": per_scene}
    for k in ("avg", "vote", "tiles"):
        iou, miou = iou_from_confmat(cm[k])
        out[k] = {"mIoU": miou, "IoU": dict(zip(CLASS_NAMES, iou.tolist())),
                  "OA": float(np.trace(cm[k]) / max(cm[k].sum(), 1)), "confmat": cm[k].tolist()}
    tot = {k: sum(s[k] for s in per_scene.values())
           for k in ("occurrences", "unique_points", "scored_points", "label_conflicts", "avg_vs_vote_disagree")}
    out["totals"] = tot

    print(f"\n[eval] {n_scenes} scenes, {tot['scored_points']:,} scored points "
          f"({tot['occurrences']:,} occurrences), label conflicts {tot['label_conflicts']}, "
          f"averaging vs vote differ on {tot['avg_vs_vote_disagree']:,} points")
    w = max(len(n) for n in CLASS_NAMES)
    print(f"{'':{w}}  {'softmax avg':>11}  {'maj. vote':>9}  {'tiles':>7}")
    for n in CLASS_NAMES:
        print(f"{n:{w}}  {out['avg']['IoU'][n]:11.4f}  {out['vote']['IoU'][n]:9.4f}  {out['tiles']['IoU'][n]:7.4f}")
    print(f"{'mIoU':{w}}  {out['avg']['mIoU']:11.4f}  {out['vote']['mIoU']:9.4f}  {out['tiles']['mIoU']:7.4f}")
    print(f"{'OA':{w}}  {out['avg']['OA']:11.4f}  {out['vote']['OA']:9.4f}  {out['tiles']['OA']:7.4f}")

    # LaTeX row, column order of the paper's table
    order = ["ground", "buildings", "cars", "trucks", "poles", "power_lines", "fences", "vegetation"]
    row = " & ".join([f"{out['avg']['mIoU']:.3f}"] + [f"{out['avg']['IoU'][c]:.3f}" for c in order])
    print(f"\nLaTeX (softmax averaging): Ours & {row} \\\\")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
