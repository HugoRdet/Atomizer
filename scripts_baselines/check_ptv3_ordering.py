"""
check_ptv3_ordering.py
======================

Sanity checks for PTv3Segmenter (pixels-as-points) on Cashew.

  0. Determinism
       Same input + same seed -> same logits. Measures the run-to-run noise
       floor used by the other checks.
  1. Metadata order
       Each row of the backbone output carries the same (batch, grid_coord)
       as the input row at the same index, i.e. "row i in == row i out".
  2. Permutation follow
       Shuffle the input rows inside each sample. If the backbone preserves
       input order, its output must follow the shuffle (out_perm ==
       out_ref[perm]). Does not rely on metadata, so it also catches a bug
       where feat gets re-indexed but grid_coord does not.
  3. Localized poke (full wrapper, incl. the final view/permute)
       Add a large perturbation to one pixel; the largest change in the
       output logit map must land on that same pixel. If the reshape put
       predictions on the wrong pixels, it would land somewhere unrelated.
  4. Batch independence (eval mode)
       Logits for sample 0 must not depend on which other samples share
       the batch.
  5. Real data (only with --ckpt and --data_dir)
       Test-split mIoU with the real input vs. with the input zeroed out.
       Zeroed input should drop to roughly majority-class level; if it
       stays high, labels are leaking in somewhere. Also saves PNGs of
       RGB / target / prediction / error for visual inspection.

Run with the SAME --ptv3_* arguments used for training. Place this file
next to script_train_cashew_baselines.py (same sys.path setup).

Usage:
    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH

    # ordering checks only (random weights), on real tiles:
    python check_ptv3_ordering.py \
        --data_dir ./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant

    # trained checkpoint: ordering checks + leakage check + plots
    python check_ptv3_ordering.py \
        --data_dir ./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant \
        --ckpt ./checkpoints/cashew_baselines/bl_ptv3_pixels_ptv3-epoch=XX-val_mIoU=0.XXXX.ckpt \
        --ptv3_enc_channels 32 64 128 256 384

Exit code is 0 if every check passes, 1 otherwise.

Note: PTv3 may randomly shuffle its serialization orders at each forward.
Every forward here is therefore run under the same fixed seed, so two
runs on the same points use the same orders.
"""

import os
import sys
import argparse
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from training.PTv3.model_ptv3_seg import PTv3Segmenter
from point_transformer_v3 import Point


NUM_CLASSES = 7
NUM_CHANNELS = 12
IGNORE_INDEX = 255
GSD = 10.0

RESULTS = []


# =============================================================================
# helpers
# =============================================================================

def report(name, ok, msg):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def seeded(fn, *a, seed=0):
    """Run fn(*a) under a fixed seed (PTv3 may shuffle serialization orders)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return fn(*a)


def make_point(m, x, perm=None):
    """Build the same Point PTv3Segmenter.forward builds, optionally with the
    rows shuffled by `perm` (perm must keep each sample's rows contiguous)."""
    B, C, H, W = x.shape
    g = m._grid(H, W, x.device)
    grid = g.repeat(B, 1)
    feat = x.permute(0, 2, 3, 1).reshape(-1, C).float()
    if perm is not None:
        grid, feat = grid[perm], feat[perm]
    return Point(
        coord=grid.float() * m.gsd,
        grid_coord=grid,
        feat=feat,
        offset=torch.arange(1, B + 1, device=x.device) * (H * W),
        grid_size=float(m.gsd),
    )


def batch_index(point):
    if "batch" in point.keys():
        return point["batch"].long()
    off = point["offset"].long()
    counts = torch.diff(off, prepend=off.new_zeros(1))
    return torch.repeat_interleave(torch.arange(len(off), device=off.device), counts)


# =============================================================================
# checks
# =============================================================================

def check_determinism(m, x):
    a = seeded(m, x).float()
    b = seeded(m, x).float()
    noise = (a - b).abs().max().item()
    scale = a.abs().mean().item()
    report("0 determinism", noise <= 1e-2 * scale + 1e-6,
           f"max |diff| between two identical runs = {noise:.2e} "
           f"(mean |logit| = {scale:.2e})")


def check_metadata(m, x):
    B, C, H, W = x.shape
    n = B * H * W
    out = seeded(m.backbone, make_point(m, x))

    if out["feat"].shape[0] != n:
        report("1 metadata order", False,
               f"backbone returned {out['feat'].shape[0]} rows, expected {n} "
               f"(output is not the level-0 point set)")
        return

    exp_grid = m._grid(H, W, x.device).repeat(B, 1)
    exp_batch = torch.arange(B, device=x.device).repeat_interleave(H * W)
    got_grid = out["grid_coord"].int()
    got_batch = batch_index(out)

    bad = (got_grid != exp_grid).any(1) | (got_batch != exp_batch)
    nbad = int(bad.sum())
    msg = f"{n - nbad}/{n} output rows sit on the same pixel as the input row"
    if nbad:
        i = int(bad.nonzero()[0])
        msg += (f"; first mismatch at row {i}: expected b={exp_batch[i].item()} "
                f"(x,y)={exp_grid[i, :2].tolist()}, got b={got_batch[i].item()} "
                f"(x,y)={got_grid[i, :2].tolist()}")
    report("1 metadata order", nbad == 0, msg)


def check_permutation(m, x):
    B, C, H, W = x.shape
    HW = H * W
    gen = torch.Generator().manual_seed(123)
    perm = torch.cat(
        [torch.randperm(HW, generator=gen) + b * HW for b in range(B)]
    ).to(x.device)

    ref = seeded(m.backbone, make_point(m, x))["feat"].float()
    ref2 = seeded(m.backbone, make_point(m, x))["feat"].float()
    out = seeded(m.backbone, make_point(m, x, perm))["feat"].float()

    scale = ref.abs().mean().item() + 1e-12
    noise = (ref - ref2).abs().mean().item() / scale
    aligned = (out - ref[perm]).abs().mean().item() / scale   # output follows input order
    naive = (out - ref).abs().mean().item() / scale           # output ignores input order

    ok = aligned < max(0.05, 10 * noise) and naive > 5 * aligned
    report("2 permutation follow", ok,
           f"rel. error assuming output follows input order = {aligned:.2e}, "
           f"assuming it ignores input order = {naive:.2e}, "
           f"run-to-run noise = {noise:.2e}")


def check_poke(m, x, n_pokes, poke):
    _, _, H, W = x.shape
    xs = x[:1]
    base = seeded(m, xs).float()
    gen = torch.Generator().manual_seed(7)

    hits, dists = 0, []
    for _ in range(n_pokes):
        py = int(torch.randint(0, H, (1,), generator=gen))
        px = int(torch.randint(0, W, (1,), generator=gen))
        x2 = xs.clone()
        x2[0, :, py, px] += poke
        d = (seeded(m, x2).float() - base).abs().sum(1)[0]    # [H, W]
        ay, ax = divmod(int(d.argmax()), W)
        dist = ((ay - py) ** 2 + (ax - px) ** 2) ** 0.5
        dists.append(dist)
        hits += int(ay == py and ax == px)

    med = float(np.median(dists))
    ok = hits >= 0.8 * n_pokes or med <= 1.0
    report("3 localized poke", ok,
           f"largest logit change landed on the poked pixel {hits}/{n_pokes} "
           f"times, median distance {med:.1f} px, max {max(dists):.1f} px")


def check_batch_independence(m, x):
    if x.shape[0] < 2:
        print("[SKIP] 4 batch independence: needs --batch_size >= 2")
        return
    a = seeded(m, x)[:1].float()
    b = seeded(m, x[:1]).float()
    rel = ((a - b).abs().mean() / (a.abs().mean() + 1e-12)).item()
    report("4 batch independence", rel < 0.05,
           f"rel. diff of sample 0 alone vs. in a batch of {x.shape[0]} = {rel:.2e}")


# =============================================================================
# real data: leakage check + plots
# =============================================================================

def update_cm(cm, pred, tgt):
    valid = tgt != IGNORE_INDEX
    idx = tgt[valid] * NUM_CLASSES + pred[valid]
    cm += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(
        NUM_CLASSES, NUM_CLASSES).cpu()


def miou_from_cm(cm):
    inter = cm.diag().float()
    union = (cm.sum(0) + cm.sum(1) - cm.diag()).float()
    present = cm.sum(1) > 0
    iou = inter / union.clamp(min=1)
    return iou[present].mean().item(), iou


def save_plot(path, img, tgt, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = img[[2, 1, 0]].float().cpu()          # B04, B03, B02 in BAND_PREFIXES order
    flat = rgb.flatten(1)
    lo = torch.quantile(flat, 0.02, dim=1)[:, None, None]
    hi = torch.quantile(flat, 0.98, dim=1)[:, None, None]
    rgb = ((rgb - lo) / (hi - lo + 1e-6)).clamp(0, 1).permute(1, 2, 0).numpy()

    tgt = tgt.cpu().numpy()
    pred = pred.cpu().numpy()
    err = (pred != tgt) & (tgt != IGNORE_INDEX)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[1].imshow(np.ma.masked_equal(tgt, IGNORE_INDEX), cmap="tab10",
                 vmin=0, vmax=9, interpolation="nearest")
    ax[1].set_title("target")
    ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax[2].set_title("prediction")
    ax[3].imshow(err, cmap="gray", interpolation="nearest")
    ax[3].set_title(f"errors ({err.mean():.1%} of pixels)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def real_data_eval(m, ds, device, batch_size, out_dir, n_plots):
    os.makedirs(out_dir, exist_ok=True)
    cm_real = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm_zero = torch.zeros_like(cm_real)
    plotted = 0

    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        x = torch.stack([it["image"]["s2"] for it in items]).to(device)
        t = torch.stack([it["target"] for it in items]).to(device)

        p = seeded(m, x).argmax(1)
        pz = seeded(m, torch.zeros_like(x)).argmax(1)
        update_cm(cm_real, p, t)
        update_cm(cm_zero, pz, t)

        for k, it in enumerate(items):
            if plotted >= n_plots:
                break
            name = it["metadata"]["sample_name"]
            save_plot(os.path.join(out_dir, f"{plotted:02d}_{name}.png"),
                      x[k], t[k], p[k])
            plotted += 1

    miou_real, iou_real = miou_from_cm(cm_real)
    miou_zero, _ = miou_from_cm(cm_zero)

    gt_counts = cm_real.sum(1).float()
    n_present = int((gt_counts > 0).sum())
    majority = (gt_counts.max() / gt_counts.sum()).item() / max(n_present, 1)

    print(f"       test mIoU, real input:   {miou_real:.4f}")
    print(f"       per-class IoU:           "
          f"{[round(v, 3) for v in iou_real.tolist()]}")
    print(f"       test mIoU, zeroed input: {miou_zero:.4f}")
    print(f"       majority-class mIoU:     {majority:.4f}")
    print(f"       plots saved to:          {out_dir}")

    report("5 leakage (zeroed input)", miou_zero <= majority + 0.05,
           f"zeroed-input mIoU {miou_zero:.4f} vs. majority-class level "
           f"{majority:.4f} (should be close; far above means labels leak in)")


# =============================================================================
# main
# =============================================================================

def load_ckpt(m, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    sd = stripped or sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model built from the --ptv3_* args.\n"
            f"  missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"  unexpected: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )
    print(f"Loaded checkpoint: {path}")


def main():
    p = argparse.ArgumentParser(description="PTv3Segmenter ordering / leakage checks")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Cashew root. If omitted, checks 0-4 use random input "
                        "and check 5 is skipped.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--ckpt", type=str, default=None,
                   help="Lightning checkpoint. If omitted, random weights "
                        "(fine for ordering checks 0-4).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--size", type=int, default=256,
                   help="Tile size for random input (ignored with --data_dir).")
    p.add_argument("--n_pokes", type=int, default=10)
    p.add_argument("--poke", type=float, default=10.0,
                   help="Perturbation added to all bands of the poked pixel "
                        "(normalized units).")
    p.add_argument("--n_plots", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./ptv3_checks")
    p.add_argument("--amp", action="store_true",
                   help="Run under bf16 autocast, as in training (bf16-mixed).")

    p.add_argument("--ptv3_patch", type=int, default=1024)
    p.add_argument("--ptv3_enc_channels", type=int, nargs=5, default=None)
    p.add_argument("--ptv3_enc_depths", type=int, nargs=5, default=None)
    p.add_argument("--ptv3_dec_channels", type=int, nargs=4, default=None)
    p.add_argument("--ptv3_dec_depths", type=int, nargs=4, default=None)
    p.add_argument("--ptv3_head_dim", type=int, default=16)
    p.add_argument("--ptv3_drop_path", type=float, default=0.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No CUDA device; PTv3's Triton kernels may not run on CPU.")

    kw = {}
    if args.ptv3_enc_channels is not None:
        kw["enc_channels"] = tuple(args.ptv3_enc_channels)
    if args.ptv3_enc_depths is not None:
        kw["enc_depths"] = tuple(args.ptv3_enc_depths)
    if args.ptv3_dec_channels is not None:
        kw["dec_channels"] = tuple(args.ptv3_dec_channels)
    if args.ptv3_dec_depths is not None:
        kw["dec_depths"] = tuple(args.ptv3_dec_depths)

    m = PTv3Segmenter(
        in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES, gsd=GSD,
        patch_size=args.ptv3_patch, head_dim=args.ptv3_head_dim,
        drop_path=args.ptv3_drop_path, **kw,
    )
    if args.ckpt:
        load_ckpt(m, args.ckpt)
    m = m.to(device).eval()

    ds = None
    if args.data_dir:
        from training.utils.datasets_baselines.utils_dataset_cashew_baselines import (
            CashewBaselineDataset,
        )
        ds = CashewBaselineDataset(root_path=args.data_dir, mode=args.split,
                                   crop_size=None, augment=False)
        n = min(args.batch_size, len(ds))
        x = torch.stack([ds[i]["image"]["s2"] for i in range(n)]).to(device)
        print(f"Using {n} real '{args.split}' tiles, shape {tuple(x.shape)}")
    else:
        x = torch.randn(args.batch_size, NUM_CHANNELS, args.size, args.size,
                        device=device)
        print(f"Using random input, shape {tuple(x.shape)}")

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.amp and device == "cuda" else contextlib.nullcontext())

    print()
    with torch.inference_mode(), amp:
        check_determinism(m, x)
        check_metadata(m, x)
        check_permutation(m, x)
        check_poke(m, x, args.n_pokes, args.poke)
        check_batch_independence(m, x)

        if args.ckpt and ds is not None:
            real_data_eval(m, ds, device, args.batch_size, args.out_dir, args.n_plots)
        else:
            print("[SKIP] 5 leakage: needs both --ckpt and --data_dir")

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
