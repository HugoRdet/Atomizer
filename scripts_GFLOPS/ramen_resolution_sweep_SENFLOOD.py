"""
RAMEN Resolution Sweep — Sen1Floods11
========================================

Evaluates a SINGLE trained RAMEN checkpoint across a range of working
resolutions (`res`), measuring GFLOPs/forward and test mIoU at each.
All modalities used, no modality dropping.

Why one checkpoint works across every resolution: RAMENBackbone's
pos_embed is a non-persistent buffer, recomputed at construction time
from `effective_size = input_size * (input_res/res)` — it is NEVER
saved into the checkpoint's state_dict. Every learnable weight
(SpectralProjector, RadarProjector, ScaleResampler, ViT blocks) is
resolution-agnostic. So rebuilding the model at a different `res` and
loading the SAME checkpoint's weights is always shape-safe — no
retraining needed to sweep resolution at eval time.

Sliding-window mechanics are unchanged across the sweep: --ramen_window_size
(pixel-space crop of the *input* image) stays fixed, so the number of
windows tiling the full 512x512 image stays fixed too. Only the number
of TOKENS per window changes with `res` (coarser res -> fewer tokens ->
cheaper forward, smaller effective receptive detail).

Usage
-----
    python script_ramen_resolution_sweep_senflood.py \
        --ckpt ./checkpoints/senflood_baselines/bl_ramen_s2s1_ramen-last.ckpt \
        --xp_name ramen_res_sweep \
        --ramen_window_size 128 --ramen_stride 96

    # Custom resolution list
    python script_ramen_resolution_sweep_senflood.py \
        --ckpt ./checkpoints/.../ramen-last.ckpt \
        --xp_name ramen_res_sweep_coarse \
        --resolutions 10 50 100 150 200
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.sliding_window import sliding_window_inference  # adjust import path
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = Sen1Floods11BaselineDataset.NUM_CLASSES   # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX  # 255
NUM_S2_BANDS = Sen1Floods11BaselineDataset.NUM_S2_BANDS  # 13
NUM_S1_BANDS = Sen1Floods11BaselineDataset.NUM_S1_BANDS  # 2
MODALITY_KEY = "s2s1"

ALL_S2 = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
ALL_S1 = ["VV","VH"]

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B08A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}
S1_POLARIZATIONS = {"VV": "asc_vv", "VH": "asc_vh"}

RAMEN_INPUT_BANDS = {"optical": ALL_S2, "sar": ALL_S1}
RAMEN_WAVELENGTHS = {"optical": S2_WAVELENGTHS_NM, "sar": S1_POLARIZATIONS}


# =============================================================================
# INPUT ADAPTER — splits the merged 's2s1' tensor into {'optical','sar'}
# =============================================================================

class RAMENInputAdapter(nn.Module):
    """
    Splits the dataset's merged image["s2s1"] : [B,15,H,W] tensor into
    RAMEN's expected {"optical": [B,13,H,W], "sar": [B,2,H,W]} — no
    modality dropping (this script always uses every band).

    Composes correctly with sliding_window_inference the same way as
    RAMENChannelDropWrapper in the modality-drop script: it crops the
    merged tensor generically per-window, and this adapter splits each
    window crop right before the RAMENUPerNet forward call.
    """
    expects_full_image_dict = True

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: dict, **kwargs):
        merged = x[MODALITY_KEY]
        optical = merged[:, :NUM_S2_BANDS]
        sar = merged[:, NUM_S2_BANDS: NUM_S2_BANDS + NUM_S1_BANDS]
        return self.model({"optical": optical, "sar": sar}, **kwargs)


# =============================================================================
# COLLATE
# =============================================================================

def senflood_collate(batch):
    images  = {k: torch.stack([s["image"][k] for s in batch])
               for k in batch[0]["image"]}
    targets  = torch.stack([s["target"]   for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# FLOPs MEASUREMENT — same harness as the other baseline scripts
# =============================================================================

def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b


@torch.no_grad()
def measure_gflops_forward(forward_fn, batches, device, n_warmup=1):
    """
    One warmup pass discarded; each measured pass profiled separately
    with with_flops=True; per-pass total summed over key_averages();
    report mean / 1e9. forward_fn internally loops over sliding-window
    tiles, so this captures the TOTAL cost of one full-image forward.
    """
    for b in batches[:n_warmup]:
        _ = forward_fn(b)
    if device == "cuda":
        torch.cuda.synchronize()

    flops_list = []
    for b in batches[n_warmup:]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True, record_shapes=True, profile_memory=True) as prof:
            _ = forward_fn(b)
            if device == "cuda":
                torch.cuda.synchronize()
        total = sum(evt.flops for evt in prof.key_averages()
                    if getattr(evt, "flops", None))
        flops_list.append(total)

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="RAMEN resolution sweep on Sen1Floods11")
parser.add_argument("--ckpt", type=str, required=True,
                    help="RAMEN checkpoint to evaluate at every resolution.")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                    help="Explicit list of `res` values (m/px) to sweep. "
                         "Default: 10, 20, ..., 200 (step 10, 20 values).")

parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of profiled forward passes per resolution (mean).")

# RAMEN architecture args — fixed across the sweep, must match the
# checkpoint's training config EXCEPT `res` itself, which is swept.
parser.add_argument("--ramen_embed_dim",   type=int, default=384)
parser.add_argument("--ramen_depth",       type=int, default=12)
parser.add_argument("--ramen_num_heads",   type=int, default=8)
parser.add_argument("--ramen_input_res",   type=float, default=10.0)
parser.add_argument("--ramen_window_size", type=int, default=128,
                    help="MUST match the checkpoint's training window size.")
parser.add_argument("--ramen_stride",      type=int, default=96,
                    help="Sliding-window stride for full 512x512 eval.")
parser.add_argument("--ramen_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--ramen_decoder_channels", type=int, default=256)

parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional YAML setting window_size/embed_dim/etc "
                         "(same format as script_train_senflood_baseline.py). "
                         "A `res` key in this file is IGNORED — resolution "
                         "is controlled by --resolutions, not the config, "
                         "since this script's entire point is to sweep it.")

args = parser.parse_args()

if args.resolutions is None:
    args.resolutions = [float(r) for r in range(10, 201, 10)]  # 10..200 step 10

# ── apply RAMEN config (everything except `res`) ───────────────────────────
if args.ramen_config is not None:
    key_map = {
        "input_size": "ramen_window_size", "input_res": "ramen_input_res",
        "embed_dim": "ramen_embed_dim", "depth": "ramen_depth",
        "num_heads": "ramen_num_heads", "stride": "ramen_stride",
    }
    import sys
    explicit = {
        tok[2:].split("=")[0].replace("-", "_")
        for tok in sys.argv[1:] if tok.startswith("--")
    }
    with open(args.ramen_config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, val in cfg.items():
        if key == "res":
            print(f"[INFO] Ignoring 'res' in {args.ramen_config} — this "
                  f"script sweeps resolution via --resolutions instead.")
            continue
        dest = key_map.get(key)
        if dest is None:
            print(f"[WARNING] Unrecognized key '{key}' in {args.ramen_config} "
                  f"— ignoring. Known keys: {sorted(key_map)} (+ 'res', ignored here).")
            continue
        if dest in explicit:
            print(f"[INFO] '{key}' in {args.ramen_config} ignored — "
                  f"--{dest} was explicitly set on the command line.")
            continue
        print(f"[INFO] {args.ramen_config}: {key}={val} -> --{dest}")
        setattr(args, dest, val)

if args.ramen_stride > args.ramen_window_size:
    print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
          f"--ramen_window_size ({args.ramen_window_size}); clamping to "
          f"window_size (non-overlapping tiling).")
    args.ramen_stride = args.ramen_window_size


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  RAMEN Resolution Sweep — Sen1Floods11")
print(f"  Checkpoint:   {args.ckpt}")
print(f"  Window size:  {args.ramen_window_size}x{args.ramen_window_size}")
print(f"  Stride:       {args.ramen_stride}")
print(f"  Resolutions:  {args.resolutions}")
print(f"{'='*60}\n")


# =============================================================================
# TEST DATASET (built once, reused across every resolution)
# =============================================================================

test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers,
    collate_fn=senflood_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)
print(f"[Eval] Test set: {len(test_ds)} samples\n")

# Fixed batch pool for FLOPs profiling — same input geometry reused
# across every resolution so the comparison is apples-to-apples.
device = "cuda" if torch.cuda.is_available() else "cpu"
_flops_raw = []
for b in test_loader:
    _flops_raw.append(_to_device(b, device))
    if len(_flops_raw) >= args.flops_n + 1:  # +1 warmup
        break


# =============================================================================
# SWEEP
# =============================================================================

results = []  # list of dicts: {res, effective_size, tokens_per_modality, gflops, miou}

for res in args.resolutions:
    effective_size = int(args.ramen_window_size * (args.ramen_input_res / res))
    effective_size = max(effective_size, 1)
    tokens_per_modality = effective_size ** 2

    print(f"\n{'─'*60}")
    print(f"  res = {res:.0f} m/px  "
          f"(effective grid {effective_size}x{effective_size} = "
          f"{tokens_per_modality} tokens/modality)")
    print(f"{'─'*60}")

    # ── Build model at this resolution ──────────────────────────────────
    model = build_ramen_upernet(
        input_bands=RAMEN_INPUT_BANDS,
        wavelengths=RAMEN_WAVELENGTHS,
        num_classes=NUM_CLASSES,
        input_size=args.ramen_window_size,
        embed_dim=args.ramen_embed_dim,
        depth=args.ramen_depth,
        num_heads=args.ramen_num_heads,
        input_res=args.ramen_input_res,
        res=res,
        output_layers=tuple(args.ramen_output_layers),
        decoder_channels=args.ramen_decoder_channels,
    )

    # ── Load the SAME checkpoint's weights — safe at any res, see module
    #    docstring: pos_embed is a non-persistent buffer, never saved ───
    trainer_module = BaselineTrainer.load_from_checkpoint(
        args.ckpt,
        strict=True,
        model=model,
        modality="optical+sar",
        temporal=False,
        task="senflood",
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        window_size=args.ramen_window_size,
        window_stride=args.ramen_stride,
    )
    trainer_module.model = RAMENInputAdapter(model)
    trainer_module.eval()

    # ── FLOPs (full sliding-window pass over the 512x512 image) ─────────
    adapter = trainer_module.model.to(device).eval()

    def fwd(b, m=adapter):
        return sliding_window_inference(
            m, b["image"],
            window_size=args.ramen_window_size,
            stride=args.ramen_stride,
            num_classes=NUM_CLASSES,
        )

    gflops = measure_gflops_forward(fwd, _flops_raw, device, n_warmup=1)
    print(f"  GFLOPs/forward (bs=1, full 512x512, sliding-window): {gflops:.2f}")

    # ── mIoU (full test split, all modalities, no dropping) ─────────────
    pl_trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        precision="bf16-mixed",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    test_results = pl_trainer.test(trainer_module, test_loader, verbose=False)
    metrics = test_results[0] if test_results else {}
    miou = metrics.get("test_mIoU", float("nan"))
    print(f"  test mIoU: {miou:.4f}")

    results.append({
        "res": res,
        "effective_size": effective_size,
        "tokens_per_modality": tokens_per_modality,
        "gflops": gflops,
        "miou": miou,
    })

    del model, trainer_module, adapter
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*70}")
print(f"  RAMEN RESOLUTION SWEEP SUMMARY — {args.xp_name}")
print(f"{'='*70}")
print(f"  {'Resolution':<14}{'Tokens/mod':<14}{'GFLOPs':<12}{'mIoU':<10}")
print(f"  {'─'*50}")
for r in results:
    print(f"  {str(int(r['res']))+' m':<14}"
          f"{r['tokens_per_modality']:<14}"
          f"{r['gflops']:<12.2f}"
          f"{r['miou']:<10.4f}")
print(f"{'='*70}\n")

# Also print in the exact "resolution XX: GFLOPS: YY mIoU: ZZ" phrasing
print("Compact summary:")
for r in results:
    print(f"resolution {int(r['res'])}: GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}")


# =============================================================================
# WRITE RESULTS
# =============================================================================

out_path = f"./results_{args.xp_name}_resolution_sweep.txt"
with open(out_path, "w") as f:
    f.write(f"RAMEN Resolution Sweep — {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n")
    f.write(f"Window size: {args.ramen_window_size}, stride: {args.ramen_stride}\n\n")
    f.write(f"{'Resolution':<14}{'Tokens/mod':<14}{'GFLOPs':<12}{'mIoU':<10}\n")
    f.write(f"{'─'*50}\n")
    for r in results:
        f.write(f"{str(int(r['res']))+' m':<14}"
                f"{r['tokens_per_modality']:<14}"
                f"{r['gflops']:<12.2f}"
                f"{r['miou']:<10.4f}\n")
    f.write("\nCompact summary:\n")
    for r in results:
        f.write(f"resolution {int(r['res'])}: GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}\n")

print(f"\n[Sweep] Results saved to {out_path}")
