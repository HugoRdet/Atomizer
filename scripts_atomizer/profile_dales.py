"""
profile_dales.py -- where does the training step spend its time?
================================================================

Location: scripts_atomizer/profile_dales.py. Run on ONE GPU (no srun with several tasks).

Measures, on real training batches (augmentation on, as in training):
    * data loading: time per batch from the DataLoader alone (workers included);
    * forward per stage (synchronised wall time): geom/columns, geom/voxel_hierarchy
      or the column stage, enc/..., unet2d, dec/..., geom/query_lookup, query_decoder;
    * forward total, backward (includes checkpoint recomputation), optimiser step;
    * peak GPU memory.

    python scripts_atomizer/profile_dales.py --config_model config_test-DALES-unet-hier.yaml \
        --data_root ./data/DALES --batch_size 8 --steps 10
    # compare variants in one go:
    python scripts_atomizer/profile_dales.py ... --no_checkpoint
    python scripts_atomizer/profile_dales.py ... --max_queries 20000
"""
import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.utils import read_yaml
import training.atomiser.Atomiser_unet_dales as AU
from training.trainer_DALES_unet import Model_DalesUnet
from training.utils.datasets.dataset_dales_points import DalesPointDataset, collate_dales

ap = argparse.ArgumentParser()
ap.add_argument("--config_model", required=True)
ap.add_argument("--data_root", default="./data/DALES")
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--num_workers", type=int, default=8)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--max_queries", type=int, default=100_000)
ap.add_argument("--max_context_points", type=int, default=256_000)
ap.add_argument("--no_checkpoint", action="store_true")
ap.add_argument("--loader_batches", type=int, default=20)
args = ap.parse_args()
torch.set_float32_matmul_precision("high")
dev = torch.device("cuda")

config = read_yaml(str(_REPO_ROOT / "training" / "configs" / args.config_model))
if args.no_checkpoint:
    config["Atomiser"]["unet_checkpoint"] = False
ds = DalesPointDataset(str(Path(args.data_root) / "train"), "train", max_queries=args.max_queries,
                       max_context_points=args.max_context_points,
                       load_partition=config.get("Atomiser", {}).get("unet_hier_nodes") == "ezsp_pre")
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                collate_fn=collate_dales, pin_memory=True, persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None)

# ---- 1. data loading alone ----
it = iter(dl); next(it)                                   # start the workers
t = time.perf_counter()
for _ in range(args.loader_batches):
    b = next(it)
load_ms = (time.perf_counter() - t) / args.loader_batches * 1000
pts_per_batch = int(b["n_points"].sum()); q_per_batch = int(b["n_queries"].sum())

# ---- 2. model step, per stage ----
lit = Model_DalesUnet(config, class_weighting="sqrt").to(dev).train()
opt = torch.optim.AdamW(lit.parameters(), lr=1e-4)
batches = [next(it) for _ in range(args.warmup + args.steps)]
fw, bw, op = [], [], []
torch.cuda.reset_peak_memory_stats()
for i, bt in enumerate(batches):
    bt = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}
    AU._TIMING["on"] = i >= args.warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = lit(bt)
        loss = lit.loss_fn(logits.reshape(-1, logits.shape[-1]).float(), bt["query_labels"].reshape(-1))
    torch.cuda.synchronize(); t1 = time.perf_counter()
    AU._TIMING["on"] = False
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(lit.parameters(), 1.0); opt.step()
    torch.cuda.synchronize(); t3 = time.perf_counter()
    if i >= args.warmup:
        fw.append(t1 - t0); bw.append(t2 - t1); op.append(t3 - t2)

S = args.steps
stages = {k: v / S * 1000 for k, v in AU._TIMING["acc"].items()}
fwd = float(np.mean(fw)) * 1000; bwd = float(np.mean(bw)) * 1000; opt_ms = float(np.mean(op)) * 1000
step = fwd + bwd + opt_ms
print(f"\nconfig {args.config_model} | batch {args.batch_size} | checkpointing {'off' if args.no_checkpoint else 'on'}"
      f" | ~{pts_per_batch:,} points, {q_per_batch:,} queries per batch")
print(f"stats: {getattr(lit.model, 'last_partition_stats', None)}")
print(f"\n{'stage (forward)':28s} {'ms':>9s} {'% fwd':>7s}")
for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
    print(f"{k:28s} {v:9.1f} {100 * v / fwd:6.1f}%")
other = fwd - sum(stages.values())
print(f"{'(unlabelled forward)':28s} {other:9.1f} {100 * other / fwd:6.1f}%")
print(f"\n{'forward':28s} {fwd:9.1f}\n{'backward (+recompute)':28s} {bwd:9.1f}\n{'optimiser':28s} {opt_ms:9.1f}")
print(f"{'GPU step total':28s} {step:9.1f}  -> {1000 / step:.2f} it/s")
print(f"{'data loading per batch':28s} {load_ms:9.1f}  ({args.num_workers} workers)"
      + ("  <-- LOADER-BOUND: the GPU waits for data" if load_ms > step else ""))
print(f"{'peak GPU memory':28s} {torch.cuda.max_memory_allocated() / 2 ** 30:9.1f} GB")
