"""
BurnScars (SKIP) — Test ONE checkpoint under a given (tpl, cross_k, decoder_k_spatial)
=======================================================================================

Inference-time density generalization for the SKIP model. Loads a single
trained Atomiser_Senflood_Skip checkpoint (shared encoder class, used for
BurnScars here) and evaluates on the BurnScars val OR test split at an
arbitrary latent density (tokens_per_latent), cross-attention budget
(cross_k), and decoder spatial-sampler budget (decoder_k_spatial).
No training.

# >>> PORTED_FROM_SENFLOOD: this is a rename+feature-port of
# script_test_senflood_skip_density.py onto BurnScars. All the same
# machinery applies here (same tags kept for traceability):
#   # >>> VAL_SPLIT        --split {val,test}, explicit dataloaders
#   # >>> LIGHT_GFLOPS      --gflops_scope {full,light,skip}, light = a
#                           small sampled forward-pass budget instead of
#                           the entire split (GFLOPs is ~content-
#                           independent for a fixed (tpl, cross_k, dks),
#                           unlike mIoU)
#   # >>> MULTI_GPU_GFLOPS  GFLOPs measurement is sharded round-robin
#                           across DDP ranks and combined via all_reduce,
#                           instead of every rank redundantly profiling
#                           the whole (shard of the) split
#   # >>> GFLOPS_METHOD     torch.utils.flop_counter.FlopCounterMode
#                           (SDPA-aware) instead of torch.profiler's
#                           with_flops=True
#   # >>> RESULTS_TRACKING  per-run JSON to ./scripts_GFLOPS/tmp_results/
#                           for resume support
#
# >>> DROPPED vs the original BurnScars script: the torch.profiler-based
# per-op-type table, Chrome trace export, and REGION_LABELS/record_function
# interval-matching breakdown. Those relied on torch.profiler's
# with_flops=True + per-event timing, which is a different measurement
# path than FlopCounterMode and isn't compatible with the no_grad
# module-tracker patch FlopCounterMode needs for per-rank sharded
# measurement under torch.no_grad(). If you need the region-level
# breakdown again, it would have to be re-added as a SEPARATE one-off
# profiling pass (not folded into the sharded GFLOPs loop), since mixing
# the two FLOPs-counting methodologies in one number is exactly what the
# GFLOPS_METHOD note in the Sen1Floods11 script warns against.

IMPORTANT — this uses the SKIP stack:
    Model_BurnScars_Skip + BurnScarsDataset + collate_grouped_skip
A previous non-skip version would silently drop the pixel_query /
pixel_cross_attn weights (strict=False), disabling the skip cascade and
lowering the score.

Density override:
    TEST reads val_sampling (sample_config(training=False) -> val_sampling).
    We set BOTH train_sampling and val_sampling to [[tpl, ck]] and pass the
    patched config to load_from_checkpoint so it overrides the checkpoint's
    saved hyperparameters. To reproduce the trained test score, use the
    config's val_sampling value from config_test-BURNSCARS.yaml.

# >>> DECODER_K: decoder_k_spatial lives at config_model["Atomiser"]["decoder_k_spatial"]
# and is likewise overridden before load_from_checkpoint so it takes effect
# regardless of what was saved in the checkpoint's hparams.

Emits a single parseable line:
    RESULT split=<val|test> tpl=<T> cross_k=<K> decoder_k_spatial=<DKS> mIoU=<V> accuracy=<A> gflops=<G> gflops_n=<N>

# >>> RESULTS_TRACKING: additionally writes a small JSON file to
# ./scripts_GFLOPS/tmp_results/ once the run completes, recording the
# checkpoint used, the (split, gflops_scope, tpl, cross_k, decoder_k_spatial)
# config, the scores, and whether config["Atomiser"]["use_quadtree_decode"] /
# config["Atomiser"]["use_adaptive_decode"] were set. The driver script
# checks for this file before launching a run, so an interrupted sweep can
# be resumed without re-scoring configs that already finished.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json                     # >>> RESULTS_TRACKING
import re                       # >>> RESULTS_TRACKING
import hashlib                  # >>> RESULTS_TRACKING
from datetime import datetime   # >>> RESULTS_TRACKING
import torch
from torch.utils.flop_counter import FlopCounterMode   # >>> GFLOPS_METHOD
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

# >>> RENAME: SKIP stack — BurnScars trainer + dataset, shared encoder class
from training.trainer_BURNSCARS import Model_BurnScars_Skip
from training.utils.datasets.utils_dataset_BURNSCARS import BurnScarsDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip


# =============================================================================
# >>> RESULTS_TRACKING: per-run result file (checkpointing the sweep itself)
# =============================================================================
TMP_RESULTS_DIR = "./scripts_GFLOPS/tmp_results"
VALID_TYPES = ("regular", "quadtree", "zoneprobe")


def _ckpt_tag(ckpt_path):
    """Short, filesystem-safe, collision-resistant tag for a checkpoint path."""
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    h = hashlib.md5(os.path.abspath(ckpt_path).encode()).hexdigest()[:8]
    return f"{safe}_{h}"


def _decode_mode(use_quadtree_decode, use_adaptive_decode):
    """Derived only for a sanity cross-check against --type; not the source
    of truth for filenames (that's --type, since callers dedicate one
    script/config per method)."""
    if use_quadtree_decode:
        return "quadtree"
    if use_adaptive_decode:
        return "zoneprobe"
    return "regular"


# >>> VAL_SPLIT / LIGHT_GFLOPS: split AND gflops_scope folded into the
# tmp_results filename so a val-split score, a light-GFLOPs score, and a
# full-GFLOPs score for the SAME (ckpt, tpl, ck, dks, type) never
# collide/overwrite each other in the resume cache.
def tmp_result_path(ckpt_path, tpl, ck, dks, decode_type, split="test",
                    gflops_scope="full", task="burnscars"):
    return os.path.join(
        TMP_RESULTS_DIR,
        f"{task}_{_ckpt_tag(ckpt_path)}_{decode_type}_{split}_{gflops_scope}_"
        f"tpl{tpl}_ck{ck}_dks{dks}.json",
    )


# =============================================================================
# >>> GFLOPS_METHOD: FlopCounterMode measurement — module-tracker no_grad
# patch + measurement helper (see script_test_senflood_skip_density.py for
# the full rationale, identical here).
# =============================================================================

def _patch_module_tracker_for_no_grad():
    """Idempotently patches torch.utils.module_tracker so its forward-pre
    hook's register_multi_grad_hook call no longer raises under
    torch.no_grad() — needed for FlopCounterMode's TOTAL to remain
    reliable even though per-module attribution becomes unreliable."""
    import torch.utils.module_tracker as _mt

    if getattr(_mt, "_flopcounter_noop_patch_applied", False):
        return
    _mt._flopcounter_noop_patch_applied = True

    _orig_register_multi_grad_hook = _mt.register_multi_grad_hook

    class _NoOpHandle:
        def remove(self):
            pass

    def _safe_register_multi_grad_hook(tensors, fn, *args, **kwargs):
        try:
            return _orig_register_multi_grad_hook(tensors, fn, *args, **kwargs)
        except AssertionError:
            return _NoOpHandle()

    _mt.register_multi_grad_hook = _safe_register_multi_grad_hook


@torch.no_grad()
def measure_gflops_forward_shard(forward_fn, loader, device, n_warmup=1):
    """
    # >>> MULTI_GPU_GFLOPS (sharded)
    Profiles every batch in `loader` (already a per-rank SHARD — see
    _shard_dataset_round_robin) with FlopCounterMode, first `n_warmup`
    batches discarded unprofiled. Returns a local SUM (not mean) + local
    count so the caller can all-reduce sums/counts across ranks and only
    then divide — correct even when shards are uneven sizes.

    Returns (sum_gflops, n_measured, fc_last) — local to this process.
    """
    flops_list = []
    fc_last = None
    n_measured = 0

    for i, raw_b in enumerate(loader):
        b = _to_device(raw_b, device)

        if i < n_warmup:
            out = forward_fn(b)
            del out
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            continue

        fc = FlopCounterMode(display=False)
        with fc:
            out = forward_fn(b)
        flops_list.append(fc.get_total_flops())
        fc_last = fc
        n_measured += 1
        del out
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if n_measured == 0:
        return 0.0, 0, None

    if all(f == 0 for f in flops_list):
        print("[measure_gflops_forward_shard] WARNING: all measured "
              "passes on this rank returned exactly 0 FLOPs. This suggests "
              "the assumption behind the no_grad patch (FlopCounterMode's "
              "'Global' bucket still accumulates totals even when its "
              "per-module attribution hook is patched to a no-op) does not "
              "hold for the installed torch version. Treat this GFLOPs "
              "number as UNRELIABLE -- please report the torch version so "
              "the patch can be adjusted.")

    return (sum(flops_list) / 1e9), n_measured, fc_last


def _shard_dataset_round_robin(dataset, rank, world_size, limit_n=None):
    """
    # >>> MULTI_GPU_GFLOPS: round-robin sharding (index i -> rank i % world_size)
    rather than torch's DistributedSampler, which pads the tail by
    REPEATING samples (would double-count in a FLOPs SUM). Round-robin
    gives shard sizes differing by at most 1, combined correctly via
    sum(flops)/sum(count) regardless of unevenness.

    # >>> LIGHT_GFLOPS: `limit_n`, if given, restricts sharding to the
    first `limit_n` GLOBAL indices before round-robin split across ranks.
    """
    n = len(dataset) if limit_n is None else min(limit_n, len(dataset))
    indices = list(range(rank, n, world_size))
    return torch.utils.data.Subset(dataset, indices)


def _allreduce_sum(value, device):
    """Sums a Python scalar across all DDP ranks (no-op if not distributed)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t.item()


def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Test one BurnScars SKIP ckpt at a given density")
parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
parser.add_argument("--xp_name", type=str, default="density_eval")
parser.add_argument("--tokens_per_latent", type=int, required=True)
parser.add_argument("--cross_k", type=int, required=True)
parser.add_argument("--decoder_k_spatial", type=int, required=True)
parser.add_argument("--type", type=str, default="regular",
                    choices=VALID_TYPES,
                    help="Decode method label: regular | quadtree | zoneprobe. "
                         "Used to keep per-run result files from colliding "
                         "across methods run at the same (tpl, cross_k, dks).")
# >>> RENAME: BurnScars config/data defaults
parser.add_argument("--config", type=str,
                    default="./training/configs/config_test-BURNSCARS.yaml")
parser.add_argument("--configs_dataset", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--bands_yaml", type=str,
                    default="./data/bands_info/bands.yaml")
parser.add_argument("--data_dir", type=str, default="./data/hls_burn_scars")
parser.add_argument("--num_workers", type=int, default=4)
# >>> LIGHT_GFLOPS
parser.add_argument("--gflops_scope", type=str, default="full",
                    choices=("full", "light"),
                    help="'full' (default): stream the ENTIRE (sharded) "
                         "split for a precise mean — use for final "
                         "reported numbers. 'light': profile only "
                         "--gflops_light_n samples total (split round-"
                         "robin across ranks like the full path) — use "
                         "for cheap per-config GFLOPs across a wide sweep.")
parser.add_argument("--gflops_light_n", type=int, default=30,
                    help="Total number of samples to profile (summed "
                         "across ALL ranks) when --gflops_scope light. "
                         "Ignored when --gflops_scope full.")
parser.add_argument("--flops_n", type=int, default=1,
                    help="Number of leading batches of THIS RANK's shard "
                         "treated as unprofiled warmup before GFLOPs "
                         "measurement begins.")
parser.add_argument("--skip_gflops", action="store_true",
                    help="Skip GFLOPs measurement entirely (mIoU/accuracy "
                         "only). Use this with --split val for a fast "
                         "scoring pass over a wide config grid.")
# >>> VAL_SPLIT
parser.add_argument("--split", type=str, default="test", choices=("val", "test"),
                    help="Which split to score on. 'val' is intended for "
                         "cheap model/config SELECTION (combine with "
                         "--gflops_scope light); 'test' (default) is for "
                         "final reported numbers on the shortlisted "
                         "configs (combine with --gflops_scope full).")
args = parser.parse_args()

tpl, ck = int(args.tokens_per_latent), int(args.cross_k)
dks = int(args.decoder_k_spatial)  # >>> DECODER_K
decode_type = args.type

# =============================================================================
# CONFIG + DENSITY OVERRIDE  (before model build)
# =============================================================================
config_model = read_yaml(args.config)
config_model.setdefault("latent_grid", {})
config_model["latent_grid"]["train_sampling"] = [[tpl, ck]]
config_model["latent_grid"]["val_sampling"]   = [[tpl, ck]]
print(f"[Test] Density override -> [[{tpl}, {ck}]] "
      f"(tokens_per_latent={tpl}, cross_k={ck})")

# >>> DECODER_K: override decoder_k_spatial so it takes effect regardless of
# what was saved in the checkpoint's hparams.
config_model.setdefault("Atomiser", {})
config_model["Atomiser"]["decoder_k_spatial"] = dks
print(f"[Test] Decoder override -> decoder_k_spatial={dks}")

# sanity: this worker targets the skip model
if not config_model.get("Atomiser", {}).get("use_decoder_skip", False):
    print("[Test][WARN] config has use_decoder_skip=False but this is the SKIP "
          "worker. Ensure the config matches the skip checkpoint.")

_uqd = bool(config_model.get("Atomiser", {}).get("use_quadtree_decode", False))
_uad = bool(config_model.get("Atomiser", {}).get("use_adaptive_decode", False))
_inferred_type = _decode_mode(_uqd, _uad)
if _inferred_type != decode_type:
    print(f"[Test][WARN] --type={decode_type} but config implies "
          f"'{_inferred_type}' (use_quadtree_decode={_uqd}, "
          f"use_adaptive_decode={_uad}). Double-check --type and --config "
          f"match the intended method.")

lookup_table = Lookup_encoding(
    read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config_model)

# =============================================================================
# LOAD CHECKPOINT WITH PATCHED CONFIG  (SKIP model)
# =============================================================================
model = Model_BurnScars_Skip.load_from_checkpoint(
    args.ckpt,
    strict=False,
    config=config_model,
    wand=False,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)
model.eval()

# =============================================================================
# DATA MODULE  (SKIP dataset + collate)
# =============================================================================
data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=args.num_workers,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(args.bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=BurnScarsDataset,
    collate_fn=collate_grouped_skip,
)

# =============================================================================
# SCORE  (>>> VAL_SPLIT: trainer.validate() on 'val', trainer.test() on 'test')
# =============================================================================
trainer = Trainer(
    devices=-1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=False,
    enable_progress_bar=True,
    enable_model_summary=False,
)

split = args.split
# >>> EXPLICIT_DATALOADER: pass the dataloader directly via `dataloaders=`
# rather than relying on trainer.validate()/trainer.test() internal
# datamodule dispatch — unambiguous for a script whose whole point is
# getting the val/test split boundary right.
data_module.setup("validate" if split == "val" else "test")
if split == "val":
    results = trainer.validate(
        model=model, dataloaders=data_module.val_dataloader(), verbose=True)
else:
    results = trainer.test(
        model=model, dataloaders=data_module.test_dataloader(), verbose=True)
metrics = results[0] if results else {}

# >>> VAL_SPLIT: try a few common logged-metric-name candidates per split
# rather than hardcoding "test_mIoU"/"test_accuracy".
_miou_candidates = [f"{split}_mIoU", "test_mIoU", "val_mIoU", "mIoU"]
_acc_candidates  = [f"{split}_accuracy", "test_accuracy", "val_accuracy", "accuracy"]

def _first_present(d, candidates):
    for k in candidates:
        if k in d:
            return d[k], k
    return float("nan"), None

miou, _miou_key = _first_present(metrics, _miou_candidates)
acc,  _acc_key  = _first_present(metrics, _acc_candidates)

if _miou_key is None or _acc_key is None:
    print(f"[Test][WARN] split='{split}': could not find expected metric "
          f"keys among {list(metrics.keys())}. Tried mIoU candidates="
          f"{_miou_candidates}, accuracy candidates={_acc_candidates}. "
          f"Reporting NaN for whichever key was not found — check your "
          f"validation_step/test_step logging names.")
else:
    print(f"[Test] split='{split}': using metric keys "
          f"mIoU<-'{_miou_key}' accuracy<-'{_acc_key}'")


# =============================================================================
# GFLOPS MEASUREMENT (FlopCounterMode, sharded across all DDP ranks,
# scope full or light — >>> MULTI_GPU_GFLOPS / LIGHT_GFLOPS)
# =============================================================================
gflops = float("nan")
gflops_n = 0

is_main = trainer.is_global_zero
world_size = trainer.world_size
rank = trainer.global_rank
device = trainer.strategy.root_device

PROFILE_DIR = "./profiler_burnscars"
tag = f"{split}_tpl{tpl}_ck{ck}_dks{dks}_{args.gflops_scope}"
out_dir = os.path.join(PROFILE_DIR, tag)
if is_main:
    os.makedirs(out_dir, exist_ok=True)
    scope_desc = (f"light, {args.gflops_light_n} samples total"
                  if args.gflops_scope == "light" else "full split")
    print(f"[GFLOPs] Saving artifacts to {out_dir}/ "
          f"(world_size={world_size} rank(s), scope={scope_desc})")

if args.skip_gflops:
    if is_main:
        print("[GFLOPs] Skipped (--skip_gflops)")
else:
    local_sum_gflops, local_n = 0.0, 0
    fc_last = None
    try:
        _patch_module_tracker_for_no_grad()

        model = model.to(device)
        model.eval()

        # >>> MULTI_GPU_GFLOPS + VAL_SPLIT: shard the SAME split used for
        # scoring, round-robin across ranks, BEFORE building the DataLoader.
        data_module.setup(split if split == "val" else "test")
        full_test_loader = (data_module.val_dataloader() if split == "val"
                            else data_module.test_dataloader())
        full_dataset = full_test_loader.dataset
        n_total = len(full_dataset)

        limit_n = args.gflops_light_n if args.gflops_scope == "light" else None
        n_warmup = args.flops_n
        if args.gflops_scope == "light":
            n_warmup = min(n_warmup, max(0, (limit_n // max(world_size, 1)) - 1))

        shard_dataset = _shard_dataset_round_robin(
            full_dataset, rank, world_size, limit_n=limit_n)
        shard_loader = torch.utils.data.DataLoader(
            shard_dataset,
            batch_size=full_test_loader.batch_size or 1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=full_test_loader.collate_fn,
            pin_memory=True,
        )
        print(f"[GFLOPs][rank {rank}/{world_size}] scope={args.gflops_scope} "
              f"shard = {len(shard_dataset)}/{n_total} samples "
              f"({n_warmup} warmup batch(es) discarded on this rank).")

        def _fwd(b, m=model):
            return m(b, training=False)

        local_sum_gflops, local_n, fc_last = measure_gflops_forward_shard(
            _fwd, shard_loader, device, n_warmup=n_warmup,
        )
        print(f"[GFLOPs][rank {rank}] local sum={local_sum_gflops:.2f} GFLOPs "
              f"over {local_n} profiled samples.")

    except Exception as e:
        import traceback
        print(f"[GFLOPs][rank {rank}] GFLOPs measurement failed: {e}")
        if is_main:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "ERROR.txt"), "w") as f:
                f.write(traceback.format_exc())
        local_sum_gflops, local_n = 0.0, 0

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    global_sum_gflops = _allreduce_sum(local_sum_gflops, device)
    global_n = int(_allreduce_sum(local_n, device))

    if global_n > 0:
        gflops = global_sum_gflops / global_n
        gflops_n = global_n
    else:
        gflops = float("nan")
        gflops_n = 0

    if is_main and gflops_n > 0:
        print(f"[GFLOPs] GLOBAL mean GFLOPs/forward "
              f"(sharded over {world_size} rank(s), {gflops_n} samples "
              f"profiled total): {gflops:.3f}  "
              f"[lower bound; matmul/conv/attention ops only]")

        summary_path = os.path.join(out_dir, f"gflops_summary_{tag}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Config: tpl={tpl} cross_k={ck} decoder_k_spatial={dks}\n")
            f.write(f"Scope: {args.gflops_scope}"
                    + (f" ({args.gflops_light_n} samples budget)"
                       if args.gflops_scope == "light" else " (entire split)")
                    + "\n")
            f.write(f"Method: torch.utils.flop_counter.FlopCounterMode "
                    f"(SDPA attention counted)\n")
            f.write(f"Scope: sharded round-robin across {world_size} DDP "
                    f"rank(s) (warmup={args.flops_n} batches/rank discarded, "
                    f"{gflops_n} total samples profiled)\n")
            f.write(f"GFLOPs/forward (global mean of {gflops_n} passes): "
                    f"{gflops:.4f}\n")
            f.write(f"Measured under torch.no_grad() via the "
                    f"module-tracker no-op patch (see script comments).\n")
        print(f"[GFLOPs] summary -> {summary_path}")

        if fc_last is not None:
            try:
                import csv
                op_flops = fc_last.flop_counts.get("Global", {})
                rows = sorted(
                    ((str(op), int(fl)) for op, fl in op_flops.items()),
                    key=lambda r: r[1], reverse=True,
                )
                csv_path = os.path.join(out_dir, f"ops_flops_{tag}.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["op", "flops", "gflops"])
                    for op, fl in rows:
                        w.writerow([op, fl, fl / 1e9])
                print(f"[GFLOPs] per-op-type FLOPs CSV (rank 0's last "
                      f"profiled batch) -> {csv_path}")
                print(f"[GFLOPs] Top FLOP op types (GFLOPs, rank 0's last "
                      f"profiled batch) [{len(rows)} op types with nonzero flops]:")
                for op, fl in rows[:10]:
                    print(f"[GFLOPs]   {op[:50]:<50} {fl / 1e9:>10.2f}")
            except Exception as ee:
                print(f"[GFLOPs] per-op-type CSV export failed: {ee}")


# RESULT line: rank-0 only.
if is_main:
    print(f"RESULT split={split} tpl={tpl} cross_k={ck} decoder_k_spatial={dks} "
          f"mIoU={miou:.6f} accuracy={acc:.6f} gflops={gflops:.6f} "
          f"gflops_n={gflops_n}")

    # =========================================================================
    # >>> RESULTS_TRACKING: write per-run result file (rank-0 only)
    # =========================================================================
    os.makedirs(TMP_RESULTS_DIR, exist_ok=True)
    run_record = {
        "task": "burnscars",
        "split": split,
        "ckpt": os.path.abspath(args.ckpt),
        "xp_name": args.xp_name,
        "tokens_per_latent": tpl,
        "cross_k": ck,
        "decoder_k_spatial": dks,
        "mIoU": miou,
        "accuracy": acc,
        "gflops": gflops,
        "gflops_n": gflops_n,
        "gflops_method": None if args.skip_gflops else "FlopCounterMode",
        "gflops_scope": None if args.skip_gflops else args.gflops_scope,
        "gflops_light_n_budget": (args.gflops_light_n
                                  if args.gflops_scope == "light" else None),
        "gflops_world_size": world_size,
        "gflops_skipped": bool(args.skip_gflops),
        "type": decode_type,
        "use_quadtree_decode": _uqd,
        "use_adaptive_decode": _uad,
        "config_path": args.config,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_result_path = tmp_result_path(
        args.ckpt, tpl, ck, dks, decode_type, split=split,
        gflops_scope=("skipped" if args.skip_gflops else args.gflops_scope),
        task="burnscars")
    with open(out_result_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"[Test] Wrote per-run result -> {out_result_path} "
          f"(type={decode_type}, split={split})")
