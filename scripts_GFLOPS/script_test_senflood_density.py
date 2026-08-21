"""
Sen1Floods11 (SKIP) — Test ONE checkpoint under a given (tpl, cross_k, decoder_k_spatial)
==========================================================================================

Inference-time density generalization for the SKIP model. Loads a single
trained Atomiser_Senflood_Skip checkpoint and evaluates on the test split at
an arbitrary latent density (tokens_per_latent), cross-attention budget
(cross_k), and decoder spatial-sampler budget (decoder_k_spatial). No training.

IMPORTANT — this uses the SKIP stack:
    Model_SenFlood_Skip + Sen1Floods11SkipDataset + collate_grouped_skip
A previous non-skip version silently dropped the pixel_query / pixel_cross_attn
weights (strict=False), disabling the skip cascade and lowering the score.

Density override:
    TEST reads val_sampling (sample_config(training=False) -> val_sampling).
    We set BOTH train_sampling and val_sampling to [[tpl, ck]] and pass the
    patched config to load_from_checkpoint so it overrides the checkpoint's
    saved hyperparameters. To reproduce the trained test score, use the
    config's val_sampling value (here: tpl=2000, cross_k=1000).

# >>> DECODER_K: decoder_k_spatial lives at config_model["Atomiser"]["decoder_k_spatial"]
# and is likewise overridden before load_from_checkpoint so it takes effect
# regardless of what was saved in the checkpoint's hparams.

# >>> RESULTS_TRACKING: ported from script_test_burnscars_density.py — see
# that file's module docstring for the full rationale. In short: this
# worker writes a small per-run JSON to ./scripts_GFLOPS/tmp_results/ once
# it finishes, recording ckpt/config used, the (tpl, cross_k,
# decoder_k_spatial, --type) it was run with, the scores, and whether
# config["Atomiser"]["use_quadtree_decode"] / ["use_adaptive_decode"] were
# set. Driver/sweep scripts check for this file before launching a run, so
# an interrupted sweep can be resumed without re-scoring finished configs.

# >>> GFLOPS_METHOD: GFLOPs measurement now uses
# torch.utils.flop_counter.FlopCounterMode (SDPA attention counted) instead
# of torch.profiler's with_flops=True kernel-reported counting. This
# matches script_universat_sweep_senflood.py / script_universat_sweep_
# burnscars.py's methodology exactly, so Atomizer's GFLOPs numbers are now
# directly comparable to UniverSat's rather than being silently produced by
# a different counting tool -- see that script's module docstring: "FLOPs
# from FlopCounterMode must never be mixed with torch.profiler-harness
# numbers." Both methods remain LOWER BOUNDS relative to true FLOPs (neither
# counts elementwise/gather ops -- Voronoi assignment, Fourier/Gaussian
# positional encodings, SKIP gather -- only matmul/conv/attention), but at
# least the comparison across models is now apples-to-apples.
#
# Because FlopCounterMode requires autograd tracking to attribute FLOPs per
# module (see _patch_module_tracker_for_no_grad below for why that's
# normally incompatible with the low-memory torch.no_grad() inference path
# this script otherwise wants), we patch that attribution hook to fail
# silently and measure under no_grad() -- matching the memory profile of
# the actual (no_grad) test forward pass above, rather than paying for a
# full backward-ready graph. This trades away a reliable PER-MODULE/region
# breakdown (the patched hook may leave module-parent bookkeeping
# inconsistent) for a reliable TOTAL (FlopCounterMode's "Global" bucket
# accumulates every op's FLOPs regardless of whether per-module attribution
# succeeds) -- see the comment block above measure_gflops_forward.

# >>> FULL_TEST_SET_GFLOPS: GFLOPs/forward is now measured by streaming
# through EVERY sample in the test dataloader instead of a small pool
# capped by --flops_n. Rationale: this model's cost is genuinely
# data-dependent (Voronoi cell sizes, SKIP gather counts, and — at low
# density in particular — how many tokens actually fall in each latent's
# cell before the min(m, |V_l|) clamp all vary per sample), so a handful
# of batches is not guaranteed to be representative of the true mean
# cost over the test distribution. --flops_n is kept only to control how
# many LEADING batches are treated as warmup (discarded, not profiled);
# every batch after that is profiled and included in the mean. This
# multiplies the GFLOPs-measurement wall-clock time by roughly
# len(test_set)/old_flops_n, so expect this pass to take noticeably
# longer than before -- it now does real measurement work proportional
# to a full test-set epoch, on top of the mIoU/accuracy epoch already
# run by trainer.test().

# >>> VAL_SPLIT: added --split {val,test} (default: test, unchanged). This
# is for cheap model SELECTION: run --split val --skip_gflops to score a
# wide grid of (tpl, cross_k, decoder_k_spatial) configs on the validation
# set fast (no GFLOPs pass, no held-out test data touched), pick your
# Pareto-optimal configs from that, and only THEN re-run the shortlisted
# few with --split test (and GFLOPs enabled) for reported numbers. Metric
# keys are looked up with a few common candidate names per split (PL's
# validate()/test() logged-metric naming isn't assumed to be identical
# across your training script versions) rather than hardcoded, so this
# doesn't silently return NaN if your validation_step logs e.g. "val_mIoU"
# instead of "test_mIoU".

Emits a single parseable line:
    RESULT split=<val|test> tpl=<T> cross_k=<K> decoder_k_spatial=<DKS> mIoU=<V> accuracy=<A> gflops=<G> gflops_n=<N>
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

# >>> SKIP stack (must match the checkpoint's architecture)
from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
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
# collide/overwrite each other in the resume cache — e.g. a 'light' run
# won't be mistaken for an already-completed 'full' run.
def tmp_result_path(ckpt_path, tpl, ck, dks, decode_type, split="test",
                    gflops_scope="full", task="senflood"):
    return os.path.join(
        TMP_RESULTS_DIR,
        f"{task}_{_ckpt_tag(ckpt_path)}_{decode_type}_{split}_{gflops_scope}_"
        f"tpl{tpl}_ck{ck}_dks{dks}.json",
    )


# =============================================================================
# >>> GFLOPS_METHOD: FlopCounterMode measurement (matches the UniverSat
# sweep scripts' methodology) -- module-tracker no_grad patch + measurement
# helper, ported from script_train_xview.py.
# =============================================================================

def _patch_module_tracker_for_no_grad():
    """Idempotently patches torch.utils.module_tracker so its forward-pre
    hook's register_multi_grad_hook call no longer raises under
    torch.no_grad() (see the module-level comment above for why this is
    safe for FlopCounterMode's TOTAL, even though it makes any per-module/
    region breakdown unreliable)."""
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
    # >>> MULTI_GPU_GFLOPS (option 2: sharded)
    Same per-batch profiling as before (FlopCounterMode per batch, first
    `n_warmup` batches discarded unprofiled), but now called with a
    per-RANK SHARD of the test set rather than the whole thing — see
    `_shard_dataset_round_robin` below. This function itself is
    rank-agnostic: it just measures whatever loader it's given and
    returns a local SUM (not mean) + local count, so the caller can
    all-reduce sums/counts across ranks and only then divide, which is
    the correct way to combine per-rank means of possibly differently-
    sized shards.

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
    rather than torch's DistributedSampler. DistributedSampler pads the
    tail by REPEATING samples so every rank gets an equal-sized shard,
    which would double-count a few samples in a FLOPs SUM. Round-robin
    with no padding means shard sizes differ by at most 1 across ranks,
    but since we combine ranks via sum(flops)/sum(count) rather than
    averaging per-rank means, uneven shard sizes are handled correctly
    with zero duplication.

    # >>> LIGHT_GFLOPS: `limit_n`, if given, restricts sharding to the
    first `limit_n` GLOBAL indices (0..limit_n-1) before round-robin
    split across ranks — so e.g. limit_n=16 with world_size=4 gives each
    rank 4 samples total, rather than each rank getting its own 16.
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


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Test one SKIP ckpt at a given density")
parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
parser.add_argument("--xp_name", type=str, default="density_eval")
parser.add_argument("--tokens_per_latent", type=int, required=True)
parser.add_argument("--cross_k", type=int, required=True)
# >>> DECODER_K: new required arg, mirrors tokens_per_latent/cross_k
parser.add_argument("--decoder_k_spatial", type=int, required=True)
# >>> RESULTS_TRACKING: which method this worker run corresponds to. Source
# of truth for tmp_results filenames (checked for a sanity mismatch against
# the config's own flags below, but not overridden by it).
parser.add_argument("--type", type=str, default="regular",
                    choices=VALID_TYPES,
                    help="Decode method label: regular | quadtree | zoneprobe. "
                         "Used to keep per-run result files from colliding "
                         "across methods run at the same (tpl, cross_k, dks).")
parser.add_argument("--config", type=str,
                    default="./training/configs/config_test-SENFLOOD.yaml")
parser.add_argument("--configs_dataset", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--bands_yaml", type=str,
                    default="./data/bands_info/bands.yaml")
parser.add_argument("--data_dir", type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers", type=int, default=4)
# >>> FULL_TEST_SET_GFLOPS: --flops_n is repurposed as a WARMUP count
# (batches run but not profiled) rather than a cap on how many batches
# get profiled. GFLOPs is now always averaged over the rest of the full
# test set. Set --flops_n_skip to skip GFLOPs measurement entirely.
parser.add_argument("--flops_n", type=int, default=1,
                    help="Number of leading batches of THIS RANK's shard "
                         "treated as unprofiled warmup before GFLOPs "
                         "measurement begins. GFLOPs is then measured over "
                         "EVERY remaining batch in the rank's shard, and "
                         "summed/counted across ranks (see MULTI_GPU_GFLOPS) "
                         "to give a full-test-set mean.")
parser.add_argument("--skip_gflops", action="store_true",
                    help="Skip GFLOPs measurement entirely (mIoU/accuracy "
                         "only).")
# >>> LIGHT_GFLOPS: GFLOPs for a given (tpl, cross_k, decoder_k_spatial) is
# almost content-independent — it's driven by the fixed attention budgets,
# not by what's in a given image (the main source of per-sample variance
# is ragged Voronoi cell sizes below the m-cap, a second-order effect).
# So for building a GFLOPS-vs-mIoU SELECTION curve across many configs, a
# small number of sampled forward passes gives a real (if slightly
# noisier) GFLOPs number far more cheaply than a full-test-set pass —
# letting every config in a wide sweep get an actual measured GFLOPs
# value instead of only the shortlisted few. Use --gflops_scope full for
# the expensive, precise, full-test-set number on final/shortlisted
# configs (unchanged behavior, still the default).
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

# >>> RESULTS_TRACKING: cross-check --type against what the config actually
# says.
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
model = Model_SenFlood_Skip.load_from_checkpoint(
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
    dataset_class=Sen1Floods11SkipDataset,
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
data_module.setup("validate" if split == "val" else "test")
if split == "val":
    results = trainer.validate(model=model, dataloaders=data_module.val_dataloader(), verbose=True)
else:
    results = trainer.test(model=model, dataloaders=data_module.test_dataloader(), verbose=True)
metrics = results[0] if results else {}

# >>> VAL_SPLIT: try a few common logged-metric-name candidates per split
# rather than hardcoding "test_mIoU"/"test_accuracy" — validation_step in
# your training script may log under "val_mIoU" or reuse "test_mIoU"
# regardless of stage. First matching key wins; if none match we fall back
# to NaN and print the metric dict we actually got so this is easy to debug
# rather than silently reporting NaN.
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
# GFLOPS MEASUREMENT (FlopCounterMode, over the FULL test set, SHARDED
# across all DDP ranks — >>> MULTI_GPU_GFLOPS)
# =============================================================================
# Rationale: trainer.test() above already scales across GPUs via PL's DDP
# (devices=-1 self-relaunches this same script as one process per GPU).
# This manual GFLOPs pass runs entirely OUTSIDE trainer.test()/trainer.fit(),
# so it gets none of that for free — without changes every rank would
# redundantly profile the ENTIRE test set (N ranks -> N x the work, not
# N x the speed). Instead, each rank profiles only its own shard
# (round-robin by sample index, see _shard_dataset_round_robin), then all
# ranks all-reduce their local (flops_sum, count) so every rank ends up
# with the identical, correct global mean. File writes and the RESULT
# line are then restricted to trainer.is_global_zero so N ranks don't
# race on the same output files.
def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b

gflops = float("nan")
gflops_n = 0

is_main = trainer.is_global_zero
world_size = trainer.world_size
rank = trainer.global_rank
# >>> MULTI_GPU_GFLOPS: use the rank's own assigned device, not a blind
# "cuda" default (which would put every rank on cuda:0 and contend).
device = trainer.strategy.root_device

PROFILE_DIR = "./profiler"
tag = f"{split}_tpl{tpl}_ck{ck}_dks{dks}_{args.gflops_scope}"  # >>> VAL_SPLIT / LIGHT_GFLOPS
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
        # scoring, round-robin across ranks, BEFORE building the DataLoader,
        # so each rank only forwards its own slice instead of the whole set.
        data_module.setup(split if split == "val" else "test")
        full_test_loader = (data_module.val_dataloader() if split == "val"
                            else data_module.test_dataloader())
        full_dataset = full_test_loader.dataset
        n_total = len(full_dataset)

        # >>> LIGHT_GFLOPS: cap the pool of GLOBAL indices before sharding
        # when scope=light, so total profiled samples across ALL ranks is
        # ~gflops_light_n rather than a full-split pass per rank.
        limit_n = args.gflops_light_n if args.gflops_scope == "light" else None
        # A couple of warmup batches eat into a small light budget fast;
        # clamp warmup so 'light' always leaves at least 1 profiled sample.
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

    # ── Combine every rank's local (sum, count) into one global mean.
    # Barrier first so a slow rank doesn't get its all_reduce paired with
    # a different iteration on a fast rank that raced ahead. ──────────────
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
              f"(sharded over {world_size} rank(s), {gflops_n} test samples "
              f"profiled total): {gflops:.3f}  "
              f"[lower bound; matmul/conv/attention ops only]")

        # ── Summary text file (rank 0 only — avoids concurrent writes) ──
        summary_path = os.path.join(out_dir, f"gflops_summary_{tag}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Config: tpl={tpl} cross_k={ck} decoder_k_spatial={dks}\n")
            f.write(f"Scope: {args.gflops_scope}"
                    + (f" ({args.gflops_light_n} samples budget)"
                       if args.gflops_scope == "light" else " (entire split)")
                    + "\n")
            f.write(f"Method: torch.utils.flop_counter.FlopCounterMode "
                    f"(SDPA attention counted)\n")
            f.write(f"Scope: FULL TEST SET, sharded round-robin across "
                    f"{world_size} DDP rank(s) "
                    f"(warmup={args.flops_n} batches/rank discarded, "
                    f"{gflops_n} total batches profiled)\n")
            f.write(f"GFLOPs/forward (global mean of {gflops_n} passes): "
                    f"{gflops:.4f}\n")
            f.write(f"Measured under torch.no_grad() via the "
                    f"module-tracker no-op patch (see script comments) "
                    f"-- matches the real test forward's memory profile.\n")
        print(f"[GFLOPs] summary -> {summary_path}")

        # ── Per-op-TYPE FLOPs CSV, from rank 0's LAST profiled batch only
        #    (reliable regardless of the patch; NOT a global breakdown
        #    across ranks — just representative op-type shares) ──
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


# RESULT line: rank-0 only (all ranks computed the same global gflops via
# all-reduce, so this isn't a correctness issue, but printing it N times
# would just clutter the driver's captured stdout).
# >>> VAL_SPLIT: 'split' is now part of the parseable RESULT line, and
# mIoU/accuracy field names are split-agnostic ("mIoU"/"accuracy" rather
# than hardcoded "test_mIoU"/"test_accuracy") since the same line format
# now covers both val and test runs.
if is_main:
    print(f"RESULT split={split} tpl={tpl} cross_k={ck} decoder_k_spatial={dks} "
          f"mIoU={miou:.6f} accuracy={acc:.6f} gflops={gflops:.6f} "
          f"gflops_n={gflops_n}")

    # =========================================================================
    # >>> RESULTS_TRACKING: write per-run result file to
    # ./scripts_GFLOPS/tmp_results/ — rank-0 only, avoids concurrent writes
    # to the same JSON path from multiple DDP processes. Filename now
    # includes split (>>> VAL_SPLIT) so val and test scores for the same
    # config never collide in the resume cache.
    # =========================================================================
    os.makedirs(TMP_RESULTS_DIR, exist_ok=True)
    run_record = {
        "task": "senflood",
        "split": split,  # >>> VAL_SPLIT
        "ckpt": os.path.abspath(args.ckpt),
        "xp_name": args.xp_name,
        "tokens_per_latent": tpl,
        "cross_k": ck,
        "decoder_k_spatial": dks,
        "mIoU": miou,
        "accuracy": acc,
        "gflops": gflops,
        "gflops_n": gflops_n,  # number of samples actually profiled (global), 0 if skipped
        "gflops_method": None if args.skip_gflops else "FlopCounterMode",
        # >>> LIGHT_GFLOPS: 'full' (entire split, precise) vs 'light' (small
        # sampled budget, cheap enough to run for every config in a sweep).
        "gflops_scope": None if args.skip_gflops else args.gflops_scope,
        "gflops_light_n_budget": (args.gflops_light_n
                                  if args.gflops_scope == "light" else None),
        "gflops_world_size": world_size,
        "gflops_skipped": bool(args.skip_gflops),  # >>> VAL_SPLIT
        "type": decode_type,
        "use_quadtree_decode": _uqd,
        "use_adaptive_decode": _uad,
        "config_path": args.config,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_result_path = tmp_result_path(
        args.ckpt, tpl, ck, dks, decode_type, split=split,
        gflops_scope=("skipped" if args.skip_gflops else args.gflops_scope),
        task="senflood")
    with open(out_result_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"[Test] Wrote per-run result -> {out_result_path} "
          f"(type={decode_type}, split={split})")
