"""
Multi-task baseline sanity-check training script.

Trains 5 tasks jointly with a round-robin schedule (one batch from each
task per optimizer step via accumulate_grad_batches=NUM_TASKS):

    Segmentation:
      - BurnScars     (binary, 6 HLS bands -> 13 canonical, single-frame)
      - Sen1Floods11  (binary, 13 S2 + 2 S1, single-frame)
      - PASTIS        (19 classes, 10 S2 -> 13 canonical, T=6)

    Classification:
      - EuroSAT       (10 classes, 13 S2 bands permuted to canonical)
      - ForestNet     (12 classes, 6 Landsat -> 13 canonical via interpolation)

Pipeline being smoke-tested:
    - per-task datasets producing the unified 15-channel format
      (single-frame for all but PASTIS, [T=6, 15, H, W] for PASTIS)
    - tagged collates injecting batch["task"]; dates carried for PASTIS;
      seg_collate vs cls_collate selected per task type
    - RoundRobinLoader cycling tasks deterministically
    - MultiTaskTrainer dispatching loss + metrics by task type
    - shared ResNet+UPerNet with per-task seg MLP heads + per-task cls heads,
      plus a PASTIS-only TimeMerge adapter (auto-built from num_frames=6)
    - per-task metrics (mIoU for seg, top-1/macro_acc for cls) + cross-task
      mean of primary metrics as the checkpoint criterion

Defaults to `--variant resnet_small` for fast iteration. For numbers
comparable to single-task baselines, use `--variant resnet50`.

Memory note: PASTIS micro-batches are [B, 6, 15, 512, 512] — 6x larger
than the single-frame tasks. EuroSAT and ForestNet are padded to
512x512 but their real content is much smaller (64 and 320 respectively),
so the encoder activations are mostly zero-padding — works but inefficient.
If you OOM, reduce --batch-size.

Single GPU only. For DDP, set up DistributedSampler manually on each
per-task DataLoader (RoundRobinLoader bypasses Lightning's auto sampler
injection — `use_distributed_sampler=False` is already set on the Trainer).
"""

import argparse
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from training.utils.datasets_baselines_MT.dataset_burnscars_mt import BurnScarsMTDataset
from training.utils.datasets_baselines_MT.dataset_senflood_mt import Sen1Floods11MTDataset
from training.utils.datasets_baselines_MT.dataset_pastis_mt import PastisMTDataset
from training.utils.datasets_baselines_MT.dataset_eurosat_mt import EuroSATMTDataset
from training.utils.datasets_baselines_MT.dataset_forestnet_mt import ForestNetMTDataset
from training.VIT.model_mt_vit_upernet import build_multitask_vit_upernet
from training.ResNet.model_mt_resnet_upernet import build_multitask_resnet_upernet
from training.utils.datasets_baselines_MT.mt_pipeline import (
    RoundRobinLoader, make_tagged_collate, seg_collate, cls_collate,
)
from training.trainer_mt_baselines import MultiTaskTrainer


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "burnscars": {"type": "seg", "num_classes": 2,  "num_frames": 1},
    "senflood":  {"type": "seg", "num_classes": 2,  "num_frames": 1},
    "pastis":    {"type": "seg", "num_classes": 19, "num_frames": 6},
    "eurosat":   {"type": "cls", "num_classes": 10, "num_frames": 1},
    "forestnet": {"type": "cls", "num_classes": 12, "num_frames": 1},
}
NUM_TASKS = len(TASK_CONFIGS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--burnscars-root", type=str, default="./data/hls_burn_scars")
    p.add_argument("--senflood-root",  type=str, default="./data/SENFLOOD")
    p.add_argument("--pastis-root",    type=str, default="./data/PASTIS-HD")
    p.add_argument("--eurosat-root",   type=str,
                   default="./data/geo-bench-1.0/classification_v1.0/m-eurosat")
    p.add_argument("--forestnet-root", type=str,
                   default="./data/geo-bench-1.0/classification_v1.0/m-forestnet")
    p.add_argument("--output-dir",     type=str, default="./runs/mt_sanity")
    p.add_argument("--batch-size",     type=int, default=4)
    p.add_argument("--num-workers",    type=int, default=16,
                   help="Workers PER per-task DataLoader. Total processes = "
                        "num_workers * num_tasks. .tif decode is the bottleneck, "
                        "so set this high (16+) if you have the cores.")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-2)
    p.add_argument("--max-epochs",     type=int, default=10)
    # Epoch length specified as images-per-task; the script derives the
    # optimizer-step count via:
    #     steps_per_epoch = images_per_task // (batch_size * devices)
    p.add_argument("--images-per-task", type=int, default=2000,
                   help="Images per task per epoch. With shuffle, smaller "
                        "datasets cycle multiple times per epoch.")
    p.add_argument("--warmup-steps",   type=int, default=200,
                   help="Linear warmup, in optimizer steps.")
    p.add_argument("--precision",      type=str, default="bf16-mixed")
    p.add_argument("--devices",        type=int, default=-1,
                   help="Number of GPUs. -1 = all visible (DDP).")
    p.add_argument("--gradient-clip",  type=float, default=1.0)
    p.add_argument("--seed",           type=int, default=42)
    # ── Test-only mode ─────────────────────────────────────
    # Skip trainer.fit() and go straight to the two test phases (shared
    # best across all tasks, then per-task best). Useful when an earlier
    # training run completed but didn't successfully run the test phase
    # (e.g. crash, timeout, or test-skipping bug). Requires the ckpt dir
    # to already contain mean_primary-best.ckpt and the per-task
    # <task>-best.ckpt files.
    p.add_argument("--test-only", action="store_true",
                   help="Skip training, run only the test phases against "
                        "existing checkpoints in ./checkpoints/mt_baselines/"
                        "{model_tag}/.")
    # ── Model ───────────────────────────────────────────────
    p.add_argument(
        "--model", type=str, default="resnet",
        choices=["resnet", "vit"],
        help="Backbone family. 'resnet' uses --variant for the ResNet "
             "depth; 'vit' uses --vit-* args for ViT config.",
    )
    p.add_argument(
        "--variant", type=str, default="resnet_small",
        choices=["resnet_super_small", "resnet_small",
                 "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone (only used when --model resnet). Default "
             "'resnet_small' for fast sanity; use 'resnet50' for runs "
             "comparable to single-task baselines.",
    )
    # ViT-specific config (only used when --model vit). Defaults match
    # the single-task ViT (embed_dim=384, depth=12, heads=6, patch=16).
    p.add_argument("--vit-img-size", type=int, default=512,
                   help="ViT input size. Must match the canonical multi-task "
                        "spatial size (datasets pad to 512x512).")
    p.add_argument("--vit-embed-dim", type=int, default=384)
    p.add_argument("--vit-depth", type=int, default=12)
    p.add_argument("--vit-num-heads", type=int, default=6)
    p.add_argument("--vit-patch-size", type=int, default=16)
    p.add_argument("--decoder-channels", type=int, default=256,
                   help="UPerNet hidden channels. Also the per-task seg "
                        "head's input/hidden dim.")
    p.add_argument("--seg-head-dropout", type=float, default=0.0,
                   help="Dropout in the per-task seg MLP head.")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Per-worker prefetch depth (PyTorch default 2). "
                        "Higher keeps the pipeline filled when GPU is faster "
                        "than I/O. Costs ~prefetch_factor x batches of RAM "
                        "per worker.")
    p.add_argument("--xp-name", type=str, default=None,
                   help="Optional run name suffix for W&B. Defaults to the "
                        "model variant (e.g. 'MT_resnet50'). With suffix: "
                        "'MT_<variant>_<xp_name>'.")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="W&B run ID to resume (e.g. 'abc123xy'). Useful "
                        "with --test-only when the original training run "
                        "completed but didn't log test metrics: pass the "
                        "ID and the new test metrics will land in the "
                        "same run instead of creating a new one. Without "
                        "this, a new W&B run is created.")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# DataLoader builders
# ────────────────────────────────────────────────────────────────────

class _DistSamplerSetEpochCallback(Callback):
    """
    Calls set_epoch() on each per-task DistributedSampler at every
    train-epoch start. Without this, DistributedSampler reuses the same
    shuffle pattern across epochs, hurting training.

    Lightning normally handles set_epoch automatically for samplers it
    injected itself; ours are manually constructed (because RoundRobinLoader
    bypasses Lightning's auto sampler injection), so we do it ourselves.

    Args:
        train_samplers: dict {task: DistributedSampler or None}.
                        None entries (single-GPU) are skipped silently.
    """

    def __init__(self, train_samplers: dict):
        super().__init__()
        self.samplers = train_samplers

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        for sampler in self.samplers.values():
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)


def build_loader(dataset, task_name, batch_size, num_workers, shuffle, drop_last,
                 prefetch_factor=4, sampler=None):
    # Pick the right collate based on task type. Both produce batches with
    # the same top-level keys (image/target/valid_mask/original_size/metadata),
    # plus a task tag added by make_tagged_collate. Seg targets are [B, H, W];
    # cls targets are [B].
    base_collate = (
        seg_collate if TASK_CONFIGS[task_name]["type"] == "seg" else cls_collate
    )
    # When a DistributedSampler is provided (DDP), DataLoader must have
    # shuffle=False — the sampler controls ordering.
    loader_shuffle = shuffle if sampler is None else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=loader_shuffle,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=make_tagged_collate(base_collate, task_name),
    )


def _make_dist_sampler(dataset, shuffle, drop_last, use_ddp):
    """
    Build a DistributedSampler when DDP is active, else return None.

    Reads RANK/WORLD_SIZE from environment variables set by Lightning's
    DDP launcher BEFORE trainer.fit() runs. We can't use torch.distributed
    here because the process group isn't initialized yet at the time loaders
    are constructed in main() — that happens later inside trainer.fit().

    Returning None means build_loader falls back to the regular shuffled
    DataLoader (single-GPU semantics).
    """
    if not use_ddp:
        return None
    from torch.utils.data.distributed import DistributedSampler
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )


_DATASET_CLASSES = {
    "burnscars": BurnScarsMTDataset,
    "senflood":  Sen1Floods11MTDataset,
    "pastis":    PastisMTDataset,
    "eurosat":   EuroSATMTDataset,
    "forestnet": ForestNetMTDataset,
}


def _construct_dataset(task, args, mode):
    """Construct one of the per-task datasets in a given split."""
    cls = _DATASET_CLASSES[task]
    if task == "burnscars":
        return cls(root_path=args.burnscars_root, mode=mode, augment=(mode == "train"))
    if task == "senflood":
        return cls(root_path=args.senflood_root, mode=mode, augment=(mode == "train"))
    if task == "pastis":
        return cls(
            root_path=args.pastis_root, mode=mode,
            num_frames=6, augment=(mode == "train"),
        )
    if task == "eurosat":
        return cls(root_path=args.eurosat_root, mode=mode, augment=(mode == "train"))
    if task == "forestnet":
        return cls(root_path=args.forestnet_root, mode=mode, augment=(mode == "train"))
    raise KeyError(task)


def build_train_loaders(args, use_ddp: bool):
    """
    Build per-task train DataLoaders.

    Returns:
        loaders:  dict {task: DataLoader}
        samplers: dict {task: DistributedSampler or None}
                  Used by main() to call set_epoch() on each per epoch
                  when running DDP.
    """
    loaders, samplers = {}, {}
    for task in TASK_CONFIGS.keys():
        ds = _construct_dataset(task, args, mode="train")
        sampler = _make_dist_sampler(
            ds, shuffle=True, drop_last=True, use_ddp=use_ddp,
        )
        loaders[task] = build_loader(
            ds, task, args.batch_size, args.num_workers,
            shuffle=True, drop_last=True,
            prefetch_factor=args.prefetch_factor,
            sampler=sampler,
        )
        samplers[task] = sampler
    return loaders, samplers


def build_val_loaders(args, use_ddp: bool):
    """
    Build per-task validation DataLoaders.

    With DDP, each rank sees a unique shard of each val set
    (DistributedSampler with shuffle=False, drop_last=False).
    Cross-rank metric aggregation happens in MultiTaskTrainer's
    _aggregate_split via sync_dist=True.

    Returned as a list — Lightning iterates each fully each epoch.
    Order matches TASK_CONFIGS so logged keys are consistent.
    """
    loaders = []
    for task in TASK_CONFIGS.keys():
        ds = _construct_dataset(task, args, mode="validation")
        sampler = _make_dist_sampler(
            ds, shuffle=False, drop_last=False, use_ddp=use_ddp,
        )
        loaders.append(build_loader(
            ds, task, args.batch_size, args.num_workers,
            shuffle=False, drop_last=False,
            prefetch_factor=args.prefetch_factor,
            sampler=sampler,
        ))
    return loaders


def build_test_loaders(args, use_ddp: bool):
    """Build per-task test DataLoaders. Same DDP semantics as val."""
    loaders = []
    for task in TASK_CONFIGS.keys():
        ds = _construct_dataset(task, args, mode="test")
        sampler = _make_dist_sampler(
            ds, shuffle=False, drop_last=False, use_ddp=use_ddp,
        )
        loaders.append(build_loader(
            ds, task, args.batch_size, args.num_workers,
            shuffle=False, drop_last=False,
            prefetch_factor=args.prefetch_factor,
            sampler=sampler,
        ))
    return loaders


# ────────────────────────────────────────────────────────────────────
# Test-phase helpers
# ────────────────────────────────────────────────────────────────────

def _collect_test_metrics(callback_metrics: dict, tasks_to_keep):
    """
    Pull the just-logged test metrics for the given tasks out of
    `trainer.callback_metrics` (a flat dict of all logged scalars).

    The trainer logs metrics under keys like:
        test/<task>/<metric>    (per-task metrics)
        test/<task>/loss        (per-task loss)
        test/mean_primary       (cross-task aggregate)

    We keep only entries for tasks in `tasks_to_keep` and return them
    grouped: {task: {metric: value, ...}, "mean_primary": value (if present)}.

    Values are converted to Python floats. Tasks that weren't tested
    in this pass either won't have entries in callback_metrics, or
    will have NaN values from torchmetrics computing on no updates
    (which we filter out).
    """
    import math

    out = {task: {} for task in tasks_to_keep}
    for k, v in callback_metrics.items():
        if not k.startswith("test/"):
            continue
        if isinstance(v, torch.Tensor):
            try:
                v_float = float(v.item())
            except Exception:
                continue
        else:
            try:
                v_float = float(v)
            except Exception:
                continue
        # Drop NaNs (untested tasks compute NaN under no metric updates).
        if math.isnan(v_float):
            continue

        # Strip the "test/" prefix and route to the right bucket.
        suffix = k[len("test/"):]
        parts = suffix.split("/", 1)
        if len(parts) == 2:
            task, metric = parts
            if task in tasks_to_keep:
                out[task][metric] = v_float
        elif suffix == "mean_primary":
            out["mean_primary"] = v_float
    return out


def _log_test_results_to_wandb(results: dict, prefix: str, wandb_logger):
    """
    Log a {task: {metric: value, ...}, ["mean_primary": value]} dict to
    W&B under a non-colliding prefix.

    Used by the multi-checkpoint test phase: shared-best results go under
    `test_shared/*`, per-task-best results go under `test_per_task/*`.
    Without separate prefixes, the second `trainer.test()` call would
    overwrite the first's metrics in W&B (since both log under `test/`).

    We log via wandb_logger.experiment (the W&B run handle that Lightning
    owns) — calling wandb.log() on the global handle would create a
    second orphan run that Lightning's metrics aren't part of.
    """
    if wandb_logger is None:
        return

    payload = {}
    for key, val in results.items():
        if key == "mean_primary" and isinstance(val, (int, float)):
            payload[f"{prefix}/mean_primary"] = val
            continue
        if isinstance(val, dict):
            for metric, v in val.items():
                payload[f"{prefix}/{key}/{metric}"] = v
    if not payload:
        return

    try:
        wandb_logger.experiment.log(payload)
    except Exception as e:
        print(f"[wandb] log failed: {e}")


def _print_summary_tables(task_configs, primary_metric_per_task,
                          shared_results, per_task_results):
    """
    Print two human-readable tables summarizing the test phase.

    Table 1 — single shared model:
        | task | primary metric | value |
        From shared_results: results of testing the mean_primary-best ckpt
        on every task. One model, evaluated on the suite.

    Table 2 — per-task best models:
        | task | primary metric | value |
        From per_task_results: each row uses the model selected by its
        own task's val metric, evaluated only on that task. Upper envelope
        when per-task model selection is allowed.
    """
    print(f"\n{'='*70}")
    print(f"  RESULTS — TABLE 1: single shared model (mean_primary-best ckpt)")
    print(f"{'='*70}")
    print(f"  {'Task':<14} {'Metric':<12} {'Value':<10}")
    print(f"  {'-'*14} {'-'*12} {'-'*10}")
    for task in task_configs:
        m = primary_metric_per_task[task]
        v = shared_results.get(task, {}).get(m, None)
        v_str = f"{v:.4f}" if v is not None else "—"
        print(f"  {task:<14} {m:<12} {v_str:<10}")
    if "mean_primary" in shared_results:
        v = shared_results["mean_primary"]
        print(f"  {'-'*14} {'-'*12} {'-'*10}")
        print(f"  {'mean_primary':<14} {'':<12} {v:.4f}")

    print(f"\n{'='*70}")
    print(f"  RESULTS — TABLE 2: per-task best ckpts")
    print(f"{'='*70}")
    print(f"  {'Task':<14} {'Metric':<12} {'Value':<10}")
    print(f"  {'-'*14} {'-'*12} {'-'*10}")
    for task in task_configs:
        m = primary_metric_per_task[task]
        v = per_task_results.get(task, {}).get(m, None)
        v_str = f"{v:.4f}" if v is not None else "—"
        print(f"  {task:<14} {m:<12} {v_str:<10}")

    # Gap (per-task best - shared best), per task
    print(f"\n{'='*70}")
    print(f"  GAP — per-task best minus shared best (positive = MT hurts)")
    print(f"{'='*70}")
    print(f"  {'Task':<14} {'Shared':<10} {'Per-task':<10} {'Gap':<10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    for task in task_configs:
        m = primary_metric_per_task[task]
        v_shared = shared_results.get(task, {}).get(m, None)
        v_pt     = per_task_results.get(task, {}).get(m, None)
        if v_shared is not None and v_pt is not None:
            gap = v_pt - v_shared
            print(f"  {task:<14} {v_shared:<10.4f} {v_pt:<10.4f} {gap:+.4f}")
        else:
            print(f"  {task:<14} {'—':<10} {'—':<10} {'—':<10}")
    print()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve --devices=-1 to actual GPU count ──
    # The schedule math, DistributedSampler injection, and W&B init
    # all need a concrete device count.
    if args.devices == -1:
        if torch.cuda.is_available():
            args.devices = max(1, torch.cuda.device_count())
        else:
            args.devices = 1
    print(f"[Devices] Using {args.devices} GPU(s)")

    # Multi-GPU training engages DDP and per-task DistributedSamplers.
    use_ddp = args.devices > 1 and torch.cuda.is_available()

    # ── Resolve schedule: images-per-task -> optimizer steps ──
    # Per optimizer step, each task sees `batch_size * devices` images
    # (one batch per task per GPU; gradients from all `num_tasks` tasks
    # merge via accumulate_grad_batches=num_tasks).
    images_per_step_per_task = args.batch_size * args.devices
    steps_per_epoch = max(1, args.images_per_task // images_per_step_per_task)
    total_optimizer_steps = steps_per_epoch * args.max_epochs
    micro_steps_per_epoch = steps_per_epoch * NUM_TASKS

    print(
        f"[Schedule] images_per_task={args.images_per_task}, "
        f"batch_size={args.batch_size}, devices={args.devices}, "
        f"num_tasks={NUM_TASKS}\n"
        f"           => steps_per_epoch={steps_per_epoch}, "
        f"micro_steps_per_epoch={micro_steps_per_epoch}, "
        f"total_optimizer_steps={total_optimizer_steps}"
    )

    # ── DataLoaders ─────────────────────────────────────────
    train_task_loaders, train_samplers = build_train_loaders(args, use_ddp=use_ddp)
    val_loaders = build_val_loaders(args, use_ddp=use_ddp)

    # Round-robin: one micro-batch per task per optimizer step.
    train_loader_mt = RoundRobinLoader(train_task_loaders, length=micro_steps_per_epoch)

    # ── Model ───────────────────────────────────────────────
    # `model_tag` is the short name used for the checkpoint folder and
    # the W&B run name. For ResNet that's the variant ("resnet50",
    # "resnet_small", ...). For ViT it's just "vit" — we don't enumerate
    # ViT sizes in --model the way we do for ResNet variants.
    if args.model == "resnet":
        model = build_multitask_resnet_upernet(
            variant=args.variant,
            in_channels=15,
            task_specs=TASK_CONFIGS,
            decoder_channels=args.decoder_channels,
            seg_head_dropout=args.seg_head_dropout,
        )
        model_tag = args.variant
    elif args.model == "vit":
        model = build_multitask_vit_upernet(
            in_channels=15,
            task_specs=TASK_CONFIGS,
            img_size=args.vit_img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            decoder_channels=args.decoder_channels,
            seg_head_dropout=args.seg_head_dropout,
        )
        model_tag = "vit"
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    # ── LightningModule ─────────────────────────────────────
    trainer_module = MultiTaskTrainer(
        model=model,
        task_configs=TASK_CONFIGS,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=total_optimizer_steps,
        warmup_steps=args.warmup_steps,
    )

    # ── Checkpoint directory ───────────────────────────────
    # ./checkpoints/mt_baselines/{model_tag}/
    # model_tag is the variant for ResNet (e.g. "resnet50") or "vit" for ViT.
    ckpt_dir = os.path.join("./checkpoints/mt_baselines", model_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[Checkpoints] Saving to {ckpt_dir}")

    # ── W&B logger ─────────────────────────────────────────
    # Don't call wandb.init() manually — it creates a run handle that
    # WandbLogger isn't connected to, leaving Lightning's metrics
    # orphaned. Pass name/project/config to WandbLogger directly.
    wandb_logger = None
    if os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            run_name = f"MT_{model_tag}"
            if args.xp_name:
                run_name = f"{run_name}_{args.xp_name}"
            # When --wandb-run-id is provided, resume that existing run.
            # WandbLogger forwards id/resume to wandb.init via kwargs.
            # resume="must" makes the run fail loudly if the id doesn't
            # exist (vs "allow" which would silently create a new run);
            # we want loud failures so you don't accidentally log into
            # a fresh run thinking you resumed.
            wandb_kwargs = dict(
                name=run_name,
                project="Atomizer_MT_Baselines",
                config=vars(args),
            )
            if args.wandb_run_id:
                wandb_kwargs["id"] = args.wandb_run_id
                wandb_kwargs["resume"] = "must"
                print(f"[wandb] Resuming run id={args.wandb_run_id}")
            wandb_logger = WandbLogger(**wandb_kwargs)
        except Exception as e:
            print(f"  WandB not available ({e}), logging to console only.")

    # Per-task primary metric names (must match what trainer logs in
    # val/<task>/<metric>). Read from MultiTaskTrainer's `primary` mapping.
    primary_per_type = trainer_module.primary
    primary_metric_per_task = {
        task: primary_per_type[cfg["type"]]
        for task, cfg in TASK_CONFIGS.items()
    }

    # ── Checkpoint callbacks: 1 for shared best + 1 per task ──
    callbacks = [
        # (1) Best joint checkpoint: max val/mean_primary
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="mean_primary-best",
            monitor="val/mean_primary",
            mode="max",
            save_top_k=1,
            save_last=True,            # also writes last.ckpt
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    # (2) Best per-task checkpoint, one per task.
    for task, metric_name in primary_metric_per_task.items():
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"{task}-best",
                monitor=f"val/{task}/{metric_name}",
                mode="max",
                save_top_k=1,
                verbose=True,
            )
        )
    # (3) DDP-only: rotate the DistributedSampler shuffle across epochs.
    #     A no-op for single-GPU (all samplers in the dict are None).
    callbacks.append(_DistSamplerSetEpochCallback(train_samplers))

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        # Multi-task heads in ModuleDicts mean each train step uses only
        # one task's head + adapters; the other tasks' parameters are
        # unused that step. Vanilla DDP errors on unused parameters, so
        # we use the variant that allows them. Static graph is NOT
        # compatible because the unused-param set changes step-to-step
        # (different task each round-robin step).
        strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
        precision=args.precision if torch.cuda.is_available() else "32-true",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=NUM_TASKS,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        # Custom round-robin loader doesn't play well with auto sampler injection.
        # On single GPU this is moot; for DDP set up DistributedSampler manually
        # on each per-task DataLoader.
        use_distributed_sampler=False,
    )

    # ── Fit (skipped when --test-only) ──────────────────────
    if not args.test_only:
        trainer.fit(
            trainer_module,
            train_dataloaders=train_loader_mt,
            val_dataloaders=val_loaders,
        )
    else:
        if trainer.global_rank == 0:
            print(f"\n[--test-only] Skipping trainer.fit(), going straight "
                  f"to test phases.\n[--test-only] Loading checkpoints from "
                  f"{ckpt_dir}/")

    # ─────────────────────────────────────────────────────────────
    # Test phase
    # ─────────────────────────────────────────────────────────────
    # Two evaluations:
    #   (1) "shared best": load mean_primary-best.ckpt and test all 5 tasks.
    #   (2) "per-task best": for each task, load <task>-best.ckpt and test
    #       only that task. Yields the upper envelope when per-task model
    #       selection is allowed.
    # The two together let us report two result tables and quantify the gap.
    #
    # DDP note: trainer.test() runs on every rank (each rank evaluates its
    # shard, metrics aggregate via sync_dist=True). Only the prints, the W&B
    # test-result logging, and the summary tables are rank-0 only.
    is_rank_zero = trainer.global_rank == 0

    test_loaders_list = build_test_loaders(args, use_ddp=use_ddp)
    # build_test_loaders returns a list ordered by TASK_CONFIGS insertion.
    test_loader_by_task = dict(zip(TASK_CONFIGS.keys(), test_loaders_list))

    # ── (1) Shared best across all tasks ───────────────────
    shared_ckpt = os.path.join(ckpt_dir, "mean_primary-best.ckpt")
    if is_rank_zero:
        print(f"\n{'='*60}")
        print(f"  TEST PHASE 1/2: shared best ckpt across all tasks")
        print(f"  Ckpt: {shared_ckpt}")
        print(f"{'='*60}")
    if not os.path.exists(shared_ckpt):
        if is_rank_zero:
            print(f"[Test] WARNING: {shared_ckpt} not found, skipping.")
        shared_results = {}
    else:
        trainer.test(
            trainer_module,
            dataloaders=test_loaders_list,
            ckpt_path=shared_ckpt,
        )
        # Collect logged metrics under a structured prefix for the summary.
        shared_results = _collect_test_metrics(
            trainer.callback_metrics, list(TASK_CONFIGS.keys()),
        )
        # Re-log under a non-colliding prefix so the per-task test pass
        # below doesn't overwrite the shared-best metrics in W&B.
        if is_rank_zero and wandb_logger is not None:
            _log_test_results_to_wandb(
                shared_results, prefix="test_shared", wandb_logger=wandb_logger,
            )

    # ── (2) Per-task best, each tested on its own task ─────
    # Note: each per-task test gets a FRESH pl.Trainer instance. The
    # shared trainer from phase 1 has logging state that Lightning
    # mutates based on whether `dataloaders` is a list-of-many vs
    # list-of-one (different `add_dataloader_idx` defaults). Reusing
    # the same trainer triggers
    #     "MisconfigurationException: You called self.log(...) twice
    #     in test_step with different arguments"
    # because phase 1 logged with idx-suffixed keys and phase 2 with
    # bare keys. A fresh trainer per task sidesteps the issue —
    # each invocation starts from clean logging state.
    def _make_test_trainer():
        return pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=args.devices,
            strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
            precision=args.precision if torch.cuda.is_available() else "32-true",
            logger=wandb_logger,
            default_root_dir=args.output_dir,
            use_distributed_sampler=False,
        )

    per_task_results = {}
    for task in TASK_CONFIGS.keys():
        task_ckpt = os.path.join(ckpt_dir, f"{task}-best.ckpt")
        if is_rank_zero:
            print(f"\n{'='*60}")
            print(f"  TEST PHASE 2/2: per-task best for '{task}'")
            print(f"  Ckpt: {task_ckpt}")
            print(f"{'='*60}")
        if not os.path.exists(task_ckpt):
            if is_rank_zero:
                print(f"[Test] WARNING: {task_ckpt} not found, skipping.")
            per_task_results[task] = {}
            continue
        per_task_trainer = _make_test_trainer()
        per_task_trainer.test(
            trainer_module,
            dataloaders=[test_loader_by_task[task]],
            ckpt_path=task_ckpt,
        )
        # _collect_test_metrics returns {task: {metric: val, ...}, "mean_primary": ...}
        # For phase 2 we only have one task per call, so unwrap to the
        # metrics dict directly. _print_summary_tables expects
        # per_task_results[task][metric_name] (not [task][task][metric_name]).
        collected = _collect_test_metrics(
            per_task_trainer.callback_metrics, [task],
        )
        per_task_results[task] = collected.get(task, {})
        if is_rank_zero and wandb_logger is not None:
            _log_test_results_to_wandb(
                {task: per_task_results[task]},
                prefix="test_per_task",
                wandb_logger=wandb_logger,
            )

    # ── Summary tables (rank 0 only) ──────────────────────
    if is_rank_zero:
        _print_summary_tables(
            TASK_CONFIGS, primary_metric_per_task,
            shared_results, per_task_results,
        )
    # WandbLogger handles its own teardown (called automatically when the
    # trainer / process exits). No manual wandb.finish() needed.


if __name__ == "__main__":
    main()