"""
Multi-task baseline sanity-check training script.

Trains BurnScars + Sen1Floods11 + PASTIS jointly with a round-robin
schedule (one batch from each task per optimizer step via
accumulate_grad_batches=NUM_TASKS).

Pipeline being smoke-tested:
    - per-task datasets producing the unified 15-channel format
      (single-frame for BurnScars/Sen1Floods11, [T=6, 15, H, W] for PASTIS)
    - tagged collates injecting batch["task"]; dates carried for PASTIS
    - RoundRobinLoader cycling tasks deterministically
    - MultiTaskTrainer dispatching and accumulating gradients
    - shared ResNet+UPerNet with per-task MLP seg heads, plus a
      PASTIS-only TimeMerge adapter (auto-built from num_frames=6)
    - per-task metrics + cross-task mean as checkpoint criterion

Defaults to `--variant resnet_small` for fast iteration. For numbers
comparable to single-task baselines, use `--variant resnet50`.

Memory note: PASTIS micro-batches are [B, 6, 15, 512, 512] — 6x larger
than the single-frame tasks. If you OOM, reduce --batch-size (or only
PASTIS-side via a separate loader factory if needed).

Single GPU only. For DDP, set up DistributedSampler manually on each
per-task DataLoader (RoundRobinLoader bypasses Lightning's auto sampler
injection — `use_distributed_sampler=False` is already set on the Trainer).
"""

import argparse
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader



from training.utils.datasets_baselines_MT.dataset_burnscars_mt import BurnScarsMTDataset
from training.utils.datasets_baselines_MT.dataset_senflood_mt import Sen1Floods11MTDataset
from training.utils.datasets_baselines_MT.dataset_pastis_mt import PastisMTDataset
from training.ResNet.model_mt_resnet_upernet import build_multitask_resnet_upernet
from training.utils.datasets_baselines_MT.mt_pipeline import RoundRobinLoader, make_tagged_collate, seg_collate
from training.trainer_mt_baselines import MultiTaskTrainer

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "burnscars": {"type": "seg", "num_classes": 2,  "num_frames": 1},
    "senflood":  {"type": "seg", "num_classes": 2,  "num_frames": 1},
    "pastis":    {"type": "seg", "num_classes": 19, "num_frames": 6},
}
NUM_TASKS = len(TASK_CONFIGS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--burnscars-root", type=str, default="./data/hls_burn_scars")
    p.add_argument("--senflood-root",  type=str, default="./data/SENFLOOD")
    p.add_argument("--pastis-root",    type=str, default="./data/PASTIS-HD")
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
    p.add_argument("--devices",        type=int, default=1)
    p.add_argument("--gradient-clip",  type=float, default=1.0)
    p.add_argument("--seed",           type=int, default=42)
    # Model.
    p.add_argument(
        "--variant", type=str, default="resnet_small",
        choices=["resnet_super_small", "resnet_small",
                 "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone. Default 'resnet_small' for fast sanity; "
             "use 'resnet50' for runs comparable to single-task baselines.",
    )
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
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# DataLoader builders
# ────────────────────────────────────────────────────────────────────

def build_loader(dataset, task_name, batch_size, num_workers, shuffle, drop_last,
                 prefetch_factor=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=make_tagged_collate(seg_collate, task_name),
    )


def build_train_loaders(args):
    burnscars_train = BurnScarsMTDataset(
        root_path=args.burnscars_root, mode="train", augment=True,
    )
    senflood_train = Sen1Floods11MTDataset(
        root_path=args.senflood_root, mode="train", augment=True,
    )
    pastis_train = PastisMTDataset(
        root_path=args.pastis_root, mode="train", num_frames=6, augment=True,
    )
    return {
        "burnscars": build_loader(
            burnscars_train, "burnscars",
            args.batch_size, args.num_workers,
            shuffle=True, drop_last=True,
            prefetch_factor=args.prefetch_factor,
        ),
        "senflood": build_loader(
            senflood_train, "senflood",
            args.batch_size, args.num_workers,
            shuffle=True, drop_last=True,
            prefetch_factor=args.prefetch_factor,
        ),
        "pastis": build_loader(
            pastis_train, "pastis",
            args.batch_size, args.num_workers,
            shuffle=True, drop_last=True,
            prefetch_factor=args.prefetch_factor,
        ),
    }


def build_val_loaders(args):
    burnscars_val = BurnScarsMTDataset(
        root_path=args.burnscars_root, mode="validation", augment=False,
    )
    senflood_val = Sen1Floods11MTDataset(
        root_path=args.senflood_root, mode="validation", augment=False,
    )
    pastis_val = PastisMTDataset(
        root_path=args.pastis_root, mode="validation", num_frames=6, augment=False,
    )
    # Returned as a list — Lightning iterates each fully each epoch.
    # Order matches TASK_CONFIGS so logged keys are consistent.
    return [
        build_loader(burnscars_val, "burnscars",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
        build_loader(senflood_val, "senflood",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
        build_loader(pastis_val, "pastis",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
    ]


def build_test_loaders(args):
    burnscars_test = BurnScarsMTDataset(
        root_path=args.burnscars_root, mode="test", augment=False,
    )
    senflood_test = Sen1Floods11MTDataset(
        root_path=args.senflood_root, mode="test", augment=False,
    )
    pastis_test = PastisMTDataset(
        root_path=args.pastis_root, mode="test", num_frames=6, augment=False,
    )
    return [
        build_loader(burnscars_test, "burnscars",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
        build_loader(senflood_test, "senflood",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
        build_loader(pastis_test, "pastis",
                     args.batch_size, args.num_workers,
                     shuffle=False, drop_last=False,
                     prefetch_factor=args.prefetch_factor),
    ]


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

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
    train_task_loaders = build_train_loaders(args)
    val_loaders        = build_val_loaders(args)

    # Round-robin: one micro-batch per task per optimizer step.
    train_loader_mt = RoundRobinLoader(train_task_loaders, length=micro_steps_per_epoch)

    # ── Model ───────────────────────────────────────────────
    model = build_multitask_resnet_upernet(
        variant=args.variant,
        in_channels=15,
        task_specs=TASK_CONFIGS,
        decoder_channels=args.decoder_channels,
        seg_head_dropout=args.seg_head_dropout,
    )

    # ── LightningModule ─────────────────────────────────────
    trainer_module = MultiTaskTrainer(
        model=model,
        task_configs=TASK_CONFIGS,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=total_optimizer_steps,
        warmup_steps=args.warmup_steps,
    )

    # ── Trainer ─────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "ckpts"),
            filename="best-{epoch}-{step}",
            monitor="val/mean_primary",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision if torch.cuda.is_available() else "32-true",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=NUM_TASKS,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        # Custom round-robin loader doesn't play well with auto sampler injection.
        # On single GPU this is moot; for DDP set up DistributedSampler manually
        # on each per-task DataLoader.
        use_distributed_sampler=False,
    )

    # ── Fit ─────────────────────────────────────────────────
    trainer.fit(
        trainer_module,
        train_dataloaders=train_loader_mt,
        val_dataloaders=val_loaders,
    )

    # ── Test on best checkpoint ─────────────────────────────
    test_loaders = build_test_loaders(args)
    trainer.test(trainer_module, dataloaders=test_loaders, ckpt_path="best")


if __name__ == "__main__":
    main()