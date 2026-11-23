"""
EuroSAT PTv3 Training Script
============================

PTv3 classifier (pixels-as-points, encoder only, mean pooling, random
init) on geo-bench m-eurosat (10 classes, 13 S2 bands, 64x64), through
ClassificationBaselineTrainer.

    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH
    export SLURM_JOB_NAME=interactive        # interactive Jean Zay session

    python script_train_eurosat_ptv3.py --xp_name ptv3_eurosat \
        --batch_size 64 --lr 1e-4 --epochs 100

    # quick check (one forward + backward on random data):
    python script_train_eurosat_ptv3.py --xp_name smoke --smoke
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_eurosat import EurosatBaselineDataset
from training.trainer_baselines_classification import ClassificationBaselineTrainer
from training.PTv3.model_ptv3_cls import PTv3Classifier


NUM_CLASSES  = EurosatBaselineDataset.NUM_CLASSES    # 10
NUM_CHANNELS = EurosatBaselineDataset.NUM_CHANNELS   # 13
MODALITY_KEY = "s2"


def eurosat_collate(batch):
    return {
        "image":  {MODALITY_KEY: torch.stack([s["image"][MODALITY_KEY] for s in batch])},
        "target": torch.stack([s["target"] for s in batch]),        # [B]
        "metadata": [s["metadata"] for s in batch],
    }


p = argparse.ArgumentParser(description="EuroSAT PTv3")
p.add_argument("--xp_name", type=str, required=True)
p.add_argument("--data_dir", type=str,
               default="./data/geo-bench-1.0/classification_v1.0/m-eurosat")
p.add_argument("--test_only", type=str, default=None)
p.add_argument("--resume", type=str, default=None)
p.add_argument("--smoke", action="store_true")

p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight_decay", type=float, default=1e-2)
p.add_argument("--label_smoothing", type=float, default=0.0)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--num_workers", type=int, default=8)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--grad_accum", type=int, default=1)
p.add_argument("--precision", type=str, default="bf16-mixed")
p.add_argument("--crop_size", type=int, default=None,
               help="None = full 64x64 (as the other EuroSAT baselines).")

p.add_argument("--ptv3_patch", type=int, default=256,
               help="Points per attention window. A 64x64 tile has 4,096 points, "
                    "so 1024 would already cover a quarter of the tile at level 0.")
p.add_argument("--ptv3_enc_channels", type=int, nargs=5, default=[32, 64, 128, 256, 512])
p.add_argument("--ptv3_enc_depths", type=int, nargs=5, default=[2, 2, 2, 6, 2])
p.add_argument("--ptv3_head_dim", type=int, default=16)
p.add_argument("--ptv3_drop_path", type=float, default=0.3)
p.add_argument("--head_dropout", type=float, default=0.0)
args = p.parse_args()

_bad = [c for c in args.ptv3_enc_channels if c % args.ptv3_head_dim]
if _bad:
    raise ValueError(f"PTv3 widths {_bad} not divisible by --ptv3_head_dim.")


def build_model():
    return PTv3Classifier(
        in_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        gsd=10.0,
        patch_size=args.ptv3_patch,
        enc_channels=tuple(args.ptv3_enc_channels),
        enc_depths=tuple(args.ptv3_enc_depths),
        head_dim=args.ptv3_head_dim,
        drop_path=args.ptv3_drop_path,
        head_dropout=args.head_dropout,
    )


if args.smoke:
    m = build_model().cuda().train()
    x = torch.randn(4, NUM_CHANNELS, 64, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    y.float().mean().backward()
    print(f"[smoke] logits {tuple(y.shape)}, "
          f"params {sum(q.numel() for q in m.parameters()) / 1e6:.1f}M")
    sys.exit(0)


# =============================================================================
# DATA
# =============================================================================

train_ds = EurosatBaselineDataset(root_path=args.data_dir, mode="train",
                                  crop_size=args.crop_size, augment=True)
val_ds   = EurosatBaselineDataset(root_path=args.data_dir, mode="validation",
                                  crop_size=args.crop_size, augment=False)
test_ds  = EurosatBaselineDataset(root_path=args.data_dir, mode="test",
                                  crop_size=args.crop_size, augment=False)

loader_kwargs = dict(
    num_workers=args.num_workers, collate_fn=eurosat_collate, pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)
test_loader  = DataLoader(test_ds, batch_size=args.batch_size, **loader_kwargs)


# =============================================================================
# MODEL
# =============================================================================

module = ClassificationBaselineTrainer(
    model=build_model(),
    modality=MODALITY_KEY,
    num_classes=NUM_CLASSES,
    lr=args.lr,
    weight_decay=args.weight_decay,
    label_smoothing=args.label_smoothing,
)

print(f"\n{'=' * 60}\n  EuroSAT PTv3 (encoder only, mean pool)   patch={args.ptv3_patch}"
      f"\n  Enc ch: {tuple(args.ptv3_enc_channels)}   BS={args.batch_size}  LR={args.lr}"
      f"  GPUs={torch.cuda.device_count()}\n{'=' * 60}\n")


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        wandb.init(name=f"BL_{args.xp_name}_ptv3_p{args.ptv3_patch}",
                   project="Atomizer_EuroSAT_Baselines", config=vars(args))
        wandb_logger = WandbLogger(project="Atomizer_EuroSAT_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN
# =============================================================================

ckpt_dir = "./checkpoints/eurosat_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir,
                        filename=f"bl_{args.xp_name}_ptv3-{{epoch:02d}}-{{val_top1:.4f}}",
                        monitor="val_top1", mode="max", save_top_k=1, verbose=True),
        ModelCheckpoint(dirpath=ckpt_dir, filename=f"bl_{args.xp_name}_ptv3-last",
                        every_n_epochs=1, save_top_k=1, save_last=True),
        EarlyStopping(monitor="val_top1", mode="max", patience=args.patience, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1, accelerator="gpu",
        max_epochs=args.epochs, precision=args.precision,
        logger=wandb_logger, log_every_n_steps=5, callbacks=callbacks,
        default_root_dir=ckpt_dir, gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    best_ckpt = trainer.checkpoint_callback.best_model_path

    import torch.distributed as dist
    is_rank_zero = trainer.is_global_zero
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if not is_rank_zero:
        if wandb_logger:
            import wandb
            wandb.finish()
        raise SystemExit(0)
else:
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(f"--test_only checkpoint not found: {args.test_only}")
    best_ckpt = args.test_only

print(f"\n{'=' * 60}\n  Testing checkpoint: {best_ckpt}\n{'=' * 60}\n")
test_trainer = Trainer(devices=1, accelerator="gpu", precision=args.precision,
                       logger=wandb_logger, default_root_dir=ckpt_dir)
test_trainer.test(module, test_loader, ckpt_path=best_ckpt)

if wandb_logger:
    import wandb
    wandb.finish()
