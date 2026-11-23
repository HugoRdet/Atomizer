"""
train_dales_unet.py -- train / test Atomiser_UnetDales on DALES
==============================================================

Location: scripts_atomizer/train_dales_unet.py (run from anywhere; the repo
root is found from this file's location).

Data: tiled patches from tile_dales.py, in
    <data_root>/train, <data_root>/val, <data_root>/test

Examples
--------
    # train (auto-resumes from the latest "last" checkpoint of this xp_name)
    python scripts_atomizer/train_dales_unet.py --xp_name vhv_v1 \
        --config_model config_test-DALES-unet-vhv.yaml --data_root /path/to/DALES

    # 2D baseline: same config with Atomiser.unet_vertical: none
    python scripts_atomizer/train_dales_unet.py --xp_name flat_v1 \
        --config_model config_test-DALES-unet-flat.yaml --data_root /path/to/DALES

    # quick checks: overfit a few tiles / short run
    python scripts_atomizer/train_dales_unet.py --xp_name dbg --overfit_tiles 4 --epochs 200 --no_wandb

    # test only (tile-level metrics; the official score is the full-scene eval)
    python scripts_atomizer/train_dales_unet.py --xp_name vhv_v1_test --test_only \
        --ckpt_path ./checkpoints/dales_unet/<best>.ckpt

Metrics logged on tiles (val_mIoU, test_tiles_mIoU) count points of
overlapping tiles several times: use them for model selection; report the
full-scene evaluation.
"""
import argparse
import datetime
import glob
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from training.utils import read_yaml
from training.trainer_DALES_unet import Model_DalesUnet
from training.utils.datasets.dataset_dales_points import DalesPointDataset, collate_dales

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")            # TF32 for the float32 parts


def str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ("yes", "true", "t", "1")


p = argparse.ArgumentParser(description="DALES Atomiser U-Net training")
p.add_argument("--xp_name", required=True)
p.add_argument("--config_model", default="config_test-DALES-unet-vhv.yaml",
               help="file in training/configs/")
p.add_argument("--data_root", default="./data/DALES", help="contains train/ val/ test/ tiles")
p.add_argument("--epochs", type=int, default=None, help="default: config trainer.epochs or 100")
p.add_argument("--batch_size", type=int, default=None, help="default: config trainer.batchsize")
p.add_argument("--num_workers", type=int, default=8)
p.add_argument("--max_context_points", type=int, default=256_000)
p.add_argument("--max_queries", type=int, default=100_000, help="supervised queries per training tile")
p.add_argument("--rotation", default="free", choices=["free", "d4", "none"])
p.add_argument("--class_weighting", default="sqrt", choices=["sqrt", "inv", "none"])
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--find_unused", type=str2bool, default=True,
               help="DDP unused-parameter search (False is faster when every parameter is used)")
p.add_argument("--ddp_timeout_min", type=int, default=10,
               help="NCCL timeout: a desynchronised or crashed rank stops the job after this")
p.add_argument("--overfit_tiles", type=int, default=0,
               help="> 0: train AND validate on the first N train tiles, no augmentation")
p.add_argument("--limit_train_batches", type=float, default=1.0)
p.add_argument("--limit_val_batches", type=float, default=1.0)
p.add_argument("--val_every", type=int, default=1, help="validate every N epochs")
p.add_argument("--ckpt_dir", default="./checkpoints/dales_unet/")
p.add_argument("--ckpt_path", default=None)
p.add_argument("--no_auto_resume", action="store_true")
p.add_argument("--test_only", action="store_true")
p.add_argument("--no_wandb", action="store_true")
p.add_argument("--wandb_project", default="Atomizer-DALES")
p.add_argument("--wandb_run_id", default=None)
args = p.parse_args()

# ----------------------------------------------------------------------------- config
config = read_yaml(str(_REPO_ROOT / "training" / "configs" / args.config_model))
tcfg = config.setdefault("trainer", {})
epochs = args.epochs or int(tcfg.get("epochs", 100))
batch_size = args.batch_size or int(tcfg.get("batchsize", 8))
tcfg["epochs"] = epochs

# ----------------------------------------------------------------------------- data
root = Path(args.data_root)


LOAD_PART = config.get("Atomiser", {}).get("unet_hier_nodes") == "ezsp_pre"


def build(split, augment=True, files=None):
    return DalesPointDataset(
        tiled_dir=str(root / split), split=split, files=files, load_partition=LOAD_PART,
        max_context_points=args.max_context_points, max_queries=args.max_queries,
        augment=augment, rotation=args.rotation)


if args.overfit_tiles > 0:
    files = sorted(list((root / "train").glob("*.laz")) + list((root / "train").glob("*.las")))[:args.overfit_tiles]
    train_ds = build("train", augment=False, files=files)
    val_ds = build("val", files=files)          # same tiles, eval mode (all points are queries)
    test_ds = val_ds
else:
    train_ds, val_ds, test_ds = build("train"), build("val"), build("test")
print(f"[DALES-Unet] tiles: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")


def make_loader(ds, shuffle):
    sampler = (DistributedSampler(ds, shuffle=shuffle)
               if dist.is_available() and dist.is_initialized() else None)
    return DataLoader(ds, batch_size=batch_size if shuffle else max(1, batch_size // 2),
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=args.num_workers, collate_fn=collate_dales, pin_memory=True,
                      persistent_workers=args.num_workers > 0,
                      prefetch_factor=2 if args.num_workers > 0 else None,
                      drop_last=shuffle and args.overfit_tiles == 0)


class DalesDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return make_loader(train_ds, True)

    def val_dataloader(self):
        return make_loader(val_ds, False)

    def test_dataloader(self):
        return make_loader(test_ds, False)


# ----------------------------------------------------------------------------- checkpoints
os.makedirs(args.ckpt_dir, exist_ok=True)
prefix = f"atomiser_unet_dales_{args.xp_name}"


def latest_last():
    m = glob.glob(os.path.join(args.ckpt_dir, f"{prefix}-last-*.ckpt"))
    if not m:
        return None
    return max(m, key=lambda s: int((re.findall(r"\d+", os.path.basename(s)) or [-1])[-1]))


resume = None
if not args.test_only:
    resume = args.ckpt_path or (None if args.no_auto_resume else latest_last())

# ----------------------------------------------------------------------------- logger
logger = False
if not args.no_wandb:
    from pytorch_lightning.loggers import WandbLogger
    run_id = args.wandb_run_id
    id_file = _REPO_ROOT / "training" / "wandb_runs" / f"{prefix}.txt"
    if run_id is None and resume is not None and id_file.exists():
        run_id = id_file.read_text().strip()
    logger = WandbLogger(project=args.wandb_project, name=f"AtomizerUnetDALES_{args.xp_name}",
                         save_dir=os.environ.get("WANDB_DIR", "./wandb"),
                         config={**config, "batch_size": batch_size, "epochs": epochs,
                                 "class_weighting": args.class_weighting, "rotation": args.rotation,
                                 "max_context_points": args.max_context_points,
                                 "max_queries": args.max_queries, "resume": resume},
                         id=run_id, resume="must" if run_id else None)

print(f"\n{'=' * 70}\n  DALES U-Net -- {args.xp_name}\n{'=' * 70}\n"
      f"  config {args.config_model} | vertical={config.get('Atomiser', {}).get('unet_vertical', 'VHV')}\n"
      f"  batch {batch_size} | epochs {epochs} | precision {args.precision} | weights {args.class_weighting}\n"
      f"  resume: {resume}\n")

# ----------------------------------------------------------------------------- model + trainer
model = Model_DalesUnet(config, name=args.xp_name, class_weighting=args.class_weighting)

callbacks = [
    ModelCheckpoint(dirpath=args.ckpt_dir, filename=f"{prefix}-best-{{epoch:03d}}-{{val_mIoU:.4f}}",
                    monitor="val_mIoU", mode="max", save_top_k=1),
    ModelCheckpoint(dirpath=args.ckpt_dir, filename=f"{prefix}-last-{{epoch:03d}}",
                    every_n_epochs=1, save_top_k=1),
]
if logger:
    callbacks.append(LearningRateMonitor(logging_interval="step"))

n_gpu = torch.cuda.device_count()
trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=args.find_unused,
                         timeout=datetime.timedelta(minutes=args.ddp_timeout_min)) if n_gpu > 1 or int(os.environ.get("SLURM_NNODES", 1)) > 1
    else "auto",
    use_distributed_sampler=False,
    accelerator="gpu" if n_gpu > 0 else "cpu",
    devices=-1 if n_gpu > 0 else 1,
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    max_epochs=epochs,
    precision=args.precision if n_gpu > 0 else "32-true",
    logger=logger,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=args.ckpt_dir,
    gradient_clip_val=1.0, gradient_clip_algorithm="norm",
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    check_val_every_n_epoch=args.val_every,
    num_sanity_val_steps=1,
)

if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only needs --ckpt_path")
    state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"[DALES-Unet] missing {len(res.missing_keys)} / unexpected {len(res.unexpected_keys)} keys")
    trainer.test(model, datamodule=DalesDataModule())
else:
    trainer.fit(model, datamodule=DalesDataModule(), ckpt_path=resume)
    if args.overfit_tiles == 0:
        trainer.test(model, datamodule=DalesDataModule(), ckpt_path="best")

if logger and trainer.is_global_zero:
    import wandb
    if getattr(wandb, "run", None) is not None:
        id_file = _REPO_ROOT / "training" / "wandb_runs" / f"{prefix}.txt"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(wandb.run.id)
