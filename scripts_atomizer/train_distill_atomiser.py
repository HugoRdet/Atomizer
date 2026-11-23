"""
Distillation training script -- Atomiser U-Net student, OlmoEarth (Sentinel) + DINOv3 (FLAIR-HUB aerial) teachers, DUNE recipe.

Where it goes: scripts_atomizer/train_distill_atomiser.py   (mirrors train_burnscars_unet.py)

    # once, before any DDP run (computes the z-score statistics; also prints shapes, timings, token layout):
    PYTHONPATH=. python training/distill/check_distill_data.py --config training/configs/config_distill.yaml --compute-stats
    # smoke run (a few optimizer steps, tiny validation):
    python scripts_atomizer/train_distill_atomiser.py --config training/configs/config_distill.yaml --xp_name smoke --smoke
    # real run:
    python scripts_atomizer/train_distill_atomiser.py --config training/configs/config_distill.yaml --xp_name distill_s42 --seed 42
"""
import argparse
import glob
import os
import sys
import time

T0 = time.time()                             # job start: the time limit counts from here

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from training.distill.trainer_distill import (DistillDataModule, Model_Distill, SaveEncoder, StopAtIters,
                                              TimeLimitCheckpoint)
from training.utils import Lookup_encoding, read_yaml

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--xp_name", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None, help="checkpoint (.ckpt) to resume from")
parser.add_argument("--auto_resume", action="store_true",
                    help="resume from the newest .ckpt in ./checkpoints/distill/<xp_name> if there is one "
                         "(ignored when --resume is given); start from scratch otherwise")
parser.add_argument("--time_limit", type=float, default=115.0,
                    help="minutes after the script starts at which to save timelimit.ckpt and stop "
                         "(default 115 = 1 h 55 for 2 h jobs; 0 disables)")
parser.add_argument("--wandb", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="wandb logging: auto = on except for --smoke; the CSV log is always written")
parser.add_argument("--wandb_project", default="Distill")
parser.add_argument("--wandb_group", default=None, help="e.g. the name of a sweep")
parser.add_argument("--wandb_tags", nargs="*", default=None)
parser.add_argument("--wandb_dir", default="/lustre/fsn1/projects/rech/ebp/uuk56xr/wandb",
                    help="wandb root; the offline runs are written to <wandb_dir>/wandb/offline-run-*")
parser.add_argument("--wandb_online", action="store_true", help="sync live (needs internet); default is OFFLINE, for compute nodes")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override keys of the distill: block, e.g. --set lr=1e-3 epochs=2 batch_per_dataset=32 epoch_size=20000")
parser.add_argument("--smoke", action="store_true", help="10 optimizer steps, validation every 5, tiny validation set")
parser.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
args = parser.parse_args()
seed_everything(args.seed, workers=True)

config = read_yaml(args.config)
dc = config["distill"]
for _k, _v in dict(steps=None, warmup_steps=None, grad_accum=1).items():   # keys added after the first config version
    dc.setdefault(_k, _v)                                                  # (steps, when set, overrides epochs x epoch_size)


def _parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return yaml.safe_load(v)


for kv in args.set:
    k, v = kv.split("=", 1)
    assert k in dc, f"--set {k}: not a key of the distill block ({sorted(dc)})"
    dc[k] = _parse(v)
    if IS_RANK0:
        print(f"[Train] override distill.{k} = {dc[k]!r}")
torch.set_float32_matmul_precision("high")
try:                                         # harmless DDP/autograd stream warning
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except AttributeError:
    pass
config["Atomiser"]["use_error_predictor"] = False
config["trainer"]["slide"] = False
if args.smoke:
    dc.update(steps=6, warmup_steps=1, val_every_steps=5, val_batches=2, ckpt_every_steps=5, workers=2)

bands = read_yaml(args.bands_yaml)
lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)
dm = DistillDataModule(config, lookup, bands)
model = Model_Distill(config, lookup, name=args.xp_name)

ckpt_dir = f"./checkpoints/distill/{args.xp_name}"
os.makedirs(ckpt_dir, exist_ok=True)
if args.resume is None and args.auto_resume:
    found = [f for f in glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if os.path.isfile(f)]
    if found:
        args.resume = max(found, key=os.path.getmtime)
    if IS_RANK0:
        print(f"[Train] auto_resume: {args.resume or 'no checkpoint in ' + ckpt_dir + ', starting from scratch'}")
use_wandb = args.wandb == "on" or (args.wandb == "auto" and not args.smoke)
offline = not args.wandb_online
if use_wandb:
    # wandb offline by default; all of its files (runs, cache, config) live under args.wandb_dir, not in $HOME
    os.environ["WANDB_DIR"] = args.wandb_dir
    os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(args.wandb_dir, "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(args.wandb_dir, "config"))
    os.environ.setdefault("WANDB_DATA_DIR", os.path.join(args.wandb_dir, "data"))
    if offline:
        os.environ["WANDB_MODE"] = "offline"
loggers = [CSVLogger("./logs_distill", name=args.xp_name)]
if use_wandb:
    os.makedirs(args.wandb_dir, exist_ok=True)
    id_file = os.path.join(ckpt_dir, "wandb_run_id.txt")
    wkw = {}
    if args.resume and os.path.exists(id_file):                       # resume the same wandb run
        wkw = dict(id=open(id_file).read().strip(), resume="allow")
    wlogger = WandbLogger(project=args.wandb_project, name=args.xp_name, group=args.wandb_group, tags=args.wandb_tags,
                          save_dir=args.wandb_dir, offline=offline, log_model=False, **wkw)
    loggers.append(wlogger)
    if IS_RANK0:
        with open(id_file, "w") as f:
            f.write(str(wlogger.experiment.id))


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _git_commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


hparams = {**_flat(config["distill"], "distill."),
           **{f"atomiser.{k}": v for k, v in config["Atomiser"].items() if k.startswith("unet_")},
           "xp_name": args.xp_name, "seed": args.seed, "config_file": args.config, "smoke": args.smoke,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_commit": _git_commit()}
for lg in loggers:
    lg.log_hyperparams(hparams)
callbacks = [
    LearningRateMonitor(logging_interval="step"),
    StopAtIters(),
    *([TimeLimitCheckpoint(ckpt_dir, args.time_limit * 60, start_time=T0)] if args.time_limit > 0 else []),
    SaveEncoder(ckpt_dir, int(dc.get("ckpt_every_steps", 1000))),
    ModelCheckpoint(dirpath=ckpt_dir, filename="step{step}", every_n_train_steps=int(dc.get("ckpt_every_steps", 1000)),
                    save_top_k=-1, save_last=True, auto_insert_metric_name=False),
]

trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",   # each micro-batch uses only one teacher head
    devices=-1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=-1,                                 # stopped by StopAtIters after DUNE's iteration count
    accumulate_grad_batches=2 * int(dc.get("grad_accum", 1)),   # grad_accum x (one Sentinel + one FLAIR micro-batch) per optimizer step
    gradient_clip_val=1.0,                         # the module's configure_gradient_clipping applies DUNE's per-parameter clipping
    use_distributed_sampler=False,                 # the round-robin sampler shards by rank itself
    check_val_every_n_epoch=None,
    val_check_interval=2 * int(dc.get("grad_accum", 1)) * int(dc.get("val_every_steps", 500)),   # in micro-batches
    limit_val_batches=int(dc.get("val_batches", 20)),
    num_sanity_val_steps=0,
    log_every_n_steps=int(dc.get("log_every", 10)),
    logger=loggers,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
)
if IS_RANK0:
    print(f"[Train] config {args.config} | xp {args.xp_name} | seed {args.seed} | smoke {args.smoke}")
trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
if IS_RANK0 and use_wandb and offline:
    print(f"[Train] wandb ran OFFLINE. From a login node with internet (after `wandb login`):  wandb sync {args.wandb_dir}/wandb/offline-run-*")
