"""
Multi-task training of Atomiser_UnetMulti (PLANTEUR + AgriPotential + Cashew)
=============================================================================

    python scripts_atomizer/train_multitask.py --xp_name mt_v1 \
        --config_model config_test-MULTITASK.yaml

Data per epoch (config multitask.tasks.<name>.samples_per_epoch: null = all):
PLANTEUR all training patches, AgriPotential 1,900 random patches (redrawn every
epoch), Cashew all training patches; one dataset per batch, round robin,
identical schedule on every DDP rank.

Per-date datasets (PLANTEUR, AgriPotential) are wrapped by PerDateView; Cashew is
single-date. The model is built BEFORE the datasets are used: it registers the
days of the year in the lookup table, which PerDateView reads.

Checkpoints: ./checkpoints/multitask/ — best by val_score (mean of the tasks'
primary validation metrics) and last. After training, the best checkpoint is
tested on every task's test split.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import argparse
import datetime
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, DistributedSampler

from training.utils import read_yaml, Lookup_encoding
from training.trainer_multitask import MultiTaskTrainer
from training.utils.datasets.multitask_data import (PerDateView, TaggedView, MultiTaskDataset,
                                                    MultiTaskBatchSampler, collate_multitask_batch)
try:
    from training.utils.resolution_registry import register_all_resolutions
except ImportError:
    from resolution_registry import register_all_resolutions

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

# ---- hang diagnostics: `kill -USR1 <pid>` prints every thread's Python stack; with
# MT_STACK_DUMP_MIN=<minutes> (default 3), a stack dump is also printed every N minutes,
# so a stuck rank shows where it waits (cancelled automatically when training ends).
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
_DUMP_MIN = float(os.environ.get("MT_STACK_DUMP_MIN", "3"))
BANDS_YAML = "./data/bands_info/bands.yaml"
CONFIGS_DATASET = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"


# =============================================================================
# Datasets
# =============================================================================
def build_view(task, tcfg, split, config, lookup):
    """One task's dataset for a split, wrapped for the multi-task pipeline."""
    T = int(config.get("dataset", {}).get("multi_temporal", 6))
    if task == "planteur":
        from training.utils.datasets.utils_dataset_PLANTEUR import PlanteurDataset
        ds = PlanteurDataset(root_path=tcfg["data_dir"], mode=split, config_model=config, look_up=lookup,
                             use_s1=True, split_protocol=tcfg.get("split_protocol", "default"),
                             temporal_window=str(tcfg.get("temporal_window", "2019")))
        return PerDateView(ds, task, T, ds.NUM_S2_BANDS, ds.NUM_S1_BANDS, lookup)
    if task == "pastis":
        from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
        ds = PastisHDDataset(root_path=tcfg["data_dir"], mode=split, config_model=config, look_up=lookup,
                             use_s1=True, use_spot=False)
        return PerDateView(ds, task, T, ds.NUM_S2_BANDS, ds.NUM_S1_BANDS, lookup)
    if task == "agripotential":
        from training.utils.datasets.utils_dataset_AGRIPOTENTIAL import AgriPotentialDataset
        ds = AgriPotentialDataset(root_path=tcfg["data_dir"], mode=split, config_model=config,
                                  look_up=lookup, crop=tcfg.get("crop", "viticulture"), n_frames=T)
        return PerDateView(ds, task, T, ds.NUM_S2_BANDS, 0, lookup)
    if task == "cashew":
        from training.utils.datasets.utils_dataset_cashew import CashewSkipDataset
        ds = CashewSkipDataset(root_path=tcfg["data_dir"], mode=split, dataset_config=read_yaml(BANDS_YAML),
                               config_model=config, look_up=lookup)
        return TaggedView(ds, task)
    raise ValueError(f"unknown task {task!r} (pastis | planteur | agripotential | cashew)")


class RankMarkers(pl.Callback):
    """Prints, from EVERY rank, when it reaches each end-of-epoch stage, and re-arms the
    periodic stack dump, so the last line of each rank shows where it stopped."""

    def _mark(self, trainer, what):
        print(f"[rank {trainer.global_rank}] epoch {trainer.current_epoch}: {what}", flush=True)
        if _DUMP_MIN > 0:
            faulthandler.dump_traceback_later(_DUMP_MIN * 60, repeat=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._mark(trainer, "train epoch start")

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._mark(trainer, "validation start")

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._mark(trainer, "validation end (checkpoint callbacks next)")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._mark(trainer, "checkpoint dict built (writing next)")

    def on_train_epoch_end(self, trainer, pl_module):
        self._mark(trainer, "train epoch end")

    # re-arm after every batch: dumps only fire when a rank makes no progress at all
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if _DUMP_MIN > 0:
            faulthandler.dump_traceback_later(_DUMP_MIN * 60, repeat=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if _DUMP_MIN > 0:
            faulthandler.dump_traceback_later(_DUMP_MIN * 60, repeat=True)

    def on_train_end(self, trainer, pl_module):
        faulthandler.cancel_dump_traceback_later()


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, config, lookup, num_workers=4):
        super().__init__()
        self.config, self.lookup, self.num_workers = config, lookup, num_workers
        self.mt = config["multitask"]
        self.tasks = list(self.mt["tasks"])
        self.train_batch_sampler = None

    def setup(self, stage=None):
        if getattr(self, "_done", False):
            return
        self._done = True
        self.views = {s: {t: build_view(t, self.mt["tasks"][t], s, self.config, self.lookup)
                          for t in self.tasks} for s in ("train", "validation", "test")}
        for s, v in self.views.items():
            print(f"[MultiTask-DM] {s}: " + ", ".join(f"{t}={len(d)}" for t, d in v.items()))

    def _loader(self, **kw):
        return DataLoader(num_workers=self.num_workers, collate_fn=collate_multitask_batch,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=2 if self.num_workers > 0 else None, **kw)

    def train_dataloader(self):
        views = self.views["train"]
        combined = MultiTaskDataset(views)
        ddp = dist.is_available() and dist.is_initialized()
        self.train_batch_sampler = MultiTaskBatchSampler(
            sizes={t: len(v) for t, v in views.items()}, offsets=combined.offsets,
            batch_sizes={t: int(self.mt["tasks"][t]["batch_size"]) for t in self.tasks},
            samples_per_epoch={t: self.mt["tasks"][t].get("samples_per_epoch") for t in self.tasks},
            order=self.mt.get("order", "round_robin"), seed=int(self.mt.get("seed", 42)),
            rank=dist.get_rank() if ddp else 0, world_size=dist.get_world_size() if ddp else 1)
        if IS_RANK0:
            counts = {}
            for t, _ in self.train_batch_sampler.schedule():
                counts[t] = counts.get(t, 0) + 1
            print(f"[MultiTask-DM] batches per epoch (per rank): {counts}, total {len(self.train_batch_sampler)}")
        return self._loader(dataset=combined, batch_sampler=self.train_batch_sampler)

    def _eval_loaders(self, split):
        ddp = dist.is_available() and dist.is_initialized()
        out = []
        for t in self.tasks:
            v = self.views[split][t]
            sampler = DistributedSampler(v, shuffle=False) if ddp else None
            out.append(self._loader(dataset=v, batch_size=int(self.mt["tasks"][t]["batch_size"]),
                                    sampler=sampler, shuffle=False))
        return out

    def val_dataloader(self):
        return self._eval_loaders("validation")

    def test_dataloader(self):
        return self._eval_loaders("test")


def close_file_handles(dm):
    """Close file handles opened in THIS process by the datasets (e.g. AgriPotential's
    HDF5 file, opened when the pre-flight check reads a sample). DataLoader workers are
    forked from this process, and h5py handles inherited through fork can block the
    workers' first read indefinitely; each worker must open its own."""
    n = 0
    for views in dm.views.values():
        for v in views.values():
            ds = getattr(v, "ds", v)
            h = getattr(ds, "_h5", None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
                ds._h5 = None
                n += 1
    if n:
        print(f"[MultiTask] closed {n} file handle(s) opened in the main process before forking workers")


def preflight(dm):
    """Every task: fixed token layout across samples, and the expected shapes."""
    for t, v in dm.views["train"].items():
        b = collate_multitask_batch([v[0], v[1 % len(v)]])
        tok = next(iter(b["groups"].values()))["tokens"]
        if not torch.equal(tok[0][..., 1:3], tok[1][..., 1:3]):
            raise RuntimeError(f"[Preflight] {t}: token positions differ between samples")
        extra = f", time_positions {b['time_positions'][0].tolist()}" if "time_positions" in b else ""
        print(f"[Preflight] {t}: tokens {tuple(tok.shape)}, queries {tuple(b['queries'].shape)}{extra} ✓")


# =============================================================================
# Main
# =============================================================================
parser = argparse.ArgumentParser(description="Multi-task Atomiser U-Net training")
parser.add_argument("--xp_name", required=True)
parser.add_argument("--config_model", default="config_test-MULTITASK.yaml")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ckpt_path", default=None, help="resume training from a Lightning checkpoint")
parser.add_argument("--test_only", default=None, help="checkpoint to test (no training)")
parser.add_argument("--skip_preflight", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed_everything(args.seed, workers=True)
config = read_yaml("./training/configs/" + args.config_model)
config["multitask"]["seed"] = args.seed

lookup = Lookup_encoding(read_yaml(CONFIGS_DATASET), read_yaml(BANDS_YAML), config)
register_all_resolutions(lookup)                     # before the model and any dataset
lookup.register_abstract_channel("VV_VH")            # PLANTEUR's S1 ratio channel

model = MultiTaskTrainer(config, lookup, name=args.xp_name)   # registers days 1..366
dm = MultiTaskDataModule(config, lookup, num_workers=args.num_workers)
dm.setup()
if IS_RANK0 and not args.skip_preflight:
    preflight(dm)
close_file_handles(dm)            # always, before any DataLoader forks its workers

wandb_logger = None
if IS_RANK0 and args.test_only is None:
    import wandb
    wandb.init(name=f"MT_{args.xp_name}", project="MULTITASK", config=config)
    wandb_logger = WandbLogger(project="MULTITASK")

ckpt_dir = "./checkpoints/multitask/"
os.makedirs(ckpt_dir, exist_ok=True)
callbacks = [
    RankMarkers(),
    ModelCheckpoint(dirpath=ckpt_dir, filename=f"mt_{args.xp_name}-{{epoch:02d}}-{{val_score:.4f}}",
                    monitor="val_score", mode="max", save_top_k=1, verbose=True),
    ModelCheckpoint(dirpath=ckpt_dir, filename=f"mt_{args.xp_name}-last-{{epoch:02d}}",
                    every_n_epochs=1, save_top_k=1, save_last=True),
    LearningRateMonitor(logging_interval="step"),
]
trainer = Trainer(
    # other tasks' heads / loss weights are idle at each step -> find_unused_parameters.
    # timeout: a blocked collective raises an error naming it instead of hanging forever
    strategy=DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(minutes=30)),
    use_distributed_sampler=False,                       # the batch sampler shards by rank itself
    devices=-1, num_nodes=int(os.environ.get("SLURM_NNODES", 1)), accelerator="gpu",
    precision="bf16-mixed", max_epochs=config["trainer"]["epochs"], logger=wandb_logger,
    log_every_n_steps=5, callbacks=callbacks, default_root_dir=ckpt_dir, gradient_clip_val=1.0,
)

if args.test_only:
    ckpt = torch.load(args.test_only, map_location="cpu", weights_only=False)
    res = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    if res.missing_keys:
        raise RuntimeError(f"checkpoint is missing weights, e.g. {res.missing_keys[:5]}")
    trainer.test(model, datamodule=dm)
    sys.exit(0)

trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
trainer.test(model, datamodule=dm, ckpt_path="best")
