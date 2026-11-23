"""
PLANTEUR Training Script — tropical crop-type segmentation (flat / SKIP / U-Net)
================================================================================

Adapted from train_pastis.py. Supported encoders (chosen by the config, exactly
as in PASTISTrainer): Atomiser_Unet (joint temporal encoding),
Atomiser_Senflood_Skip (use_decoder_skip), Atomiser_Senflood.

NOT yet supported: the per-date paths (use_temporal_transformer /
unet_temporal: per_level). They need a PLANTEUR port of PastisHDTemporalDataset.

Splits:
    --split_protocol default          Fold 1 / 4 / 5 (train / val / test)
    --split_protocol loio:reunion     leave-one-island-out (guadeloupe,
                                      martinique, guyane, reunion, mayotte)

Examples:
    python train_planteur.py --xp_name planteur_skip --use_s1 \
        --config_model config_test-PLANTEUR.yaml --multi_temporal 6

    python train_planteur.py --xp_name planteur_unet_loio_reu --use_s1 \
        --config_model config_test-PLANTEUR-unet.yaml --split_protocol loio:reunion
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import statistics
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_PLANTEUR import PlanteurTrainer
from training.utils.datasets.utils_dataset_PLANTEUR import PlanteurDataset, TEMPORAL_WINDOWS, _slug
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

ALL_KNOWN_RESOLUTIONS = {1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


N_TEST_RUNS = 2
TEST_SEEDS = list(range(1, N_TEST_RUNS + 1))


def load_pretrained_encoder(model, ckpt_path):
    print(f"\n[PLANTEUR] Loading pretrained encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        enc = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items()
               if k.startswith("encoder.")}
    elif "encoder" in ckpt:
        enc = ckpt["encoder"]
    else:
        raise ValueError(f"Checkpoint keys: {list(ckpt.keys())}")
    own = model.encoder.state_dict()
    compatible = {k: v for k, v in enc.items() if k in own and v.shape == own[k].shape}
    skipped = [(k, v.shape, own[k].shape) for k, v in enc.items()
               if k in own and v.shape != own[k].shape]
    res = model.encoder.load_state_dict(compatible, strict=False)
    print(f"  Loaded: {len(compatible)}, Skipped: {len(skipped)}, "
          f"Fresh: {len(res.missing_keys) - len(skipped)}")
    for k, s, d in skipped:
        print(f"    - {k}: {s} != {d}")
    return model


def planteur_collate(samples):
    batch = collate_multitask(samples)
    if "queries" not in batch and "tasks" in batch:
        task_data = next(iter(batch["tasks"].values()))
        batch["queries"] = task_data["queries"]
        batch["queries_mask"] = task_data["queries_mask"]
    return batch


class PlanteurDataModule(pl.LightningDataModule):

    def __init__(self, root_path, config_model, look_up, batch_size=4,
                 num_workers=4, dataset_kwargs=None):
        super().__init__()
        self.root_path      = root_path
        self.config_model   = config_model
        self.look_up        = look_up
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.dataset_kwargs = dataset_kwargs or {}

    def _make_dataset(self, mode):
        return PlanteurDataset(root_path=self.root_path, mode=mode,
                               config_model=self.config_model, look_up=self.look_up,
                               **self.dataset_kwargs)

    def setup(self, stage=None):
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True
        self.train_dataset = self._make_dataset("train")   # first: computes norm stats
        self.val_dataset   = self._make_dataset("validation")
        self.test_dataset  = self._make_dataset("test")
        print(f"\n[PLANTEUR-DM] train={len(self.train_dataset)} "
              f"val={len(self.val_dataset)} test={len(self.test_dataset)}")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=planteur_collate,
            pin_memory=True, persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="PLANTEUR training")
parser.add_argument("--xp_name",            type=str, required=True)
parser.add_argument("--config_model",       type=str, default="config_test-PLANTEUR.yaml")
parser.add_argument("--data_dir",           type=str, default="./data/PLANTEUR")
parser.add_argument("--ckpt_path",          type=str, default=None)
parser.add_argument("--pretrained_encoder", type=str, default=None)
parser.add_argument("--num_workers",        type=int, default=4)
parser.add_argument("--grad_accum",         type=int, default=1)
parser.add_argument("--deterministic",      action="store_true")
parser.add_argument("--wandb_run_id",       type=str, default=None)
parser.add_argument("--test_only",          action="store_true")
parser.add_argument("--use_s1",             action="store_true", help="add Sentinel-1 (default: S2 only)")
parser.add_argument("--multi_temporal",     type=int, default=None, help="frames per sensor")
parser.add_argument("--split_protocol",     type=str, default="default",
                    help="'default' or 'loio:<island>'")
parser.add_argument("--temporal_window",    type=str, default="2019", choices=list(TEMPORAL_WINDOWS))
parser.add_argument("--s1_orbit",           type=str, default="all", choices=["all", "A", "D"])
parser.add_argument("--mask_s1_nodata",     action="store_true")
parser.add_argument("--balanced_labels",    action="store_true",
                    help="train on ANNOTATIONS_BALANCED (class-balanced protocol)")
parser.add_argument("--skip_preflight",     action="store_true")
args = parser.parse_args()

# =============================================================================
# CONFIG & CHECKS
# =============================================================================
config_model = read_yaml("./training/configs/" + args.config_model)
if args.multi_temporal is not None:
    config_model.setdefault("dataset", {})["multi_temporal"] = args.multi_temporal
multi_temporal = config_model.get("dataset", {}).get("multi_temporal", 10)

if config_model["trainer"].get("num_classes") != 8:
    print(f"[PLANTEUR] trainer.num_classes={config_model['trainer'].get('num_classes')} -> 8")
    config_model["trainer"]["num_classes"] = 8

a_cfg = config_model.setdefault("Atomiser", {})
is_unet = "unet" in str(config_model.get("encoder", "")).lower() or "unet_task" in a_cfg
if a_cfg.get("use_temporal_transformer", False) or a_cfg.get("unet_temporal", "none") == "per_level":
    raise NotImplementedError(
        "Per-date encoding (use_temporal_transformer / unet_temporal: per_level) needs a "
        "PLANTEUR port of PastisHDTemporalDataset, which is not written yet.")
a_cfg["use_temporal_transformer"] = False

if is_unet:
    if a_cfg.get("unet_task", "segmentation") != "segmentation":
        raise ValueError("PLANTEUR is segmentation: set Atomiser.unet_task: segmentation")
    a_cfg["unet_task"] = "segmentation"
    a_cfg["use_error_predictor"] = False
    config_model["trainer"]["slide"] = False
    if "unet" not in str(config_model.get("encoder", "")).lower():
        config_model["encoder"] = f"{config_model.get('encoder', 'ATOMIZER')}_Unet"

lookup_table = Lookup_encoding(
    read_yaml("./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"),
    read_yaml("./data/bands_info/bands.yaml"), config_model)
register_all_resolutions(lookup_table)
if args.use_s1:
    lookup_table.register_abstract_channel("VV_VH")

modality_str = "S2+S1" if args.use_s1 else "S2"
proto_slug = _slug(args.split_protocol)

if IS_RANK0:
    enc = ("Atomiser_Unet" if is_unet else
           "Atomiser_Senflood_Skip" if a_cfg.get("use_decoder_skip", False) else
           "Atomiser_Senflood")
    print(f"\n[PLANTEUR] xp={args.xp_name} encoder={enc} modalities={modality_str}")
    print(f"[PLANTEUR] protocol={args.split_protocol} window={args.temporal_window} "
          f"frames/sensor={multi_temporal} s1_orbit={args.s1_orbit}")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if IS_RANK0 and not args.test_only:
    import wandb
    pretrain_tag = "pretrained" if args.pretrained_encoder else "scratch"
    kw = dict(
        name=f"PLANTEUR_{args.xp_name}_{modality_str}_{proto_slug}_{pretrain_tag}",
        project="PLANTEUR",
        config={**config_model, "use_s1": args.use_s1, "multi_temporal": multi_temporal,
                "split_protocol": args.split_protocol, "temporal_window": args.temporal_window,
                "s1_orbit": args.s1_orbit, "mask_s1_nodata": args.mask_s1_nodata,
                "balanced_labels": args.balanced_labels},
    )
    if args.wandb_run_id is not None:
        kw.update(id=args.wandb_run_id, resume="must")
    wandb.init(**kw)
    wandb_logger = WandbLogger(project="PLANTEUR")

# =============================================================================
# DATA & MODEL
# =============================================================================
data_module = PlanteurDataModule(
    root_path=args.data_dir, config_model=config_model, look_up=lookup_table,
    batch_size=config_model["trainer"]["batchsize"], num_workers=args.num_workers,
    dataset_kwargs=dict(
        use_s1=args.use_s1, split_protocol=args.split_protocol,
        temporal_window=args.temporal_window, s1_orbit=args.s1_orbit,
        mask_s1_nodata=args.mask_s1_nodata, balanced_labels=args.balanced_labels,
    ),
)
data_module.setup()

model = PlanteurTrainer(config=config_model, wand=(not args.test_only), name=args.xp_name,
                        transform=None, lookup_table=lookup_table)
if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)


def preflight_token_layout(dm):
    ds = dm.train_dataset
    batch = planteur_collate([ds[0], ds[1 % len(ds)]])
    if len(batch["groups"]) > 1:
        raise RuntimeError("[Preflight] more than one resolution group")
    for res, grp in batch["groups"].items():
        t = grp["tokens"]
        if not torch.equal(t[0, :, 1:3], t[1, :, 1:3]):
            raise RuntimeError(f"[Preflight] group {res}: token positions differ between samples")
        n_dates = int(torch.unique(t[0][..., 7]).numel())
        print(f"[Preflight] group {res}: tokens {tuple(t.shape)}, layout fixed ✓ "
              f"({n_dates} distinct time indices in sample 0)")
    print(f"[Preflight] queries {tuple(batch['queries'].shape)} ✓")


if is_unet and IS_RANK0 and not args.skip_preflight and not args.test_only:
    preflight_token_layout(data_module)

# =============================================================================
# TRAINER
# =============================================================================
ckpt_dir = f"./checkpoints/planteur/{proto_slug}/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(dirpath=ckpt_dir,
                    filename=f"planteur_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
                    monitor="val_mIoU", mode="max", save_top_k=1, verbose=True),
    ModelCheckpoint(dirpath=ckpt_dir, filename=f"planteur_{args.xp_name}-last-{{epoch:02d}}",
                    every_n_epochs=1, save_top_k=1, save_last=True, verbose=True),
    LearningRateMonitor(logging_interval="step"),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True), use_distributed_sampler=False,
    devices=-1, num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    max_epochs=config_model["trainer"]["epochs"], accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5, callbacks=callbacks, default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum, deterministic=args.deterministic,
)

# =============================================================================
# TEST-ONLY
# =============================================================================
if args.test_only:
    ckpt_to_load = args.ckpt_path or config_model.get("trainer", {}).get("checkpoint_path")
    if ckpt_to_load is None:
        raise ValueError("--test_only requires --ckpt_path or trainer.checkpoint_path.")
    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    if result.missing_keys:
        raise RuntimeError(f"[PLANTEUR] checkpoint is missing weights, e.g. {result.missing_keys[:5]}")
    model.eval()

    miou_runs, acc_runs = [], []
    for run_idx, seed in enumerate(TEST_SEEDS, 1):
        seed_everything(seed, workers=True)
        run_trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=True), use_distributed_sampler=False,
            devices=-1, accelerator="gpu", precision="bf16-mixed", logger=False,
            enable_model_summary=False, deterministic=args.deterministic,
        )
        results = run_trainer.test(model, datamodule=data_module, verbose=True)
        if IS_RANK0:
            m = results[0] if results else {}
            miou_runs.append(m.get("test_mIoU", float("nan")))
            acc_runs.append(m.get("test_accuracy", float("nan")))
            print(f"RESULT run={run_idx} seed={seed} test_mIoU={miou_runs[-1]:.6f} "
                  f"test_accuracy={acc_runs[-1]:.6f}")

    if IS_RANK0:
        def _fmt(vals):
            vals = [v for v in vals if v == v]
            if not vals:
                return "n=0"
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return (f"mean={statistics.mean(vals):.6f} max={max(vals):.6f} "
                    f"min={min(vals):.6f} std={sd:.6f} (n={len(vals)})")
        print(f"\nTEST-ONLY SUMMARY ckpt={ckpt_to_load} protocol={args.split_protocol}")
        print(f"  test_mIoU     : {_fmt(miou_runs)}")
        print(f"  test_accuracy : {_fmt(acc_runs)}")
    sys.exit(0)

# =============================================================================
# TRAIN
# =============================================================================
trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
trainer.test(model, datamodule=data_module)

if wandb_logger and IS_RANK0:
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/planteur_{args.xp_name}_{proto_slug}.txt", "w") as f:
        f.write(wandb.run.id)
