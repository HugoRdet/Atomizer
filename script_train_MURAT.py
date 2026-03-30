"""
MuRA-T Training Script — Cross-Sensor Building Segmentation
=============================================================

Train Atomizer-IO on MuRA-T/SpaceNet-7 with leave-one-out sensor protocol.

Examples:
    # Upper bound: all three sensors
    python train_murat.py --xp_name all_sensors \
        --train_sensors planet sentinel2 landsat8

    # Leave-out Planet
    python train_murat.py --xp_name leave_planet \
        --train_sensors sentinel2 landsat8

    # Leave-out Landsat
    python train_murat.py --xp_name leave_landsat \
        --train_sensors sentinel2 planet

    # Leave-out S2
    python train_murat.py --xp_name leave_s2 \
        --train_sensors landsat8 planet

    # Eval (same checkpoint on all sensors)
    python eval_murat.py --ckpt_path best.ckpt --sensor planet
    python eval_murat.py --ckpt_path best.ckpt --sensor sentinel2
    python eval_murat.py --ckpt_path best.ckpt --sensor landsat8
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_MURAT import MuRATSegmentation
from training.utils.datasets.token_grouping import collate_multitask


# =============================================================================
# BAND & RESOLUTION REGISTRATION
# =============================================================================

def register_murat_bands(lookup_table):
    """
    Pre-register all MuRA-T sensor bands into the lookup table.
    Must be called BEFORE Model_Pretrain so the spectral encoder
    codebook is built at the correct size.
    """
    from training.utils.datasets.utils_dataset_MURAT import (
        PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO,
    )

    n_new = 0
    for bands_info in [PLANET_BANDS_INFO, S2_BANDS_INFO, LANDSAT_BANDS_INFO]:
        for band_name, data in bands_info.items():
            bw = int(data["bandwidth"])
            wl = int(data["central_wavelength"])
            key = (bw, wl)
            if key not in lookup_table.table_wave:
                lookup_table.table_wave[key] = len(lookup_table.table_wave)
                n_new += 1

    print(f"[MuRA-T] Pre-registered {n_new} new bands "
          f"(total: {len(lookup_table.table_wave)})")
    return n_new


def register_murat_resolutions(lookup_table):
    """
    Pre-register all MuRA-T resolutions into the lookup table.
    Must be called BEFORE Model_Pretrain so geometry buffers
    are allocated at the correct size.

    Uses register_resolution() to add entries to table_resolution
    (for the resolution encoder) and register_modality() to add
    entries to the position/query tables (for coordinate offsets).

    All GSDs are rounded to 2 decimals to prevent float-precision
    cache misses in Voronoi computation.
    """
    from training.utils.datasets.utils_dataset_MURAT import (
        SENSOR_QUERY_RES, S2_RES_GROUPS,
    )

    all_gsds = set()
    all_gsds.add(round(SENSOR_QUERY_RES["planet"], 2))    # 4.78
    all_gsds.add(round(SENSOR_QUERY_RES["landsat8"], 2))  # 30.0
    for gsd in S2_RES_GROUPS.keys():                       # 10.0, 20.0, 60.0
        all_gsds.add(round(gsd, 2))

    for gsd in sorted(all_gsds):
        # Register in resolution table (for resolution encoder)
        lookup_table.register_resolution(gsd)
        # Register in position/query tables (for coordinate offsets)
        lookup_table.register_modality(gsd, 2048)

    print(f"[MuRA-T] Pre-registered resolutions:")
    for gsd in sorted(all_gsds):
        idx = lookup_table.get_resolution_idx(gsd)
        print(f"  {gsd:>8.2f} m/px → idx {idx}")


# =============================================================================
# DATAMODULE
# =============================================================================

class MuRATDataModule(pl.LightningDataModule):
    """
    MuRA-T DataModule with leave-one-out sensor support.

    Creates one dataset per training sensor and concatenates them.
    Each sample = one (AOI, sensor, month) triplet.
    """

    def __init__(
        self,
        index_csv: str,
        stats_json: str,
        look_up,
        train_sensors: list,
        config_model: dict,
        batch_size: int = 1,
        num_workers: int = 4,
        max_queries: int = 65_536,
        label_cache_dir: str = None,
        data_root: str = None,
    ):
        super().__init__()
        self.index_csv = index_csv
        self.stats_json = stats_json
        self.look_up = look_up
        self.train_sensors = train_sensors
        self.config_model = config_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queries = max_queries
        self.label_cache_dir = label_cache_dir
        self.data_root = data_root

    def setup(self, stage=None):
        common = dict(
            index_csv=self.index_csv,
            stats_json=self.stats_json,
            look_up=self.look_up,
            config_model=self.config_model,
            max_queries=self.max_queries,
            label_cache_dir=self.label_cache_dir,
            data_root=self.data_root,
        )

        # ── Training: one dataset per sensor, concatenated ───────────
        train_datasets = []
        for sensor in self.train_sensors:
            ds = MuRATSegmentation(
                mode="train",
                sensors=[sensor],
                augment=True,
                **common,
            )
            train_datasets.append(ds)
            print(f"[MuRA-T DM] Train {sensor}: {len(ds)} samples")

        if len(train_datasets) == 1:
            self.train_dataset = train_datasets[0]
        else:
            self.train_dataset = ConcatDataset(train_datasets)

        # ── Validation: same training sensors ────────────────────────
        val_datasets = []
        for sensor in self.train_sensors:
            ds = MuRATSegmentation(
                mode="val",
                sensors=[sensor],
                augment=False,
                **common,
            )
            val_datasets.append(ds)
            print(f"[MuRA-T DM] Val   {sensor}: {len(ds)} samples")

        if len(val_datasets) == 1:
            self.val_dataset = val_datasets[0]
        else:
            self.val_dataset = ConcatDataset(val_datasets)

        total_train = sum(len(d) for d in train_datasets)
        total_val = sum(len(d) for d in val_datasets)
        print(f"[MuRA-T DM] Total train: {total_train} "
              f"({'+'.join(self.train_sensors)})")
        print(f"[MuRA-T DM] Total val:   {total_val}")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="MuRA-T Cross-Sensor Training")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str, default="config_test-Atomiser_Atos_One.yaml")

# Sensors
parser.add_argument("--train_sensors",  type=str, nargs="+",
                    default=["planet", "sentinel2", "landsat8"],
                    help="Sensors for training "
                         "(e.g., --train_sensors sentinel2 landsat8)")

# Paths
parser.add_argument("--data_root",      type=str,
                    default="./data/MURAT")
parser.add_argument("--index_csv",      type=str, default=None)
parser.add_argument("--stats_json",     type=str, default=None)
parser.add_argument("--label_cache_dir", type=str, default=None)

# Training
parser.add_argument("--ckpt_path",      type=str, default=None,
                    help="Resume from checkpoint")
parser.add_argument("--pretrained_ckpt", type=str, default=None,
                    help="Load pretrained encoder weights (e.g., from FLAIR-HUB MAE)")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--max_queries",    type=int, default=65_536)
parser.add_argument("--grad_accum",     type=int, default=16)
parser.add_argument("--max_epochs",     type=int, default=None,
                    help="Override config epochs")

args = parser.parse_args()

# Resolve default paths
if args.index_csv is None:
    args.index_csv = os.path.join(args.data_root, "murat_index.csv")
if args.stats_json is None:
    args.stats_json = os.path.join(args.data_root, "murat_norm_stats.json")

# =============================================================================
# CONFIG
# =============================================================================
config_model = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path),
    read_yaml(bands_yaml_path),
    config_model,
)

# Pre-register MuRA-T bands and resolutions before model construction
register_murat_bands(lookup_table)
register_murat_resolutions(lookup_table)

sensors_str = "+".join(args.train_sensors)
print(f"[MuRA-T] Train sensors: {sensors_str}")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"MURAT_{args.xp_name}_{sensors_str}",
        project="Atomizer_MURAT",
        config={
            **config_model,
            "train_sensors": args.train_sensors,
            "xp_name": args.xp_name,
        },
    )
    wandb_logger = WandbLogger(project="Atomizer_MURAT")

# =============================================================================
# MODEL
# =============================================================================
model = Model_Pretrain(
    config=config_model,
    wand=True,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)

# Load pretrained encoder if provided (e.g., from MAE pretraining)
if args.pretrained_ckpt:
    model.load_encoder_for_downstream(args.pretrained_ckpt)
    print(f"[MuRA-T] Loaded pretrained encoder from {args.pretrained_ckpt}")

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = MuRATDataModule(
    index_csv=args.index_csv,
    stats_json=args.stats_json,
    look_up=lookup_table,
    train_sensors=args.train_sensors,
    config_model=config_model,
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=args.num_workers,
    max_queries=args.max_queries,
    label_cache_dir=args.label_cache_dir,
    data_root=args.data_root,
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")

checkpoint_best = ModelCheckpoint(
    dirpath="./checkpoints/murat/",
    filename=f"murat_{args.xp_name}-{{epoch:02d}}-{{val_murat_segmentation_mIoU:.4f}}",
    monitor="val_murat_segmentation_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

checkpoint_last = ModelCheckpoint(
    dirpath="./checkpoints/murat/",
    filename=f"murat_{args.xp_name}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    verbose=True,
)

callbacks = [checkpoint_best, checkpoint_last, lr_monitor]

# =============================================================================
# SETUP
# =============================================================================
data_module.setup()

# =============================================================================
# TRAINER
# =============================================================================
max_epochs = args.max_epochs or config_model["trainer"]["epochs"]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1,
    max_epochs=max_epochs,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/murat/",
    gradient_clip_val=1.0,
    accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# TRAIN
# =============================================================================
trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/murat_{args.xp_name}.txt", "w") as f:
        f.write(run_id)