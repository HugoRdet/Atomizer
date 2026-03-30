"""
C2Seg Training Script — Cross-City Multi-Sensor Segmentation
=============================================================

Train Atomizer-IO on C2Seg with all sensors simultaneously.
  - Germany: train Augsburg → val/test Berlin
  - China:   train Beijing  → val/test Wuhan

Supports multi-city training via --extra_train to combine cities
from different subsets (e.g., Berlin + Wuhan → test Augsburg).

Validation uses the VAL split for early stopping.

Examples:
    # Single city, from scratch
    python train_c2seg.py --xp_name germany_scratch \
        --subset germany --train_sensors hsi msi sar --fusion

    # Multi-city: Augsburg + Wuhan → test Berlin
    python train_c2seg.py --xp_name aug_wuhan_fusion \
        --subset germany --train_sensors hsi msi sar --fusion \
        --extra_train china:wuhan:./data/CrossCity/China/wuhan.mat \
        --max_queries 16384 --query_upsample_factor 4
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
from training.utils.datasets.utils_dataset_C2SEG import (
    C2SegDataset, create_c2seg_bands_info, register_c2seg_bands,
    preregister_spectral_merges,
)
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# SUBSET CONFIGURATION
# =============================================================================

SUBSET_CONFIG = {
    "germany": {
        "train_city": "augsburg",
        "test_city": "berlin",
        "train_mat": "augsburg_multimodal.mat",
        "test_mat": "berlin_multimodal.mat",
        "available_sensors": ["hsi", "msi", "sar"],
    },
    "china": {
        "train_city": "beijing",
        "test_city": "wuhan",
        "train_mat": "beijing.mat",
        "test_mat": "wuhan.mat",
        "available_sensors": ["hsi", "msi", "sar"],
    },
}

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048, 2.5: 2048, 4.78: 2048, 5.0: 2048,
    10.0: 2048, 20.0: 2048, 30.0: 2048, 60.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# PRETRAINED ENCODER LOADING
# =============================================================================

def load_pretrained_encoder(model, ckpt_path):
    print(f"\n{'='*60}")
    print(f"  Loading pretrained encoder from: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        full_state = ckpt["state_dict"]
        encoder_state = {k[len("encoder."):]: v
                         for k, v in full_state.items() if k.startswith("encoder.")}
        print(f"  Extracted {len(encoder_state)} encoder keys from Lightning checkpoint")
    elif "encoder" in ckpt:
        encoder_state = ckpt["encoder"]
        print(f"  Loaded {len(encoder_state)} encoder keys from raw checkpoint")
    else:
        raise ValueError(f"Checkpoint keys: {list(ckpt.keys())}")

    model_state = model.encoder.state_dict()
    compatible = {}
    skipped = []
    for k, v in encoder_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
        elif k in model_state:
            skipped.append((k, v.shape, model_state[k].shape))

    result = model.encoder.load_state_dict(compatible, strict=False)
    print(f"  Loaded: {len(compatible)}, Skipped: {len(skipped)}, "
          f"Fresh: {len(result.missing_keys) - len(skipped)}")
    for k, s, d in skipped:
        print(f"    - {k}: {s} ≠ {d}")
    print(f"{'='*60}\n")
    return model


# =============================================================================
# EXTRA TRAIN PARSER
# =============================================================================

def parse_extra_train(extra_train_args):
    """
    Parse --extra_train arguments.
    Format: subset:city:mat_path
    Example: china:wuhan:./data/CrossCity/China/wuhan.mat
    """
    if not extra_train_args:
        return []
    extra = []
    for arg in extra_train_args:
        parts = arg.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--extra_train format is subset:city:mat_path, got '{arg}'\n"
                f"Example: china:wuhan:./data/CrossCity/China/wuhan.mat")
        extra.append({"subset": parts[0], "city": parts[1], "mat_path": parts[2]})
    return extra


# =============================================================================
# DATAMODULE
# =============================================================================

class C2SegDataModule(pl.LightningDataModule):

    def __init__(
        self, subset, data_dir, train_sensors, test_sensors,
        crop_index_path, stats_path, spectral_meta_path,
        look_up, dataset_config,
        batch_size=1, num_workers=4, max_queries=16_384,
        resolution_augment_factors=None, fusion=False,
        query_upsample_factor=0, query_upsample_fraction=0.5,
        query_upsample_prob=0.5, query_boundary_fraction=0.7,
        query_boundary_dilation=2,
        spectral_aug_prob=0.0, spectral_aug_groups=None,
        spectral_aug_pool=None,
        norm_mode="raw",
        test_subset=None, test_city=None, test_mat=None,
        train_city=None, train_mat=None,
        extra_train_configs=None,
    ):
        super().__init__()
        self.subset = subset
        self.data_dir = data_dir
        self.train_sensors = train_sensors
        self.test_sensors = test_sensors
        self.crop_index_path = crop_index_path
        self.stats_path = stats_path
        self.spectral_meta_path = spectral_meta_path
        self.look_up = look_up
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queries = max_queries
        self.resolution_augment_factors = resolution_augment_factors
        self.fusion = fusion
        self.extra_train_configs = extra_train_configs or []

        self.query_kwargs = {}
        if query_upsample_factor > 1:
            self.query_kwargs = {
                "query_upsample_factor": query_upsample_factor,
                "query_upsample_fraction": query_upsample_fraction,
                "query_upsample_prob": query_upsample_prob,
                "query_boundary_fraction": query_boundary_fraction,
                "query_boundary_dilation": query_boundary_dilation,
            }
        # Spectral augmentation (flows through **kwargs to dataset)
        if spectral_aug_prob > 0:
            self.query_kwargs["spectral_aug_prob"] = spectral_aug_prob
            self.query_kwargs["spectral_aug_groups"] = (
                spectral_aug_groups or [32, 64, 128]
            )
            if spectral_aug_pool is not None:
                self.query_kwargs["spectral_aug_pool"] = spectral_aug_pool

        # Normalization mode (must be consistent across train/val/test)
        self.norm_mode = norm_mode

        self.test_subset = test_subset or subset
        self.test_city = test_city
        self.test_mat = test_mat

        cfg = SUBSET_CONFIG[subset]
        self.train_city = train_city or cfg["train_city"]
        if self.test_city is None:
            self.test_city = cfg["test_city"]
        self.train_mat_path = (train_mat if train_mat
                               else os.path.join(data_dir, cfg["train_mat"]))
        self.test_mat_path = (self.test_mat if self.test_mat
                              else os.path.join(data_dir, cfg["test_mat"]))

    def _make_train_ds(self, mat_path, subset, city, sensors):
        """Create training dataset(s) for one city."""
        common = dict(
            crop_index_path=self.crop_index_path,
            stats_path=self.stats_path,
            spectral_meta_path=self.spectral_meta_path,
            look_up=self.look_up,
            dataset_config=self.dataset_config,
            max_queries=self.max_queries,
        )
        if self.norm_mode != "raw":
            common["norm_mode"] = self.norm_mode
        kwargs = {**common, **self.query_kwargs}

        if self.fusion:
            ds = C2SegDataset(
                mat_path=mat_path, subset=subset, city=city,
                split="train", sensors=sensors, mode="train", augment=True,
                resolution_augment_factors=self.resolution_augment_factors,
                **kwargs)
            print(f"[C2Seg-DM] Train fusion [{'+'.join(sensors)}]: "
                  f"{len(ds)} crops ({city})")
            return ds
        else:
            datasets = []
            for sensor in sensors:
                ds = C2SegDataset(
                    mat_path=mat_path, subset=subset, city=city,
                    split="train", sensors=[sensor], mode="train", augment=True,
                    resolution_augment_factors=self.resolution_augment_factors,
                    **kwargs)
                datasets.append(ds)
                print(f"[C2Seg-DM] Train {sensor}: {len(ds)} crops ({city})")
            return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def setup(self, stage=None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return
        self._setup_done = True

        common = dict(
            crop_index_path=self.crop_index_path,
            stats_path=self.stats_path,
            spectral_meta_path=self.spectral_meta_path,
            look_up=self.look_up,
            dataset_config=self.dataset_config,
            max_queries=self.max_queries,
        )
        # norm_mode must be in common so val/test datasets match training
        if self.norm_mode != "raw":
            common["norm_mode"] = self.norm_mode

        # ── Training: primary city + extra cities ────────────────────
        all_train = []
        all_train.append(self._make_train_ds(
            self.train_mat_path, self.subset, self.train_city, self.train_sensors))

        for ec in self.extra_train_configs:
            print(f"\n[C2Seg-DM] Adding extra training city: {ec['city']} ({ec['subset']})")
            all_train.append(self._make_train_ds(
                ec["mat_path"], ec["subset"], ec["city"], self.train_sensors))

        self.train_dataset = ConcatDataset(all_train) if len(all_train) > 1 else all_train[0]

        # ── Validation: held-out val split ──────────────────────────
        if self.fusion:
            self.val_dataset = C2SegDataset(
                mat_path=self.test_mat_path, subset=self.test_subset,
                city=self.test_city, split="val",
                sensors=self.train_sensors, mode="test", augment=False, **common)
            print(f"[C2Seg-DM] Val   fusion [{'+'.join(self.train_sensors)}]: "
                  f"{len(self.val_dataset)} crops ({self.test_city})")
        else:
            vds = []
            for sensor in self.train_sensors:
                vd = C2SegDataset(
                    mat_path=self.test_mat_path, subset=self.test_subset,
                    city=self.test_city, split="val",
                    sensors=[sensor], mode="test", augment=False, **common)
                vds.append(vd)
                print(f"[C2Seg-DM] Val   {sensor}: {len(vd)} crops ({self.test_city})")
            self.val_dataset = ConcatDataset(vds) if len(vds) > 1 else vds[0]

        # ── Test datasets ────────────────────────────────────────────
        self.test_datasets = {}
        for sensor in self.test_sensors:
            self.test_datasets[sensor] = C2SegDataset(
                mat_path=self.test_mat_path, subset=self.test_subset,
                city=self.test_city, split="test",
                sensors=[sensor], mode="test", augment=False, **common)
            print(f"[C2Seg-DM] Test  {sensor}: "
                  f"{len(self.test_datasets[sensor])} crops ({self.test_city})")

        if self.fusion and len(self.test_sensors) > 1:
            fk = "+".join(self.test_sensors)
            self.test_datasets[fk] = C2SegDataset(
                mat_path=self.test_mat_path, subset=self.test_subset,
                city=self.test_city, split="test",
                sensors=self.test_sensors, mode="test", augment=False, **common)
            print(f"[C2Seg-DM] Test  fusion [{fk}]: "
                  f"{len(self.test_datasets[fk])} crops ({self.test_city})")

        # ── Summary ──────────────────────────────────────────────────
        train_cities = [self.train_city] + [ec["city"] for ec in self.extra_train_configs]
        print(f"\n[C2Seg-DM] Summary:")
        print(f"  Train: {len(self.train_dataset)} crops "
              f"({'+'.join(self.train_sensors)}, {' + '.join(train_cities)})")
        print(f"  Val:   {len(self.val_dataset)} crops ({self.test_city}) "
              f"← VAL split for early stopping")
        print(f"  Test:  {sum(len(d) for d in self.test_datasets.values())} crops "
              f"({'+'.join(self.test_sensors)}, {self.test_city})")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(
            self.test_datasets[self.test_sensors[0]], shuffle=False)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="C2Seg Training")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str,
                    default="config_test-Atomiser_Atos_One.yaml")
parser.add_argument("--subset",         type=str, required=True,
                    choices=["germany", "china"])
parser.add_argument("--train_sensors",  type=str, nargs="+",
                    default=["hsi", "msi", "sar"])
parser.add_argument("--test_sensors",   type=str, nargs="+", default=None)
parser.add_argument("--data_dir",       type=str, default=None)
parser.add_argument("--processed_dir",  type=str,
                    default="./data/CrossCity/c2seg_processed")
parser.add_argument("--test_subset",    type=str, default=None,
                    choices=["germany", "china"])
parser.add_argument("--test_city",      type=str, default=None)
parser.add_argument("--test_mat",       type=str, default=None)
parser.add_argument("--train_city",     type=str, default=None,
                    help="Override train city (default: from subset config)")
parser.add_argument("--train_mat",      type=str, default=None,
                    help="Override train mat file path")
parser.add_argument("--extra_train",    type=str, nargs="+", default=None,
                    help="Extra training cities. Format: subset:city:mat_path")
parser.add_argument("--ckpt_path",      type=str, default=None)
parser.add_argument("--pretrained_encoder", type=str, default=None)
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--max_queries",    type=int, default=16_384)
parser.add_argument("--grad_accum",     type=int, default=1)
parser.add_argument("--res_augment",    type=int, nargs="+", default=None)
parser.add_argument("--fusion",         action="store_true")
parser.add_argument("--query_upsample_factor", type=int, default=0)
parser.add_argument("--query_upsample_fraction", type=float, default=0.5)
parser.add_argument("--query_upsample_prob", type=float, default=0.5)
parser.add_argument("--query_boundary_fraction", type=float, default=0.7)
parser.add_argument("--query_boundary_dilation", type=int, default=2)

# Spectral augmentation (simulates real sensor configs: S2, Landsat, MODIS, etc.)
parser.add_argument("--spectral_aug_prob", type=float, default=0.0,
                    help="Probability of sensor simulation per sample (0=off)")
parser.add_argument("--spectral_aug_groups", type=int, nargs="+",
                    default=[32, 64, 128],
                    help="Additional uniform group counts (supplements sensor library)")

# Normalization
parser.add_argument("--norm_mode", type=str, default="raw",
                    choices=["raw", "band_minmax", "zscore"],
                    help="raw: ÷10000+clamp (default). band_minmax: per-band min-max. "
                         "zscore: per-band z-score, sensor-agnostic")

args = parser.parse_args()

if args.data_dir is None:
    subset_dir = "Germany" if args.subset == "germany" else "China"
    args.data_dir = f"./data/CrossCity/{subset_dir}"
if args.test_sensors is None:
    args.test_sensors = SUBSET_CONFIG[args.subset]["available_sensors"]

extra_train_configs = parse_extra_train(args.extra_train)

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

spectral_meta_path = os.path.join(args.processed_dir, "c2seg_spectral_meta.json")
dataset_config = read_yaml(bands_yaml_path)
c2seg_bands = create_c2seg_bands_info(spectral_meta_path)
dataset_config.update(c2seg_bands)
register_c2seg_bands(lookup_table, dataset_config)

# Pre-register all possible merged spectral entries for augmentation
_spectral_aug_pool = None
if args.spectral_aug_prob > 0:
    _spectral_aug_pool = preregister_spectral_merges(
        lookup_table, spectral_meta_path,
        groups=args.spectral_aug_groups,
        sensors=args.train_sensors,
    )

# Print config
sensors_str = "+".join(args.train_sensors)
test_sensors_str = "+".join(args.test_sensors)
primary_train_city = args.train_city or SUBSET_CONFIG[args.subset]["train_city"]
train_cities = [primary_train_city] + [e["city"] for e in extra_train_configs]

print(f"\n[C2Seg] Subset:        {args.subset}")
print(f"[C2Seg] Train cities:  {' + '.join(train_cities)}")
print(f"[C2Seg] Train sensors: {sensors_str}")
print(f"[C2Seg] Test sensors:  {test_sensors_str}")
print(f"[C2Seg] Max queries:   {args.max_queries}")
if extra_train_configs:
    for ec in extra_train_configs:
        print(f"[C2Seg] Extra train:   {ec['city']} ({ec['subset']}) → {ec['mat_path']}")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    cfg = SUBSET_CONFIG[args.subset]
    pretrain_tag = "BEN" if args.pretrained_encoder else "scratch"
    run_name = (f"C2Seg_{args.xp_name}_{'+'.join(train_cities)}_"
                f"{sensors_str}→{cfg['test_city']}_{pretrain_tag}")
    wandb.init(
        name=run_name, project="Atomizer_C2Seg",
        config={**config_model, "subset": args.subset,
                "train_sensors": args.train_sensors,
                "test_sensors": args.test_sensors,
                "train_cities": train_cities,
                "test_city": cfg["test_city"],
                "extra_train": [e["city"] for e in extra_train_configs],
                "fusion": args.fusion,
                "max_queries": args.max_queries,
                "query_upsample_factor": args.query_upsample_factor})
    wandb_logger = WandbLogger(project="Atomizer_C2Seg")

# =============================================================================
# DATA MODULE
# =============================================================================
crop_index_path = os.path.join(args.processed_dir, "c2seg_crop_index_split.csv")
stats_path = os.path.join(args.processed_dir, "c2seg_norm_stats.json")

data_module = C2SegDataModule(
    subset=args.subset, data_dir=args.data_dir,
    train_sensors=args.train_sensors, test_sensors=args.test_sensors,
    crop_index_path=crop_index_path, stats_path=stats_path,
    spectral_meta_path=spectral_meta_path,
    look_up=lookup_table, dataset_config=dataset_config,
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=args.num_workers, max_queries=args.max_queries,
    resolution_augment_factors=args.res_augment, fusion=args.fusion,
    query_upsample_factor=args.query_upsample_factor,
    query_upsample_fraction=args.query_upsample_fraction,
    query_upsample_prob=args.query_upsample_prob,
    query_boundary_fraction=args.query_boundary_fraction,
    query_boundary_dilation=args.query_boundary_dilation,
    spectral_aug_prob=args.spectral_aug_prob,
    spectral_aug_groups=args.spectral_aug_groups,
    spectral_aug_pool=_spectral_aug_pool,
    norm_mode=args.norm_mode,
    test_subset=args.test_subset, test_city=args.test_city,
    test_mat=args.test_mat,
    train_city=args.train_city, train_mat=args.train_mat,
    extra_train_configs=extra_train_configs,
)
data_module.setup()
print(f"[C2Seg] Lookup table: {len(lookup_table.table_wave)} entries")

# =============================================================================
# MODEL
# =============================================================================
model = Model_Pretrain(
    config=config_model, wand=True, name=args.xp_name,
    transform=None, lookup_table=lookup_table)

if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================
ckpt_dir = f"./checkpoints/c2seg/{args.subset}/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"c2seg_{args.xp_name}-{{epoch:02d}}-{{val_c2seg_segmentation_mIoU:.4f}}",
        monitor="val_c2seg_segmentation_mIoU", mode="max", save_top_k=1, verbose=True),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"c2seg_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1, save_top_k=1, save_last=True, verbose=True),
    LearningRateMonitor(logging_interval="step"),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1, max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    gradient_clip_val=1.0, accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# TRAIN
# =============================================================================
cfg = SUBSET_CONFIG[args.subset]
print(f"\n{'='*60}")
print(f"  Train: {' + '.join(train_cities)} ({sensors_str})")
print(f"  Val:   {data_module.test_city} ← early stopping")
print(f"  Test:  {data_module.test_city} ({test_sensors_str})")
print(f"{'='*60}\n")

trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/c2seg_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)