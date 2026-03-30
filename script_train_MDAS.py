"""
MDAS Training Script — Multi-Sensor Segmentation
==================================================

Train Atomizer-IO on MDAS with one or more sensors simultaneously.
Supports cross-sensor zero-shot transfer evaluation.

Examples:
    # Single sensor: HySpex only
    python train_mdas.py --xp_name exp_hyspex \
        --train_sensors hyspex --test_sensor hyspex

    # Multi-sensor: HySpex + EnMAP 10m (recommended)
    python train_mdas.py --xp_name exp_multi \
        --train_sensors hyspex enmap_10m --test_sensor sentinel2

    # All four experiments from one multi-sensor checkpoint:
    python eval_mdas.py --ckpt_path best.ckpt --sensor hyspex     --output_dir results/exp1
    python eval_mdas.py --ckpt_path best.ckpt --sensor enmap_10m  --output_dir results/exp2
    python eval_mdas.py --ckpt_path best.ckpt --sensor sentinel2  --output_dir results/exp3
    python eval_mdas.py --ckpt_path best.ckpt --sensor enmap_30m  --output_dir results/exp4
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
from training.utils.datasets.utils_dataset_MDAS import MDASSegmentation, create_mdas_bands_info, register_mdas_bands
from training.utils.datasets.token_grouping import collate_multitask


# =============================================================================
# DATAMODULE
# =============================================================================

class MDASDataModule(pl.LightningDataModule):
    """
    MDAS segmentation DataModule with multi-sensor training support.

    When multiple train_sensors are provided, creates one dataset per
    sensor and concatenates them. With batch_size=1 + gradient accumulation,
    each sample is processed independently regardless of token count
    (HySpex: 1.5M tokens, EnMAP: 47K tokens).
    """

    def __init__(
        self,
        root: str,
        train_sensors: list,
        test_sensor: str,
        train_sub_areas: list,
        test_sub_areas: list,
        crop_index_path: str,
        stats_path: str,
        spectral_meta_path: str,
        config_model: dict,
        look_up,
        dataset_config: dict,
        batch_size: int = 1,
        num_workers: int = 4,
        max_queries: int = 65_536,
        val_sub_areas: list = None,
        resolution_augment_factors: list = None,
        spectral_configs: list = None,
        max_spectral_group_size: int = 35,
        crop_size_ref: int = 64,
    ):
        super().__init__()
        self.root = root
        self.train_sensors = train_sensors
        self.test_sensor = test_sensor
        self.train_sub_areas = train_sub_areas
        self.val_sub_areas = val_sub_areas
        self.test_sub_areas = test_sub_areas
        self.crop_index_path = crop_index_path
        self.stats_path = stats_path
        self.spectral_meta_path = spectral_meta_path
        self.config_model = config_model
        self.look_up = look_up
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queries = max_queries
        self.resolution_augment_factors = resolution_augment_factors
        self.spectral_configs = spectral_configs
        self.max_spectral_group_size = max_spectral_group_size
        self.crop_size_ref = crop_size_ref

    def setup(self, stage=None):
        # Guard against double setup (Lightning calls setup() again)
        if hasattr(self, '_setup_done') and self._setup_done:
            return
        self._setup_done = True

        common = dict(
            root=self.root,
            crop_index_path=self.crop_index_path,
            stats_path=self.stats_path,
            spectral_meta_path=self.spectral_meta_path,
            look_up=self.look_up,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            max_queries=self.max_queries,
            crop_size_ref=self.crop_size_ref,
        )

        # ── Build one dataset per training sensor ────────────────────
        train_datasets = []
        val_datasets = []

        for sensor in self.train_sensors:
            if self.val_sub_areas is not None:
                # Separate sub_areas for val
                train_ds = MDASSegmentation(
                    sensor=sensor,
                    sub_areas=self.train_sub_areas,
                    mode="train",
                    augment=True,
                    resolution_augment_factors=self.resolution_augment_factors,
                    spectral_configs=self.spectral_configs,
                    max_spectral_group_size=self.max_spectral_group_size,
                    **common,
                )

                val_ds = MDASSegmentation(
                    sensor=sensor,
                    sub_areas=self.val_sub_areas,
                    mode="test",
                    augment=False,
                    **common,
                )
            else:
                # 90/10 split from training sub_areas
                full_ds = MDASSegmentation(
                    sensor=sensor,
                    sub_areas=self.train_sub_areas,
                    mode="train",
                    augment=True,
                    resolution_augment_factors=self.resolution_augment_factors,
                    spectral_configs=self.spectral_configs,
                    max_spectral_group_size=self.max_spectral_group_size,
                    **common,
                )

                val_ds = MDASSegmentation(
                    sensor=sensor,
                    sub_areas=self.train_sub_areas,
                    mode="train",
                    augment=False,
                    **common,
                )

                # Deterministic 90/10 split
                full_len = len(full_ds)
                val_len = max(8, int(full_len * 0.1))
                train_len = full_len - val_len

                generator = torch.Generator().manual_seed(42)
                all_indices = torch.randperm(full_len, generator=generator).tolist()

                full_ds.crops = [full_ds.crops[i] for i in all_indices[:train_len]]
                val_ds.crops = [val_ds.crops[i] for i in all_indices[train_len:]]

                train_ds = full_ds

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

            val_info = f"sub_areas={self.val_sub_areas}" if self.val_sub_areas else "10% held-out"
            print(f"[MDAS-DM] Train {sensor}: {len(train_ds)} crops "
                  f"(sub_areas={self.train_sub_areas})")
            print(f"[MDAS-DM] Val   {sensor}: {len(val_ds)} crops "
                  f"({val_info})")

        # ── Concatenate across sensors ───────────────────────────────
        if len(train_datasets) == 1:
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
        else:
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)

        # ── Test: specified test sensor ──────────────────────────────
        self.test_dataset = MDASSegmentation(
            sensor=self.test_sensor,
            sub_areas=self.test_sub_areas,
            mode="test",
            augment=False,
            **common,
        )

        total_train = sum(len(d) for d in train_datasets)
        total_val = sum(len(d) for d in val_datasets)
        val_label = f"SA{self.val_sub_areas}" if self.val_sub_areas else "held-out 10%"
        print(f"[MDAS-DM] Total train: {total_train} crops "
              f"({'+'.join(self.train_sensors)}, SA{self.train_sub_areas})")
        print(f"[MDAS-DM] Total val:   {total_val} crops ({val_label})")
        print(f"[MDAS-DM] Test:        {len(self.test_dataset)} crops "
              f"({self.test_sensor}, SA{self.test_sub_areas})")

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

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="MDAS Segmentation Training")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str, default="config_test-Atomiser_Atos_One.yaml")

# Sensors
parser.add_argument("--train_sensors",  type=str, nargs="+", default=["hyspex"],
                    help="One or more sensors for training "
                         "(e.g., --train_sensors hyspex enmap_10m)")
parser.add_argument("--test_sensor",    type=str, default="hyspex",
                    choices=["hyspex", "enmap_10m", "enmap_30m", "sentinel2"])

# Sub-areas
parser.add_argument("--train_sub_areas", type=int, nargs="+", default=[1, 2])
parser.add_argument("--val_sub_areas",   type=int, nargs="+", default=None,
                    help="Sub-areas for validation. If not set, uses 10%% held-out from train.")
parser.add_argument("--test_sub_areas",  type=int, nargs="+", default=[3])

# Paths
parser.add_argument("--mdas_root",       type=str,
                    default="./data/MDAS/Augsburg_data_4_publication")
parser.add_argument("--crop_index",      type=str, default=None)
parser.add_argument("--stats",           type=str, default=None)
parser.add_argument("--spectral_meta",   type=str, default=None)

# Training
parser.add_argument("--ckpt_path",       type=str, default=None)
parser.add_argument("--num_workers",     type=int, default=4)
parser.add_argument("--max_queries",     type=int, default=65_536)
parser.add_argument("--grad_accum",      type=int, default=16)
parser.add_argument("--crop_size_ref",   type=int, default=64,
                    help="Crop size on the 2.2m reference grid. "
                         "64 → 14×14 at 10m, 128 → 28×28 at 10m")
parser.add_argument("--res_augment",     type=int, nargs="+", default=None,
                    help="Resolution augmentation pool factors "
                         "(e.g., --res_augment 1 2 3 4 5 for 2.2m to 11m)")
parser.add_argument("--spectral_configs", type=int, nargs="+", default=None,
                    help="Fixed spectral band counts for merging augmentation "
                         "(e.g., --spectral_configs 12 32 128 256 368)")
parser.add_argument("--max_group_size",  type=int, default=35,
                    help="Max bands per merged group (default: 35 ≈ 200nm for HySpex)")

args = parser.parse_args()

# Resolve default paths
if args.crop_index is None:
    args.crop_index = os.path.join(args.mdas_root, "mdas_crop_index.csv")
if args.stats is None:
    args.stats = os.path.join(args.mdas_root, "mdas_norm_stats.json")
if args.spectral_meta is None:
    args.spectral_meta = os.path.join(args.mdas_root, "mdas_spectral_meta.json")

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

# =============================================================================
# DATASET CONFIG — merge MDAS band definitions
# =============================================================================
dataset_config = read_yaml(bands_yaml_path)
mdas_bands = create_mdas_bands_info(args.spectral_meta)
dataset_config.update(mdas_bands)

# Pre-register base MDAS bands
register_mdas_bands(lookup_table, dataset_config)

sensors_str = "+".join(args.train_sensors)
print(f"[MDAS] Train sensors: {sensors_str}")
print(f"[MDAS] Test sensor:   {args.test_sensor}")
print(f"[MDAS] Registered band configs: {list(mdas_bands.keys())}")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"MDAS_{args.xp_name}_{sensors_str}→{args.test_sensor}",
        project="Atomizer_MDAS",
        config={
            **config_model,
            "train_sensors": args.train_sensors,
            "test_sensor": args.test_sensor,
            "train_sub_areas": args.train_sub_areas,
            "val_sub_areas": args.val_sub_areas,
            "test_sub_areas": args.test_sub_areas,
        },
    )
    wandb_logger = WandbLogger(project="Atomizer_MDAS")

# =============================================================================
# DATA MODULE — setup FIRST to register all spectral merge keys
# =============================================================================
# CRITICAL: setup() must run BEFORE Model_Pretrain() because:
# - Spectral merge configs (--spectral_configs 12 32 128 256 368) register
#   new quantized wave keys in the lookup table
# - Resolution augmentation registers new GSD values
# - Model_Pretrain builds embeddings sized to the lookup table
# If model is built first, the codebook is too small → CUDA index OOB
# =============================================================================
data_module = MDASDataModule(
    root=args.mdas_root,
    train_sensors=args.train_sensors,
    test_sensor=args.test_sensor,
    train_sub_areas=args.train_sub_areas,
    val_sub_areas=args.val_sub_areas,
    test_sub_areas=args.test_sub_areas,
    crop_index_path=args.crop_index,
    stats_path=args.stats,
    spectral_meta_path=args.spectral_meta,
    config_model=config_model,
    look_up=lookup_table,
    dataset_config=dataset_config,
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=args.num_workers,
    max_queries=args.max_queries,
    resolution_augment_factors=args.res_augment,
    spectral_configs=args.spectral_configs,
    max_spectral_group_size=args.max_group_size,
    crop_size_ref=args.crop_size_ref,
)

data_module.setup()

print(f"[MDAS] Lookup table size after data setup: {len(lookup_table.table_wave)}")

# =============================================================================
# MODEL — now codebook will be the correct size
# =============================================================================
model = Model_Pretrain(
    config=config_model,
    wand=True,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")

checkpoint_best = ModelCheckpoint(
    dirpath="./checkpoints/mdas/",
    filename=f"mdas_{args.xp_name}-{{epoch:02d}}-{{val_mdas_segmentation_mIoU:.4f}}",
    monitor="val_mdas_segmentation_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

checkpoint_last = ModelCheckpoint(
    dirpath="./checkpoints/mdas/",
    filename=f"mdas_{args.xp_name}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    verbose=True,
)

callbacks = [checkpoint_best, checkpoint_last, lr_monitor]

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/mdas/",
    gradient_clip_val=1.0,
    accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# WARMUP: Precompute all Voronoi caches
# =============================================================================

def warmup_voronoi_caches(model, data_module, device):
    """
    Run one forward pass for each unique (spectral_config × res_factor)
    combination to populate the Voronoi cache before training.
    """
    from training.utils.datasets.token_grouping import collate_multitask
    
    print("\n" + "=" * 60)
    print("  Warming up Voronoi caches")
    print("=" * 60)
    
    model.eval()
    ds = data_module.train_dataset
    
    # Get one base sample (no augmentation)
    original_augment = ds.augment
    original_spectral = ds._spectral_merge_configs
    original_res = ds.augment_gsd_map
    
    ds.augment = False
    ds._spectral_merge_configs = None
    ds.augment_gsd_map = None
    
    base_sample = ds[0]
    
    ds.augment = original_augment
    ds._spectral_merge_configs = original_spectral
    ds.augment_gsd_map = original_res
    
    base_image = ds._read_sensor_crop(
        ds.crops[0]["sub_area"], ds.crops[0]["r0"], ds.crops[0]["c0"]
    )
    base_label = ds._read_label_crop(
        ds.crops[0]["sub_area"], ds.crops[0]["r0"], ds.crops[0]["c0"]
    )
    
    # Collect all (spectral, resolution) combos
    spectral_list = ds.spectral_configs if ds.spectral_configs else [ds.n_bands]
    res_factors = ds.resolution_augment_factors if ds.resolution_augment_factors else [1]
    
    n_configs = len(spectral_list) * len(res_factors)
    print(f"  Configs to warm up: {len(spectral_list)} spectral × "
          f"{len(res_factors)} resolution = {n_configs}")
    
    with torch.no_grad():
        for si, n_spec in enumerate(spectral_list):
            # Merge bands if needed
            if ds._spectral_merge_configs and n_spec in ds._spectral_merge_configs:
                img, spec_idx = ds._apply_spectral_config(base_image.clone(), n_spec)
            else:
                img = base_image.clone()
                spec_idx = ds.spectral_indices
            
            for ri, factor in enumerate(res_factors):
                # Downsample if needed
                if factor > 1:
                    from training.utils.datasets.utils_dataset_MDAS import (
                        downsample_image, downsample_label_majority,
                    )
                    img_ds = downsample_image(img, factor)
                    label_ds = downsample_label_majority(base_label.clone(), factor)
                else:
                    img_ds = img
                    label_ds = base_label.clone()
                
                gsd = ds.sensor_gsd * factor
                res_idx = ds.look_up.get_resolution_idx(gsd)
                
                # Build tokens
                tokens = ds.token_builder.build_tokens(
                    image=img_ds.contiguous(),
                    label=label_ds.contiguous(),
                    resolution=gsd,
                    spectral_indices=spec_idx,
                    resolution_idx=res_idx,
                    time_idx=-1,
                )
                token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)
                
                H, W = img_ds.shape[1], img_ds.shape[2]
                n_bands = img_ds.shape[0]
                
                groups = {
                    gsd: {
                        "tokens": tokens,
                        "mask": token_mask,
                        "shape": (n_bands, H, W),
                    }
                }
                
                # Build dummy queries
                queries = ds.token_builder.build_queries(
                    label=label_ds.contiguous(),
                    resolution=gsd,
                    first_spectral_idx=spec_idx[0].item(),
                    resolution_idx=res_idx,
                    time_idx=-1,
                )
                queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
                
                sample = {
                    "groups": groups,
                    "tasks": {
                        "mdas_segmentation": {
                            "queries": queries,
                            "queries_mask": queries_mask,
                        },
                    },
                    "target_resolution": gsd,
                    "dataset_name": "MDAS",
                }
                
                # Collate and forward
                batch = collate_multitask([sample])
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                # Move nested dicts to device
                for key in ["groups", "tasks"]:
                    if key in batch and isinstance(batch[key], dict):
                        for k2 in batch[key]:
                            if isinstance(batch[key][k2], dict):
                                for k3 in batch[key][k2]:
                                    if isinstance(batch[key][k2][k3], torch.Tensor):
                                        batch[key][k2][k3] = batch[key][k2][k3].to(device)
                            elif isinstance(batch[key][k2], torch.Tensor):
                                batch[key][k2] = batch[key][k2].to(device)
                
                try:
                    _ = model.forward_multitask(batch, training=False)
                    n_tokens = tokens.shape[0]
                    print(f"  [{si*len(res_factors)+ri+1}/{n_configs}] "
                          f"spec={n_spec}, factor={factor} → "
                          f"{n_tokens:,} tokens ✓")
                except Exception as e:
                    print(f"  [{si*len(res_factors)+ri+1}/{n_configs}] "
                          f"spec={n_spec}, factor={factor} → FAILED: {e}")
    
    model.train()
    print(f"  Voronoi warmup complete.\n")


device = torch.device("cpu")
warmup_voronoi_caches(model, data_module, device)

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
    with open(f"training/wandb_runs/mdas_{args.xp_name}.txt", "w") as f:
        f.write(run_id)