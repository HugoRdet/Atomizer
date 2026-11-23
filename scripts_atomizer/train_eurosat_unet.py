"""
Atomiser EuroSAT / EuroSAT-SAR Training Script — U-Net or baseline
====================================================================

10-class land cover classification with the Model_ForestNet trainer.

    --dataset eurosat      geo-bench m-eurosat (EurosatDataset): 13 S2 bands, 64x64
    --dataset eurosat_sar  EuroSAT-MS + EuroSAT-SAR (EuroSATSARDataset): 13 S2 + VV/VH

The encoder is chosen by the trainer from the config: Atomiser_Unet when
config["encoder"] contains "unet" or Atomiser.unet_task is set, the baseline
Atomiser_Senflood otherwise. Same script, same trainer for both.

Changes vs script_train_eurosat_sar.py are tagged  # >>> UNET.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training entirely
    and run test on the provided checkpoint with a single GPU.

Modality ablation at inference (eurosat_sar):
    Set trainer.bands.drop: [VV, VH] in the config to zero+mask the SAR
    channels. Masked atoms keep their positions, so the U-Net geometry is
    unchanged; its leaf sampler skips masked atoms.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from collections import Counter

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    EarlyStopping,
)

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils.Lookup_positional import Lookup_encoding   # >>> UNET: direct import

# >>> UNET: trainer that builds Atomiser_Unet or the baseline from the config
#     (trainer_forestnet_unet.py saved as training/trainer_FORESTNET.py, or
#      change this import to training.trainer_forestnet_unet)
from training.trainer_FORESTNET import Model_ForestNet
from training.utils.datasets.dataloaders import UnifiedDataModule

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser EuroSAT / EuroSAT-SAR training")
parser.add_argument("--xp_name",      type=str, required=True)
# >>> UNET: choose the dataset and the config
parser.add_argument("--dataset", type=str, default="eurosat",
                    choices=["eurosat", "eurosat_sar"],
                    help="eurosat = geo-bench m-eurosat (13 S2 bands); "
                         "eurosat_sar = EuroSAT-MS + EuroSAT-SAR (S2 + VV/VH)")
parser.add_argument("--config", type=str,
                    default="./training/configs/config_test-EUROSAT_unet.yaml")
parser.add_argument("--skip_preflight", action="store_true",
                    help="Skip the token-layout sanity check")

parser.add_argument("--clipping",     action="store_true")
parser.add_argument("--use_class_weights", action="store_true")
parser.add_argument("--label_smoothing",   type=float, default=0.0)

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

parser.add_argument("--resume", action="store_true",
                    help="Resume training from the '-last' checkpoint for this "
                         "xp_name, if one exists. Restores full trainer state. "
                         "Ignored if --test_only is set.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Explicit checkpoint path to resume from (overrides "
                         "--resume auto-detection).")

args = parser.parse_args()

xp_name         = args.xp_name
config_model    = read_yaml(args.config)
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml      = "./data/bands_info/bands.yaml"


# =============================================================================
# >>> UNET: DATASET SELECTION
# =============================================================================

if args.dataset == "eurosat":
    from training.utils.datasets.utils_dataset_Eurosat import EurosatDataset as DatasetClass
    data_root = "./data/geo-bench-1.0/classification_v1.0/m-eurosat"
    n_atoms_per_image = 13 * 64 * 64
    ckpt_dir = "./checkpoints/eurosat/"
    wandb_project = "Atomizer_EuroSAT"
else:
    from training.utils.datasets.utils_dataset_EUROSAT_SAR import EuroSATSARDataset as DatasetClass
    data_root = "./data"   # parent of EuroSAT_MS/ and EuroSAT-SAR/
    n_atoms_per_image = 15 * 64 * 64
    ckpt_dir = "./checkpoints/eurosat_sar/"
    wandb_project = "Atomizer_EuroSAT_SAR"

# EuroSATSARDataset is registered in UnifiedDataModule.GROUPED_DATASET_CLASSES
# already; EurosatDataset may not be (the original ForestNet/EuroSAT scripts
# registered it manually). Register it if the registry is a set.
_registry = getattr(UnifiedDataModule, "GROUPED_DATASET_CLASSES", None)
if _registry is not None and DatasetClass not in _registry:
    if isinstance(_registry, set):
        UnifiedDataModule.GROUPED_DATASET_CLASSES |= {DatasetClass}
        if IS_RANK0:
            print(f"[Train] Registered {DatasetClass.__name__} in GROUPED_DATASET_CLASSES")
    elif IS_RANK0:
        print(f"[Train] WARNING: {DatasetClass.__name__} is not in "
              f"UnifiedDataModule.GROUPED_DATASET_CLASSES ({type(_registry).__name__}). "
              f"Register it the way script_train_eurosat.py does.")


# =============================================================================
# >>> UNET: CONFIG CHECKS
# =============================================================================

is_unet = ("unet" in str(config_model.get("encoder", "")).lower()
           or "unet_task" in config_model.get("Atomiser", {}))

if is_unet:
    a = config_model.setdefault("Atomiser", {})
    if a.get("unet_task", "classification") != "classification":
        raise ValueError(f"EuroSAT is classification but Atomiser.unet_task="
                         f"{a['unet_task']!r}")
    a["unet_task"] = "classification"
    a["use_error_predictor"] = False
    # distinct name -> checkpoints / wandb runs never collide with the baseline
    if "unet" not in str(config_model.get("encoder", "")).lower():
        config_model["encoder"] = f"{config_model.get('encoder', 'ATOMIZER')}_Unet"

# The shared hierarchy geometry requires every image to keep ALL its tokens in a
# fixed order. The datasets randomly subsample (and so reorder) tokens when an
# image has more than trainer.max_tokens.
max_tokens = int(config_model["trainer"]["max_tokens"])
if is_unet and max_tokens < n_atoms_per_image:
    raise ValueError(
        f"trainer.max_tokens={max_tokens} < {n_atoms_per_image} atoms per image: "
        f"the dataset would randomly subsample tokens per sample, which breaks the "
        f"U-Net's shared geometry. Set max_tokens >= {n_atoms_per_image}.")

if IS_RANK0:
    print(f"[Train] Dataset: {args.dataset} ({DatasetClass.__name__}) at {data_root}")
    print(f"[Train] Config:  {args.config}")
    print(f"[Train] Encoder: {config_model['encoder']} ({'U-Net' if is_unet else 'baseline'})")
    if args.test_only:
        print(f"[Train] Test-only mode: {args.test_only}")
    else:
        print(f"[Train] Gradient clipping: {'ON' if args.clipping else 'OFF'}")
        print(f"[Train] Class weights:     {'ON' if args.use_class_weights else 'OFF'}")
        print(f"[Train] Label smoothing:   {args.label_smoothing}")
        print(f"[Train] Resume:            "
              f"{'ON (' + (args.resume_from or 'auto-detect last ckpt') + ')' if (args.resume or args.resume_from) else 'OFF'}")
    bands_cfg = config_model.get("trainer", {}).get("bands", {})
    print(f"[Train] Bands keep: {bands_cfg.get('keep', 'ALL')}")
    print(f"[Train] Bands drop: {bands_cfg.get('drop', 'none')}")


# =============================================================================
# LOOKUP TABLE
# =============================================================================

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset),
    read_yaml(bands_yaml),
    config_model,
)


# =============================================================================
# DATA MODULE
# =============================================================================

data_module = UnifiedDataModule(
    path=data_root,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=DatasetClass,
)


# =============================================================================
# >>> UNET: PRE-FLIGHT — shared token layout across samples
# =============================================================================

def preflight_token_layout():
    ds = DatasetClass(root_path=data_root, mode="train",
                      dataset_config=read_yaml(bands_yaml),
                      config_model=config_model, look_up=lookup_table)
    s0, s1 = ds[0], ds[1 % len(ds)]
    for res in s0["groups"]:
        t0, t1 = s0["groups"][res]["tokens"], s1["groups"][res]["tokens"]
        if t0.shape != t1.shape or not torch.equal(t0[:, 1:3], t1[:, 1:3]):
            raise RuntimeError(
                f"[Preflight] group {res}: token positions differ between samples "
                f"(shapes {tuple(t0.shape)} vs {tuple(t1.shape)}). The U-Net's shared "
                f"geometry needs a fixed token layout. Rerun with --skip_preflight "
                f"only if you know why this is safe.")
        print(f"[Preflight] group {res}: tokens {tuple(t0.shape)}, "
              f"positions identical across samples ✓")
    lab = s0["label"]
    assert lab.dim() == 0, f"expected a scalar class label, got shape {tuple(lab.shape)}"
    print(f"[Preflight] label: scalar ✓  task: {s0.get('task')}")
    del ds


if is_unet and IS_RANK0 and not args.skip_preflight:
    preflight_token_layout()


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if IS_RANK0:
    import wandb
    wandb.init(
        name=config_model["encoder"] + "_" + xp_name,
        project=wandb_project,
        config=config_model,
    )
    wandb_logger = WandbLogger(project=wandb_project)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

class_weights = None
if args.use_class_weights and args.test_only is None:
    tmp_train = DatasetClass(
        root_path=data_root,
        mode="train",
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
    )
    # >>> UNET: EuroSATSARDataset exposes sample_list = [(cls, ...)];
    #     EurosatDataset stores labels inside the HDF5 files, so read them.
    if hasattr(tmp_train, "sample_list"):
        counts = Counter(cls for cls, _ in tmp_train.sample_list)
    else:
        counts = Counter(int(tmp_train[i]["label"]) for i in range(len(tmp_train)))
    weights = torch.zeros(tmp_train.NUM_CLASSES, dtype=torch.float32)
    for c in range(tmp_train.NUM_CLASSES):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)
    class_weights = weights / weights.sum() * tmp_train.NUM_CLASSES
    if IS_RANK0:
        print(f"[Train] Class weights: {class_weights.tolist()}")
    del tmp_train


# =============================================================================
# MODEL
# =============================================================================

model = Model_ForestNet(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
    class_weights=class_weights,
    label_smoothing=args.label_smoothing,
)


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    resume_ckpt_path = None
    if args.resume_from is not None:
        resume_ckpt_path = args.resume_from
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(
                f"--resume_from checkpoint not found: {resume_ckpt_path}"
            )
        print(f"[Resume] Using explicit checkpoint: {resume_ckpt_path}")
    elif args.resume:
        auto_path = os.path.join(ckpt_dir, f"{config_model['encoder']}_{xp_name}-last.ckpt")
        if os.path.exists(auto_path):
            resume_ckpt_path = auto_path
            print(f"[Resume] Found existing checkpoint, resuming: {auto_path}")
        else:
            print(f"[Resume] --resume set but no checkpoint found at "
                  f"{auto_path} — starting fresh.")

    lr_monitor   = LearningRateMonitor(logging_interval="step")
    accumulator  = GradientAccumulationScheduler(scheduling={0: 1})

    checkpoint_val = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-{{epoch:02d}}-{{val_top1:.4f}}",
        monitor="val_top1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-last",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_top1",
        mode="max",
        patience=int(config_model["trainer"].get("patience", 20)),
        verbose=True,
    )

    callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]

    trainer = Trainer(
        # find_unused_parameters: the U-Net's level-0 seed and the zero-initialised
        # gates get no gradient at some steps; keep it on for both encoders.
        strategy="ddp_find_unused_parameters_true",
        devices=-1,
        max_epochs=config_model["trainer"]["epochs"],
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0 if args.clipping else None,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    best_ckpt = checkpoint_val.best_model_path

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
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys "
          f"(runtime caches — recreated automatically)")
# >>> UNET: strict=False also hides MISSING weights; a partially loaded model
#     would silently produce meaningless test numbers.
if missing:
    print(f"[load_state_dict] WARNING: {len(missing)} missing keys, e.g. {missing[:5]}")
    raise RuntimeError("Checkpoint does not match the model (missing weights). "
                       "Check that --config is the one used for training.")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(model, datamodule=data_module)


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger and IS_RANK0:
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
    wandb.finish()
