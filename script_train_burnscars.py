"""
HLS Burn Scars Training Script — Atomizer
==========================================

Trains Atomizer on HLS BurnScars binary segmentation.

  - 6 HLS bands @ 30m
  - 512×512 native, kept full (no crop)
  - Reuses Model_SenFlood (generic seg trainer)
  - Reuses collate_grouped + UnifiedDataModule

Splits:
  - Train: 90% of training/ folder (stratified, random_state=23, PANGAEA-compat)
  - Val:   10% of training/ folder (same stratified split)
  - Test:  all of validation/ folder

Examples:
    # From scratch
    python train_burnscars_atomiser.py \
        --xp_name burnscars_v1 \
        --config_model config_atomiser_burnscars.yaml \
        --dataset_name u_regular

    # Resume training from a Lightning checkpoint
    python train_burnscars_atomiser.py \
        --xp_name burnscars_v1_resumed \
        --config_model config_atomiser_burnscars.yaml \
        --dataset_name u_regular \
        --ckpt_path ./checkpoints/atomiserburnscars_v1-...ckpt \
        --wandb_run_id <run_id_of_the_aborted_run>

    # Test-only on a saved checkpoint
    python train_burnscars_atomiser.py \
        --xp_name burnscars_v1_test \
        --config_model config_atomiser_burnscars.yaml \
        --dataset_name u_regular \
        --ckpt_path ./checkpoints/atomiserburnscars_v1-best.ckpt \
        --test_only
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import argparse

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)

seed_everything(42, workers=True)

# --- Project imports ---
from training.utils import read_yaml
from training.utils import Lookup_encoding
from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_BURNSCARS import BurnScarsDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# RESOLUTION REGISTRATION
# =============================================================================

# Same set as PASTIS-HD's register_all_resolutions — covers HLS (30m),
# Sentinel-2 (10m, 20m), SPOT (1m), and a few intermediate values for
# any future modality. ref_size=2048 is the maximum image size at any
# resolution for the positional encoding's coordinate range.
ALL_KNOWN_RESOLUTIONS = {
    1.0:  2048,
    2.5:  2048,
    10.0: 2048,
    20.0: 2048,
    30.0: 2048,
}


def register_all_resolutions(lookup_table):
    """
    Register every known resolution in the lookup table + TokenBuilder.

    Required so the dataset's `lookup_table.get_resolution_idx(30.0)`
    call inside `__init__` succeeds. Same pattern as PASTIS-HD.
    """
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="HLS BurnScars Atomizer training")
parser.add_argument("--xp_name",      type=str, required=True,
                    help="Experiment name")
parser.add_argument("--config_model", type=str, default="config_test-BURNSCARS.yaml",
                    help="Model config yaml under ./training/configs/")

parser.add_argument("--data_dir",     type=str, default="./data/hls_burn_scars",
                    help="Path to HLS BurnScars data root")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--ckpt_path",    type=str, default=None,
                    help="Resume training from Lightning checkpoint")
parser.add_argument("--wandb_run_id", type=str, default=None,
                    help="Wandb run ID to resume into (use with --ckpt_path)")
parser.add_argument("--test_only",    action="store_true",
                    help="Skip training, load --ckpt_path and run val+test only")
args = parser.parse_args()

xp_name         = args.xp_name
config_model    = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml      = "./data/bands_info/bands.yaml"

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(
    read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)
register_all_resolutions(lookup_table)

print(f"\n[BurnScars] Experiment:   {xp_name}")
print(f"[BurnScars] Config model:  {args.config_model}")
print(f"[BurnScars] Data dir:      {args.data_dir}")
print(f"[BurnScars] Lookup table:  {len(lookup_table.table_wave)} entries")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb

    wandb_init_kwargs = dict(
        name=f"BurnScars_{config_model['encoder']}_{xp_name}",
        project="BurnScars",
        config=config_model,
    )
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"[BurnScars] Resuming wandb run: {args.wandb_run_id}")
    else:
        print(f"[BurnScars] Starting new wandb run: {wandb_init_kwargs['name']}")

    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="BurnScars")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss",   step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
# Class names for cleaner logging — turns "test_IoU_class_1" into
# "test_IoU_burn". Optional but nice to have.
class_names = ["no_burn", "burn"]

model = Model_SenFlood(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
    class_names=class_names,
)

# =============================================================================
# DATA MODULE
# =============================================================================
# `dataset_config` is the bands.yaml dict — BurnScarsDataset reads
# `dataset_config["bands_hls_info"]` from it. Same convention as
# Sen1Floods11 (which reads `dataset_config["bands_senflood"]`).
data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["batchsize"],
    num_workers=args.num_workers,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=BurnScarsDataset,
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor  = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_best = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)
checkpoint_last = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    verbose=True,
)

callbacks = [accumulator, checkpoint_best, checkpoint_last, lr_monitor]

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/",
)

# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError(
            "--test_only requires --ckpt_path to load weights from."
        )

    print(f"\n{'='*60}")
    print(f"  BurnScars — TEST ONLY")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"{'='*60}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[BurnScars] Loaded checkpoint — "
          f"missing: {len(result.missing_keys)}, "
          f"unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"[BurnScars] First 5 missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[BurnScars] First 5 unexpected: {result.unexpected_keys[:5]}")

    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
else:
    print(f"\n{'='*60}")
    print(f"  BurnScars — TRAINING")
    if args.ckpt_path is not None:
        print(f"  RESUMING from: {args.ckpt_path}")
        if args.wandb_run_id:
            print(f"  Wandb run:     {args.wandb_run_id}")
    print(f"{'='*60}\n")

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
    trainer.test(model, datamodule=data_module)


# =============================================================================
# COMPLEXITY MEASUREMENT
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    """Recursively move a nested batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


# Skip complexity measurement in test_only mode — not the primary purpose,
# and the timing/FLOP block can be slow on large datasets.
if not args.test_only and os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        FlopCountAnalysis = None
        print("[BurnScars] fvcore not installed — skipping FLOP measurement")

    print("\n" + "=" * 80)
    print("MEASURING MODEL COMPLEXITY")
    print("=" * 80 + "\n")

    data_module.setup("test")
    test_dataset = data_module.test_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(test_dataset))
    samples = [test_dataset[i] for i in range(num_samples)]

    results = []

    for input_size in [512]:
        print(f"\nTesting input size: {input_size}x{input_size}")

        batch_0 = collate_grouped([samples[0]])
        batch_0 = _batch_to_device(batch_0, device)

        with torch.no_grad():
            _ = model(batch_0)

        # FLOPs
        gflops = -1
        if FlopCountAnalysis is not None:
            batch_1 = collate_grouped([samples[1]])
            batch_1 = _batch_to_device(batch_1, device)
            try:
                with torch.no_grad():
                    flops = FlopCountAnalysis(model, (batch_1,))
                    gflops = flops.total() / 1e9
            except Exception as e:
                print(f"  FLOPs measurement failed: {e}")

        # Inference time
        num_warmup, num_runs = 3, 20
        with torch.no_grad():
            for i in range(num_warmup):
                b = collate_grouped([samples[i % num_samples]])
                b = _batch_to_device(b, device)
                _ = model(b)

            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_runs):
                idx = (i + num_warmup) % num_samples
                b = collate_grouped([samples[idx]])
                b = _batch_to_device(b, device)
                _ = model(b)
                torch.cuda.synchronize()
            end = time.time()

        avg_time_ms = (end - start) / num_runs * 1000

        first_res  = next(iter(batch_0["groups"]))
        num_tokens = batch_0["groups"][first_res]["tokens"].shape[1]

        results.append({
            "input_size":         input_size,
            "gsd_target":         30,
            "num_tokens":         num_tokens,
            "gflops":             gflops,
            "inference_time_ms":  avg_time_ms,
        })

        print(f"  Tokens: {num_tokens}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Inference time: {avg_time_ms:.2f} ms/tile")

    # Summary
    print("\n" + "=" * 80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']})")
    print("=" * 80)
    print(f"{'Input':<10} {'Tokens':<12} {'GFLOPs':<12} {'Time (ms)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['input_size']:<10} {r['num_tokens']:<12} "
              f"{r['gflops']:<12.2f} {r['inference_time_ms']:<12.2f}")
    print("=" * 80)


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/burnscars_{xp_name}.txt", "w") as f:
        f.write(run_id)