"""
BurnScars Training Script — Atomiser U-Net (single timestep), with --test_only mode

Changes vs the skip-variant script are tagged  # >>> UNET.

  * Config: U-Net defaults are filled in, slide / error predictor forced off,
    and the encoder name is made distinct so checkpoints / wandb runs never
    collide with the baseline.
  * Pre-flight check: token positions must be identical across samples of a
    batch (the hierarchy geometry is built once from sample 0 and shared).
  * DDP: always find_unused_parameters=True (the zero-initialised skip gates
    leave their LayerNorm without gradient on the very first step).
  * --test_only: reports missing / unexpected checkpoint keys instead of
    silently ignoring them (strict=False).
  * Unused TokenAssignmentCallbackSenFlood removed.

The trainer (training/trainer_BURNSCARS.py) must instantiate Atomiser_Unet:
    from training.atomiser.Atomiser_unet import Atomiser_Unet
    self.encoder = Atomiser_Unet(config=self.config, lookup_table=self.lookup_table)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import torch
import numpy as np
from collections import defaultdict


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)

from training.utils import read_yaml
from training.utils import Lookup_encoding

from training.trainer_BURNSCARS import Model_BurnScars_Skip
from training.utils.datasets.utils_dataset_BURNSCARS import BurnScarsDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip


IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Training script (Atomiser U-Net)")
parser.add_argument("--xp_name",      type=str, required=True, help="Experiment name")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset used")
parser.add_argument("--clipping",     action="store_true",     help="Enable gradient clipping at 1.0")
parser.add_argument("--resume",        type=str, default=None,
                    help="Path to checkpoint to resume training from")
parser.add_argument("--deterministic", action="store_true",
                    help="Force deterministic CUDA ops")
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a checkpoint: load it and run ONLY trainer.test "
                         "(skips fit and the complexity block).")
# >>> UNET: config path is now an argument (same default as before)
parser.add_argument("--config", type=str,
                    default="./training/configs/config_test-BURNSCARS.yaml",
                    help="Model config YAML")
parser.add_argument("--skip_preflight", action="store_true",
                    help="Skip the token-layout sanity check")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed (default 42, as before); use several per variant")
args = parser.parse_args()
seed_everything(args.seed, workers=True)

xp_name = args.xp_name
config_model = read_yaml(args.config)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

# =============================================================================
# >>> UNET: CONFIG DEFAULTS + SAFETY OVERRIDES
# =============================================================================
UNET_DEFAULTS = {
    "unet_dim": 512,
    "unet_heads": 8,
    "unet_levels": 3,
    "unet_depth": 2,
    "unet_fusion": "gate",
    "unet_atom_dim": 128,
    "unet_query_dim": 256,
    "unet_km_train": [[1000, 1000], [1500, 1500], [2000, 2000]],
    "unet_km_eval": [1500, 1500],
    "unet_decoder_neighbors": 9,
    "unet_leaf_chunk": 256,
    "unet_query_chunk": 16384,
}
config_model.setdefault("Atomiser", {})
for key, val in UNET_DEFAULTS.items():
    config_model["Atomiser"].setdefault(key, val)

# Not supported by the U-Net encoder: force off rather than silently half-run.
config_model["Atomiser"]["use_error_predictor"] = False
config_model["trainer"]["slide"] = False

# Distinct encoder name -> distinct checkpoint filenames, save_model path, wandb run.
enc_name = str(config_model.get("encoder", "Atomiser"))
if "unet" not in enc_name.lower():
    config_model["encoder"] = f"{enc_name}_Unet"

if IS_RANK0:
    print(f"[Train] Config: {args.config}")
    print(f"[Train] Encoder name: {config_model['encoder']}")
    print(f"[Train] U-Net: levels={config_model['Atomiser']['unet_levels']} "
          f"dim={config_model['Atomiser']['unet_dim']} "
          f"depth={config_model['Atomiser']['unet_depth']} "
          f"fusion={config_model['Atomiser']['unet_fusion']}")
    print(f"[Train]        km_train={config_model['Atomiser']['unet_km_train']} "
          f"km_eval={config_model['Atomiser']['unet_km_eval']}")
    print(f"[Train] Gradient clipping: ON (val=1.0, fixed in Trainer)")
    print(f"[Train] Deterministic ops: {'ON' if args.deterministic else 'OFF'}")
    if args.resume:
        print(f"[Train] RESUMING from: {args.resume}")
    if args.test_only:
        print(f"[Train] TEST-ONLY mode, loading: {args.test_only}")

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/hls_burn_scars",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=BurnScarsDataset,
    collate_fn=collate_grouped_skip,
)

# =============================================================================
# >>> UNET: PRE-FLIGHT CHECK — shared token layout across samples
# =============================================================================
def preflight_token_layout(dm):
    """The hierarchy geometry is built from sample 0's token positions and reused
    for the whole batch. That is only valid if token i is the same position in
    every sample. Check it on two collated training samples."""
    dm.setup("fit")
    ds = dm.train_dataset
    batch = collate_grouped_skip([ds[0], ds[1 % len(ds)]])
    for res, grp in batch["groups"].items():
        t = grp["tokens"]
        assert t.dim() == 3 and t.shape[-1] >= 8, \
            f"group {res}: expected tokens [B, N, 8], got {tuple(t.shape)}"
        if not torch.equal(t[0, :, 1:3], t[1, :, 1:3]):
            raise RuntimeError(
                f"[Preflight] group {res}: token positions differ between samples. "
                "collate_grouped_skip (or the dataset) reorders/pads tokens per sample, "
                "which breaks the shared hierarchy geometry. Fix the token order, "
                "or rerun with --skip_preflight only if you know why this is safe.")
        print(f"[Preflight] group {res}: tokens {tuple(t.shape)}, "
              f"positions identical across samples ✓")
    q = batch["queries"]
    assert q.dim() == 3 and q.shape[-1] >= 8, f"expected queries [B, M, 8], got {tuple(q.shape)}"
    print(f"[Preflight] queries {tuple(q.shape)} ✓")


if IS_RANK0 and not args.skip_preflight:
    preflight_token_layout(data_module)

# =============================================================================
# WANDB  (skip in test-only mode)
# =============================================================================
wandb_logger = None
if IS_RANK0 and args.test_only is None:
    import wandb
    wandb.init(name=config_model["encoder"], project="BurnScars", config=config_model)
    wandb_logger = WandbLogger(project="BurnScars")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL  (load ckpt if test-only, else fresh)
# =============================================================================
if args.test_only is not None:
    model = Model_BurnScars_Skip.load_from_checkpoint(
        args.test_only, strict=False, config=config_model, wand=False,
        name=xp_name, transform=None, lookup_table=lookup_table)
    model.eval()
    # >>> UNET: strict=False silently ignores mismatches — report them.
    if IS_RANK0:
        ckpt_keys = set(torch.load(args.test_only, map_location="cpu",
                                   weights_only=False)["state_dict"].keys())
        model_keys = set(model.state_dict().keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        print(f"[TestOnly] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        for k in missing[:20]:
            print(f"  missing:    {k}")
        for k in unexpected[:20]:
            print(f"  unexpected: {k}")
        if missing:
            print("[TestOnly] WARNING: model is only partially loaded; "
                  "test metrics will not reflect the checkpoint.")
else:
    model = Model_BurnScars_Skip(
        config=config_model, wand=True, name=xp_name,
        transform=None, lookup_table=lookup_table)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)



callbacks = [accumulator, checkpoint_val, lr_monitor]

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    # >>> UNET: always find unused params (gate LayerNorms have no grad at init)
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/",
    gradient_clip_val=1.0,
    deterministic=args.deterministic,
)

# =============================================================================
# TEST-ONLY EARLY EXIT
# =============================================================================
if args.test_only is not None:
    results = trainer.test(model, datamodule=data_module, verbose=True)
    if IS_RANK0:
        metrics = results[0] if results else {}
        miou = metrics.get("test_mIoU", float("nan"))
        acc  = metrics.get("test_accuracy", float("nan"))
        print(f"RESULT test_only ckpt={args.test_only} "
              f"test_mIoU={miou:.6f} test_accuracy={acc:.6f}")
    sys.exit(0)

# =============================================================================
# TRAIN & TEST
# =============================================================================
trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
trainer.test(model, datamodule=data_module)


# =============================================================================
# MEASURE COMPLEXITY
# =============================================================================
def _batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


if IS_RANK0:
    from fvcore.nn import FlopCountAnalysis

    print("\n" + "=" * 80)
    print("MEASURING MODEL COMPLEXITY")
    print("=" * 80 + "\n")

    data_module.setup("fit")
    train_dataset = data_module.train_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(train_dataset))
    samples = [train_dataset[i] for i in range(num_samples)]

    results = []

    for input_size in [512]:
        print(f"\nTesting input size: {input_size}x{input_size}")
        batch_0 = collate_grouped_skip([samples[0]])
        batch_0 = _batch_to_device(batch_0, device)
        with torch.no_grad():
            _ = model(batch_0)

        batch_1 = collate_grouped_skip([samples[1]])
        batch_1 = _batch_to_device(batch_1, device)
        # >>> UNET: tracing may fail on data-dependent indexing; timing still runs.
        try:
            with torch.no_grad():
                flops = FlopCountAnalysis(model, (batch_1,))
                gflops = flops.total() / 1e9
        except Exception as e:
            print(f"  FLOPs measurement failed: {e}")
            gflops = -1

        num_warmup, num_runs = 3, 20
        with torch.no_grad():
            for i in range(num_warmup):
                b = collate_grouped_skip([samples[i % num_samples]])
                b = _batch_to_device(b, device)
                _ = model(b)
            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_runs):
                idx = (i + num_warmup) % num_samples
                b = collate_grouped_skip([samples[idx]])
                b = _batch_to_device(b, device)
                _ = model(b)
                torch.cuda.synchronize()
            end = time.time()

        avg_time_ms = (end - start) / num_runs * 1000
        first_res = next(iter(batch_0["groups"]))
        num_tokens = batch_0["groups"][first_res]["tokens"].shape[1]

        results.append({
            "input_size": input_size, "gsd_target": 30,
            "num_tokens": num_tokens, "gflops": gflops,
            "inference_time_ms": avg_time_ms,
        })
        print(f"  Tokens: {num_tokens}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Inference time: {avg_time_ms:.2f} ms/tile")

    print("\n" + "=" * 80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']})")
    print("=" * 80)
    print(f"{'Input':<10} {'Tokens':<12} {'GFLOPs':<12} {'Time (ms)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['input_size']:<10} {r['num_tokens']:<12} {r['gflops']:<12.2f} {r['inference_time_ms']:<12.2f}")
    print("=" * 80)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and IS_RANK0:
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
