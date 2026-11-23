"""
Sen1Floods11 Training Script (SKIP / U-Net variant) — with --test_only mode

Runs the baseline (Atomiser_Senflood_Skip) or Atomiser_Unet, chosen by the
trainer from the config (encoder name containing "unet", or Atomiser.unet_task).
Changes vs the previous script are tagged  # >>> UNET:
    * --config argument (default: config_test-SENFLOOD.yaml)
    * U-Net safety overrides: slide / error predictor / temporal off, distinct
      encoder name (checkpoints and wandb never collide with the baseline)
    * pre-flight check: token positions identical across samples (the U-Net's
      hierarchy geometry is shared by every sample)
    * DDP always find_unused_parameters=True (the old line switched to "auto"
      for a U-Net, which crashes: some parameters get no gradient at step 1)
    * --test_only reports missing / unexpected checkpoint keys
    * unused TokenAssignmentCallbackSenFlood removed

# >>> MULTI_SEED_TEST: --test_only runs N_TEST_RUNS repeats with different
# seeds and reports mean/max/min/std. This matters for the baseline, whose
# spatial sampler draws a random subset per latent at test time. The U-Net's
# evaluation is deterministic (fixed atom order per leaf cell at eval), so its
# repeats give identical numbers; the summary is kept for a uniform protocol.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import statistics
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

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils.Lookup_positional import Lookup_encoding          # >>> UNET: direct import

from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip

from training.utils.callbacks.segmentation_viz_callback import SegmentationVizCallback

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

# =============================================================================
# >>> MULTI_SEED_TEST: config for the repeated test-only runs
# =============================================================================
N_TEST_RUNS = 2
TEST_SEEDS = list(range(1, N_TEST_RUNS + 1))  # seed == run number (1, 2, 3, ...)

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Training script (skip / U-Net variant)")
parser.add_argument("--xp_name",      type=str, required=True, help="Experiment name")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset used")
parser.add_argument("--clipping",     action="store_true",     help="Enable gradient clipping at 1.0")
parser.add_argument("--resume",        type=str, default=None,
                    help="Path to checkpoint to resume training from")
parser.add_argument("--deterministic", action="store_true",
                    help="Force deterministic CUDA ops")
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a checkpoint: load it and run ONLY trainer.test "
                         "(skips fit and the complexity block). Runs N_TEST_RUNS "
                         "repeats under different seeds and reports summary stats.")
# >>> UNET
parser.add_argument("--config", type=str, default="./training/configs/config_test-SENFLOOD.yaml",
                    help="Model config YAML")
parser.add_argument("--skip_preflight", action="store_true",
                    help="Skip the token-layout sanity check (U-Net only)")
# >>> END UNET
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml(args.config)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

# =============================================================================
# >>> UNET: detection + safety overrides
# =============================================================================
a_cfg = config_model.setdefault("Atomiser", {})
is_unet = ("unet" in str(config_model.get("encoder", "")).lower() or "unet_task" in a_cfg)
if is_unet:
    if a_cfg.get("unet_task", "segmentation") != "segmentation":
        raise ValueError("Sen1Floods11 is segmentation: set Atomiser.unet_task: segmentation")
    a_cfg["unet_task"] = "segmentation"
    a_cfg["use_error_predictor"] = False
    a_cfg["use_temporal_transformer"] = False
    config_model["trainer"]["slide"] = False
    if config_model["trainer"].get("mode", "segmentation") != "segmentation":
        raise ValueError("Atomiser_Unet supports trainer.mode: segmentation only")
    if "unet" not in str(config_model.get("encoder", "")).lower():
        config_model["encoder"] = f"{config_model.get('encoder', 'ATOMIZER')}_Unet"
# >>> END UNET

if IS_RANK0:
    print(f"[Train] Config: {args.config}")
    print(f"[Train] Encoder: {config_model['encoder']} ({'U-Net' if is_unet else 'baseline'})")
    print(f"[Train] Gradient clipping: ON (val=1.0, fixed in Trainer)")
    print(f"[Train] Deterministic ops: {'ON' if args.deterministic else 'OFF'}")
    if args.resume:
        print(f"[Train] RESUMING from: {args.resume}")
    if args.test_only:
        print(f"[Train] TEST-ONLY mode, loading: {args.test_only}")
        print(f"[Train] TEST-ONLY will run {N_TEST_RUNS} repeats "
              f"with seeds {TEST_SEEDS}")
    if not is_unet:
        _skip_on = a_cfg.get("use_decoder_skip", False)
        print(f"[Train] Decoder pixel-skip: {'ON' if _skip_on else 'OFF (baseline)'}")

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/SENFLOOD",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=Sen1Floods11SkipDataset,
    collate_fn=collate_grouped_skip,
)

# =============================================================================
# >>> UNET: PRE-FLIGHT — shared token layout across samples
# =============================================================================
def preflight_token_layout(dm):
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
                "The U-Net's hierarchy geometry needs a fixed token layout.")
        print(f"[Preflight] group {res}: tokens {tuple(t.shape)}, "
              f"positions identical across samples ✓")
    print(f"[Preflight] queries {tuple(batch['queries'].shape)} ✓")


if is_unet and IS_RANK0 and not args.skip_preflight and args.test_only is None:
    preflight_token_layout(data_module)

# =============================================================================
# WANDB  (skip in test-only mode)
# =============================================================================
wandb_logger = None
if IS_RANK0 and args.test_only is None:
    import wandb
    wandb.init(name=config_model["encoder"], project="SenFlood", config=config_model)
    wandb_logger = WandbLogger(project="SenFlood")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL  (load ckpt if test-only, else fresh)
# =============================================================================
if args.test_only is not None:
    model = Model_SenFlood_Skip.load_from_checkpoint(
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
            raise RuntimeError("[TestOnly] checkpoint does not match the model built from "
                               "--config (missing weights). Use the training config.")
    # >>> END UNET
else:
    model = Model_SenFlood_Skip(
        config=config_model, wand=True, name=xp_name,
        transform=None, lookup_table=lookup_table)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 2})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

viz_callback = SegmentationVizCallback(
    sample_indices=[0, 1, 2], log_every_n_epochs=1, use_wandb=True)

callbacks = [accumulator, checkpoint_val, lr_monitor, viz_callback]

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    # >>> UNET: always find unused parameters (both encoders need it)
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
# TEST-ONLY EARLY EXIT  (>>> MULTI_SEED_TEST: N repeats + summary stats)
# =============================================================================
if args.test_only is not None:
    miou_runs, acc_runs = [], []

    for run_idx, seed in enumerate(TEST_SEEDS, 1):
        seed_everything(seed, workers=True)

        if IS_RANK0:
            print(f"\n[Train][TEST-ONLY] Run {run_idx}/{N_TEST_RUNS} "
                  f"(seed={seed})")

        run_trainer = Trainer(
            strategy="ddp_find_unused_parameters_true",
            devices=-1,
            accelerator="gpu",
            precision="bf16-mixed",
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=args.deterministic,
        )
        results = run_trainer.test(model, datamodule=data_module, verbose=True)

        if IS_RANK0:
            metrics = results[0] if results else {}
            miou = metrics.get("test_mIoU", float("nan"))
            acc  = metrics.get("test_accuracy", float("nan"))
            miou_runs.append(miou)
            acc_runs.append(acc)
            print(f"RESULT test_only ckpt={args.test_only} run={run_idx} "
                  f"seed={seed} test_mIoU={miou:.6f} test_accuracy={acc:.6f}")

    if IS_RANK0:
        def _stats(vals):
            vals = [v for v in vals if v == v]  # drop nan
            if not vals:
                return dict(mean=float("nan"), max=float("nan"),
                            min=float("nan"), std=float("nan"), n=0)
            return dict(
                mean=statistics.mean(vals),
                max=max(vals),
                min=min(vals),
                std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
                n=len(vals),
            )

        miou_stats = _stats(miou_runs)
        acc_stats  = _stats(acc_runs)

        print("\n" + "=" * 78)
        print(f"TEST-ONLY SUMMARY over {N_TEST_RUNS} runs "
              f"(seeds={TEST_SEEDS}) — ckpt={args.test_only}")
        print("=" * 78)
        print(f"  test_mIoU     : mean={miou_stats['mean']:.6f}  "
              f"max={miou_stats['max']:.6f}  min={miou_stats['min']:.6f}  "
              f"std={miou_stats['std']:.6f}  (n={miou_stats['n']})")
        print(f"  test_accuracy : mean={acc_stats['mean']:.6f}  "
              f"max={acc_stats['max']:.6f}  min={acc_stats['min']:.6f}  "
              f"std={acc_stats['std']:.6f}  (n={acc_stats['n']})")
        print("=" * 78)

        print(f"RESULT_SUMMARY test_only ckpt={args.test_only} "
              f"n_runs={miou_stats['n']} "
              f"test_mIoU_mean={miou_stats['mean']:.6f} "
              f"test_mIoU_max={miou_stats['max']:.6f} "
              f"test_mIoU_min={miou_stats['min']:.6f} "
              f"test_mIoU_std={miou_stats['std']:.6f} "
              f"test_accuracy_mean={acc_stats['mean']:.6f} "
              f"test_accuracy_max={acc_stats['max']:.6f} "
              f"test_accuracy_min={acc_stats['min']:.6f} "
              f"test_accuracy_std={acc_stats['std']:.6f}")

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
            "input_size": input_size, "gsd_target": 10,
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
