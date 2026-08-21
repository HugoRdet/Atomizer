"""
Cashew Training Script (SKIP variant)

Mirrors the Sen1Floods11 skip launcher, retargeted at Cashew:
  - Model_Cashew_Skip / CashewSkipDataset instead of the SenFlood versions
  - dataset root -> ./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant
  - wandb project "Cashew"
  - complexity measurement uses input_size=256 (Cashew's native patch size)
    instead of 512

Per your note, this version does NOT do multi-seed --test_only repeats —
a single seeded test run instead, no summary-stats aggregation.

Assumes collate_grouped_skip is generic over the "groups"/"queries"/
"query_token_idx"/"query_token_valid" dict shape (it is, for Sen1Floods11Skip)
and needs no Cashew-specific changes. If your collate fn hardcodes anything
Sen1Floods11-specific (band counts, resolution keys), flag it — none of what
I've seen so far suggests it does.
"""

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

from training.utils import read_yaml
from training.utils import Lookup_encoding

from training.trainer_Cashew import Model_Cashew_Skip
from training.utils.datasets.utils_dataset_cashew import CashewSkipDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip

from training.utils.callbacks.stability_monitor import StabilityMonitor
from training.utils.callbacks.token_assignement import TokenAssignmentCallbackSenFlood


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Cashew training script (skip variant)")
parser.add_argument("--xp_name",      type=str, required=True, help="Experiment name")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset used")
parser.add_argument("--clipping",     action="store_true",     help="Enable gradient clipping at 1.0")
parser.add_argument("--resume",        type=str, default=None,
                    help="Path to checkpoint to resume training from")
parser.add_argument("--deterministic", action="store_true",
                    help="Force deterministic CUDA ops")
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a checkpoint: load it and run ONLY trainer.test "
                         "(skips fit and the complexity block). Single run, no "
                         "multi-seed aggregation.")
args = parser.parse_args()

xp_name = args.xp_name
# NOTE: this config file doesn't exist yet in what I've seen — create
# ./training/configs/config_test-CASHEW.yaml (copy config_test-SENFLOOD.yaml
# and set trainer.num_classes=7, plus any Cashew-specific max_tokens/crop
# settings) before running this script.
config_model = read_yaml("./training/configs/config_test-CASHEW.yaml")

bands_yaml = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"[Train] Gradient clipping: {'ON (val=1.0)' if args.clipping else 'OFF'}")
    print(f"[Train] Deterministic ops: {'ON' if args.deterministic else 'OFF'}")
    if args.resume:
        print(f"[Train] RESUMING from: {args.resume}")
    if args.test_only:
        print(f"[Train] TEST-ONLY mode, loading: {args.test_only}")
    _skip_on = config_model.get("Atomiser", {}).get("use_decoder_skip", False)
    print(f"[Train] Decoder pixel-skip: {'ON' if _skip_on else 'OFF (baseline)'}")

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(None, read_yaml(bands_yaml), config_model)

# =============================================================================
# WANDB  (skip in test-only mode)
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and args.test_only is None:
    import wandb
    wandb.init(name=config_model["encoder"], project="Cashew", config=config_model)
    wandb_logger = WandbLogger(project="Cashew")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL  (load ckpt if test-only, else fresh)
# =============================================================================
is_unet = False
if args.test_only is not None:
    model = Model_Cashew_Skip.load_from_checkpoint(
        args.test_only, strict=False, config=config_model, wand=False,
        name=xp_name, transform=None, lookup_table=lookup_table)
    model.eval()
else:
    model = Model_Cashew_Skip(
        config=config_model, wand=True, name=xp_name,
        transform=None, lookup_table=lookup_table)

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=CashewSkipDataset,
    collate_fn=collate_grouped_skip,
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/cashew/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

# NOTE: TokenAssignmentCallbackSenFlood is reused as-is — it's presumably
# generic over the batch/groups format rather than Sen1Floods11-specific.
# If it hardcodes SenFlood band counts/names for its visualization, it'll
# need a Cashew-specific variant; nothing I've seen so far suggests it does.
token_assign = TokenAssignmentCallbackSenFlood(
    log_every_n_epochs=1, sample_indices=[0],
    save_dir="./viz_token_assignment", use_wandb=True)



callbacks = [accumulator, checkpoint_val, lr_monitor,
             StabilityMonitor(log_every_n_steps=1)]

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy="ddp_find_unused_parameters_true" if not is_unet else "auto",
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
# TEST-ONLY EARLY EXIT  (single run, no multi-seed aggregation)
# =============================================================================
if args.test_only is not None:
    results = trainer.test(model, datamodule=data_module, verbose=True)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        metrics = results[0] if results else {}
        miou = metrics.get("test_mIoU", float("nan"))
        acc  = metrics.get("test_accuracy", float("nan"))
        print(f"RESULT test_only ckpt={args.test_only} "
              f"test_mIoU={miou:.6f} test_accuracy={acc:.6f}")

    import sys
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


if os.environ.get("LOCAL_RANK", "0") == "0":
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

    # Cashew's native patch size is 256x256 (vs. Sen1Floods11's 512x512).
    for input_size in [256]:
        print(f"\nTesting input size: {input_size}x{input_size}")
        batch_0 = collate_grouped_skip([samples[0]])
        batch_0 = _batch_to_device(batch_0, device)
        with torch.no_grad():
            _ = model(batch_0)

        batch_1 = collate_grouped_skip([samples[1]])
        batch_1 = _batch_to_device(batch_1, device)
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
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
