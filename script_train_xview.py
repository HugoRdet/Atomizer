"""
Atomiser xView2 Training Script (SKIP variant)
=================================================

5-class damage segmentation on xView2 (PANGAEA setup):
    Input:  pre + post RGB, [T=2, C=3, H=512, W=512]
    Target: 5-class damage mask {0..4}, no IGNORE
    Native resolution: 0.5 m (sub-meter aerial WorldView-3)

Uses Model_SenFlood_Skip (training.trainer_SENFLOOD_skip) — the SAME
trainer class Sen1Floods11's skip variant uses (script_train_senflood_skip.py)
-- xView2 is just another multi-temporal segmentation task from its
perspective, same as the non-skip Model_SenFlood was before. Paired with
collate_grouped_skip (training.utils.datasets.collate_grouped_skip), the
generic grouped-skip collate that already handles variable-length
query_token_idx/query_token_valid padding across a batch -- no bespoke
xView2-specific collate needed, unlike BioMassters' script, which used its
own collate_biomassters_skip because it isn't built on UnifiedDataModule at
all. Passing collate_fn explicitly also means the GROUPED_DATASET_CLASSES
registration the previous (non-skip) version of this script needed is no
longer necessary -- Sen1Floods11's skip script doesn't do that either.

XView2Dataset itself must already build query_token_idx/query_token_valid
per sample (the per-pixel SKIP gather index) for this to work -- see that
dataset's _build_query_token_index / _build_full_pixel_index.

Requires the config (--config_model) to set Atomiser.use_decoder_skip: true
(and whatever else Model_SenFlood_Skip's decoder needs) for skip mode to
actually activate -- this script only reports what the config says, it
doesn't set it. Prints the Atomiser.use_decoder_skip value at startup so
it's visible if the config is off by mistake.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training and run
    test on a saved checkpoint.

--resume_from mode:
    Pass --resume_from <path/to/checkpoint.ckpt> to resume training (full
    trainer state) via Trainer.fit(ckpt_path=...). If the file doesn't
    exist yet, use --resume_wait_seconds to poll for it instead of failing
    immediately (useful for chained SLURM jobs) -- ported from
    train_biomassters.py's wait_for_checkpoint, reused here for
    --test_only's checkpoint wait too. Mutually exclusive with --test_only.

Required:
    - bands_xview section in ./data/bands_info/bands.yaml (3 RGB bands)
    - configs_dataset_xview.yaml under ./data/Tiny_BigEarthNet/ (or wherever
      you keep dataset configs)
    - atomiser_xview.yaml under ./training/configs/ (with
      Atomiser.use_decoder_skip: true if you want the skip path active)

Examples:
    python script_train_xview.py --xp_name v1 \\
        --config_model atomiser_xview.yaml

    # Resume a chained SLURM job, waiting up to 10 minutes for the checkpoint
    python script_train_xview.py --xp_name v1 --config_model atomiser_xview.yaml \\
        --resume_from ./checkpoints/xview/atomiser_v1-last.ckpt \\
        --resume_wait_seconds 600
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
    EarlyStopping,
)

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.utils.datasets.utils_dataset_xview import XView2Dataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip


# =============================================================================
# RESUME HELPER (ported from train_biomassters.py)
# =============================================================================

def wait_for_checkpoint(path: str, wait_seconds: int, poll_interval: int = 15) -> str:
    """
    Polls for `path` to exist, up to `wait_seconds` total, checking every
    `poll_interval` seconds. Useful for chained SLURM jobs where the next
    job in the chain can start before the previous job's checkpoint write
    (and any filesystem sync delay, common on Lustre) has actually landed.

    wait_seconds=0 means "check once, don't wait" -- fails fast if the
    file isn't there, matching the old plain os.path.exists() behavior.

    Raises FileNotFoundError if the checkpoint never appears within the
    timeout, rather than silently falling back to training from scratch --
    resuming should be an explicit, verified action.
    """
    if os.path.exists(path):
        return path

    if wait_seconds <= 0:
        raise FileNotFoundError(
            f"Checkpoint not found: {path} "
            f"(use --resume_wait_seconds > 0 to poll for it instead of failing immediately)"
        )

    print(f"[xView2] Checkpoint not found yet: {path}")
    print(f"[xView2] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[xView2] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[xView2]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"Checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser xView2 training (skip variant)")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,  default="config_test-xview.yaml",
                    help="Model config yaml (e.g. atomiser_xview.yaml)")
parser.add_argument("--clipping",     action="store_true")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Resume-training mode (ported from train_biomassters.py)
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint to resume TRAINING from (full "
                         "trainer state via Trainer.fit(ckpt_path=...)). If "
                         "the file doesn't exist yet, use --resume_wait_seconds "
                         "to poll for it instead of failing immediately.")
parser.add_argument("--resume_wait_seconds", type=int, default=0,
                    help="How long to poll for --resume_from (or --test_only, "
                         "reused for both) to appear before giving up "
                         "(0 = check once, fail immediately if missing).")
parser.add_argument("--resume_poll_interval", type=int, default=15,
                    help="Seconds between polls while waiting for the checkpoint.")

parser.add_argument("--data_dir", type=str, default="./data/xview")

args = parser.parse_args()

if args.test_only is not None and args.resume_from is not None:
    raise ValueError(
        "--test_only and --resume_from are mutually exclusive: --test_only "
        "skips training entirely (loads weights, runs test), --resume_from "
        "continues training from a checkpoint. Pick one."
    )

xp_name           = args.xp_name
config_model      = read_yaml("./training/configs/" + args.config_model)
configs_dataset   = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    if args.test_only:
        print(f"[Train] Test-only mode: {args.test_only}")
    else:
        print(f"[Train] Gradient clipping: {'ON' if args.clipping else 'OFF'}")
    if args.resume_from:
        print(f"[Train] Resume requested: {args.resume_from} "
              f"(wait up to {args.resume_wait_seconds}s if not found yet)")
    _skip_on = config_model.get("Atomiser", {}).get("use_decoder_skip", False)
    print(f"[Train] Decoder pixel-skip: {'ON' if _skip_on else 'OFF (baseline)'}")


# =============================================================================
# LOOKUP TABLE
# =============================================================================

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset),
    read_yaml(bands_yaml),
    config_model,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and args.test_only is None:
    import wandb
    wandb.init(
        name=config_model["encoder"] + "_" + xp_name,
        project="Atomizer_xView2",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="Atomizer_xView2")


# =============================================================================
# DATA MODULE
# =============================================================================

data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=XView2Dataset,
    collate_fn=collate_grouped_skip,
)


# =============================================================================
# MODEL
# =============================================================================

model = Model_SenFlood_Skip(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
)


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/xview/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    accumulator  = GradientAccumulationScheduler(scheduling={0: 4})

    checkpoint_val = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
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
        monitor="val_mIoU",
        mode="max",
        patience=int(config_model["trainer"].get("patience", 20)),
        verbose=True,
    )

    callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]

    # Lightning's SLURM environment plugin cross-checks Trainer(num_nodes=...)
    # against SLURM_NNODES -- if num_nodes isn't passed explicitly it
    # defaults to 1, which raises the moment this runs as a multi-node
    # sbatch job (same bug fixed in script_train_baselines_xview.py; this
    # script had it too, just never triggered by a single-node run).
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    print(f"[xView2] num_nodes: {num_nodes} (from SLURM_NNODES, default 1 if unset)")

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        devices=-1, num_nodes=num_nodes,
        max_epochs=config_model["trainer"]["epochs"],
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
    )

    # Resolve --resume_from (with polling) right before fit -- dataset/model/
    # wandb setup above happens regardless of whether we're resuming, so
    # only the actual fit() call blocks waiting for the checkpoint.
    fit_ckpt_path = None
    if args.resume_from is not None:
        fit_ckpt_path = wait_for_checkpoint(
            args.resume_from, args.resume_wait_seconds, args.resume_poll_interval)
        print(f"[xView2] RESUMING from: {fit_ckpt_path}")

    trainer.fit(model, datamodule=data_module, ckpt_path=fit_ckpt_path)

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
    best_ckpt = wait_for_checkpoint(
        args.test_only, args.resume_wait_seconds, args.resume_poll_interval)
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================
# Note: For now we use Lightning's test() with the cropped val-style protocol
# (center crop 512×512). Sliding-window full-image evaluation for Atomiser
# requires custom logic since the model takes tokenized queries rather than
# spatial logits — see docstring for future work.

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys "
          f"(runtime caches — recreated automatically)")

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

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
    wandb.finish()
