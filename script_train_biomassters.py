"""
BioMassters Training Script — Multi-temporal AGB Regression
==============================================================

Train Atomizer on BioMassters for above-ground biomass (AGB) regression.
  - S2 (10 physical bands, CLP excluded, multi-temporal, fixed T via pad-by-
    replication) — always enabled
  - S1 (4 bands: VV/VH x ascending/descending, multi-temporal, same fixed T)
    — always enabled

Splits:
  - Train / Validation: carved from train_features/train_agbm (BioMassters
    ships no official val split -- see BioMasstersSkipDataset._carve_val_split,
    a deterministic 10% chip-level split, seed=42)
  - Test: test_features/test_agbm

Modality/band dropping (NEW):
  - STATIC eval-time drop: config trainer.bands.drop, or override via
    --drop_bands on the command line (takes precedence over the config
    value if given) -- applied at EVERY split by BioMasstersSkipDataset.
    This is how you drive a modality-drop ablation run without editing the
    YAML, e.g.:
        --test_only --resume_from <ckpt> \
            --drop_bands VV_asc VH_asc VV_desc VH_desc   # "S2 only"
  - STOCHASTIC training-time augmentation: config
    trainer.band_dropout_augmentation (enabled/p_dropout_applied/
    p_whole_modality/p_band_drop), read directly by the dataset -- no CLI
    override here, set it in the YAML per training run.

Example:
    python train_biomassters.py --xp_name biomassters_run1

    # resume from a checkpoint that may not exist yet (e.g. chained SLURM
    # jobs where the previous job's checkpoint write hasn't landed on disk
    # yet when this job starts): poll for up to 10 minutes before giving up
    python train_biomassters.py --xp_name biomassters_run1 \
        --resume_from ./checkpoints/biomassters/biomassters_run1-last.ckpt \
        --resume_wait_seconds 600

    # Modality-drop ablation (test-only, S2-only i.e. drop all S1 bands)
    python train_biomassters.py --xp_name biomassters_run1 --test_only \
        --resume_from ./checkpoints/biomassters/biomassters_run1-last.ckpt \
        --drop_bands VV_asc VH_asc VV_desc VH_desc
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
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
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from training.utils import Lookup_encoding
seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.utils_dataset_biomasters import BioMasstersSkipDataset
from training.utils.datasets.collate_biomassters_skip import collate_biomassters_skip
from training.trainer_biomassters import Model_BioMassters_Skip

# NOTE: create_biomassters_bands_info lives in Lookup_encoding.py per the
# earlier patch (ABSTRACT_CHANNELS + this helper added there).
from training.utils.lookup_positional import create_biomassters_bands_info


# =============================================================================
# COLLATE
# =============================================================================
# collate_biomassters_skip already handles the tasks->queries bridge and
# query_token_idx/valid padding -- imported above, used directly.


# =============================================================================
# DATAMODULE
# =============================================================================

class BioMasstersDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root_path: str,
        config_model: dict,
        look_up,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root_path    = root_path
        self.config_model = config_model
        self.look_up      = look_up
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    def _make_dataset(self, mode: str):
        return BioMasstersSkipDataset(
            root_path=self.root_path,
            mode=mode,
            config_model=self.config_model,
            look_up=self.look_up,
        )

    def setup(self, stage=None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return
        self._setup_done = True

        self.train_dataset = self._make_dataset("train")
        self.val_dataset   = self._make_dataset("validation")
        self.test_dataset  = self._make_dataset("test")

        print(f"\n[BioMassters-DM] Summary:")
        print(f"  Train: {len(self.train_dataset)} chips")
        print(f"  Val:   {len(self.val_dataset)} chips (10% held out from train)")
        print(f"  Test:  {len(self.test_dataset)} chips")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=collate_biomassters_skip,
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
# RESUME HELPER
# =============================================================================

def wait_for_checkpoint(path: str, wait_seconds: int, poll_interval: int = 15) -> str:
    """
    Polls for `path` to exist, up to `wait_seconds` total, checking every
    `poll_interval` seconds. Useful for chained SLURM jobs where the next
    job in the chain can start before the previous job's checkpoint write
    (and any filesystem sync delay, common on Lustre) has actually landed.

    wait_seconds=0 means "check once, don't wait" -- matches the old
    --ckpt_path behavior of just failing fast if the file isn't there.

    Raises FileNotFoundError if the checkpoint never appears within the
    timeout, rather than silently falling back to training from scratch --
    resuming should be an explicit, verified action.
    """
    if os.path.exists(path):
        return path

    if wait_seconds <= 0:
        raise FileNotFoundError(
            f"--resume_from checkpoint not found: {path} "
            f"(use --resume_wait_seconds > 0 to poll for it instead of failing immediately)"
        )

    print(f"[BioMassters] Checkpoint not found yet: {path}")
    print(f"[BioMassters] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[BioMassters] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[BioMassters]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"--resume_from checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="BioMassters Training")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str,
                    default="config_test-Biomassters.yaml")
parser.add_argument("--data_dir",       type=str, default="./data/biomassters")
parser.add_argument("--ckpt_path",      type=str, default=None,
                    help="[Deprecated alias for --resume_from, kept for backward compat] "
                         "Path to checkpoint to resume training from.")
parser.add_argument("--resume_from",    type=str, default=None,
                    help="Path to a checkpoint to resume training from. If the file "
                         "doesn't exist yet, use --resume_wait_seconds to poll for it "
                         "instead of failing immediately (useful for chained SLURM jobs).")
parser.add_argument("--resume_wait_seconds", type=int, default=0,
                    help="How long to poll for --resume_from to appear before giving up "
                         "(0 = check once, fail immediately if missing).")
parser.add_argument("--resume_poll_interval", type=int, default=15,
                    help="Seconds between polls while waiting for --resume_from.")
parser.add_argument("--pretrained_encoder", type=str, default=None,
                    help="Load pretrained encoder weights (no head)")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--grad_accum",     type=int, default=4)

# Wandb resume
parser.add_argument("--wandb_run_id",   type=str, default=None,
                    help="Wandb run ID to resume logging into (use with --resume_from)")

# Test-only mode
parser.add_argument("--test_only",      action="store_true",
                    help="Skip training, load checkpoint from config trainer.checkpoint_path "
                         "(or --resume_from) and run test split only")

# Temporal config
parser.add_argument("--num_timesteps",  type=int, default=None,
                    help="Fixed number of timesteps per sensor (overrides config, "
                         "pad-by-replication / evenly-spaced subsample to this count)")

# Modality/band-drop override (static eval-time drop -- overrides config
# trainer.bands.drop if given). Named band strings, e.g.:
#   --drop_bands VV_asc VH_asc VV_desc VH_desc     (drop all S1 -> "S2 only")
#   --drop_bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12   (drop all S2 -> "S1 only")
parser.add_argument("--drop_bands", type=str, nargs="+", default=None,
                    help="Band names to statically drop at every split "
                         "(overrides config trainer.bands.drop if given). "
                         "Valid names: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12, "
                         "VV_asc,VH_asc,VV_desc,VH_desc. Use this for "
                         "modality-drop ablation runs, e.g. --test_only "
                         "--drop_bands VV_asc VH_asc VV_desc VH_desc.")

args = parser.parse_args()

# --resume_from takes precedence; --ckpt_path kept as a deprecated alias so
# existing launch commands/scripts don't break.
resume_ckpt_path = args.resume_from or args.ckpt_path
if args.ckpt_path and not args.resume_from:
    print("[BioMassters] NOTE: --ckpt_path is deprecated, use --resume_from instead "
          "(still honored for backward compat).")

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model         = read_yaml("./training/configs/" + args.config_model)
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# Override num_timesteps in config if specified via CLI
if args.num_timesteps is not None:
    if "dataset" not in config_model:
        config_model["dataset"] = {}
    config_model["dataset"]["num_timesteps"] = args.num_timesteps

# Override static band-drop config if --drop_bands was given. Applied
# BEFORE dataset construction, since BioMasstersSkipDataset reads
# trainer.bands.drop once at __init__ time (_resolve_drop_indices).
if args.drop_bands is not None:
    if "trainer" not in config_model:
        config_model["trainer"] = {}
    if "bands" not in config_model["trainer"]:
        config_model["trainer"]["bands"] = {}
    config_model["trainer"]["bands"]["drop"] = args.drop_bands
    print(f"[BioMassters] --drop_bands override: {args.drop_bands} "
          f"(overrides config trainer.bands.drop)")

# NOTE: unlike PASTIS/Sen1Floods11 (which load bands.yaml from disk),
# BioMassters' bands_info is built in Python via create_biomassters_bands_info()
# -- it references ABSTRACT_CHANNELS codes directly rather than a YAML file.
lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), create_biomassters_bands_info(), config_model)

fixed_T = config_model.get("dataset", {}).get("num_timesteps", BioMasstersSkipDataset.N_MONTHS)

print(f"\n[BioMassters] Experiment:   {args.xp_name}")
print(f"[BioMassters] Data dir:     {args.data_dir}")
print(f"[BioMassters] Timesteps:    {fixed_T} (fixed, pad-by-replication if a chip has fewer)")
print(f"[BioMassters] num_classes:  {config_model['trainer']['num_classes']} "
      f"(must be 1 for regression -- check config_test-Biomassters.yaml if this looks wrong)")
_skip_on = config_model.get("Atomiser", {}).get("use_decoder_skip", False)
print(f"[BioMassters] Decoder pixel-skip: {'ON' if _skip_on else 'OFF (baseline)'}")

# Band-drop / dropout-augmentation visibility -- read the SAME config keys
# BioMasstersSkipDataset itself reads, so this print can never drift from
# what the dataset actually does.
_drop_cfg = (config_model.get("trainer", {}).get("bands", {}) or {}).get("drop", None)
if _drop_cfg:
    print(f"[BioMassters] Static bands dropped (every split): {_drop_cfg}")
else:
    print(f"[BioMassters] Static band drop: none")
_aug_cfg = config_model.get("trainer", {}).get("band_dropout_augmentation", {}) or {}
if bool(_aug_cfg.get("enabled", True)):
    print(f"[BioMassters] Band-dropout augmentation (train only): ON "
          f"(p_applied={_aug_cfg.get('p_dropout_applied', 0.5)}, "
          f"p_whole_modality={_aug_cfg.get('p_whole_modality', 0.5)}, "
          f"p_band_drop={_aug_cfg.get('p_band_drop', 0.15)})")
else:
    print(f"[BioMassters] Band-dropout augmentation (train only): OFF")

if resume_ckpt_path:
    print(f"[BioMassters] Resume requested: {resume_ckpt_path} "
          f"(wait up to {args.resume_wait_seconds}s if not found yet)")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and not args.test_only:
    import wandb
    pretrain_tag = "pretrained" if args.pretrained_encoder else "scratch"
    run_name = f"BioMassters_{args.xp_name}_{pretrain_tag}"

    wandb_init_kwargs = dict(
        name=run_name,
        project="BioMassters",
        config={**config_model, "num_timesteps": fixed_T},
    )

    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"[BioMassters] Resuming wandb run: {args.wandb_run_id}")
    else:
        print(f"[BioMassters] Starting new wandb run: {run_name}")

    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="BioMassters")

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = BioMasstersDataModule(
    root_path=args.data_dir,
    config_model=config_model,
    look_up=lookup_table,
    batch_size=config_model["trainer"]["batchsize"],
    num_workers=args.num_workers,
)
data_module.setup()
print(f"[BioMassters] Lookup table: {len(lookup_table.table_wave)} entries")

# Target normalization stats (plain z-score), computed/cached by
# BioMasstersSkipDataset alongside the input band stats in the same
# normalization_stats.pt -- pulled from the already-constructed train
# dataset rather than re-reading the file separately.
_agb_mean = data_module.train_dataset.norm_stats["agb_mean"].item()
_agb_std  = data_module.train_dataset.norm_stats["agb_std"].item()
print(f"[BioMassters] AGB target normalization: z-score "
      f"(mean={_agb_mean:.4f}, std={_agb_std:.4f})")

# =============================================================================
# MODEL
# =============================================================================
model = Model_BioMassters_Skip(
    config=config_model, wand=True, name=args.xp_name,
    transform=None, lookup_table=lookup_table,
    agb_mean=_agb_mean, agb_std=_agb_std,
)

if args.pretrained_encoder:
    print(f"\n{'='*60}")
    print(f"  Loading pretrained encoder from: {args.pretrained_encoder}")
    print(f"{'='*60}")
    ckpt = torch.load(args.pretrained_encoder, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        full_state = ckpt["state_dict"]
        encoder_state = {k[len("encoder."):]: v
                         for k, v in full_state.items() if k.startswith("encoder.")}
    elif "encoder" in ckpt:
        encoder_state = ckpt["encoder"]
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
        print(f"    - {k}: {s} != {d}")
    print(f"{'='*60}\n")

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================
ckpt_dir = "./checkpoints/biomassters/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"biomassters_{args.xp_name}-{{epoch:02d}}-{{val_RMSE:.4f}}",
        monitor="val_RMSE", mode="min",
        save_top_k=1, verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"biomassters_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1, save_top_k=1, save_last=True, verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

num_nodes = int(os.environ.get("SLURM_NNODES", 1))
print(f"[BioMassters] num_nodes: {num_nodes} (from SLURM_NNODES, default 1 if unset)")

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1, num_nodes=num_nodes,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    ckpt_to_load = (
        resume_ckpt_path
        or config_model.get("trainer", {}).get("checkpoint_path")
    )
    if ckpt_to_load is None:
        raise ValueError(
            "--test_only requires a checkpoint. Either pass --resume_from "
            "or set trainer.checkpoint_path in the config YAML."
        )
    ckpt_to_load = wait_for_checkpoint(
        ckpt_to_load, args.resume_wait_seconds, args.resume_poll_interval)

    print(f"\n{'='*60}")
    print(f"  BioMassters — TEST ONLY")
    print(f"  Checkpoint: {ckpt_to_load}")
    print(f"  Timesteps:  {fixed_T} (fixed)")
    if args.drop_bands:
        print(f"  Modality-drop ablation: {args.drop_bands}")
    print(f"{'='*60}\n")

    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[BioMassters] Loaded checkpoint — "
          f"missing: {len(result.missing_keys)}, "
          f"unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"[BioMassters] First 5 missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[BioMassters] First 5 unexpected: {result.unexpected_keys[:5]}")

    trainer.validate(model, datamodule=data_module)
    results = trainer.test(model, datamodule=data_module)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        metrics = results[0] if results else {}
        rmse = metrics.get("test_RMSE", float("nan"))
        mae  = metrics.get("test_MAE", float("nan"))
        drop_str = ",".join(args.drop_bands) if args.drop_bands else "none"
        print(f"RESULT test_only ckpt={ckpt_to_load} drop={drop_str} "
              f"test_RMSE={rmse:.6f} test_MAE={mae:.6f}")
else:
    fit_ckpt_path = None
    if resume_ckpt_path:
        fit_ckpt_path = wait_for_checkpoint(
            resume_ckpt_path, args.resume_wait_seconds, args.resume_poll_interval)

    print(f"\n{'='*60}")
    print(f"  BioMassters")
    print(f"  Timesteps: {fixed_T} (fixed, pad-by-replication)")
    print(f"  Train/Val: carved from train_features (10% held out) → Test: test_features")
    if fit_ckpt_path is not None:
        print(f"  RESUMING from: {fit_ckpt_path}")
        if args.wandb_run_id:
            print(f"  Wandb run:     {args.wandb_run_id}")
    print(f"{'='*60}\n")

    trainer.fit(model, datamodule=data_module, ckpt_path=fit_ckpt_path)
    trainer.test(model, datamodule=data_module)

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/biomassters_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
