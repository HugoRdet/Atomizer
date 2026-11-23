"""
PASTIS-HD Training Script — Multi-temporal Crop Segmentation (SKIP / U-Net variant)
====================================================================================

Train Atomiser-IO on PASTIS-HD for crop type segmentation, with support for
the decoder pixel-skip cascade (Atomiser_Senflood_Skip), selected via
config["Atomiser"]["use_decoder_skip"]. PASTISTrainer already branches on
that flag, so this script does not need a separate "Skip" trainer class.

>>> UNET: when the config is an Atomiser_Unet config (encoder name contains
"unet", or Atomiser.unet_task is set), PASTISTrainer builds Atomiser_Unet;
the PASTIS metric (crop mIoU over classes 1-18) is the same for every
encoder. The U-Net uses JOINT temporal encoding: all
dates' atoms go into one set, acquisition time is atom metadata, and every
pixel is one site with T x C atoms. Constraints enforced below:
    * no --use_temporal_transformer (the U-Net is single-pass)
    * no --use_spot (a 1 m group would dominate the hierarchy geometry)
    * a fixed token layout across samples (pre-flight check)

>>> TEMPORAL: also supports config["Atomiser"]["use_temporal_transformer"]
(or --use_temporal_transformer), which swaps the dataset/collate pair to
PastisHDTemporalDataset + pastis_temporal_collate_fn (per-timestep groups,
required by AtomiserTemporal). It ALWAYS requires S1 and NEVER uses SPOT.

Dataset (utils_dataset_PASTIS.PastisHDDataset) emits, on every sample:
    "query_token_idx":   [N_q, Ts2*C2 + Ts1*C1] long
    "query_token_valid": [N_q] bool
which index into groups[10.0]["tokens"] (the SAT group only -- NOT SPOT).
Atomiser_Unet ignores them.

Splits (fold-based):
  - Train: folds 1, 2, 3
  - Val:   fold 4
  - Test:  fold 5

# >>> MULTI_SEED_TEST: --test_only runs N_TEST_RUNS repeats with different
# seeds and reports mean/max/min/std. (The U-Net's evaluation is
# deterministic, so its repeats give identical numbers.)

Examples:
    # U-Net, S2 + S1, per-date encoding + per-level temporal aggregation
    python train_pastis.py --xp_name pastis_unet_temporal --use_s1 \
        --config_model config_test-PASTIS-unet-legacy-temporal.yaml

    # U-Net, S2 + S1, joint temporal encoding
    python train_pastis.py --xp_name pastis_unet --use_s1 \
        --config_model config_test-PASTIS-unet.yaml

    # Baseline, S2 + S1, TemporalTransformer aggregation
    python train_pastis.py --xp_name pastis_temporal \
        --use_s1 --use_temporal_transformer --multi_temporal 6
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import statistics
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

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_PASTIS import PASTISTrainer          # builds Atomiser_Unet for U-Net configs
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder

# >>> TEMPORAL
from training.utils.datasets.pastis_temporal_dataset import (
    PastisHDTemporalDataset,
    pastis_temporal_collate_fn,
)

from training.utils.callbacks.segmentation_viz_callback import SegmentationVizCallback

IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

# =============================================================================
# KNOWN RESOLUTIONS
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# MULTI-SEED TEST CONFIG
# =============================================================================
N_TEST_RUNS = 2
TEST_SEEDS = list(range(1, N_TEST_RUNS + 1))


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
        print(f"    - {k}: {s} != {d}")
    print(f"{'='*60}\n")
    return model


# =============================================================================
# COLLATE (flat/SKIP variant — used when NOT use_temporal_transformer)
# =============================================================================

def pastis_collate(samples):
    batch = collate_multitask(samples)
    if "queries" not in batch and "tasks" in batch:
        task_data = next(iter(batch["tasks"].values()))
        batch["queries"] = task_data["queries"]
        batch["queries_mask"] = task_data["queries_mask"]
    return batch


# =============================================================================
# DATAMODULE
# =============================================================================

class PastisDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root_path: str,
        config_model: dict,
        look_up,
        batch_size: int = 4,
        num_workers: int = 4,
        use_s1: bool = True,
        use_spot: bool = True,
        dataset_cls=PastisHDDataset,
        collate_fn=pastis_collate,
    ):
        super().__init__()
        self.root_path    = root_path
        self.config_model = config_model
        self.look_up      = look_up
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.use_s1       = use_s1
        self.use_spot     = use_spot
        self.dataset_cls  = dataset_cls
        self.collate_fn   = collate_fn

    def _make_dataset(self, mode: str):
        return self.dataset_cls(
            root_path=self.root_path,
            mode=mode,
            config_model=self.config_model,
            look_up=self.look_up,
            use_s1=self.use_s1,
            use_spot=self.use_spot,
        )

    def setup(self, stage=None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return
        self._setup_done = True

        self.train_dataset = self._make_dataset("train")
        self.val_dataset   = self._make_dataset("validation")
        self.test_dataset  = self._make_dataset("test")

        modalities = ["S2"]
        if self.use_s1:
            modalities.append("S1")
        if self.use_spot:
            modalities.append("SPOT")

        print(f"\n[PASTIS-DM] Summary:")
        print(f"  Dataset:    {self.dataset_cls.__name__}")
        print(f"  Modalities: {' + '.join(modalities)}")
        print(f"  Train: {len(self.train_dataset)} patches (folds 1,2,3)")
        print(f"  Val:   {len(self.val_dataset)} patches (fold 4)")
        print(f"  Test:  {len(self.test_dataset)} patches (fold 5)")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=self.collate_fn,
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
parser = argparse.ArgumentParser(description="PASTIS-HD Training (skip / U-Net variant)")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str,
                    default="config_test-PASTIS.yaml")
parser.add_argument("--data_dir",       type=str, default="./data/PASTIS-HD")
parser.add_argument("--ckpt_path",      type=str, default=None,
                    help="Resume training from Lightning checkpoint, "
                         "or checkpoint to load for --test_only")
parser.add_argument("--pretrained_encoder", type=str, default=None,
                    help="Load pretrained encoder weights (no head)")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--grad_accum",     type=int, default=1)
parser.add_argument("--deterministic",  action="store_true",
                    help="Force deterministic CUDA ops")
parser.add_argument("--wandb_run_id",   type=str, default=None,
                    help="Wandb run ID to resume logging into (use with --ckpt_path)")
parser.add_argument("--test_only",      action="store_true",
                    help="Skip training, load checkpoint (--ckpt_path or config "
                         f"trainer.checkpoint_path) and run test only, {N_TEST_RUNS} repeats.")
parser.add_argument("--use_s1",         action="store_true",
                    help="Enable S1A SAR data (default: S2-only)")
parser.add_argument("--use_spot",       action="store_true",
                    help="Enable SPOT6 RGB data (not supported by the U-Net)")
parser.add_argument("--use_temporal_transformer", action="store_true",
                    help="Use AtomiserTemporal (baseline only; requires --use_s1)")
parser.add_argument("--multi_temporal", type=int, default=None,
                    help="Number of temporal frames (overrides config)")
# >>> UNET
parser.add_argument("--skip_preflight", action="store_true",
                    help="Skip the token-layout sanity check (U-Net only)")
# >>> END UNET

args = parser.parse_args()

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model         = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path      = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

if args.multi_temporal is not None:
    config_model.setdefault("dataset", {})["multi_temporal"] = args.multi_temporal

config_model.setdefault("Atomiser", {})

_use_temporal_transformer = (
    args.use_temporal_transformer
    or config_model["Atomiser"].get("use_temporal_transformer", False)
)
config_model["Atomiser"]["use_temporal_transformer"] = _use_temporal_transformer

# >>> UNET: detection + constraints
a_cfg = config_model["Atomiser"]
is_unet = ("unet" in str(config_model.get("encoder", "")).lower() or "unet_task" in a_cfg)
unet_temporal = is_unet and a_cfg.get("unet_temporal", "none") == "per_level"
if is_unet:
    if _use_temporal_transformer:
        raise ValueError("Atomiser_Unet: drop --use_temporal_transformer / "
                         "Atomiser.use_temporal_transformer (for per-date encoding set "
                         "Atomiser.unet_temporal: per_level instead).")
    if unet_temporal and not args.use_s1:
        raise ValueError("unet_temporal: per_level uses PastisHDTemporalDataset, which "
                         "requires --use_s1.")
    if args.use_spot:
        raise ValueError("Atomiser_Unet on PASTIS: SPOT is not supported (a 1 m group "
                         "would dominate the hierarchy geometry). Drop --use_spot.")
    if a_cfg.get("unet_task", "segmentation") != "segmentation":
        raise ValueError("PASTIS is segmentation: set Atomiser.unet_task: segmentation")
    a_cfg["unet_task"] = "segmentation"
    a_cfg["use_error_predictor"] = False
    config_model["trainer"]["slide"] = False
    if "unet" not in str(config_model.get("encoder", "")).lower():
        config_model["encoder"] = f"{config_model.get('encoder', 'ATOMIZER')}_Unet"
# >>> END UNET

if _use_temporal_transformer:
    if not args.use_s1:
        raise ValueError("--use_temporal_transformer requires --use_s1.")
    if args.use_spot:
        print("[PASTIS] [WARNING] --use_spot is ignored with --use_temporal_transformer.")

_use_decoder_skip = a_cfg.get("use_decoder_skip", False)
if IS_RANK0:
    if is_unet:
        print(f"[PASTIS] Encoder: Atomiser_Unet "
              f"({'per-level temporal aggregation' if unet_temporal else 'joint temporal encoding'})")
    elif _use_temporal_transformer:
        print(f"[PASTIS] Temporal transformer: ON (AtomiserTemporal)")
    elif _use_decoder_skip:
        print(f"[PASTIS] Decoder pixel-skip: ON (Atomiser_Senflood_Skip)")
    else:
        print(f"[PASTIS] Decoder pixel-skip: OFF (PASTISTrainer default encoder)")
    print(f"[PASTIS] Deterministic ops: {'ON' if args.deterministic else 'OFF'}")
    if args.test_only:
        print(f"[PASTIS] TEST-ONLY mode, {N_TEST_RUNS} repeats with seeds {TEST_SEEDS}")

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

if args.use_s1:
    lookup_table.register_abstract_channel("VV_VH")

modalities = ["S2"]
if args.use_s1:
    modalities.append("S1")
if args.use_spot and not _use_temporal_transformer:
    modalities.append("SPOT")
modality_str = "+".join(modalities)

multi_temporal = config_model.get("dataset", {}).get("multi_temporal", 10)

print(f"\n[PASTIS] Experiment:   {args.xp_name}")
print(f"[PASTIS] Data dir:     {args.data_dir}")
print(f"[PASTIS] Modalities:   {modality_str}")
print(f"[PASTIS] Temporal:     {multi_temporal} frames (uniform via linspace)")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if IS_RANK0 and not args.test_only:
    import wandb
    pretrain_tag = "pretrained" if args.pretrained_encoder else "scratch"
    run_name = f"PASTIS_{args.xp_name}_{modality_str}_{pretrain_tag}"

    wandb_init_kwargs = dict(
        name=run_name,
        project="PASTIS",
        config={
            **config_model,
            "modalities":               modalities,
            "use_s1":                   args.use_s1,
            "use_spot":                 args.use_spot and not _use_temporal_transformer,
            "use_temporal_transformer": _use_temporal_transformer,
            "multi_temporal":           multi_temporal,
        },
    )
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"[PASTIS] Resuming wandb run: {args.wandb_run_id}")
    else:
        print(f"[PASTIS] Starting new wandb run: {run_name}")

    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="PASTIS")

# =============================================================================
# DATA MODULE
# =============================================================================
# >>> UNET: per-date tokens for unet_temporal: per_level too
_per_date = _use_temporal_transformer or unet_temporal
_dataset_cls = PastisHDTemporalDataset if _per_date else PastisHDDataset
_collate_fn  = pastis_temporal_collate_fn if _per_date else pastis_collate

data_module = PastisDataModule(
    root_path=args.data_dir,
    config_model=config_model,
    look_up=lookup_table,
    batch_size=config_model["trainer"]["batchsize"],
    num_workers=args.num_workers,
    use_s1=args.use_s1,
    use_spot=args.use_spot,
    dataset_cls=_dataset_cls,
    collate_fn=_collate_fn,
)
data_module.setup()
print(f"[PASTIS] Lookup table: {len(lookup_table.table_wave)} entries")

# =============================================================================
# MODEL
# =============================================================================
# >>> UNET: PASTISTrainer builds Atomiser_Unet for U-Net configs and keeps the
# PASTIS metric (crop mIoU over classes 1-18) for every encoder
model = PASTISTrainer(
    config=config_model, wand=(not args.test_only), name=args.xp_name,
    transform=None, lookup_table=lookup_table,
)
# >>> END UNET

if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)


# =============================================================================
# >>> UNET: PRE-FLIGHT — shared token layout across samples
# (built AFTER the model, so the lookup table already holds every day of year)
# =============================================================================
def preflight_token_layout(dm):
    ds = dm.train_dataset
    s0, s1 = ds[0], ds[1 % len(ds)]
    batch = _collate_fn([s0, s1])
    for res, grp in batch["groups"].items():
        t = grp["tokens"]
        if t.dim() == 4:                                    # [B, T, N, 8]: per-date tokens
            same_dates = all(torch.equal(t[0, 0, :, 1:3], t[0, d, :, 1:3]) for d in range(t.shape[1]))
            if not same_dates:
                raise RuntimeError(f"[Preflight] group {res}: token positions differ between "
                                   "dates; the per-date U-Net needs one layout for all dates.")
            print(f"[Preflight] group {res}: {t.shape[1]} dates x {t.shape[2]} atoms, "
                  f"positions identical across dates ✓")
            t = t[:, 0]
        if not torch.equal(t[0, :, 1:3], t[1, :, 1:3]):
            raise RuntimeError(
                f"[Preflight] group {res}: token positions differ between samples "
                f"(e.g. a series shorter than multi_temporal={multi_temporal}). "
                "The U-Net's hierarchy geometry needs a fixed token layout.")
        n_dates = int(torch.unique(grp["tokens"][0][..., 7]).numel())
        print(f"[Preflight] group {res}: tokens {tuple(grp['tokens'].shape)}, positions identical "
              f"across samples ✓  ({n_dates} distinct time indices in sample 0)")
    if unet_temporal:
        tp = batch.get("time_positions", batch.get("time_indices"))
        print(f"[Preflight] time positions for the temporal transformers: "
              f"{'missing (ordinal fallback)' if tp is None else tuple(tp.shape)}")
    if len(batch["groups"]) > 1:
        raise RuntimeError("[Preflight] more than one resolution group (SPOT?) — "
                           "not supported by the U-Net here.")
    print(f"[Preflight] queries {tuple(batch['queries'].shape)} ✓")


if is_unet and IS_RANK0 and not args.skip_preflight and not args.test_only:
    preflight_token_layout(data_module)
# >>> END UNET

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================
ckpt_dir = f"./checkpoints/pastis/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"pastis_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU", mode="max",
        save_top_k=1, verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"pastis_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1, save_top_k=1, save_last=True, verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

num_nodes = int(os.environ.get("SLURM_NNODES", 1))

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1, num_nodes=num_nodes, max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum,
    deterministic=args.deterministic,
)

# =============================================================================
# TEST-ONLY EARLY EXIT
# =============================================================================
if args.test_only:
    ckpt_to_load = (
        args.ckpt_path
        or config_model.get("trainer", {}).get("checkpoint_path")
    )
    if ckpt_to_load is None:
        raise ValueError("--test_only requires --ckpt_path or trainer.checkpoint_path.")

    if IS_RANK0:
        print(f"\n{'='*60}")
        print(f"  PASTIS-HD — TEST ONLY")
        print(f"  Checkpoint: {ckpt_to_load}")
        print(f"  Modalities: {modality_str}")
        print(f"  Temporal:   {multi_temporal} frames")
        print(f"{'='*60}\n")

    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    if IS_RANK0:
        print(f"[PASTIS] Loaded checkpoint — "
              f"missing: {len(result.missing_keys)}, "
              f"unexpected: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"[PASTIS] First 5 missing: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"[PASTIS] First 5 unexpected: {result.unexpected_keys[:5]}")
    # >>> UNET: a partially loaded model gives meaningless numbers
    if result.missing_keys:
        raise RuntimeError("[PASTIS] checkpoint does not match the model built from "
                           "--config_model (missing weights).")
    model.eval()

    miou_runs, acc_runs = [], []

    for run_idx, seed in enumerate(TEST_SEEDS, 1):
        seed_everything(seed, workers=True)
        if IS_RANK0:
            print(f"\n[PASTIS][TEST-ONLY] Run {run_idx}/{N_TEST_RUNS} (seed={seed})")

        run_trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=True),
            use_distributed_sampler=False,
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
            print(f"RESULT test_only ckpt={ckpt_to_load} run={run_idx} "
                  f"seed={seed} test_mIoU={miou:.6f} test_accuracy={acc:.6f}")

    if IS_RANK0:
        def _stats(vals):
            vals = [v for v in vals if v == v]
            if not vals:
                return dict(mean=float("nan"), max=float("nan"),
                            min=float("nan"), std=float("nan"), n=0)
            return dict(mean=statistics.mean(vals), max=max(vals), min=min(vals),
                        std=statistics.stdev(vals) if len(vals) > 1 else 0.0, n=len(vals))

        miou_stats = _stats(miou_runs)
        acc_stats  = _stats(acc_runs)

        print("\n" + "=" * 78)
        print(f"TEST-ONLY SUMMARY over {N_TEST_RUNS} runs "
              f"(seeds={TEST_SEEDS}) — ckpt={ckpt_to_load}")
        print("=" * 78)
        print(f"  test_mIoU     : mean={miou_stats['mean']:.6f}  "
              f"max={miou_stats['max']:.6f}  min={miou_stats['min']:.6f}  "
              f"std={miou_stats['std']:.6f}  (n={miou_stats['n']})")
        print(f"  test_accuracy : mean={acc_stats['mean']:.6f}  "
              f"max={acc_stats['max']:.6f}  min={acc_stats['min']:.6f}  "
              f"std={acc_stats['std']:.6f}  (n={acc_stats['n']})")
        print("=" * 78)
        print(f"RESULT_SUMMARY test_only ckpt={ckpt_to_load} "
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
# TRAIN
# =============================================================================
print(f"\n{'='*60}")
print(f"  PASTIS-HD — {modality_str}")
print(f"  Temporal: {multi_temporal} frames (linspace)")
print(f"  Train: folds 1,2,3 → Val: fold 4 → Test: fold 5")
if args.ckpt_path is not None:
    print(f"  RESUMING from: {args.ckpt_path}")
    if args.wandb_run_id:
        print(f"  Wandb run:     {args.wandb_run_id}")
print(f"{'='*60}\n")

trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
trainer.test(model, datamodule=data_module)

if wandb_logger and IS_RANK0:
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/pastis_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
