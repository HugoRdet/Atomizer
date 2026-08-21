"""
PASTIS-HD Training Script — Multi-temporal Crop Segmentation (SKIP variant)
=============================================================================

Train Atomiser-IO on PASTIS-HD for crop type segmentation, with support for
the decoder pixel-skip cascade (Atomiser_Senflood_Skip), selected via
config["Atomiser"]["use_decoder_skip"]. PASTISTrainer already branches on
that flag, so this script does not need a separate "Skip" trainer class --
PASTISTrainer + a skip-enabled config + the skip dataset/collate is enough.

>>> TEMPORAL: also supports config["Atomiser"]["use_temporal_transformer"]
(or --use_temporal_transformer), which swaps the dataset/collate pair to
PastisHDTemporalDataset + pastis_temporal_collate_fn. That dataset builds
per-timestep [T, N, 8] token groups (S2 anchor, S1 nearest-calendar-date
matched) instead of one flat concatenated pool, required by AtomiserTemporal
(training/atomiser/Atomiser_temporal.py). It ALWAYS requires S1 (raises if
use_s1=False) and NEVER uses SPOT (forces use_spot=False internally,
regardless of --use_spot). PASTISTrainer's encoder-selection branches on the
same config flag, so setting use_temporal_transformer: true in the config
(or passing --use_temporal_transformer) is sufficient for both sides to
switch in lockstep.

Dataset (utils_dataset_PASTIS.PastisHDDataset) now emits, on every sample:
    "query_token_idx":   [N_q, Ts2*C2 + Ts1*C1] long
    "query_token_valid": [N_q] bool
which index into groups[10.0]["tokens"] (the SAT group only -- NOT SPOT).

Collate (token_grouping.collate_multitask) is already skip-aware: it pads
query_token_idx/query_token_valid in lockstep with queries and generalizes
over however many resolution groups a sample has (SAT@10m, optionally
SPOT@1m). No PASTIS-specific collate changes were needed.

  - S2 (10 bands, multi-temporal) — always enabled
  - S1A (3 bands, multi-temporal) — optional via --use_s1 (REQUIRED if
    --use_temporal_transformer is set)
  - SPOT6 (3 bands, single frame) — optional via --use_spot (ignored if
    --use_temporal_transformer is set)

Splits (fold-based):
  - Train: folds 1, 2, 3
  - Val:   fold 4
  - Test:  fold 5

# >>> MULTI_SEED_TEST: --test_only now runs N_TEST_RUNS repeats, each with a
# different seed (list hardcoded below), and reports mean/max/min/std across
# runs for test_mIoU and test_accuracy. Mirrors the SenFlood skip script:
# test-time forward passes are not fully deterministic across seeds because
# of per-step random subsampling in the spatial sampler, so repeating under
# different seeds captures that sampling variance.

Examples:
    # S2-only, from scratch
    python train_pastis.py --xp_name pastis_s2only

    # S2 + S1 + SPOT, full temporal (flat token, no TemporalTransformer)
    python train_pastis.py --xp_name pastis_full \
        --use_s1 --use_spot --multi_temporal 10

    # S2 + S1, TemporalTransformer aggregation (per-timestep groups)
    python train_pastis.py --xp_name pastis_temporal \
        --use_s1 --use_temporal_transformer --multi_temporal 6

    # Test-only, N_TEST_RUNS repeats with summary stats
    python train_pastis.py --xp_name pastis_full --test_only \
        --ckpt_path ./checkpoints/pastis/pastis_full-last-epoch=42.ckpt
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
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
from training.trainer_PASTIS import PASTISTrainer
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder

# >>> TEMPORAL
from training.utils.datasets.pastis_temporal_dataset import (
    PastisHDTemporalDataset,
    pastis_temporal_collate_fn,
)

from training.utils.callbacks.segmentation_viz_callback import SegmentationVizCallback

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
TEST_SEEDS = list(range(1, N_TEST_RUNS + 1))  # seed == run number (1, 2, 3, ...)


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
#
# >>> SKIP: collate_multitask (token_grouping.py) already pads
# query_token_idx / query_token_valid in lockstep with queries and puts them
# at the TOP LEVEL of the returned dict (not nested under "tasks"), so no
# extra handling is needed here beyond the pre-existing queries lift.
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
#
# >>> TEMPORAL: dataset_cls / collate_fn are now parameters instead of being
# hardcoded, so the SAME datamodule class serves both the flat/SKIP path and
# the per-timestep TemporalTransformer path. Which pair gets used is decided
# once, in the __main__ block below, from --use_temporal_transformer /
# config["Atomiser"]["use_temporal_transformer"].
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
parser = argparse.ArgumentParser(description="PASTIS-HD Training (skip variant)")
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

# Wandb resume
parser.add_argument("--wandb_run_id",   type=str, default=None,
                    help="Wandb run ID to resume logging into (use with --ckpt_path)")

# Test-only mode
parser.add_argument("--test_only",      action="store_true",
                    help="Skip training, load checkpoint from config trainer.checkpoint_path "
                         "(or --ckpt_path) and run test split only. "
                         f"Runs {N_TEST_RUNS} repeats under different seeds "
                         "and reports mean/max/min/std, same as the SenFlood "
                         "skip script.")

# Modality toggles
parser.add_argument("--use_s1",         action="store_true",
                    help="Enable S1A SAR data (default: S2-only)")
parser.add_argument("--use_spot",       action="store_true",
                    help="Enable SPOT6 RGB data (default: S2-only; ignored "
                         "if --use_temporal_transformer is set)")

# >>> TEMPORAL
parser.add_argument("--use_temporal_transformer", action="store_true",
                    help="Use AtomiserTemporal: per-timestep "
                         "Atomiser_Senflood_Skip encoding + TemporalTransformer "
                         "aggregation, with S2/S1 aligned via nearest-calendar-"
                         "date matching (PastisHDTemporalDataset). Requires "
                         "--use_s1. SPOT is not supported in this mode.")

# Temporal config
parser.add_argument("--multi_temporal", type=int, default=None,
                    help="Number of temporal frames (overrides config)")

args = parser.parse_args()

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model         = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path      = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# Override multi_temporal in config if specified via CLI
if args.multi_temporal is not None:
    if "dataset" not in config_model:
        config_model["dataset"] = {}
    config_model["dataset"]["multi_temporal"] = args.multi_temporal

# >>> TEMPORAL: CLI flag wins, but keep the config value in sync so
# PASTISTrainer's encoder-selection branch (which reads
# config["Atomiser"]["use_temporal_transformer"]) matches what this script
# actually wires up on the data side. Also validate the S1/SPOT constraints
# up front rather than letting PastisHDTemporalDataset raise mid-__init__.
if "Atomiser" not in config_model:
    config_model["Atomiser"] = {}

_use_temporal_transformer = (
    args.use_temporal_transformer
    or config_model["Atomiser"].get("use_temporal_transformer", False)
)
config_model["Atomiser"]["use_temporal_transformer"] = _use_temporal_transformer

if _use_temporal_transformer:
    if not args.use_s1:
        raise ValueError(
            "--use_temporal_transformer requires --use_s1: "
            "PastisHDTemporalDataset always pairs S2 with S1 via "
            "nearest-calendar-date matching."
        )
    if args.use_spot:
        print("[PASTIS] [WARNING] --use_spot is ignored with "
              "--use_temporal_transformer (SPOT is out of scope for "
              "PastisHDTemporalDataset).")

# >>> SKIP: sanity check -- the skip gather index built by PastisHDDataset is
# only useful (and only matches what Atomiser_Senflood_Skip expects) if the
# skip encoder is actually selected. Warn loudly rather than silently
# training/testing with the base encoder while paying the extra dataset cost.
_use_decoder_skip = config_model.get("Atomiser", {}).get("use_decoder_skip", False)
if os.environ.get("LOCAL_RANK", "0") == "0":
    if _use_temporal_transformer:
        print(f"[PASTIS] Temporal transformer: ON "
              f"(AtomiserTemporal, PastisHDTemporalDataset, S2-anchored "
              f"nearest-date S1 matching)")
        if _use_decoder_skip:
            print(f"[PASTIS] Decoder pixel-skip: ON (active inside AtomiserTemporal)")
        else:
            print(f"[PASTIS] Decoder pixel-skip: OFF "
                  f"(query_token_idx/query_token_valid computed and collated "
                  f"but ignored by AtomiserTemporal.forward)")
    elif _use_decoder_skip:
        print(f"[PASTIS] Decoder pixel-skip: ON "
              f"(Atomiser_Senflood_Skip, via config['Atomiser']['use_decoder_skip'])")
    else:
        print(f"[PASTIS] Decoder pixel-skip: OFF -- "
              f"query_token_idx/query_token_valid will be computed by the "
              f"dataset and collated, but PASTISTrainer will NOT use them "
              f"(falls back to AtomiserLTAE or Atomiser_Senflood). Set "
              f"config['Atomiser']['use_decoder_skip'] = true to enable the skip cascade.")
    print(f"[PASTIS] Deterministic ops: {'ON' if args.deterministic else 'OFF'}")
    if args.test_only:
        print(f"[PASTIS] TEST-ONLY mode")
        print(f"[PASTIS] TEST-ONLY will run {N_TEST_RUNS} repeats "
              f"with seeds {TEST_SEEDS}")

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

# Register VV-VH SAR channel (not in bands.yaml, needed for S1 3rd band)
if args.use_s1:
    lookup_table.register_abstract_channel("VV_VH")

# Build modality description for logging
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
# WANDB  (skip in test-only mode, same convention as the SenFlood skip script)
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and not args.test_only:
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

    # Resume an existing wandb run if requested
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]      = args.wandb_run_id
        wandb_init_kwargs["resume"]  = "must"        # fail if run doesn't exist
        print(f"[PASTIS] Resuming wandb run: {args.wandb_run_id}")
    else:
        print(f"[PASTIS] Starting new wandb run: {run_name}")

    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="PASTIS")

# =============================================================================
# DATA MODULE
#
# >>> TEMPORAL: dataset_cls / collate_fn swapped based on
# use_temporal_transformer. Everything else about the datamodule is shared.
# =============================================================================
_dataset_cls = PastisHDTemporalDataset if _use_temporal_transformer else PastisHDDataset
_collate_fn  = pastis_temporal_collate_fn if _use_temporal_transformer else pastis_collate

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
model = PASTISTrainer(
    config=config_model, wand=(not args.test_only), name=args.xp_name,
    transform=None, lookup_table=lookup_table,
)

if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)

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
    devices=-1,num_nodes=num_nodes, max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum,
    deterministic=args.deterministic,
)

# =============================================================================
# TEST-ONLY EARLY EXIT  (>>> MULTI_SEED_TEST: N repeats + summary stats)
# =============================================================================
if args.test_only:
    ckpt_to_load = (
        args.ckpt_path
        or config_model.get("trainer", {}).get("checkpoint_path")
    )
    if ckpt_to_load is None:
        raise ValueError(
            "--test_only requires a checkpoint. Either pass --ckpt_path "
            "or set trainer.checkpoint_path in the config YAML."
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"\n{'='*60}")
        print(f"  PASTIS-HD — TEST ONLY")
        print(f"  Checkpoint: {ckpt_to_load}")
        print(f"  Modalities: {modality_str}")
        print(f"  Temporal:   {multi_temporal} frames")
        print(f"{'='*60}\n")

    # Load weights into the already-constructed model once, up front --
    # each repeat reuses these weights, only the seed (and therefore the
    # stochastic spatial sampling at eval time) changes.
    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[PASTIS] Loaded checkpoint — "
              f"missing: {len(result.missing_keys)}, "
              f"unexpected: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"[PASTIS] First 5 missing: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"[PASTIS] First 5 unexpected: {result.unexpected_keys[:5]}")
    model.eval()

    miou_runs, acc_runs = [], []

    for run_idx, seed in enumerate(TEST_SEEDS, 1):
        # Reseed before each run so any stochastic eval-time behavior (e.g.
        # spatial sampler subsampling) draws a fresh sample per run rather
        # than repeating the same draw N times.
        seed_everything(seed, workers=True)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"\n[PASTIS][TEST-ONLY] Run {run_idx}/{N_TEST_RUNS} "
                  f"(seed={seed})")

        # Fresh Trainer per run: avoids any state (e.g. logged metrics,
        # internal loop counters) carrying over between repeated .test()
        # calls on the same Trainer instance.
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

        if os.environ.get("LOCAL_RANK", "0") == "0":
            metrics = results[0] if results else {}
            miou = metrics.get("test_mIoU", float("nan"))
            acc  = metrics.get("test_accuracy", float("nan"))
            miou_runs.append(miou)
            acc_runs.append(acc)
            print(f"RESULT test_only ckpt={ckpt_to_load} run={run_idx} "
                  f"seed={seed} test_mIoU={miou:.6f} test_accuracy={acc:.6f}")

    if os.environ.get("LOCAL_RANK", "0") == "0":
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

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/pastis_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
