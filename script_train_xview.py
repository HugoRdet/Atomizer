"""
Atomiser xView2 Training Script (SKIP + optional TEMPORAL variant)
=====================================================================

5-class damage segmentation on xView2 (PANGAEA setup):
    Input:  pre + post RGB, [T=2, C=3, H=512, W=512]
    Target: 5-class damage mask {0..4}, no IGNORE
    Native resolution: 0.5 m (sub-meter aerial WorldView-3)

Uses Model_SenFlood_Skip (training.trainer_SENFLOOD_skip) -- the SAME
trainer class Sen1Floods11's skip variant uses (script_train_senflood_skip.py)
-- xView2 is just another multi-temporal segmentation task from its
perspective, same as the non-skip Model_SenFlood was before.

>>> TEMPORAL: also supports config["Atomiser"]["use_temporal_transformer"]
(or --use_temporal_transformer), which swaps the dataset/collate pair to
XView2TemporalDataset + xview2_temporal_collate_fn (both in
training/utils/datasets/xview2_temporal_dataset.py). That dataset builds
per-timestep [T, N, 8] token groups (T=2, pre/post; single RGB modality, no
cross-sensor alignment needed unlike PASTIS) instead of one flat
concatenated pool, required by AtomiserTemporal
(training/atomiser/Atomiser_temporal.py). Model_SenFlood_Skip's
encoder-selection branches on the same config flag, so setting
use_temporal_transformer: true in the config (or passing
--use_temporal_transformer) is sufficient for both the dataset/collate side
here and the encoder side in the trainer to switch in lockstep -- same
pattern as script_train_PASTIS.py.

NOTE on "sliding window" -- two DIFFERENT mechanisms share this name here:
  1. config["trainer"]["slide"]: drives Model_SenFlood_Skip's
     _sliding_window_step/_forward_crop path. INCOMPATIBLE with
     use_decoder_skip and/or use_temporal_transformer (the trainer raises
     at construction if both are set -- _forward_crop's mini_batch drops
     query_token_idx/valid and time_positions).
  2. THIS SCRIPT's own sliding-window TEST evaluation
     (SlidingWindowTileDataset / run_sliding_window_test, "Sliding-window
     test (additional, runs by default)" section below): tiles via
     get_tile_sample -> the normal _build_sample_from_crop path (full,
     self-contained samples, including time_positions and a complete
     query_token_idx for the temporal path) -> the ORDINARY test_step, not
     _forward_crop. This one IS compatible with use_decoder_skip and
     use_temporal_transformer and runs regardless of config["trainer"]["slide"].
Keep trainer.slide=False for both the skip and temporal runs; this
script's own sliding-window test (on by default, --skip_sliding_window to
disable) is unaffected by that setting either way.

Uses Model_SenFlood_Skip's trainer, paired with either collate_grouped_skip
(flat token, default) or xview2_temporal_collate_fn (per-timestep, when
--use_temporal_transformer is set) -- the generic grouped-skip collate
already handles variable-length query_token_idx/query_token_valid padding
for the flat case; xview2_temporal_collate_fn does the analogous padding
for the [T, N, 8] case (see its own docstring for why it pads rather than
assumes fixed sizes).

XView2Dataset / XView2TemporalDataset must already build
query_token_idx/query_token_valid per sample (the per-pixel SKIP gather
index) for this to work -- see that dataset's _build_query_token_index /
_build_full_pixel_index.

Requires the config (--config_model) to set Atomiser.use_decoder_skip: true
(and whatever else Model_SenFlood_Skip's decoder needs) for skip mode to
actually activate, and Atomiser.use_temporal_transformer: true for the
temporal module -- this script only reports what the config says (and
syncs --use_temporal_transformer into it), it doesn't set use_decoder_skip
for you.

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

FLOPs at test time:
    Right before the Lightning test() call, we measure GFLOPs/forward with
    torch.utils.flop_counter.FlopCounterMode -- same convention as the
    UniverSat sweep scripts and script_train_xview_baselines.py: one
    warmup pass discarded, mean over --flops_n counted passes, GFLOPs =
    total_flops / 1e9. Unlike the baseline scripts (plain [N,T,C,H,W]
    tensor in, logits out), Atomiser's forward takes a whole batch dict
    (tokenized queries + query_token_idx/query_token_valid from the active
    collate function), so the FLOPs pass pulls a few real batches from
    data_module's test dataloader and calls model(batch) directly --
    the same call Lightning's test_step makes under the hood for this
    trainer class.

    IMPORTANT: this script's test protocol evaluates a SINGLE center crop
    per xView2 sample (see "SINGLE-GPU TEST" below), not a sliding window
    over the full 1024x1024 native image the way
    script_train_xview_baselines.py's evaluate_sliding_window does. So the
    raw measured number is per-crop, not per-image. We additionally report
    a full-image-equivalent figure:
        gflops_full_image = gflops_per_crop * (native_side / crop_size)**2
    computed analytically (per-crop cost x tile count), NOT by an actual
    tiled forward pass -- see measure_test_gflops's docstring for the
    assumption this relies on (crops are processed independently with no
    cross-crop attention/shared latents). Use --xview_native_side /
    --xview_crop_size to override the defaults if needed.

    Set --flops_n 0 to skip FLOPs measurement entirely.

Sliding-window test (additional, runs by default):
    In addition to the whole-image/center-crop test above, this script now
    ALSO runs a genuine sliding-window evaluation: each test image is
    tiled into non-overlapping --sliding_window_tile_size crops (default
    512, matching script_train_xview_baselines.py's evaluate_sliding_window
    tile size), and each tile is fed through the SAME already-correct
    Lightning test_step one at a time (see SlidingWindowTileDataset's
    docstring near measure_test_gflops for why accumulating per-tile
    metrics this way is mathematically identical to stitching predictions
    into the full image first and scoring once -- no need to know
    Atomiser's internal output format to get a correct result). This gives
    a directly comparable full-image test number to the baseline scripts'
    sliding-window protocol, on top of the whole-image single-forward
    number the earlier test already reports. A matching per-tile /
    per-image GFLOPs measurement is reported alongside it (measured, not
    analytically scaled, since sliding-window evaluation genuinely does
    n_tiles separate forward passes). Pass --skip_sliding_window to skip
    it, or --sliding_window_tile_size / --sliding_window_stride to change
    the tiling (stride defaults to tile_size, i.e. non-overlapping;
    changing this breaks the metric equivalence the tile dataset relies
    on -- see its docstring).

Required:
    - bands_xview section in ./data/bands_info/bands.yaml (3 RGB bands)
    - configs_dataset_xview.yaml under ./data/Tiny_BigEarthNet/ (or wherever
      you keep dataset configs)
    - atomiser_xview.yaml under ./training/configs/ (with
      Atomiser.use_decoder_skip: true if you want the skip path active,
      Atomiser.use_temporal_transformer: true for the temporal module)

Examples:
    python script_train_xview.py --xp_name v1 \\
        --config_model atomiser_xview.yaml

    # Temporal transformer variant
    python script_train_xview.py --xp_name v1_temporal \\
        --config_model atomiser_xview.yaml --use_temporal_transformer

    # Resume a chained SLURM job, waiting up to 10 minutes for the checkpoint
    python script_train_xview.py --xp_name v1 --config_model atomiser_xview.yaml \\
        --resume_from ./checkpoints/xview/atomiser_v1-last.ckpt \\
        --resume_wait_seconds 600

    # Test-only, skip GFLOPs measurement
    python script_train_xview.py --xp_name v1 --config_model atomiser_xview.yaml \\
        --test_only ./checkpoints/xview/atomiser_v1-last.ckpt --flops_n 0

    # Test-only, skip the sliding-window evaluation (whole-image test only)
    python script_train_xview.py --xp_name v1 --config_model atomiser_xview.yaml \\
        --test_only ./checkpoints/xview/atomiser_v1-last.ckpt --skip_sliding_window
"""

import os
import time
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.flop_counter import FlopCounterMode

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

# >>> TEMPORAL
from training.utils.datasets.xview2_temporal_dataset import (
    XView2TemporalDataset,
    xview2_temporal_collate_fn,
)


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
# FLOPs MEASUREMENT — FlopCounterMode (counts SDPA attention)
# =============================================================================

def _patch_module_tracker_for_no_grad():
    """Idempotently patches torch.utils.module_tracker so its forward-pre
    hook's register_multi_grad_hook call no longer raises under
    torch.no_grad() (see the module-level comment above for why this is
    safe for FlopCounterMode's total, even though it makes the unused
    per-module breakdown incomplete)."""
    import torch.utils.module_tracker as _mt

    if getattr(_mt, "_flopcounter_noop_patch_applied", False):
        return
    _mt._flopcounter_noop_patch_applied = True

    _orig_register_multi_grad_hook = _mt.register_multi_grad_hook

    class _NoOpHandle:
        def remove(self):
            pass

    def _safe_register_multi_grad_hook(tensors, fn, *args, **kwargs):
        try:
            return _orig_register_multi_grad_hook(tensors, fn, *args, **kwargs)
        except AssertionError:
            return _NoOpHandle()

    _mt.register_multi_grad_hook = _safe_register_multi_grad_hook


@torch.no_grad()
def measure_gflops_forward(forward_fn, batches, device, n_warmup=1):
    """One warmup pass discarded; each measured pass counted with
    FlopCounterMode; report mean / 1e9."""
    _patch_module_tracker_for_no_grad()

    for b in batches[:n_warmup]:
        out = forward_fn(b)
        del out
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    flops_list = []
    for b in batches[n_warmup:]:
        fc = FlopCounterMode(display=False)
        with fc:
            out = forward_fn(b)
        flops_list.append(fc.get_total_flops())
        del out
        if device == "cuda":
            torch.cuda.empty_cache()

    if not flops_list:
        return float("nan")

    if all(f == 0 for f in flops_list):
        print("[measure_gflops_forward] WARNING: all measured passes "
              "returned exactly 0 FLOPs. Treat this GFLOPs number as "
              "UNRELIABLE -- please report the torch version so the patch "
              "can be adjusted.")

    return (sum(flops_list) / len(flops_list)) / 1e9


def _batch_to_device(batch, device):
    """Recursively move a (possibly nested dict/list) batch onto `device`."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_device(v, device) for v in batch)
    return batch


# =============================================================================
# SLIDING-WINDOW TEST EVALUATION (tile-level, via the existing test_step)
# =============================================================================
# See the module docstring's "NOTE on sliding window" section for why this
# is a DIFFERENT mechanism from config["trainer"]["slide"], and why it
# remains compatible with use_decoder_skip / use_temporal_transformer.

def _compute_tile_grid(H: int, W: int, tile_size: int, stride: int):
    """Non-overlapping (or overlapping, if stride < tile_size) tile
    top-left coordinates covering an HxW image."""
    tops  = list(range(0, H - tile_size + 1, stride))
    lefts = list(range(0, W - tile_size + 1, stride))
    if not tops:
        tops = [0]
    if not lefts:
        lefts = [0]
    if tops[-1] + tile_size < H:
        tops.append(H - tile_size)
    if lefts[-1] + tile_size < W:
        lefts.append(W - tile_size)
    return [(top, left) for top in tops for left in lefts]


class SlidingWindowTileDataset(Dataset):
    """
    Wraps an XView2Dataset (or XView2TemporalDataset) test split so each
    item is ONE deterministic, non-overlapping tile (via get_tile_sample)
    instead of one whole image. len(this) == len(base) * len(tile_grid).

    Works unchanged for the temporal path: get_tile_sample is inherited
    from XView2Dataset and calls self._build_sample_from_crop under the
    hood, which XView2TemporalDataset overrides -- so tiles built through
    THIS class automatically come out as [T, N, 8] groups + time_positions
    for the temporal dataset, no separate handling needed here.
    """

    def __init__(self, base_dataset, tile_size: int, stride: int, native_side: int = 1024):
        self.base = base_dataset
        self.tile_size = tile_size
        self.stride = stride
        self.native_side = native_side
        self.tile_grid = _compute_tile_grid(native_side, native_side, tile_size, stride)

        if stride != tile_size:
            print(f"[xView2-Atom-SW] WARNING: stride ({stride}) != "
                  f"tile_size ({tile_size}) -- overlapping tiles break the "
                  f"metric-accumulation-equals-stitching equivalence this "
                  f"class relies on.")

        n_tiles = len(self.tile_grid)
        print(f"[xView2-Atom-SW] {len(self.base)} images x {n_tiles} tiles "
              f"({tile_size}x{tile_size}, stride={stride}) = "
              f"{len(self.base) * n_tiles} tile-samples")

    def __len__(self):
        return len(self.base) * len(self.tile_grid)

    def __getitem__(self, idx):
        img_idx, tile_idx = divmod(idx, len(self.tile_grid))
        top, left = self.tile_grid[tile_idx]
        return self.base.get_tile_sample(img_idx, top, left, self.tile_size)


def run_sliding_window_test(
    trainer_module, base_test_dataset, collate_fn,
    tile_size: int, stride: int, native_side: int,
    ckpt_dir: str, wandb_logger, num_workers: int = 4,
):
    """Runs the existing (already-correct) Lightning test_step over a
    sliding-window tiling of the xView2 test set."""
    tile_dataset = SlidingWindowTileDataset(
        base_test_dataset, tile_size=tile_size, stride=stride,
        native_side=native_side,
    )
    tile_loader = DataLoader(
        tile_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    sw_trainer = Trainer(
        devices=1,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        default_root_dir=ckpt_dir,
    )
    return sw_trainer.test(trainer_module, dataloaders=tile_loader)


def measure_sliding_window_gflops(
    model, base_test_dataset, collate_fn, device, flops_n,
    tile_size: int, stride: int, native_side: int, n_warmup: int = 1,
):
    """Tile-level GFLOPs measurement for the sliding-window test protocol."""
    result = {
        "gflops_per_tile": float("nan"),
        "gflops_total": float("nan"),
        "tile_size_px": tile_size,
        "n_tiles": None,
    }
    if flops_n <= 0:
        return result

    model = model.to(device).eval()

    tile_dataset = SlidingWindowTileDataset(
        base_test_dataset, tile_size=tile_size, stride=stride,
        native_side=native_side,
    )
    n_tiles_per_image = len(tile_dataset.tile_grid)
    result["n_tiles"] = n_tiles_per_image

    n_needed = flops_n + n_warmup
    n_avail = min(n_needed, len(tile_dataset))
    if n_avail < n_needed:
        print(f"[xView2-Atom-SW] WARNING: tile dataset yielded only "
              f"{n_avail} tile(s), needed {n_needed} (1 warmup + "
              f"{flops_n} counted). Measuring with what's available.")

    cpu_batches = [collate_fn([tile_dataset[i]]) for i in range(n_avail)]

    def _fwd(cpu_batch, m=model):
        device_batch = _batch_to_device(cpu_batch, device)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return m(device_batch)
        return m(device_batch)

    gflops_per_tile = measure_gflops_forward(_fwd, cpu_batches, device, n_warmup=n_warmup)
    result["gflops_per_tile"] = gflops_per_tile
    if gflops_per_tile == gflops_per_tile:
        result["gflops_total"] = gflops_per_tile * n_tiles_per_image

    del cpu_batches
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def measure_test_gflops(
    model, data_module, device, flops_n, n_warmup=1,
    native_side_px=1024, crop_size_px=None,
):
    """Measures GFLOPs/forward at test time, using real batches from
    data_module's test dataloader (so it automatically uses whichever
    collate_fn was passed into UnifiedDataModule -- flat or temporal)."""
    result = {
        "gflops_per_crop": float("nan"),
        "gflops_full_image": float("nan"),
        "crop_size_px": crop_size_px,
        "native_side_px": native_side_px,
        "n_tiles": None,
        "is_full_image": False,
    }

    if flops_n <= 0:
        return result

    model = model.to(device).eval()

    if not hasattr(data_module, "test_dataset"):
        data_module.setup(stage="test")
    if not hasattr(data_module, "test_dataset"):
        dtype = getattr(data_module, "_dataset_type", None)
        setup_fn = {
            "h5": "_setup_h5_datasets",
            "simple": "_setup_simple_datasets",
            "grouped": "_setup_grouped_datasets",
        }.get(dtype)
        if setup_fn is not None and hasattr(data_module, setup_fn):
            getattr(data_module, setup_fn)()
    if not hasattr(data_module, "test_dataset"):
        raise AttributeError(
            "measure_test_gflops: data_module still has no 'test_dataset' "
            "after calling setup(stage='test') (and the dataset-type-"
            "specific setup fallback)."
        )

    is_full_image = getattr(data_module.test_dataset, "full_image", False)
    result["is_full_image"] = is_full_image
    if is_full_image:
        if crop_size_px is not None and crop_size_px != native_side_px:
            print(f"[xView2] NOTE: test_dataset.full_image is True, "
                  f"overriding the explicitly-passed "
                  f"--xview_crop_size={crop_size_px}.")
        crop_size_px = native_side_px
        result["crop_size_px"] = crop_size_px
    elif crop_size_px is None:
        crop_size_px = getattr(data_module.test_dataset, "crop_size", None)
        if crop_size_px is None:
            crop_size_px = getattr(data_module.test_dataset, "crop_size_px", None)
        result["crop_size_px"] = crop_size_px

    test_loader = data_module.test_dataloader()
    n_needed = flops_n + n_warmup

    cpu_batches = []
    for batch in test_loader:
        cpu_batches.append(batch)
        if len(cpu_batches) >= n_needed:
            break

    if len(cpu_batches) < n_needed:
        print(f"[xView2] WARNING: test dataloader yielded only "
              f"{len(cpu_batches)} batch(es), needed {n_needed}.")

    def _fwd(cpu_batch, m=model):
        device_batch = _batch_to_device(cpu_batch, device)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return m(device_batch)
        return m(device_batch)

    gflops_per_crop = measure_gflops_forward(_fwd, cpu_batches, device, n_warmup=n_warmup)
    result["gflops_per_crop"] = gflops_per_crop

    del cpu_batches
    if device == "cuda":
        torch.cuda.empty_cache()

    if crop_size_px is not None and crop_size_px > 0 and gflops_per_crop == gflops_per_crop:
        if native_side_px % crop_size_px != 0:
            print(f"[xView2] WARNING: native_side_px={native_side_px} is not "
                  f"evenly divisible by crop_size_px={crop_size_px}.")
        tiles_per_side = native_side_px / crop_size_px
        n_tiles = tiles_per_side ** 2
        result["n_tiles"] = n_tiles
        result["gflops_full_image"] = gflops_per_crop * n_tiles
    else:
        print(f"[xView2] Could not resolve crop_size_px (found "
              f"{crop_size_px!r}) -- reporting per-crop GFLOPs only.")

    return result


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser xView2 training (skip + temporal variant)")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,  default="config_test-xview.yaml",
                    help="Model config yaml (e.g. atomiser_xview.yaml)")
parser.add_argument("--clipping",     action="store_true")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint to resume TRAINING from.")
parser.add_argument("--resume_wait_seconds", type=int, default=0)
parser.add_argument("--resume_poll_interval", type=int, default=15)

parser.add_argument("--data_dir", type=str, default="./data/xview")

# >>> TEMPORAL
parser.add_argument("--use_temporal_transformer", action="store_true",
                    help="Use AtomiserTemporal: per-timestep "
                         "Atomiser_Senflood_Skip encoding + RoPE "
                         "TemporalTransformer aggregation, with per-timestep "
                         "[T, N, 8] token groups (XView2TemporalDataset + "
                         "xview2_temporal_collate_fn) instead of one flat "
                         "concatenated pool.")

# FLOPs measurement (test time only)
parser.add_argument("--flops_n", type=int, default=3)
parser.add_argument("--xview_native_side", type=int, default=1024)
parser.add_argument("--xview_crop_size", type=int, default=None)

# Sliding-window test evaluation
parser.add_argument("--skip_sliding_window", action="store_true")
parser.add_argument("--sliding_window_tile_size", type=int, default=512)
parser.add_argument("--sliding_window_stride", type=int, default=None)

args = parser.parse_args()

if args.sliding_window_stride is None:
    args.sliding_window_stride = args.sliding_window_tile_size

if args.test_only is not None and args.resume_from is not None:
    raise ValueError(
        "--test_only and --resume_from are mutually exclusive."
    )

xp_name           = args.xp_name
config_model      = read_yaml("./training/configs/" + args.config_model)
configs_dataset   = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"

# >>> TEMPORAL: CLI flag wins, but keep the config value in sync so
# Model_SenFlood_Skip's encoder-selection branch (which reads
# config["Atomiser"]["use_temporal_transformer"]) matches what this script
# actually wires up on the data side -- same pattern as script_train_PASTIS.py.
if "Atomiser" not in config_model:
    config_model["Atomiser"] = {}
_use_temporal_transformer = (
    args.use_temporal_transformer
    or config_model["Atomiser"].get("use_temporal_transformer", False)
)
config_model["Atomiser"]["use_temporal_transformer"] = _use_temporal_transformer

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
    print(f"[Train] Temporal transformer: {'ON' if _use_temporal_transformer else 'OFF'}"
          + (" (per-timestep [T,N,8] groups, XView2TemporalDataset)"
             if _use_temporal_transformer else ""))
    if _use_temporal_transformer and not _skip_on:
        print(f"[Train] [WARNING] use_temporal_transformer=True but "
              f"use_decoder_skip=False -- the SKIP cascade will be ignored "
              f"by AtomiserTemporal.forward.")
    print(f"[Train] FLOPs at test: n={args.flops_n} counted passes"
          + (" (skipped)" if args.flops_n == 0 else ""))


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
#
# >>> TEMPORAL: dataset_class / collate_fn swapped based on
# use_temporal_transformer. Everything else about the datamodule call is
# shared -- UnifiedDataModule only needs XView2Dataset/XView2TemporalDataset
# registered in GROUPED_DATASET_CLASSES (dataloaders.py) for correct
# routing, which is assumed already done.
# =============================================================================

_dataset_class = XView2TemporalDataset if _use_temporal_transformer else XView2Dataset
_collate_fn    = xview2_temporal_collate_fn if _use_temporal_transformer else collate_grouped_skip

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
    dataset_class=_dataset_class,
    collate_fn=_collate_fn,
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
        patience=int(config_model["trainer"].get("patience", 50)),
        verbose=True,
    )

    callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]

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

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys "
          f"(runtime caches — recreated automatically)")
if missing:
    print(f"[load_state_dict] {len(missing)} missing keys — if this is "
          f"unexpected, verify --use_temporal_transformer matches how this "
          f"checkpoint was trained (see the key-check pattern used earlier "
          f"in this conversation for PASTIS: search state_dict keys for "
          f"'temporal' to confirm).")

_flops_device = "cuda" if torch.cuda.is_available() else "cpu"
flops_result = measure_test_gflops(
    model=model,
    data_module=data_module,
    device=_flops_device,
    flops_n=args.flops_n,
    native_side_px=args.xview_native_side,
    crop_size_px=args.xview_crop_size,
)
gflops_per_crop = flops_result["gflops_per_crop"]
gflops_full_image = flops_result["gflops_full_image"]
_crop_px = flops_result["crop_size_px"]
_n_tiles = flops_result["n_tiles"]
_is_full_image = flops_result["is_full_image"]

if args.flops_n == 0:
    print(f"[xView2] GFLOPs at test: skipped (--flops_n 0)")
elif _is_full_image:
    print(f"[xView2] GFLOPs/forward, full native image "
          f"({_crop_px}x{_crop_px}, "
          f"batch_size={config_model['trainer']['train_batch_size']}, "
          f"n={args.flops_n} measured passes): "
          + (f"{gflops_per_crop:.2f}" if gflops_per_crop == gflops_per_crop
             else "nan"))
else:
    print(f"[xView2] GFLOPs/forward per crop "
          f"({_crop_px}x{_crop_px} if known, "
          f"batch_size={config_model['trainer']['train_batch_size']}, "
          f"n={args.flops_n} measured passes): "
          + (f"{gflops_per_crop:.2f}" if gflops_per_crop == gflops_per_crop
             else "nan"))
    if gflops_full_image == gflops_full_image:
        print(f"[xView2] GFLOPs full-image-equivalent (ANALYTICAL) "
              f"({args.xview_native_side}x{args.xview_native_side}, "
              f"{_n_tiles:.2f} crop-tiles of {_crop_px}px): "
              f"{gflops_full_image:.2f}")
    else:
        print(f"[xView2] GFLOPs full-image-equivalent: unavailable")

if wandb_logger is not None:
    try:
        import wandb
        wandb.log({
            "test_gflops_per_crop": gflops_per_crop,
            "test_gflops_full_image": gflops_full_image,
            "test_gflops_crop_size_px": _crop_px,
            "test_gflops_native_side_px": args.xview_native_side,
            "test_gflops_n_tiles": _n_tiles,
            "test_gflops_is_full_image": _is_full_image,
            "use_temporal_transformer": _use_temporal_transformer,
        })
    except Exception as e:
        print(f"[xView2] wandb log of test_gflops failed: {e}")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(model, datamodule=data_module)


# =============================================================================
# SLIDING-WINDOW TEST (additional, non-overlapping tile-based evaluation)
# =============================================================================

if not args.skip_sliding_window:
    print(f"\n{'='*60}")
    print(f"  SLIDING-WINDOW TEST "
          f"(tile={args.sliding_window_tile_size}px, "
          f"stride={args.sliding_window_stride}px)")
    print(f"{'='*60}\n")

    # base_test_dataset is the SAME dataset (XView2Dataset or
    # XView2TemporalDataset, depending on _use_temporal_transformer)
    # already built by data_module.setup() above.
    base_test_dataset = data_module.test_dataset

    sw_flops_result = measure_sliding_window_gflops(
        model=model,
        base_test_dataset=base_test_dataset,
        collate_fn=_collate_fn,
        device=_flops_device,
        flops_n=args.flops_n,
        tile_size=args.sliding_window_tile_size,
        stride=args.sliding_window_stride,
        native_side=args.xview_native_side,
    )
    if args.flops_n == 0:
        print(f"[xView2-Atom-SW] GFLOPs at test: skipped (--flops_n 0)")
    else:
        print(f"[xView2-Atom-SW] GFLOPs/forward per tile "
              f"({sw_flops_result['tile_size_px']}x"
              f"{sw_flops_result['tile_size_px']}, "
              f"n={args.flops_n} measured passes): "
              + (f"{sw_flops_result['gflops_per_tile']:.2f}"
                 if sw_flops_result['gflops_per_tile'] == sw_flops_result['gflops_per_tile']
                 else "nan"))
        print(f"[xView2-Atom-SW] GFLOPs total per image "
              f"({sw_flops_result['n_tiles']} tiles, MEASURED): "
              + (f"{sw_flops_result['gflops_total']:.2f}"
                 if sw_flops_result['gflops_total'] == sw_flops_result['gflops_total']
                 else "nan"))

    sw_test_results = run_sliding_window_test(
        trainer_module=model,
        base_test_dataset=base_test_dataset,
        collate_fn=_collate_fn,
        tile_size=args.sliding_window_tile_size,
        stride=args.sliding_window_stride,
        native_side=args.xview_native_side,
        ckpt_dir=ckpt_dir,
        wandb_logger=wandb_logger,
    )

    print(f"\n{'='*60}")
    print(f"  SLIDING-WINDOW TEST RESULTS "
          f"(tile={args.sliding_window_tile_size}px)")
    print(f"{'='*60}")
    if sw_test_results:
        for k, v in sw_test_results[0].items():
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    if wandb_logger is not None:
        try:
            import wandb
            log_payload = {
                "sw_test_gflops_per_tile": sw_flops_result["gflops_per_tile"],
                "sw_test_gflops_total": sw_flops_result["gflops_total"],
                "sw_test_tile_size_px": sw_flops_result["tile_size_px"],
                "sw_test_n_tiles": sw_flops_result["n_tiles"],
            }
            if sw_test_results:
                log_payload.update({
                    f"sw_{k}": v for k, v in sw_test_results[0].items()
                })
            wandb.log(log_payload)
        except Exception as e:
            print(f"[xView2-Atom-SW] wandb log failed: {e}")
else:
    print(f"[xView2-Atom-SW] Skipped (--skip_sliding_window)")


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
