"""
xView2 Damage Assessment — Baseline Training
==============================================

4-class damage segmentation at known building locations (Variant B):
  Input:  pre + post RGB, [T=2, C=3, H, W]
  Target: 4-class damage at building pixels, IGNORE elsewhere
          {0=NoDamage, 1=Minor, 2=Major, 3=Destroyed, 255=ignore}

Models:
  - unet     : UNetMT (channel-concat T*C → C via DoubleConv) + final 1×1 conv
  - resnet_upernet_mt : ResNet (channel-concat early fusion via TimeMerge) + UPerNet
  - vit_upernet_ltae  : ViT per-frame + UPerNet + LTAE (late fusion)
  - ramen    : RAMENUPerNet (spectral tokenization encoder, per-modality LTAE
               temporal fusion) + UPerNet decoder, segmentation head
               (num_classes=NUM_CLASSES). xView2 is RGB-only (no SAR), so
               only the "optical" modality is used. Wrapped in
               RAMENXView2Adapter so it presents the SAME forward(x) plain-
               tensor signature as every other model here -- xview_collate
               and evaluate_sliding_window() (which calls model.model(x)
               with a raw [N,T,C,H,W] tensor) work unmodified. Contrast
               with the BioMassters baseline script's RAMEN path, which
               needs its own collate because that trainer passes per-
               modality dicts straight through instead.
  - universat : UniverSat encoder (gastruc/UniverSat, AnySat v2) trained
               FROM SCRATCH (random init) + linear per-token head, via
               training.Universat.universat_augmenter.UniverSatSegmenter,
               wrapped in UniverSatXView2Adapter to present the same
               forward(x) plain-tensor signature (so xview_collate and
               evaluate_sliding_window work unmodified, like RAMEN).

               This is the first dataset exercising UniverSat's TEMPORAL
               axis: the pre/post pair goes in as a (B, T=2, C=3, H, W)
               time series for a single "vhr" modality, with SYNTHETIC
               day-of-year dates [0, 180] injected by the adapter
               (maximally separated on UniverSat's circular annual
               encoding) purely as an ordering signal -- xView2 has no
               real acquisition dates, same caveat as RAMEN's [0, 1]
               placeholder below. The UPE's temporal-axis block, inert on
               the snapshot datasets, is active here.

               CHANNEL ORDER: the dataset keeps cv2's BGR (see its
               docstring), so UniverSat's wavelength list is
               [490, 560, 665] nm in B,G,R order -- note this differs
               from the RAMEN table below, which labels the channels
               R,G,B and therefore mislabels the wavelengths for this
               BGR dataset (pre-existing; flagged, not silently changed).

               No window is baked at construction: the latent grid is
               recomputed per input (H / patch_px per side), so the same
               weights handle the 512 train crops and the 512 sliding-
               window test tiles. Geometry: every input side must be
               divisible by the patch size in px (--universat_patch_m /
               0.5 = 8 px at the 4 m default), by
               --universat_output_stride, AND by --universat_subpatch_px
               (patch_px must itself be a multiple of subpatch_px -- see
               below). Default output_stride is 8 (native 4 m prediction
               granularity at 0.5 m GSD): the BurnScars (patch x stride)
               sweep showed eval mIoU is nearly stride-insensitive down to
               the latent-grid scale while CA_Sub cost scales as 1/os^2,
               and at VHR the os=4 CA_Sub bill (~6.6 TFLOP per 512 tile)
               is the dominant training cost -- set --universat_output_stride
               4 to match the other datasets' training convention instead.

               SUB-PATCHING (--universat_subpatch_px, default 4): UniverSat's
               own convention for VHR aerial imagery (FLAIR) groups pixels
               into subpatch_px × subpatch_px sub-patches; the UPE's S1 axis
               attends within each group, and sub-patch tokens (not raw
               pixels) become the CA_Sub keys -- a subpatch_px=4 default
               means 16x fewer spatial tokens feeding cross-attention, with
               fine detail summarized at subpatch_px * GSD (= 2 m here)
               before the skip connection. subpatch_px=1 recovers pixel-
               level keys (the other, coarser-GSD datasets' setting --
               appropriate at 10-30 m GSD, wasteful at 0.5 m VHR). patch_px
               must be an integer multiple of subpatch_px.

Channel-concat early fusion baselines mirror Atomiser's architectural strategy
(temporal info fused before the spatial encoder). ViT uses LTAE because its
patch embed bottleneck makes channel-concat ineffective. RAMEN uses its own
internal per-modality LTAE for the same reason. UniverSat fuses time inside
its patch encoder (axial temporal attention), before the spatial trunk --
closest in spirit to the channel-concat early-fusion group.

RAMEN caveats specific to xView2 (see RAMENXView2Adapter docstring):
  - No real acquisition dates exist for the pre/post pair -- it's a single
    before/after snapshot, not a periodic time series. Placeholder
    sequential values [0, 1] are passed as "dates" purely to give RAMEN's
    LTAE an ordering signal, NOT a calendar day-of-year value.
  - --ramen_window_size must equal --crop_size: evaluate_sliding_window()
    tiles test images using --crop_size as both tile_size and stride
    (hardcoded), so a mismatch would silently train/build RAMEN at one
    spatial extent and evaluate it at another.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training and run
    test on a saved checkpoint (single set of ranks, no special handling
    needed since DDP is torn down before the sliding-window eval anyway).

--resume_from mode:
    Pass --resume_from <path/to/checkpoint.ckpt> to resume training (full
    trainer state: model, optimizer, LR scheduler, epoch/step counters,
    callback state) from that checkpoint via Trainer.fit(ckpt_path=...).
    If the file doesn't exist yet, use --resume_wait_seconds to poll for
    it instead of failing immediately (useful for chained SLURM jobs where
    the next job in the chain can start before the previous job's
    checkpoint write has landed on disk -- --resume_wait_seconds is also
    reused for --test_only's checkpoint wait).
    Mutually exclusive with --test_only.

FLOPs at test time:
    Two GFLOPs/forward numbers are measured as part of the sliding-window
    test pass (see evaluate_sliding_window / measure_gflops_forward below),
    using torch.utils.flop_counter.FlopCounterMode -- same method and
    reporting convention (mean over --flops_n counted passes, 1 warmup pass
    discarded, GFLOPs = total_flops / 1e9) as
    script_universat_sweep_burnscars.py / script_universat_sweep_senflood.py,
    so numbers are comparable across scripts:
      - gflops_per_crop:   ONE tile_size x tile_size crop forwarded alone.
      - gflops_full_image: the WHOLE sliding-window batch (all N_tiles
                           tiles of one native 1024x1024 image, forwarded
                           together) -- exactly what the main test loop
                           does, so this is a real measurement, not a
                           scaled estimate.
    Both call model.model(x) on real sliding-window tile batches ([N_tiles
    or 1, T, C, tile, tile] at image bs=1), the exact call the main test
    loop makes, so this works unmodified for all five model choices. Set
    --flops_n 0 to skip both measurements entirely.

Examples:
    # ResNet50 + channel-concat MT
    python script_train_xview_baselines.py --xp_name resnet50 \\
        --model resnet_upernet_mt --resnet_variant resnet50 \\
        --batch_size 4 --epochs 100

    # ViT + LTAE
    python script_train_xview_baselines.py --xp_name vit_ltae \\
        --model vit_upernet_ltae --batch_size 4 --epochs 100

    # RAMEN
    python script_train_baselines_xview.py --xp_name ramen \\
        --model ramen --batch_size 4 --epochs 100

    # UniverSat-S from scratch (temporal pre/post, os=8, subpatch=4 defaults)
    python script_train_baselines_xview.py --xp_name universat \\
        --model universat --batch_size 2 --grad_accum 2 --epochs 100

    # UniverSat at os=4 (other-datasets convention; ~4x the CA_Sub cost)
    python script_train_baselines_xview.py --xp_name universat_os4 \\
        --model universat --universat_output_stride 4 \\
        --batch_size 1 --grad_accum 4 --epochs 100

    # UniverSat with pixel-level sub-patch keys (no sub-patch grouping)
    python script_train_baselines_xview.py --xp_name universat_sp1 \\
        --model universat --universat_subpatch_px 1 \\
        --batch_size 2 --grad_accum 2 --epochs 100

    # Resume a chained SLURM job, waiting up to 10 minutes for the checkpoint
    python script_train_baselines_xview.py --xp_name ramen --model ramen \\
        --resume_from ./checkpoints/xview_baselines/bl_ramen_ramen-last.ckpt \\
        --resume_wait_seconds 600

    # Test-only, skip GFLOPs measurement
    python script_train_baselines_xview.py --xp_name ramen --model ramen \\
        --test_only ./checkpoints/xview_baselines/bl_ramen_ramen-last.ckpt \\
        --flops_n 0
"""

import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping,
)
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_xview_baseline import (
    XView2BaselineDataset, NUM_CLASSES, IGNORE_INDEX,
)
from training.ltae.ltae import UNetLTAE
from training.VIT.model_vit_upernet import (
    ViTUPerNet, ViTLTAEUPerNet, ViTUPerNetLTAE, build_vit_upernet_mt,
)
from training.ResNet.model_resnet_upernet import build_resnet_upernet_mt
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.Universat.universat_augmenter import build_universat_segmenter
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# RESUME HELPER (mirrors train_biomassters.py / script_train_biomassters_baseline.py)
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

    print(f"[xView2-BL] Checkpoint not found yet: {path}")
    print(f"[xView2-BL] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[xView2-BL] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[xView2-BL]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"Checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# COLLATE — multi-temporal segmentation
# =============================================================================

def xview_collate(batch):
    """Collate xView2 samples into batched tensors.

    Output:
        image:  [B, T=2, C=3, H, W]
        target: [B, H, W]
    """
    images  = torch.stack([s["image"]  for s in batch], dim=0)
    targets = torch.stack([s["target"] for s in batch], dim=0)
    metadata = [s["metadata"] for s in batch]
    return {
        "image":    {"s2": images},   # use 's2' key for compat with BaselineTrainer
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# RAMEN BAND METADATA
# =============================================================================
# xView2 is RGB-only (no SAR), single "optical" modality. Band order MUST
# match the tensor's channel axis -- xview_collate stacks whatever channel
# order XView2BaselineDataset returns; this assumes standard R,G,B. If the
# dataset actually stores BGR (common with cv2-based loaders), swap the
# order here to match.
#
# KNOWN INCONSISTENCY: XView2BaselineDataset explicitly keeps cv2's BGR
# order (see its docstring/normalization), so this R,G,B labeling
# mislabels the wavelengths (channel 0 = Blue ~490nm is tagged Red
# 665nm). Left as-is to preserve comparability with already-trained RAMEN
# checkpoints -- fix the order AND retrain RAMEN together if you correct
# it. The UniverSat table below uses the correct BGR order.
#
# Wavelengths are approximate visible-spectrum centers, reused from the
# same convention already used elsewhere in this codebase for RGB/S2
# visible bands (see the BioMassters/ForestNet baseline scripts' RAMEN
# band tables) -- xView2 is VHR RGB, not multispectral, so these are
# nominal centers rather than sensor-calibrated values.
RAMEN_RGB_BAND_ORDER = ["Red", "Green", "Blue"]
RAMEN_RGB_WAVELENGTHS = {"Red": 665.0, "Green": 560.0, "Blue": 490.0}

RAMEN_INPUT_BANDS = {"optical": RAMEN_RGB_BAND_ORDER}
RAMEN_WAVELENGTHS = {"optical": RAMEN_RGB_WAVELENGTHS}


# =============================================================================
# UNIVERSAT BAND METADATA
# =============================================================================
# Single VHR modality, TEMPORAL (T=2). Channel order is the dataset's
# actual BGR (cv2), so wavelengths are listed Blue, Green, Red. Nominal
# visible-spectrum centers in nm; the adapter converts to UniverSat's µm
# convention internally.
UNIVERSAT_MODALITY = "vhr"
UNIVERSAT_INPUT_BANDS = {UNIVERSAT_MODALITY: ["Blue", "Green", "Red"]}
UNIVERSAT_WAVELENGTHS = {UNIVERSAT_MODALITY: [490.0, 560.0, 665.0]}  # BGR order

XVIEW_GSD_M = 0.5   # VHR native ground sampling distance

# Synthetic day-of-year values for the pre/post pair: no real acquisition
# dates exist (same situation as RAMEN's [0, 1] placeholder). [0, 180]
# puts the two frames maximally apart on UniverSat's circular annual
# temporal encoding, giving the UPE's temporal block an unambiguous
# ordering signal. This is a labeled ordering, NOT calendar time --
# report it as such.
UNIVERSAT_SYNTHETIC_DOY = [0, 180]


# =============================================================================
# RAMEN ADAPTER
# =============================================================================

class RAMENXView2Adapter(nn.Module):
    """
    Wraps RAMENUPerNet so it presents the same forward(x) signature as
    every other model in this script: x is a plain [B, T, C, H, W] tensor
    (xview_collate's "s2" key), not a per-modality dict.

    This keeps xView2's existing collate (xview_collate, unmodified) and
    the sliding-window test harness (evaluate_sliding_window, which calls
    model.model(tile_batch) with a raw 5D tensor, bypassing any dict-based
    collate) working unchanged for RAMEN too -- unlike the BioMassters
    baseline script's RAMEN path, which needed its own collate because
    that codebase's trainer passes per-modality dicts straight through.

    xView2 is RGB-only (no SAR), so only the "optical" modality is built.

    xView2 has no real acquisition dates for pre/post (they're two
    snapshots around a single event, not a periodic time series) -- this
    passes placeholder sequential "dates" [0, 1, ..., T-1] purely to give
    RAMEN's per-modality LTAE a consistent ordering signal, NOT a
    calendar day-of-year value. Revisit if RAMENUPerNet's LTAE assumes
    genuinely periodic (day-of-year) input internally, since [0, 1] is
    about as far from periodic as it gets.
    """

    def __init__(self, ramen_upernet: nn.Module):
        super().__init__()
        self.model = ramen_upernet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]  (RGB pre/post)
        B, T, C, H, W = x.shape
        optical = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        dates = (
            torch.arange(T, dtype=torch.float32, device=x.device)
            .unsqueeze(0).expand(B, T).contiguous()
        )
        return self.model({"optical": optical}, dates={"optical": dates})


# =============================================================================
# UNIVERSAT ADAPTER
# =============================================================================

class UniverSatXView2Adapter(nn.Module):
    """
    Wraps UniverSatSegmenter so it presents the same forward(x) plain-
    tensor signature as every other model here: x is [B, T, C, H, W]
    (xview_collate's "s2" key). Same rationale as RAMENXView2Adapter --
    evaluate_sliding_window() calls model.model(tile_batch) with a raw
    5D tensor, so a tensor-in adapter keeps the whole harness unmodified.

    Builds the modality dict UniverSatSegmenter expects:
        {"vhr": [B, T, C, H, W], "vhr_dates": [B, T]}
    UniverSat's own format for time series -- its UPE detects the
    "<mod>_dates" key and runs the temporal-axis block (inert on the
    snapshot datasets, active here). Dates are the synthetic [0, 180]
    day-of-year pair (see UNIVERSAT_SYNTHETIC_DOY above).
    """

    def __init__(self, segmenter: nn.Module):
        super().__init__()
        self.model = segmenter
        self.register_buffer(
            "_doy",
            torch.tensor(UNIVERSAT_SYNTHETIC_DOY, dtype=torch.int64),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]  (BGR pre/post)
        B, T = x.shape[0], x.shape[1]
        dates = self._doy[:T].unsqueeze(0).expand(B, T).contiguous()
        return self.model({
            UNIVERSAT_MODALITY: x,
            f"{UNIVERSAT_MODALITY}_dates": dates,
        })


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, args):
    """Build segmentation model. T=2 multi-temporal input."""
    NUM_FRAMES = 2

    if model_name == "unet":
        return UNetLTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=NUM_FRAMES,
            use_ltae=False,
        )

    elif model_name == "resnet_upernet_mt":
        return build_resnet_upernet_mt(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=NUM_FRAMES,
        )

    elif model_name == "vit_upernet_mt":
        return build_vit_upernet_mt(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=NUM_FRAMES,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "vit_upernet_ltae":
        return ViTUPerNetLTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.ltae_n_head,
            ltae_d_k=args.ltae_d_k,
            ltae_dropout=args.ltae_dropout,
        )

    elif model_name == "ramen":
        # in_channels is unused here (like the BioMassters RAMEN branch) --
        # RAMEN derives channel count from RAMEN_INPUT_BANDS/RAMEN_WAVELENGTHS.
        base = build_ramen_upernet(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=num_classes,
            input_size=args.ramen_window_size,
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )
        return RAMENXView2Adapter(base)

    elif model_name == "universat":
        # Random init, no pretrained weights. in_channels unused (channel
        # count comes from UNIVERSAT_INPUT_BANDS). No window is baked at
        # construction; the same weights handle 512 train crops and 512
        # sliding-window test tiles (both go through the tensor adapter).
        # subpatch_px controls the UPE's S1-axis sub-patch grouping for
        # CA_Sub keys (see module docstring / --universat_subpatch_px help).
        base = build_universat_segmenter(
            input_bands=UNIVERSAT_INPUT_BANDS,
            wavelengths=UNIVERSAT_WAVELENGTHS,
            num_classes=num_classes,
            input_res={UNIVERSAT_MODALITY: XVIEW_GSD_M},
            patch_size_m=args.universat_patch_m,
            output_stride=args.universat_output_stride,
            size=args.universat_size,
            subpatch_px=args.universat_subpatch_px,
        )
        return UniverSatXView2Adapter(base)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: 'unet', 'resnet_upernet_mt', "
            f"'vit_upernet_mt', 'vit_upernet_ltae', 'ramen', 'universat'"
        )


# =============================================================================
# FLOPs MEASUREMENT — FlopCounterMode (counts SDPA attention)
# =============================================================================
# Same convention as script_universat_sweep_burnscars.py /
# script_universat_sweep_senflood.py: one warmup pass discarded, mean over
# the remaining counted passes, GFLOPs = total_flops / 1e9.
#
# MEMORY: naively running FlopCounterMode WITHOUT torch.no_grad() (an
# earlier version of this fix) makes every measured pass retain a full
# backward-ready autograd graph -- every intermediate activation kept
# alive for a .backward() we never call -- which can OOM on large inputs
# (e.g. UniverSat/RAMEN at VHR resolutions) even with bf16 autocast and
# aggressive cache-clearing between passes.
#
# The actual reason no_grad() was dropped: FlopCounterMode attributes
# FLOPs to individual modules via torch.utils.module_tracker, which
# registers an autograd hook (register_multi_grad_hook) on each module's
# output tensors in its forward-pre-hook. Under no_grad(), an op whose
# inputs include a requires_grad=True parameter still produces an output
# with requires_grad=True (torch propagates that flag even under
# no_grad), but WITHOUT a grad_fn (no_grad skips building the graph) --
# so that output is neither a leaf (has no grad-accumulator) nor has a
# real grad_fn, and the hook's internal `_get_grad_fn_or_grad_acc` lookup
# asserts ("Expected gradient function to be set").
#
# That assertion only breaks the PER-MODULE flop attribution (used solely
# for FlopCounterMode's optional pretty-printed breakdown table, which we
# never request -- display=False below). It does NOT affect the overall
# total: every counted op is attributed to every name currently in
# ModuleTracker.parents, which always includes a "Global" bucket set once
# at construction and never removed, and get_total_flops() sums exactly
# that bucket. So we monkeypatch register_multi_grad_hook to fail
# silently instead of raising, and go back to measuring under
# torch.no_grad() -- matching the real test loop's memory profile
# (no_grad + bf16 autocast) instead of paying for a full graph we never
# use.

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
    FlopCounterMode; report mean / 1e9 (analytic and deterministic per
    shape -- the mean is a sanity check, not a statistical average over
    noise).

    Runs under torch.no_grad() -- see the module-level comment above for
    why this no longer breaks FlopCounterMode's total (the module-tracker
    patch applied at import time makes its per-module attribution hook
    fail silently instead of asserting). This keeps memory usage in line
    with the model's normal (no_grad, bf16-autocast) inference forward
    pass rather than a full backward-ready graph.
    """
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
              "returned exactly 0 FLOPs. This suggests the assumption "
              "behind the no_grad patch (FlopCounterMode's 'Global' "
              "bucket still accumulates totals even when its per-module "
              "attribution hook is patched to a no-op) does not hold for "
              "the installed torch version. Treat this GFLOPs number as "
              "UNRELIABLE -- please report the torch version so the patch "
              "can be adjusted.")

    return (sum(flops_list) / len(flops_list)) / 1e9


def _build_tile_batch(image: torch.Tensor, tile_size: int, stride: int):
    """Split one [1, T, C, H, W] full image into its sliding-window tile
    batch [N_tiles, T, C, tile, tile] and the (top, left) coords used to
    stitch it back -- factored out so both the FLOPs pre-pass and the main
    test loop build tiles identically (single source of truth for the
    tiling geometry)."""
    B, T, C, H, W = image.shape
    tops  = list(range(0, H - tile_size + 1, stride))
    lefts = list(range(0, W - tile_size + 1, stride))
    if tops[-1] + tile_size < H:
        tops.append(H - tile_size)
    if lefts[-1] + tile_size < W:
        lefts.append(W - tile_size)

    tiles = []
    coords = []
    for top in tops:
        for left in lefts:
            tile = image[:, :, :, top:top + tile_size, left:left + tile_size]
            tiles.append(tile.squeeze(0))
            coords.append((top, left))
    tile_batch = torch.stack(tiles, dim=0)  # [N_tiles, T, C, tile, tile]
    return tile_batch, coords


# =============================================================================
# SLIDING-WINDOW TEST EVALUATION
# =============================================================================

def evaluate_sliding_window(
    model: torch.nn.Module,
    test_loader,
    num_classes: int,
    tile_size: int = 512,
    stride: int = 512,
    device: str = "cuda",
    wandb_logger=None,
    class_names=None,
    ignore_index: int = 255,
    flops_n: int = 3,
):
    """
    Evaluate model on full-resolution xView2 test images using non-overlapping
    sliding window. Each 1024×1024 test image is tiled into 512×512 patches,
    forward through the model in a single batch, then stitched back into a
    full-image prediction. mIoU is computed over the full test set.

    Also measures GFLOPs/forward via FlopCounterMode, at test time only:
    one warmup pass discarded, mean over `flops_n` counted passes, using
    real sliding-window tile batches ([N_tiles, T, C, tile, tile] built
    from actual test images at image bs=1) forwarded through
    model.model(tile_batch) -- the exact call the main test loop below
    makes -- so this works unmodified for every model choice. Set
    flops_n=0 to skip.

    Two GFLOPs numbers are reported:
      - gflops_per_crop:   a single tile_size x tile_size crop forwarded
                           alone (tile_batch sliced to 1 tile).
      - gflops_full_image: the WHOLE sliding-window batch (all N_tiles
                           tiles of one native image forwarded together
                           in one call) -- this is a REAL measurement of
                           what the main test loop below actually does,
                           not a scaled estimate. (Contrast with the
                           Atomiser xView2 script, script_train_xview.py,
                           whose test protocol only ever forwards a
                           single crop, so its "full image" figure is an
                           analytical gflops_per_crop * n_tiles scaling
                           rather than an actual multi-tile forward.)

    Args:
        model:        Lightning module wrapping the segmentation model.
                      We call model.model(image) to bypass Lightning's
                      step hooks (we want raw logits). For RAMEN, model.model
                      is a RAMENXView2Adapter (and for UniverSat a
                      UniverSatXView2Adapter), so this still receives a plain
                      [N, T, C, t, t] tensor like every other model.
        test_loader:  DataLoader returning {"image": {"s2": [1, T, C, H, W]},
                      "target": [1, H, W]} per batch.
        num_classes:  Total classes including background.
        tile_size:    Spatial tile size. Default 512.
        stride:       Tile stride. Default 512 (non-overlapping).
        device:       Where to run the model.
        wandb_logger: Optional WandbLogger to record results.
        class_names:  Optional list of class names for per-class logging.
        flops_n:      Number of counted FLOPs passes (mean reported). 1
                      extra warmup pass is run and discarded. 0 disables
                      FLOPs measurement.
    Returns:
        Dict with 'mIoU', 'IoU_per_class', 'mean_acc', 'overall_acc',
        'gflops_per_crop', 'gflops_full_image', 'n_tiles'.
    """
    from torchmetrics.classification import (
        MulticlassJaccardIndex, MulticlassAccuracy,
    )
    from tqdm import tqdm

    model = model.to(device).eval()

    # Metrics — accumulate state across all tiles of all images.
    # ignore_index is ESSENTIAL here: Variant B targets are IGNORE (255)
    # everywhere except building pixels, so without it torchmetrics either
    # raises on out-of-range target values or silently corrupts every
    # metric by scoring the ignored region.
    iou_metric = MulticlassJaccardIndex(
        num_classes=num_classes, average="none", ignore_index=ignore_index,
    ).to(device)
    macro_acc_metric = MulticlassAccuracy(
        num_classes=num_classes, average="macro", ignore_index=ignore_index,
    ).to(device)
    overall_acc_metric = MulticlassAccuracy(
        num_classes=num_classes, average="micro", ignore_index=ignore_index,
    ).to(device)

    print(f"\n[sliding-window] Evaluating {len(test_loader)} test samples "
          f"({tile_size}×{tile_size} tiles, stride={stride})...")

    # ── FLOPs measurement (test time only) ───────────────────────────────
    # Pulls the first (flops_n + 1) test images, tiles each exactly as the
    # main loop does below (via the shared _build_tile_batch helper), and
    # counts FLOPs on model.model(tile_batch). This does NOT consume the
    # test_loader for the mIoU pass below -- the main loop re-iterates
    # test_loader from scratch afterward.
    #
    # Two numbers are reported, for comparability with the Atomiser xView2
    # script (script_train_xview.py), whose test protocol only ever
    # forwards a single crop:
    #   - gflops_per_crop:   FLOPs for ONE tile_size x tile_size crop
    #                        (tile_batch sliced down to 1 tile).
    #   - gflops_full_image: FLOPs for the WHOLE sliding-window batch
    #                        (all N_tiles tiles of the native image,
    #                        forwarded together in one call) -- this is
    #                        what the main test loop below actually does,
    #                        so it's a real measurement, not a scaled
    #                        estimate (unlike Atomiser's analytical
    #                        gflops_per_crop * n_tiles scaling).
    gflops_per_crop = float("nan")
    gflops_full_image = float("nan")
    n_tiles = None
    if flops_n > 0:
        # Keep tile batches on CPU here -- moving every measured pass's
        # (potentially large, for gflops_full_image) tile batch to GPU up
        # front means several device-resident batches coexist at once,
        # adding to the memory pressure of the grad-enabled forward passes
        # below. Instead each batch is moved to device only inside _fwd,
        # right before its own forward call.
        _flops_batches_cpu = []       # full tile_batch per sample (all tiles)
        _flops_batches_1tile_cpu = [] # first tile only, per sample
        n_needed = flops_n + 1  # +1 warmup, discarded by measure_gflops_forward
        for batch in test_loader:
            image = batch["image"]["s2"]  # [1, T, C, H, W], stays on CPU
            tile_batch, _ = _build_tile_batch(image, tile_size, stride)
            _flops_batches_cpu.append(tile_batch)
            _flops_batches_1tile_cpu.append(tile_batch[:1])
            if len(_flops_batches_cpu) >= n_needed:
                break

        if n_tiles is None and _flops_batches_cpu:
            n_tiles = _flops_batches_cpu[0].shape[0]

        # bf16 autocast matches the precision the main sliding-window loop
        # below uses (torch.autocast(..., dtype=torch.bfloat16)) -- running
        # this measurement in fp32 instead would noticeably increase peak
        # memory for no benefit (FlopCounterMode counts FLOPs analytically
        # from op shapes, so precision doesn't affect the reported number).
        def _fwd(tb_cpu, m=model):
            tb = tb_cpu.to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return m.model(tb)
            return m.model(tb)

        gflops_full_image = measure_gflops_forward(
            _fwd, _flops_batches_cpu, device, n_warmup=1)
        gflops_per_crop = measure_gflops_forward(
            _fwd, _flops_batches_1tile_cpu, device, n_warmup=1)

        print(f"[sliding-window] GFLOPs/forward per crop "
              f"({tile_size}×{tile_size} single tile, dense, "
              f"n={flops_n} measured passes): {gflops_per_crop:.2f}")
        print(f"[sliding-window] GFLOPs/forward full image "
              f"({tile_size}×{tile_size} tiles x {n_tiles} = full "
              f"sliding-window batch, image bs=1, "
              f"n={flops_n} measured passes): {gflops_full_image:.2f}\n")

        del _flops_batches_cpu, _flops_batches_1tile_cpu
        if device == "cuda":
            torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            image  = batch["image"]["s2"].to(device)   # [1, T, C, H, W]
            target = batch["target"].to(device)        # [1, H, W]

            B, T, C, H, W = image.shape
            assert B == 1, f"Test loader must have batch_size=1, got {B}"

            # Build the sliding-window tile batch (same helper the FLOPs
            # pre-pass above uses, so tiling geometry can't drift between
            # the two).
            tile_batch, coords = _build_tile_batch(image, tile_size, stride)

            # Forward all tiles in one go (memory permitting; for 1024×1024
            # with stride=512 there are only 4 tiles → cheap).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_tiles = model.model(tile_batch)  # [N_tiles, num_classes, t, t]
            logits_tiles = logits_tiles.float()

            # UniverSat emits logits at tile_size/output_stride per side;
            # upsample to the tile size before stitching (BaselineTrainer
            # does the same during train/val; here we do it explicitly
            # since this harness bypasses the Lightning step).
            if logits_tiles.shape[-1] != tile_size:
                logits_tiles = torch.nn.functional.interpolate(
                    logits_tiles, size=(tile_size, tile_size),
                    mode="bilinear", align_corners=False,
                )

            # Stitch into a full prediction map. For non-overlapping stride,
            # each pixel belongs to exactly one tile — direct copy.
            full_logits = torch.zeros(
                (1, num_classes, H, W), device=device,
            )
            count = torch.zeros((1, 1, H, W), device=device)
            for i, (top, left) in enumerate(coords):
                full_logits[:, :, top:top+tile_size, left:left+tile_size] += logits_tiles[i:i+1]
                count[:, :, top:top+tile_size, left:left+tile_size] += 1.0
            # Average where overlap occurred (only relevant for stride < tile_size).
            full_logits = full_logits / count.clamp(min=1.0)

            # Per-pixel argmax → predictions
            preds = full_logits.argmax(dim=1)  # [1, H, W]

            # Update metrics
            iou_metric.update(preds, target)
            macro_acc_metric.update(preds, target)
            overall_acc_metric.update(preds, target)

    # Compute final scores
    iou_per_class = iou_metric.compute().cpu().tolist()
    mean_iou = sum(iou_per_class) / len(iou_per_class)
    macro_acc = macro_acc_metric.compute().item()
    overall_acc = overall_acc_metric.compute().item()

    # Damage-only mIoU (drop class 0 = background) — useful auxiliary metric.
    damage_iou = iou_per_class[1:]
    damage_mIoU = sum(damage_iou) / len(damage_iou) if damage_iou else 0.0

    # Default class names if none provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # ── Print results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SLIDING-WINDOW TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"  Macro accuracy:     {macro_acc:.4f}")
    print(f"  mIoU (5-class):     {mean_iou:.4f}")
    print(f"  mIoU (damage):      {damage_mIoU:.4f}  (excludes background)")
    if flops_n > 0:
        print(f"  GFLOPs/forward (per crop, {tile_size}x{tile_size}): "
              f"{gflops_per_crop:.2f}")
        print(f"  GFLOPs/forward (full image, {n_tiles} tiles):       "
              f"{gflops_full_image:.2f}")
    else:
        print(f"  GFLOPs/forward:     skipped (--flops_n 0)")
    print(f"  Per-class IoU:")
    for name, iou in zip(class_names, iou_per_class):
        print(f"    {name:>12s}: {iou:.4f}")
    print(f"{'='*60}\n")

    # ── Log to wandb ─────────────────────────────────────────
    if wandb_logger is not None:
        try:
            import wandb
            wandb.log({
                "test_mIoU":            mean_iou,
                "test_damage_mIoU":     damage_mIoU,
                "test_macro_acc":       macro_acc,
                "test_overall_acc":     overall_acc,
                "test_gflops_per_crop":   gflops_per_crop,
                "test_gflops_full_image": gflops_full_image,
                "test_gflops_n_tiles":    n_tiles,
                **{f"test_IoU_{name}": iou
                   for name, iou in zip(class_names, iou_per_class)},
            })
        except Exception as e:
            print(f"[sliding-window] wandb log failed: {e}")

    return {
        "mIoU":              mean_iou,
        "damage_mIoU":       damage_mIoU,
        "IoU_per_class":     iou_per_class,
        "macro_acc":         macro_acc,
        "overall_acc":       overall_acc,
        "gflops_per_crop":   gflops_per_crop,
        "gflops_full_image": gflops_full_image,
        "n_tiles":           n_tiles,
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="xView2 Baseline Damage Segmentation")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--model",   type=str, default="resnet_upernet_mt",
                    choices=["unet", "resnet_upernet_mt",
                             "vit_upernet_mt", "vit_upernet_ltae", "ramen",
                             "universat"])
parser.add_argument("--data_dir", type=str, default="./data/xview")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Resume-training mode
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

# Training hyperparams
parser.add_argument("--batch_size",   type=int, default=2)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=100)
parser.add_argument("--grad_accum",   type=int, default=1)

# Dataset
parser.add_argument("--crop_size", type=int, default=512,
                    help="Spatial crop size (default 512). 1024 = no crop. "
                         "For UniverSat, must be divisible by "
                         "lcm(patch_px, output_stride, subpatch_px) -- see "
                         "the UNIVERSAT SANITY CHECK block below (512 "
                         "satisfies this at the defaults).")
parser.add_argument("--no_oversample", action="store_true",
                    help="Disable PANGAEA-style oversampling of building-damage tiles.")

# Image size for ViT
parser.add_argument("--img_size", type=int, default=512,
                    help="Should equal --crop_size for ViT positional embeddings.")

# ViT params
parser.add_argument("--vit_embed_dim",  type=int, default=384)
parser.add_argument("--vit_depth",      type=int, default=12)
parser.add_argument("--vit_num_heads",  type=int, default=6)
parser.add_argument("--vit_patch_size", type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)

# LTAE params (for vit_upernet_ltae)
parser.add_argument("--ltae_n_head",         type=int,   default=16)
parser.add_argument("--ltae_d_k",            type=int,   default=4)
parser.add_argument("--ltae_dropout",        type=float, default=0.2)
parser.add_argument("--ltae_num_freq_bands", type=int,   default=24)
parser.add_argument("--ltae_cycle_period",   type=float, default=365.0)

# ResNet params
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN params
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=0.5,
                    help="Native GSD (m/px) of xView2 VHR imagery.")
parser.add_argument("--ramen_res",       type=float, default=2,
                    help="Working resolution (m/px) RAMEN resamples to "
                         "before its shared ViT stack. Default equals "
                         "--ramen_input_res (no resampling, full native "
                         "detail) -- increase to trade detail for speed / "
                         "fewer tokens if training is too slow at native "
                         "0.5m resolution.")
parser.add_argument("--ramen_window_size", type=int, default=None,
                    help="Spatial size RAMEN is built at. Must equal "
                         "--crop_size: evaluate_sliding_window() tiles test "
                         "images using --crop_size as both tile_size and "
                         "stride (hardcoded at the bottom of this script), "
                         "so a mismatch would silently train/build RAMEN at "
                         "one spatial extent and evaluate it at another. "
                         "Defaults to --crop_size if left unset -- only "
                         "override this if you know what you're doing.")

# UniverSat params (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="'small' (384-d, 12 SA blocks, heads=6) is ~36.1M "
                         "total -- parameter-matched to the ViT-S / RAMEN "
                         "~34M budget. 'tiny' ~6.2M; 'base' ~201M.")
parser.add_argument("--universat_patch_m", type=float, default=4.0,
                    help="Patch size in METRES. 4 m = 8 px at xView2's "
                         "0.5 m VHR GSD -- the same 8 px patch as the "
                         "other datasets' defaults (240 m @ 30 m HLS, "
                         "80 m @ 10 m S1/S2). Must be an integer number "
                         "of pixels (multiple of 0.5), and every input "
                         "side must be divisible by that pixel count.")
parser.add_argument("--universat_output_stride", type=int, default=1,
                    help="Logits at H/stride per side (upsampled to full "
                         "res: by BaselineTrainer in train/val, explicitly "
                         "in evaluate_sliding_window at test). Default 8 "
                         "here (unlike 4 on the other datasets): native "
                         "4 m prediction granularity at 0.5 m GSD, and the "
                         "BurnScars (patch x stride) sweep showed eval "
                         "mIoU is nearly stride-insensitive while CA_Sub "
                         "cost scales as 1/os^2 -- at VHR the os=4 bill "
                         "(~6.6 TFLOP per 512 tile) would dominate "
                         "training. Set 4 to match the other datasets' "
                         "convention.")
parser.add_argument("--universat_subpatch_px", type=int, default=8,
                    help="Sub-patch size in PIXELS for the UPE's sub-patch "
                         "spatial skip. UniverSat's own convention for VHR "
                         "aerial imagery (FLAIR) is 4 -- pixels are grouped "
                         "into 4x4 sub-patches, the S1 axis attends within "
                         "each group, and sub-patch tokens (not pixels) "
                         "become the CA_Sub keys: 16x fewer spatial tokens, "
                         "fine detail summarized at subpatch_px * GSD "
                         "(= 2 m here) before the skip. 1 = pixel-level "
                         "keys (the other datasets' setting -- appropriate "
                         "at 10-30 m GSD, wasteful at 0.5 m). patch_px must "
                         "be a multiple of this.")

# FLOPs measurement (test time only)
parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of counted FlopCounterMode forward passes "
                         "at test time (mean GFLOPs/forward reported). One "
                         "extra warmup pass is run first and discarded. "
                         "Set 0 to skip FLOPs measurement entirely.")

args = parser.parse_args()

if args.test_only is not None and args.resume_from is not None:
    raise ValueError(
        "--test_only and --resume_from are mutually exclusive: --test_only "
        "skips training entirely (loads weights, runs test), --resume_from "
        "continues training from a checkpoint. Pick one."
    )


# =============================================================================
# RAMEN SANITY CHECK
# =============================================================================

if args.model == "ramen":
    if args.ramen_window_size is None:
        args.ramen_window_size = args.crop_size
    if args.ramen_window_size != args.crop_size:
        print(f"[WARNING] --ramen_window_size={args.ramen_window_size} != "
              f"--crop_size={args.crop_size}. evaluate_sliding_window() tiles "
              f"test images using --crop_size as both tile_size and stride, "
              f"so RAMEN would be built for one spatial extent but evaluated "
              f"on tiles of a different size, forcing a silent internal "
              f"resample at test time rather than a crash. Leave "
              f"--ramen_window_size unset (defaults to --crop_size) unless "
              f"you have a specific reason to diverge.")


# =============================================================================
# UNIVERSAT SANITY CHECK
# =============================================================================

if args.model == "universat":
    universat_patch_px = args.universat_patch_m / XVIEW_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {XVIEW_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of {XVIEW_GSD_M}."
        )
    universat_patch_px = int(round(universat_patch_px))

    if universat_patch_px % args.universat_subpatch_px:
        raise ValueError(
            f"patch ({universat_patch_px} px) must be a multiple of "
            f"--universat_subpatch_px ({args.universat_subpatch_px}): the "
            f"S1 axis groups whole sub-patches into each patch."
        )

    import math as _math
    _lcm = _math.lcm(universat_patch_px, args.universat_output_stride)
    _lcm = _math.lcm(_lcm, args.universat_subpatch_px)

    # Train crops AND the sliding-window test tiles are both --crop_size,
    # so one divisibility check covers train and test geometry.
    if args.crop_size % _lcm:
        raise ValueError(
            f"--crop_size ({args.crop_size}) must be divisible by "
            f"lcm(patch_px={universat_patch_px}, "
            f"output_stride={args.universat_output_stride}, "
            f"subpatch_px={args.universat_subpatch_px})={_lcm} for "
            f"UniverSat (the same size is used for train crops and "
            f"sliding-window test tiles)."
        )


# =============================================================================
# DATA SETUP
# =============================================================================

# 3 RGB channels per frame, T=2 frames → channel-concat baselines see 6 channels
NUM_CHANNELS_PER_FRAME = 3

is_temporal_model = args.model in (
    "unet", "resnet_upernet_mt", "vit_upernet_mt", "vit_upernet_ltae", "ramen",
    "universat",
)

print(f"\n{'='*60}")
print(f"  xView2 Damage Assessment — Baseline (PANGAEA setup, 5-class)")
print(f"  Model:        {args.model}")
if args.model in ("resnet_upernet_mt",):
    print(f"  Variant:      {args.resnet_variant}")
if args.model == "ramen":
    print(f"  RAMEN:        embed_dim={args.ramen_embed_dim}, "
          f"depth={args.ramen_depth}, heads={args.ramen_num_heads}")
    print(f"  RAMEN res:    input_res={args.ramen_input_res}m/px, "
          f"res={args.ramen_res}m/px "
          f"({'no resampling' if args.ramen_input_res == args.ramen_res else 'resampled'})")
    print(f"  RAMEN window: {args.ramen_window_size}×{args.ramen_window_size} "
          f"(tied to --crop_size)")
if args.model == "universat":
    print(f"  UniverSat:    {args.universat_size} (from scratch, random init)")
    print(f"  Patch:        {args.universat_patch_m:.1f} m "
          f"({int(args.universat_patch_m / XVIEW_GSD_M)} px @ "
          f"{XVIEW_GSD_M} m VHR)")
    print(f"  Sub-patch:    {args.universat_subpatch_px} px "
          f"({args.universat_subpatch_px * XVIEW_GSD_M:.1f} m skip "
          f"granularity; S1 axis "
          f"{'active' if args.universat_subpatch_px > 1 else 'inert'})")
    print(f"  Out stride:   {args.universat_output_stride} "
          f"(logits at H/{args.universat_output_stride}, native "
          f"{args.universat_output_stride * XVIEW_GSD_M:.1f} m granularity)")
    print(f"  Temporal:     T=2 via UPE temporal axis, synthetic DOY "
          f"{UNIVERSAT_SYNTHETIC_DOY}")
    print(f"  Channels:     BGR (dataset order), wavelengths "
          f"{UNIVERSAT_WAVELENGTHS[UNIVERSAT_MODALITY]} nm")
print(f"  Channels/fr:  {NUM_CHANNELS_PER_FRAME} (RGB)")
print(f"  Frames T:     2 (pre + post)")
print(f"  Classes:      {NUM_CLASSES} (BG, NoDamage, Minor, Major, Destroyed)")
print(f"  Crop size:    {args.crop_size}×{args.crop_size}")
print(f"  Epochs:       {args.epochs}")
print(f"  Batch size:   {args.batch_size}")
print(f"  LR:           {args.lr}")
print(f"  FLOPs at test: n={args.flops_n} counted passes"
      + (" (skipped)" if args.flops_n == 0 else ""))
print(f"  GPUs:         {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = XView2BaselineDataset(
    root_path=args.data_dir,
    mode="train",
    crop_size=args.crop_size,
    augment=True,
    oversample_building_damage=not args.no_oversample,
)
val_ds = XView2BaselineDataset(
    root_path=args.data_dir,
    mode="validation",
    crop_size=args.crop_size,
    augment=False,
    oversample_building_damage=False,
)
test_ds = XView2BaselineDataset(
    root_path=args.data_dir,
    mode="test",
    crop_size=args.crop_size,        # informational; full_image overrides
    augment=False,
    oversample_building_damage=False,
    full_image=True,                  # ← return full 1024×1024 for sliding window
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=xview_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, **loader_kwargs)
# Test loader: batch_size=1 because each sample is full 1024×1024×T=2.
# Sliding window inside the test loop will tile each sample into 4 patches
# of 512×512 and forward them in a single mini-batch.
test_loader  = DataLoader(test_ds,  batch_size=1,
                          shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model,
    in_channels=NUM_CHANNELS_PER_FRAME,
    num_classes=NUM_CLASSES,
    args=args,
)

trainer_module = BaselineTrainer(
    model=model,
    modality="s2",
    temporal=is_temporal_model,
    task="xview",
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_{args.model}"
        if args.model == "resnet_upernet_mt":
            run_name += f"_{args.resnet_variant}"
        if args.model == "universat":
            run_name += (f"_{args.universat_size}"
                         f"_os{args.universat_output_stride}"
                         f"_sp{args.universat_subpatch_px}")
        wandb.init(
            name=run_name,
            project="Atomizer_xView2_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_xView2_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/xview_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_mIoU:.4f}}",
            monitor="val_mIoU",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-last",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
        ),
        #EarlyStopping(
        #    monitor="val_mIoU",
        #    mode="max",
        #    patience=args.patience,
        #    verbose=True,
        #),
        LearningRateMonitor(logging_interval="step"),
    ]

    # `vit_upernet_ltae` builds an unused decoder.conv_seg internally
    # (the forward pass uses return_features=True then a separate self.head).
    # `ramen` similarly has unused parameters here: RAMENUPerNet's
    # RadarProjector registers polarization params (for a "sar" modality
    # this script never builds, since xView2 is RGB-only) that never
    # appear in the forward graph -- same DDP unused-parameter situation
    # as the other RAMEN integrations in this codebase.
    # `universat` has structurally inert params too: SAR/DEM channel codes
    # (VV/HH/.../nDEM) and the Bi_ACA_in fusion attention (single
    # modality). NOTE the T-axis block, inert on the snapshot datasets, IS
    # active here (T=2), and the S1-axis (sub-patch) block is ACTIVE at
    # the subpatch_px=4 default (only inert if --universat_subpatch_px 1
    # is passed). DDP needs find_unused_parameters=True for all three
    # model choices below regardless, since the exact set of inert params
    # can vary with the flags above.
    needs_unused = args.model in ("vit_upernet_ltae", "ramen", "universat")

    # Lightning's SLURM environment plugin reads SLURM_NNODES from the job
    # env and cross-checks it against Trainer(num_nodes=...) -- if
    # num_nodes isn't passed explicitly it defaults to 1, which raises
    # ("num_nodes=1 ... does not match SLURM --nodes=N") the moment this
    # runs as a multi-node sbatch job. Single-node runs never hit this
    # (SLURM_NNODES=1 matches the default), which is why it went unnoticed
    # until now. Mirrors train_biomassters.py / the BioMassters baseline
    # script's num_nodes handling.
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    print(f"[xView2-BL] num_nodes: {num_nodes} (from SLURM_NNODES, default 1 if unset)")

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=needs_unused),
        devices=-1, num_nodes=num_nodes,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )

    # Resolve --resume_from (with polling) right before fit -- dataset/model/
    # wandb setup above happens regardless of whether we're resuming, so
    # only the actual fit() call blocks waiting for the checkpoint.
    fit_ckpt_path = None
    if args.resume_from is not None:
        fit_ckpt_path = wait_for_checkpoint(
            args.resume_from, args.resume_wait_seconds, args.resume_poll_interval)
        print(f"[xView2-BL] RESUMING from: {fit_ckpt_path}")

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=fit_ckpt_path)

    best_ckpt = trainer.checkpoint_callback.best_model_path

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
    print(f"\n[test-only mode] Skipping training, testing checkpoint:")
    print(f"  {best_ckpt}\n")


# =============================================================================
# SLIDING-WINDOW TEST (mIoU + GFLOPs/forward)
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

# Load checkpoint into the trainer module (handles state dict naming, etc.)
ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = trainer_module.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys.")
if missing:
    print(f"[load_state_dict] {len(missing)} missing keys (likely OK if these "
          f"are runtime caches): {missing[:5]}{'...' if len(missing) > 5 else ''}")

class_names = ["BG", "NoDamage", "Minor", "Major", "Destroyed"]

results = evaluate_sliding_window(
    model=trainer_module,
    test_loader=test_loader,
    num_classes=NUM_CLASSES,
    tile_size=args.crop_size,
    stride=args.crop_size,           # non-overlapping
    device="cuda",
    wandb_logger=wandb_logger,
    class_names=class_names,
    ignore_index=IGNORE_INDEX,
    flops_n=args.flops_n,
)

print(f"[xView2-BL] Test GFLOPs/forward per crop:   "
      f"{results['gflops_per_crop']:.2f}")
print(f"[xView2-BL] Test GFLOPs/forward full image: "
      f"{results['gflops_full_image']:.2f} "
      f"({results['n_tiles']} tiles)")

if wandb_logger:
    import wandb
    wandb.finish()
