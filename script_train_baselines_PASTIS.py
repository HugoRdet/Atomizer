"""
PASTIS-HD Baseline Training Script
=====================================

Train temporal segmentation models on PASTIS-HD.

Supported models (via --model):
  - unet_ltae           : Classic UNet (per-frame, shared) + LTAE temporal
                          aggregation at full output resolution + 1×1 head
  - vit_ltae            : ViT per-frame + LTAE BETWEEN encoder and decoder
                          (one LTAE per FPN feature layer) + UPerNet
  - vit_upernet_ltae    : ViT per-frame + UPerNet (per-frame features)
                          + LTAE AFTER decoder at FPN resolution + 1×1 head
  - vit_upernet_mt      : ViT + UPerNet with channel-concat early fusion
                          (TimeMerge DoubleConv before encoder).
                          Mirror of resnet_upernet_mt for the ViT family —
                          PANGAEA-style early fusion.
  - vit                 : ViT encoder (channel-stacked frames, non-temporal) + UPerNet
  - prithvi              : Prithvi 3D ViT (3D tubelet conv) + UPerNet
  - resnet_upernet_mt   : ResNet (channel-concat early fusion via TimeMerge
                          DoubleConv) + UPerNet — PANGAEA-style.
                          Replaces the old resnet_upernet_ltae (late fusion)
                          for direct comparability with PANGAEA's reported
                          UNetMT/ViT numbers on PASTIS.
  - ramen                : RAMENUPerNet (spectral tokenization encoder) +
                          UPerNet decoder, using RAMEN's OWN per-modality
                          LTAE for temporal fusion — the SAME mechanism as
                          the xView2 RAMEN integration
                          (script_train_xview_baselines.py), just fed REAL
                          per-frame day-of-year dates here instead of
                          xView2's synthetic [0, 1] ordering placeholder
                          (PastisBaselineDataset._dates_to_doy gives every
                          modality genuine per-frame acquisition dates —
                          see the UniverSat section below, which predates
                          this integration and already established the
                          dated-collate pattern this reuses).

                          RAMENPastisAdapter wraps RAMENUPerNet so it
                          accepts the SAME merged dict shape UniverSat's
                          integration already produces —
                          {"s2": [B,T,10,H,W], "s2_dates": [B,T],
                          "s1": [B,T,2,H,W], "s1_dates": [B,T]} — via
                          make_universat_pastis_collate (reused unmodified;
                          despite the name it's really "dated multimodal
                          PASTIS collate", not UniverSat-specific). The
                          adapter permutes each modality to RAMEN's
                          expected [B, C, T, H, W] layout, RENAMES the
                          dict keys "s2"->"optical"/"s1"->"sar" (RAMEN's
                          forward() branches on the literal string "sar"
                          internally — confirmed by the BioMassters RAMEN
                          integration's make_ramen_collate docstring; an
                          earlier version of this adapter kept the
                          dataset's own "s2"/"s1" naming and crashed with
                          a torch.tensor() error on the SAR pol_map's
                          string values, since RAMEN silently routed it
                          through the numeric-wavelength optical path
                          instead), and forwards dates as its own kwarg,
                          mirroring RAMENXView2Adapter's permute-and-call
                          pattern.

                          SAR BAND CODES: RAMEN's Sen1Floods11 integration
                          uses pol_map string codes ("asc_vv", "asc_vh"),
                          NOT UniverSat's own codes ("VV", "VH") — these
                          are CONFIRMED different conventions within this
                          codebase (see the Sen1Floods11 RAMEN script's
                          S1_POLARIZATIONS dict). PASTIS's S1 has a 3rd
                          derived-ratio channel (VV-VH) with no established
                          RAMEN code anywhere in this project, so only the
                          first 2 raw channels (VV, VH) are fed to RAMEN
                          regardless of --universat_use_s1_ratio (that flag
                          is UniverSat-only).

                          RAMEN tokenizes per-pixel, so --ramen_window_size
                          must equal PASTIS-HD's fixed 128x128 tile size —
                          same constraint as the xView2 integration's
                          --ramen_window_size == --crop_size requirement,
                          just against a fixed tile instead of a
                          configurable crop. At that default, RAMEN's
                          window equals the whole tile, so this is a single
                          dense forward per tile — no sliding-window
                          tiling actually happens, matching how a 512
                          window on Sen1Floods11's native 512 image
                          degenerates the same way.
  - universat            : UniverSat encoder (gastruc/UniverSat, AnySat v2)
                          trained FROM SCRATCH (random init) + linear
                          per-token head, via
                          training.Universat.universat_augmenter.UniverSatSegmenter.

                          TRUE multimodal + TEMPORAL path: unlike the
                          Sen1Floods11 integration (single-frame, no dates
                          needed), PASTIS gives every modality its own
                          real per-frame acquisition dates
                          (PastisBaselineDataset._dates_to_doy — genuine
                          day-of-year, not a synthetic placeholder like
                          xView2's [0, 180]). UniverSat's UPE detects a
                          "<mod>_dates" key alongside "<mod>" and runs its
                          temporal-axis block on the REAL dates, so this
                          integration builds a DEDICATED collate
                          (make_universat_pastis_collate) that embeds dates
                          directly into the image dict at collation time
                          -- {"s2": [B,T,10,H,W], "s2_dates": [B,T],
                          "s1": [B,T,2,H,W], "s1_dates": [B,T]} -- rather
                          than routing through the existing
                          pastis_collate/make_fused_collate (which either
                          drops S1's own dates by reusing S2's truncated
                          dates, or never surfaces dates to the model at
                          all). This sidesteps any assumption about
                          whether BaselineTrainer forwards a separate
                          "dates" dict alongside "image" for
                          expects_full_image_dict models -- the merged
                          dict IS the full forward-time input, exactly
                          the {"vhr": ..., "vhr_dates": ...} shape
                          UniverSatXView2Adapter builds by hand for
                          xView2's (synthetic-date) case. RAMEN's PASTIS
                          integration above reuses this same collate.

                          SAR BAND CODES: PASTIS's S1 data has 3 channels
                          -- VV, VH, VV-VH (a DERIVED RATIO, not a real
                          polarization -- see PastisBaselineDataset's
                          docstring/_load_s1). UniverSat's full supported
                          SAR/elevation channel-code list is given in
                          universat_augmenter.py's UniverSatSegmenter
                          docstring: "VV", "VH", "HH", "HV",
                          "Ratio_VV_VH", "Ratio_HH_HV", "DSM", "nDEM" --
                          so the ratio band's correct code is
                          "Ratio_VV_VH" (NOT the dataset's own "VV-VH"
                          spelling). By default this integration feeds
                          UniverSat only the first 2 S1 channels (VV, VH);
                          pass --universat_use_s1_ratio to additionally
                          feed the 3rd channel under "Ratio_VV_VH".

                          No window is baked at construction: the latent
                          grid is recomputed per input (side / patch_px),
                          so the same weights handle every PASTIS tile
                          size. Geometry: PASTIS-HD tiles are a FIXED
                          128x128 (no --crop_size knob on this dataset,
                          unlike xView2/Sen1Floods11) -- every side must
                          be divisible by lcm(patch_px, output_stride,
                          subpatch_px); validated below (128 satisfies
                          the defaults: patch=16px, os=4, subpatch=1).

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training and run
    test on a saved checkpoint (single GPU, no DDP).

GFLOPs: measured once after testing completes, using the same
FlopCounterMode-based harness as the xView2 baseline script (counts SDPA
attention correctly, unlike torch.profiler(with_flops=True) used in the
Sen1Floods11 script — see that script's docstring warning: the profiler
silently drops every attention FLOP term, so DO NOT mix these numbers
with GFLOPs pulled from the Sen1Floods11/BurnScars sweep scripts in the
same table without re-measuring them under FlopCounterMode). Rank-zero
only. Disable with --flops_n 0. Currently measured for --model universat
AND --model ramen (both were added together; other models had no FLOPs
harness in this script before the UniverSat integration).

Measured strictly at bs=1 via a DEDICATED DataLoader built for the FLOPs
pass (not the training/test bs=args.batch_size loader) -- matching every
other GFLOPs harness in this codebase (script_universat_sweep_senflood.py
/ _burnscars.py's make_loader(batch_size=1), the xView2 baseline script's
_build_tile_batch, script_train_xview.py's measure_test_gflops). An
earlier version of this block reused `test_loader` (built at
args.batch_size, default 4) for the FLOPs forward pass while still
printing/logging the result as "bs=1" -- since UniverSat's per-sample
compute is ~linear in batch size, that silently reported
~batch_size x the true per-sample GFLOPs under a "bs=1" label. Fixed here.

Examples:
    # New ViT + channel-concat MT, S2-only, 6 frames (PANGAEA convention)
    python script_train_pastis_baseline.py --xp_name vit_mt_t6 \
        --model vit_upernet_mt --multi_temporal 6 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # ResNet + channel-concat MT, S2-only, 6 frames (PANGAEA convention)
    python script_train_pastis_baseline.py --xp_name resnet50_mt_s2 \
        --model resnet_upernet_mt --resnet_variant resnet50 \
        --multi_temporal 6 --batch_size 4 --lr 1e-4 --epochs 100

    # UNet + LTAE
    python script_train_pastis_baseline.py --xp_name unet_ltae_s2 \
        --model unet_ltae --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100

    # ViT + LTAE
    python script_train_pastis_baseline.py --xp_name vit_ltae_s2 \
        --model vit_ltae --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # ViT (non-temporal, channel-stacked), S2-only, 3 frames
    python script_train_pastis_baseline.py --xp_name vit_s2_t3 \
        --model vit --multi_temporal 3 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # UNet+LTAE with S2+S1 fusion
    python script_train_pastis_baseline.py --xp_name unet_ltae_s2s1 \
        --model unet_ltae --use_s1 --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100

    # RAMEN, S2-only, real per-frame dates via its own LTAE
    python script_train_pastis_baseline.py --xp_name ramen_s2 \
        --model ramen --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # RAMEN, S2+S1 (VV/VH only — no established code for the ratio band)
    python script_train_pastis_baseline.py --xp_name ramen_s2s1 \
        --model ramen --use_s1 --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # UniverSat-S from scratch, S2+S1, real dates, email-convention config
    # (patch=16px/160m, subpatch=1, os=4)
    python script_train_pastis_baseline.py --xp_name universat_s2s1 \
        --model universat --use_s1 --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # UniverSat, S2-only
    python script_train_pastis_baseline.py --xp_name universat_s2 \
        --model universat --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_PASTIS import (
    PastisBaselineDataset, NUM_CLASSES, IGNORE_INDEX, NUM_S2_BANDS, NUM_S1_BANDS,
    S2_WAVELENGTHS, S2_BANDWIDTHS,
)
from training.ltae.ltae import UNetLTAE
from training.VIT.model_vit_upernet import (
    ViTUPerNet, ViTLTAEUPerNet, ViTUPerNetLTAE, build_vit_upernet_mt,
)
from training.ResNet.model_resnet_upernet import build_resnet_upernet_mt
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.prithvi.prithvi import PrithviUPerNet
from training.Universat.universat_augmenter import build_universat_segmenter
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

PASTIS_GSD_M = 10.0   # Sentinel-1/2 common grid, matches the Sen1Floods11 default
NATIVE_TILE_PX = 128  # PASTIS-HD tiles are fixed-size — no --crop_size knob


# =============================================================================
# RAMEN BAND METADATA
# =============================================================================
# S2: same continuous-wavelength band names as UniverSat's table below,
# WITHOUT per-frame suffixing — unlike a channel-stacking integration,
# RAMEN's own per-modality LTAE handles the temporal axis internally
# (same mechanism as its xView2 integration), so each band appears once
# and the tensor's TIME axis (not extra channels) carries the T frames.
RAMEN_S2_BAND_NAMES = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
]
assert len(RAMEN_S2_BAND_NAMES) == NUM_S2_BANDS == len(S2_WAVELENGTHS), (
    "RAMEN_S2_BAND_NAMES must line up 1:1 with S2_WAVELENGTHS from "
    "utils_dataset_PASTIS — check for band-count drift."
)
RAMEN_S2_WAVELENGTHS_NM = dict(zip(RAMEN_S2_BAND_NAMES, S2_WAVELENGTHS))

# S1: RAMEN's OWN pol_map convention (CONFIRMED via its Sen1Floods11
# integration's S1_POLARIZATIONS dict), NOT UniverSat's "VV"/"VH" string
# codes — these are two different conventions within this codebase. Only
# the first 2 raw S1 channels (VV, VH) are used; PASTIS's 3rd channel
# (a derived VV-VH ratio) has no established RAMEN code anywhere in this
# project, unlike UniverSat's confirmed "Ratio_VV_VH".
RAMEN_S1_BAND_NAMES = ["VV", "VH"]
RAMEN_S1_POL_MAP = {"VV": "asc_vv", "VH": "asc_vh"}


# =============================================================================
# UNIVERSAT BAND METADATA
# =============================================================================
# S2: continuous wavelengths (nm), same 10-band set the dataset already
# reports in each sample's metadata (S2_WAVELENGTHS/S2_BANDWIDTHS from
# utils_dataset_PASTIS). Band "names" here are only dict keys — UniverSat's
# optical path goes through continuous-wavelength MP-Fourier encoding, no
# registry membership required (same as the Sen1Floods11 integration).
UNIVERSAT_S2_BAND_NAMES = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
]
assert len(UNIVERSAT_S2_BAND_NAMES) == NUM_S2_BANDS == len(S2_WAVELENGTHS), (
    "UNIVERSAT_S2_BAND_NAMES must line up 1:1 with S2_WAVELENGTHS from "
    "utils_dataset_PASTIS — check for band-count drift."
)
UNIVERSAT_S2_WAVELENGTHS_NM = dict(zip(UNIVERSAT_S2_BAND_NAMES, S2_WAVELENGTHS))

# S1: PASTIS gives 3 raw channels [VV, VH, VV-VH] (see PastisBaselineDataset
# docstring/_load_s1), where the 3rd is a DERIVED RATIO, not a raw
# polarization. UniverSat's supported SAR/elevation channel codes are
# given directly in universat_augmenter.py's UniverSatSegmenter docstring:
# "VV", "VH", "HH", "HV", "Ratio_VV_VH", "Ratio_HH_HV", "DSM", "nDEM" --
# looked up as learned Encoding_<code> embeddings in the UPE. So the ratio
# band's code is CONFIRMED to be "Ratio_VV_VH" (not the dataset's own
# "VV-VH" naming) -- by default only the first 2 channels (VV, VH) are
# fed to UniverSat; --universat_use_s1_ratio additionally feeds channel 3
# under "Ratio_VV_VH".
UNIVERSAT_S1_BAND_NAMES_BASE = ["VV", "VH"]
UNIVERSAT_S1_BAND_NAMES_WITH_RATIO = ["VV", "VH", "Ratio_VV_VH"]
# The codes ARE the names, same convention as the Sen1Floods11 integration
# (UNIVERSAT_WAVELENGTHS["sar"] = S1_BAND_NAMES there) -- wavelengths for
# string codes pass through UniverSatSegmenter.__init__ untouched (only
# numeric values >100 get the nm->µm conversion applied).


# =============================================================================
# COLLATE — handles nested dicts (image, dates, target, metadata)
# =============================================================================

def pastis_collate(batch):
    images = {}
    dates = {}
    sensor_keys = list(batch[0]["image"].keys())

    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
        dates[key] = torch.stack([s["dates"][key] for s in batch])

    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "dates": dates,
        "target": targets,
        "metadata": metadata,
    }


def make_fused_collate(use_s1: bool):
    """Returns a collate that fuses S2 and S1 along the channel dim."""
    if not use_s1:
        return pastis_collate

    def fused_collate(batch):
        out = pastis_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T,  2, H, W]
        T = min(s2.shape[1], s1.shape[1])
        fused = torch.cat([s2[:, :T], s1[:, :T]], dim=2)  # [B, T, 12, H, W]
        out["image"] = {"s2": fused}
        out["dates"] = {"s2": out["dates"]["s2"][:, :T]}
        return out

    return fused_collate


def make_channel_stack_collate(base_collate):
    """[B, T, C, H, W] → [B, T*C, H, W] for non-temporal models."""
    def stacked_collate(batch):
        out = base_collate(batch)
        for key, img in out["image"].items():
            if img.dim() == 5:
                B, T, C, H, W = img.shape
                out["image"][key] = img.reshape(B, T * C, H, W)
        return out
    return stacked_collate


def make_universat_pastis_collate(use_s1: bool, use_s1_ratio: bool):
    """
    Dated multimodal PASTIS collate (despite the name, shared by BOTH the
    UniverSat AND RAMEN integrations — see each model's section in the
    module docstring): embeds each modality's REAL per-frame dates
    directly into the image dict as "<mod>_dates" keys, producing
    {"s2": [B,T,10,H,W], "s2_dates": [B,T], "s1": [B,T,S1C,H,W],
    "s1_dates": [B,T]}. UniverSat's UPE consumes this directly
    (`model(batch["image"])`); RAMENPastisAdapter consumes the same shape
    and does its own permute + dates-kwarg call before forwarding into
    RAMENUPerNet.

    S2 is passed through unchanged (all 10 bands). S1, if requested, is
    sliced to 2 channels (VV, VH) unless use_s1_ratio is set, in which
    case all 3 raw channels (VV, VH, VV-VH) are passed — see module
    docstring for why the 3rd channel is UniverSat-only (RAMEN has no
    established code for it, so the RAMEN branch always calls this with
    use_s1_ratio=False regardless of --universat_use_s1_ratio).

    NOTE this is intentionally NOT built on top of pastis_collate's
    per-sample stacking loop, to avoid re-deriving dates twice; it stacks
    image/dates directly per modality, same structure as pastis_collate.
    """
    s1_channels = slice(None) if use_s1_ratio else slice(0, 2)

    def collate(batch):
        image = {
            "s2": torch.stack([s["image"]["s2"] for s in batch]),       # [B,T,10,H,W]
            "s2_dates": torch.stack([s["dates"]["s2"] for s in batch]), # [B,T]
        }
        if use_s1:
            s1_full = torch.stack([s["image"]["s1"] for s in batch])       # [B,T,3,H,W]
            s1_dates = torch.stack([s["dates"]["s1"] for s in batch])      # [B,T]
            image["s1"] = s1_full[:, :, s1_channels]
            image["s1_dates"] = s1_dates

        targets = torch.stack([s["target"] for s in batch])
        metadata = [s["metadata"] for s in batch]
        return {"image": image, "target": targets, "metadata": metadata}

    return collate


# =============================================================================
# RAMEN ADAPTER
# =============================================================================

class RAMENPastisAdapter(nn.Module):
    """
    Wraps RAMENUPerNet so it accepts the SAME merged dict shape the
    (misleadingly-named, shared) make_universat_pastis_collate produces —
    {"s2": [B,T,10,H,W], "s2_dates": [B,T], "s1": [B,T,2,H,W],
    "s1_dates": [B,T]} — permuting each modality to RAMEN's expected
    [B, C, T, H, W] layout AND remapping the dict keys "s2"->"optical",
    "s1"->"sar" before calling into RAMENUPerNet, then forwarding dates
    as RAMEN's own `dates` kwarg. Same permute-and-call pattern as
    RAMENXView2Adapter (script_train_xview_baselines.py), just multimodal
    (S2 + optional S1) and fed REAL per-frame dates instead of xView2's
    synthetic [0, 1] ordering placeholder.

    # >>> KEY_RENAME_FIX: RAMENUPerNet.forward() branches on the LITERAL
    # string "sar" to route that modality through RadarProjector's
    # pol_map path; anything else falls through to the generic spectral/
    # wavelength path, which does torch.tensor() on the wavelengths value
    # -- crashing if that value is a dict of pol_map strings (e.g.
    # {"VV": "asc_vv", ...}) rather than numbers. This is CONFIRMED by the
    # BioMassters RAMEN integration's own make_ramen_collate docstring:
    # "Renamed from 's2'/'s1' to 'optical'/'sar' -- RAMEN's forward()
    # branches on the literal string 'sar' ... so the key name is
    # load-bearing, not cosmetic." An earlier version of this adapter kept
    # the dataset's own "s2"/"s1" naming (matching UniverSat's convention,
    # which does NOT branch on modality name) and crashed with exactly
    # this torch.tensor(string dict) error. The COLLATE still produces
    # "s2"/"s1" keys (shared with UniverSat, which needs no remapping) --
    # only this adapter's OWN dict, built just before calling into
    # RAMENUPerNet, uses "optical"/"sar".

    expects_full_image_dict=True so BaselineTrainer forwards batch["image"]
    (the merged dict, dates already embedded by the collate) straight
    through as this module's single positional argument — same duck-typed
    convention as the RAMEN/UniverSat integrations on every other dataset
    in this codebase.
    """
    expects_full_image_dict = True

    def __init__(self, ramen_upernet: nn.Module, use_s1: bool):
        super().__init__()
        self.model = ramen_upernet
        self.use_s1 = use_s1

    def forward(self, x: dict, **kwargs):
        s2 = x["s2"].permute(0, 2, 1, 3, 4).contiguous()  # [B,T,10,H,W] -> [B,10,T,H,W]
        image = {"optical": s2}          # >>> KEY_RENAME_FIX: NOT "s2"
        dates = {"optical": x["s2_dates"]}
        if self.use_s1 and "s1" in x:
            s1 = x["s1"].permute(0, 2, 1, 3, 4).contiguous()
            image["sar"] = s1            # >>> KEY_RENAME_FIX: NOT "s1"
            dates["sar"] = x["s1_dates"]
        return self.model(image, dates=dates, **kwargs)


# =============================================================================
# PRITHVI ADAPTER
# =============================================================================

class PrithviAdapter(nn.Module):
    """Wraps PrithviUPerNet to accept [B, T, C, H, W] (Prithvi expects [B, C, T, H, W])."""
    def __init__(self, prithvi_model: nn.Module):
        super().__init__()
        self.prithvi = prithvi_model

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.prithvi(x, doy=doy)


# =============================================================================
# FLOPs MEASUREMENT — FlopCounterMode (counts SDPA attention)
# =============================================================================
# Same convention, same methodology, and the SAME counting tool as the
# xView2 baseline script (script_train_xview_baselines.py) and Atomiser's
# xView2 / Sen1Floods11 scripts (script_train_xview.py /
# script_test_senflood_density_skip.py): torch.utils.flop_counter.
# FlopCounterMode (SDPA attention counted), one warmup pass discarded, mean
# over the remaining counted passes, GFLOPs = total_flops / 1e9, measured
# under torch.no_grad() via the module-tracker no-op patch below (the
# per-module attribution hook otherwise asserts under no_grad; the overall
# total, which is all we use here, is unaffected -- see that hook's
# docstring in the other scripts for the full explanation). This is the
# SAME counter/methodology across every script in this codebase now that
# the Sen1Floods11 density script has also been switched off
# torch.profiler(with_flops=True) -- numbers from this script ARE directly
# comparable to those.

def _patch_module_tracker_for_no_grad():
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


def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b


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
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, args):
    """Dispatch to the requested model architecture."""
    if model_name == "unet_ltae":
        return UNetLTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=tuple(args.unet_topology),
            n_heads=args.n_heads,
            d_k=args.d_k,
            d_model=args.d_model,
        )

    elif model_name == "vit_ltae":
        return ViTLTAEUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.n_heads,
            ltae_d_k=args.d_k,
            ltae_d_model=args.d_model,
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
            ltae_n_head=args.n_heads,
            ltae_d_k=args.d_k,
            ltae_d_model=args.d_model,
        )

    elif model_name == "vit_upernet_mt":
        # ViT + UPerNet with channel-concat early fusion (TimeMerge DoubleConv).
        # Mirror of resnet_upernet_mt for the ViT family.
        return build_vit_upernet_mt(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=args.multi_temporal,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name in ("resnet_upernet_mt", "resnet_upernet_ltae"):
        # Channel-concat early fusion via TimeMerge DoubleConv (PANGAEA-style).
        # The legacy 'resnet_upernet_ltae' name is kept as an alias for
        # back-compat, but it now refers to early-fusion MT (LTAE variant
        # was removed when we switched to PANGAEA-style temporal handling).
        if model_name == "resnet_upernet_ltae":
            print("[WARN] 'resnet_upernet_ltae' is now an alias for "
                  "'resnet_upernet_mt' (channel-concat early fusion). "
                  "The LTAE late-fusion variant has been removed.")
        return build_resnet_upernet_mt(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=args.multi_temporal,
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "prithvi":
        prithvi = PrithviUPerNet(
            in_chans=in_channels,
            num_frames=args.multi_temporal,
            img_size=args.img_size,
            patch_size=args.vit_patch_size,
            tubelet_size=args.prithvi_tubelet_size,
            embed_dim=args.prithvi_embed_dim,
            depth=args.prithvi_depth,
            num_heads=args.prithvi_num_heads,
            num_classes=num_classes,
            decoder_channels=args.vit_decoder_channels,
            output_layers=tuple(args.vit_output_layers),
        )
        return PrithviAdapter(prithvi)

    elif model_name == "ramen":
        # in_channels is unused here (like the UniverSat branch below) —
        # RAMEN derives channel count from the band-name dicts. Temporal
        # fusion is RAMEN's OWN per-modality LTAE (real per-frame dates,
        # via RAMENPastisAdapter), not channel-stacking or TimeMerge.
        # >>> KEY_RENAME_FIX: dict keys MUST be "optical"/"sar" (RAMEN
        # branches on the literal string "sar" internally), NOT "s2"/"s1"
        # -- see RAMENPastisAdapter's docstring for the crash this fixes.
        input_bands = {"optical": RAMEN_S2_BAND_NAMES}
        wavelengths = {"optical": RAMEN_S2_WAVELENGTHS_NM}
        if args.use_s1:
            input_bands["sar"] = RAMEN_S1_BAND_NAMES
            wavelengths["sar"] = RAMEN_S1_POL_MAP   # RAMEN's own pol_map codes

        base = build_ramen_upernet(
            input_bands=input_bands,
            wavelengths=wavelengths,
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
        return RAMENPastisAdapter(base, use_s1=args.use_s1)

    elif model_name == "universat":
        # Random init, no pretrained weights. in_channels is unused here
        # (channel count/identity comes from the band-name dicts below).
        # No window baked at construction — same weights handle the fixed
        # 128x128 PASTIS-HD tile size, and would handle any other tile
        # size too since the latent grid is recomputed per input.
        input_bands = {"s2": UNIVERSAT_S2_BAND_NAMES}
        wavelengths = {"s2": UNIVERSAT_S2_WAVELENGTHS_NM}
        input_res = {"s2": PASTIS_GSD_M}
        if args.use_s1:
            s1_names = (UNIVERSAT_S1_BAND_NAMES_WITH_RATIO
                        if args.universat_use_s1_ratio
                        else UNIVERSAT_S1_BAND_NAMES_BASE)
            input_bands["s1"] = s1_names
            wavelengths["s1"] = s1_names   # codes ARE the names (Sen1Floods11 convention)
            input_res["s1"] = PASTIS_GSD_M

        return build_universat_segmenter(
            input_bands=input_bands,
            wavelengths=wavelengths,
            num_classes=num_classes,
            input_res=input_res,
            patch_size_m=args.universat_patch_m,
            output_stride=args.universat_output_stride,
            size=args.universat_size,
            subpatch_px=args.universat_subpatch_px,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: 'unet_ltae', 'vit_ltae', 'vit_upernet_ltae', "
            f"'vit_upernet_mt', 'vit', 'prithvi', 'resnet_upernet_mt', "
            f"'ramen', 'universat'"
        )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PASTIS-HD Baseline Training")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--model",      type=str, default="unet_ltae",
                    choices=["unet_ltae", "vit_ltae", "vit_upernet_ltae",
                             "vit_upernet_mt",
                             "vit", "prithvi", "resnet_upernet_mt",
                             "resnet_upernet_ltae",  # legacy alias kept
                             "ramen", "universat"])
parser.add_argument("--data_dir",   type=str, default="./data/PASTIS-HD")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Modality
parser.add_argument("--use_s1",     action="store_true",
                    help="Include S1A SAR data (default: S2-only)")

# Temporal
parser.add_argument("--multi_temporal", type=int, default=10,
                    help="Number of temporal frames to use")
parser.add_argument("--temporal_last",  action="store_true",
                    help="Take last N timesteps instead of uniform sampling")

# Training
parser.add_argument("--batch_size",  type=int, default=4)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",    type=int, default=30)
parser.add_argument("--grad_accum",  type=int, default=1)

# UNet+LTAE / LTAE shared params (still used by unet_ltae, vit_ltae, vit_upernet_ltae)
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])
parser.add_argument("--n_heads",    type=int, default=16)
parser.add_argument("--d_k",        type=int, default=4)
parser.add_argument("--d_model",    type=int, default=256)

# ViT-specific
parser.add_argument("--img_size",          type=int, default=128)
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+", default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)

# Prithvi-specific
parser.add_argument("--prithvi_embed_dim",     type=int, default=768)
parser.add_argument("--prithvi_depth",         type=int, default=12)
parser.add_argument("--prithvi_num_heads",     type=int, default=12)
parser.add_argument("--prithvi_tubelet_size",  type=int, default=1)

# ResNet-specific
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN-specific
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=PASTIS_GSD_M,
                    help="Native GSD (m/px) of PASTIS S2/S1, 10.0.")
parser.add_argument("--ramen_res",       type=float, default=80,
                    help="Working resolution (m/px) RAMEN resamples to. "
                         "Defaults to --ramen_input_res (no resampling) "
                         "if left unset.")
parser.add_argument("--ramen_window_size", type=int, default=None,
                    help="Spatial size RAMEN is built at. Must equal "
                         "PASTIS-HD's fixed 128x128 tile size — RAMEN "
                         "tokenizes per-pixel and this script does no "
                         "sliding-window tiling for it (unlike the xView2 "
                         "integration's evaluate_sliding_window), so a "
                         "mismatch would silently resample rather than "
                         "tile. Defaults to 128 (NATIVE_TILE_PX) if unset "
                         "— only override if you know what you're doing.")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="'small' (384-d, 12 SA blocks, heads=6) is ~36.1M "
                         "total -- parameter-matched to the ViT-S / RAMEN "
                         "~34M budget. 'tiny' ~6.2M; 'base' ~201M.")
parser.add_argument("--universat_patch_m", type=float, default=160.0,
                    help="Patch size in METRES. 160 m = 16 px at PASTIS's "
                         "10 m GSD -- the email-convention default "
                         "(patch size (px)=16 in the authors' table). "
                         "Must be an integer number of pixels, and the "
                         "128x128 tile side must be divisible by it.")
parser.add_argument("--universat_output_stride", type=int, default=1,
                    help="Logits at side/stride per side (upsampled to "
                         "full res by BaselineTrainer). Default 4 matches "
                         "the email-convention 'stride decodeur (px)' "
                         "column for every dataset except xView2.")
parser.add_argument("--universat_subpatch_px", type=int, default=1,
                    help="Sub-patch size in PIXELS for the UPE's S1-axis "
                         "sub-patch skip. Default 1 (pixel-level CA_Sub "
                         "keys) matches the email-convention 'subpatch' "
                         "column for every dataset except xView2 (VHR, "
                         "0.5 m GSD, subpatch=8 there) -- at PASTIS's "
                         "10 m GSD there's little to gain from grouping "
                         "pixels into coarser sub-patches. patch_px must "
                         "be a multiple of this.")
parser.add_argument("--universat_use_s1_ratio", action="store_true",
                    help="Also feed UniverSat the 3rd S1 channel (the "
                         "dataset's 'VV-VH' derived ratio) under "
                         "UniverSat's confirmed channel code "
                         "'Ratio_VV_VH' (see UniverSatSegmenter's "
                         "docstring for the full supported code list). "
                         "Off by default: UniverSat sees only VV/VH even "
                         "when --use_s1 pulls in the ratio band for other "
                         "models -- flip on deliberately if you want "
                         "channel parity with those baselines. UniverSat-"
                         "only: RAMEN has no established code for this "
                         "band and always uses only VV/VH.")

# FLOPs measurement (test time only)
parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of counted FlopCounterMode forward passes "
                         "at test time (mean GFLOPs/forward reported), "
                         "measured at bs=1 via a dedicated DataLoader "
                         "(NOT args.batch_size -- see module docstring). "
                         "One extra warmup pass is run first and discarded. "
                         "Set 0 to skip FLOPs measurement entirely. "
                         "Measured for --model universat and --model ramen "
                         "(other models had no FLOPs harness in this "
                         "script prior to the UniverSat integration).")

args = parser.parse_args()


# =============================================================================
# RAMEN SANITY CHECK
# =============================================================================

if args.model == "ramen":
    if args.ramen_window_size is None:
        args.ramen_window_size = NATIVE_TILE_PX
    if args.ramen_window_size != NATIVE_TILE_PX:
        print(f"[WARNING] --ramen_window_size={args.ramen_window_size} != "
              f"PASTIS-HD's fixed tile size ({NATIVE_TILE_PX}). This script "
              f"does no sliding-window tiling for RAMEN (unlike the xView2 "
              f"integration), so a mismatch means RAMEN is built for one "
              f"spatial extent but forwarded {NATIVE_TILE_PX}x{NATIVE_TILE_PX} "
              f"tiles at train/test time, forcing a silent internal "
              f"resample rather than a crash. Leave --ramen_window_size "
              f"unset (defaults to {NATIVE_TILE_PX}) unless you have a "
              f"specific reason to diverge.")
    if args.ramen_res is None:
        args.ramen_res = args.ramen_input_res


# =============================================================================
# UNIVERSAT SANITY CHECK
# =============================================================================

if args.model == "universat":
    universat_patch_px = args.universat_patch_m / PASTIS_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {PASTIS_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of {PASTIS_GSD_M}."
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

    # PASTIS-HD tiles are a FIXED 128x128 — no --crop_size knob on this
    # dataset (unlike xView2/Sen1Floods11), so the one geometry check is
    # against NATIVE_TILE_PX directly.
    if NATIVE_TILE_PX % _lcm:
        raise ValueError(
            f"PASTIS-HD's fixed {NATIVE_TILE_PX}x{NATIVE_TILE_PX} tile size "
            f"must be divisible by lcm(patch_px={universat_patch_px}, "
            f"output_stride={args.universat_output_stride}, "
            f"subpatch_px={args.universat_subpatch_px})={_lcm}. Adjust "
            f"--universat_patch_m / --universat_output_stride / "
            f"--universat_subpatch_px."
        )

    if args.universat_use_s1_ratio and not args.use_s1:
        print("[WARNING] --universat_use_s1_ratio has no effect without "
              "--use_s1 (no S1 data is loaded at all).")


# =============================================================================
# CONFIG
# =============================================================================

modalities = ["S2"]
per_frame_channels = NUM_S2_BANDS  # 10

if args.use_s1:
    modalities.append("S1")
    per_frame_channels = NUM_S2_BANDS + NUM_S1_BANDS  # 12

modality_str = "+".join(modalities)
temporal_str = f"{args.multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"

# Models that accept 5D [B, T, C, H, W] input directly (with their own
# internal temporal handling — LTAE, 3D conv, TimeMerge DoubleConv, or
# (for universat/ramen) their own dict-based temporal-axis handling).
is_temporal_model = args.model in (
    "unet_ltae", "vit_ltae", "vit_upernet_ltae", "vit_upernet_mt",
    "prithvi", "resnet_upernet_mt", "resnet_upernet_ltae",
    "ramen", "universat",
)
if args.model in ("universat", "ramen"):
    model_in_channels = per_frame_channels  # unused by either builder
elif is_temporal_model:
    model_in_channels = per_frame_channels         # model sees [B, T, C, H, W]
else:
    model_in_channels = per_frame_channels * args.multi_temporal  # [B, T*C, H, W]

# Print summary
if args.test_only:
    print(f"\n[Train] Test-only mode: {args.test_only}\n")

print(f"\n{'='*60}")
print(f"  PASTIS-HD Baseline Training")
print(f"  Model:      {args.model} ({'temporal' if is_temporal_model else 'non-temporal'})")
print(f"  Modalities: {modality_str} ({per_frame_channels} bands/frame)")
print(f"  Temporal:   {temporal_str}")
if not is_temporal_model:
    print(f"  In channels (stacked): {model_in_channels}")
if args.model == "ramen":
    print(f"  RAMEN:      embed_dim={args.ramen_embed_dim}, "
          f"depth={args.ramen_depth}, heads={args.ramen_num_heads}")
    print(f"  Resolution: input_res={args.ramen_input_res}, res={args.ramen_res} "
          f"({'no resampling' if args.ramen_input_res == args.ramen_res else 'resampled'})")
    print(f"  Window:     {args.ramen_window_size}x{args.ramen_window_size} "
          f"(= fixed PASTIS tile size, no tiling)")
    print(f"  S1 bands:   {RAMEN_S1_BAND_NAMES if args.use_s1 else 'none'} "
          f"(pol_map: {RAMEN_S1_POL_MAP if args.use_s1 else 'n/a'})")
    print(f"  Temporal:   RAMEN's own per-modality LTAE, real per-frame "
          f"day-of-year dates")
if args.model == "universat":
    print(f"  UniverSat:  {args.universat_size} (from scratch, random init)")
    print(f"  Patch:      {args.universat_patch_m:.0f} m "
          f"({int(args.universat_patch_m / PASTIS_GSD_M)} px @ "
          f"{PASTIS_GSD_M:.0f} m)")
    print(f"  Sub-patch:  {args.universat_subpatch_px} px "
          f"(S1 axis {'active' if args.universat_subpatch_px > 1 else 'inert'})")
    print(f"  Out stride: {args.universat_output_stride} "
          f"(logits at {NATIVE_TILE_PX}/{args.universat_output_stride}, "
          f"native {args.universat_output_stride * PASTIS_GSD_M:.0f} m granularity)")
    print(f"  S1 bands:   "
          f"{(UNIVERSAT_S1_BAND_NAMES_WITH_RATIO if args.universat_use_s1_ratio else UNIVERSAT_S1_BAND_NAMES_BASE) if args.use_s1 else 'none'}")
    print(f"  Temporal:   real per-frame day-of-year dates (per modality)")
print(f"  Epochs:     {args.epochs}")
print(f"  BS:         {args.batch_size}")
print(f"  LR:         {args.lr}")
print(f"  Grad acc:   {args.grad_accum}")
print(f"  FLOPs:      n={args.flops_n} counted passes at bs=1"
      + (" (skipped)" if args.flops_n == 0 else "")
      + (" (universat/ramen only)"
         if args.model not in ("universat", "ramen") and args.flops_n > 0 else ""))
print(f"  GPUs:       {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

common = dict(
    root_path=args.data_dir,
    use_s1=args.use_s1,
    multi_temporal=args.multi_temporal,
    temporal_last=args.temporal_last,
    temporal_mode="sequence",
)

train_ds = PastisBaselineDataset(mode="train",      augment=True,  **common)
val_ds   = PastisBaselineDataset(mode="validation", augment=False, **common)
test_ds  = PastisBaselineDataset(mode="test",       augment=False, **common)

print(f"  Train: {len(train_ds)} patches")
print(f"  Val:   {len(val_ds)} patches")
print(f"  Test:  {len(test_ds)} patches")


# =============================================================================
# COLLATE SELECTION
# =============================================================================

if args.model in ("universat", "ramen"):
    print(f"[PASTIS-BL] {args.model}: dedicated collate embedding real "
          f"per-modality dates as '<mod>_dates' keys")
    # RAMEN never gets the S1 ratio band (no established RAMEN code for
    # it) regardless of --universat_use_s1_ratio, which is UniverSat-only.
    collate_fn = make_universat_pastis_collate(
        use_s1=args.use_s1,
        use_s1_ratio=(args.universat_use_s1_ratio
                      if args.model == "universat" else False),
    )
else:
    base_collate = make_fused_collate(args.use_s1)
    if args.use_s1:
        print("[PASTIS-BL] S2+S1 fusion: concatenating bands in collate")

    if not is_temporal_model:
        print("[PASTIS-BL] Non-temporal model: stacking T frames into channels")
        collate_fn = make_channel_stack_collate(base_collate)
    else:
        collate_fn = base_collate


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model,
    in_channels=model_in_channels,
    num_classes=NUM_CLASSES,
    args=args,
)

trainer_module = BaselineTrainer(
    model=model,
    # UniverSat/RAMEN both consume the full {"s2":..., "s2_dates":...,
    # ["s1":..., "s1_dates":...]} dict directly (expects_full_image_dict-
    # style) — "modality" is unused in that case, only shown in the
    # startup print, same as senflood's "optical+sar" placeholder.
    modality=("s2" if args.model not in ("universat", "ramen")
              else "s2+s1" if args.use_s1 else "s2"),
    temporal=is_temporal_model,
    task="pastis",
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
        run_name = f"BL_{args.xp_name}_{args.model}_{modality_str}"
        if args.model == "universat":
            run_name += (f"_{args.universat_size}"
                         f"_os{args.universat_output_stride}"
                         f"_sp{args.universat_subpatch_px}")
        wandb.init(
            name=run_name,
            project="Atomizer_PASTIS_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_PASTIS_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/pastis_baselines/"
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
        EarlyStopping(
            monitor="val_mIoU",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # universat/ramen both have structurally inert params (universat: SAR/
    # DEM channel codes, single-modality fusion attention, S1-axis block
    # at subpatch_px=1; ramen: RadarProjector polarization params when
    # --use_s1 is off) — same find_unused_parameters requirement as every
    # other RAMEN/UniverSat integration in this codebase.
    needs_unused = args.model in ("universat", "ramen") or True  # was unconditional already

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=needs_unused),
        devices=-1,
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

    print(f"\n{'='*60}")
    print(f"  Starting: {args.model} — {modality_str}")
    print(f"  Temporal: {temporal_str}")
    print(f"  Train: folds 1,2,3 → Val: fold 4 → Test: fold 5")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path

    # ─────────────────────────────────────────────────────────────────────
    # Destroy DDP process group BEFORE the test trainer is built.
    # Rank 1 exits cleanly here; only rank 0 proceeds to test.
    # ─────────────────────────────────────────────────────────────────────
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
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing checkpoint:")
    print(f"  {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(trainer_module, test_loader, ckpt_path=best_ckpt)


# =============================================================================
# GFLOPs (universat/ramen only for now; rank-zero, after best-checkpoint load)
# =============================================================================
# Measured at bs=1 via a DEDICATED DataLoader, never the args.batch_size
# `test_loader` above -- see module docstring for the bug this fixes (an
# earlier version forwarded full args.batch_size batches through the FLOPs
# harness while still labeling the result "bs=1").

if (args.model in ("universat", "ramen") and args.flops_n > 0
        and os.environ.get("LOCAL_RANK", "0") == "0"):
    print(f"\n{'='*60}")
    print(f"  GFLOPs measurement — {args.model}")
    print(f"{'='*60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = trainer_module.model.to(device).eval()

    # Dedicated bs=1 loader for FLOPs -- reuses the SAME collate_fn (so the
    # per-sample dict shape the model expects is unchanged), just at
    # batch_size=1 instead of args.batch_size, matching every other GFLOPs
    # harness in this codebase (script_universat_sweep_senflood.py /
    # _burnscars.py's make_loader, the xView2 baseline script's
    # _build_tile_batch, script_train_xview.py's measure_test_gflops).
    flops_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    flops_raw = []
    for b in flops_loader:
        flops_raw.append(_to_device(b, device))
        if len(flops_raw) >= args.flops_n + 1:  # +1 warmup
            break

    if not flops_raw:
        print("[FLOPs] No test batches available; skipping GFLOPs measurement.")
    else:
        if len(flops_raw) < args.flops_n + 1:
            print(f"[FLOPs] WARNING: only got {len(flops_raw)} bs=1 batch(es), "
                  f"needed {args.flops_n + 1} (1 warmup + {args.flops_n} "
                  f"counted). Measuring with what's available.")

        def fwd(b, m=eval_model):
            return m(b["image"])

        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        n_measured = max(0, len(flops_raw) - 1)
        print(f"  GFLOPs/forward (bs=1, full {NATIVE_TILE_PX}x{NATIVE_TILE_PX} "
              f"dense forward): {gflops:.2f}  "
              f"(mean of {n_measured} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops, "test_gflops_bs": 1})

    del flops_raw
    if device == "cuda":
        torch.cuda.empty_cache()

if wandb_logger:
    import wandb
    wandb.finish()
