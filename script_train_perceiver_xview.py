"""
xView2 Perceiver-IO Single-Task Training Script
==================================================

Trains the multi-temporal Perceiver-IO baseline on xView2 — 5-class
post-disaster damage segmentation with bi-temporal RGB inputs.

Token layout (matches PerceiverSeg.forward):
    Per-token feature = [reflectance(C=3), Fourier(x, y), Fourier(DOY)]
    Input tokens     = [B, 2*H*W, C + pos_dim + time_dim]
    Output queries   = [B, H*W, query_dim]    # time-agnostic

Synthetic DOY scheme:
    xView2 has no calendar dates — pre/post is positional, not temporal.
    The collate synthesizes ordinal day-of-year values:
        pre  -> doy = 1   (Jan 1, normalized to ~-1.0)
        post -> doy = 183 (Jul 2, normalized to ~0.0)
    Half-year separation in Fourier-space lets the time encoder produce
    maximally distinguishable embeddings for the two frames. Day 1 (not 0)
    avoids collision with the padding sentinel used by PASTIS in the
    multi-task setup.

Train/val protocol:
    - PANGAEA splits (90/10 stratified train/val from train+tier3)
    - Random building-biased 512x512 crops (train), same oversampling for
      damage classes as the baseline script.

Test protocol (CHANGED — now matches script_train_xview_baselines.py):
    Test images are loaded FULL RESOLUTION (1024x1024, full_image=True,
    batch_size=1) and evaluated via sliding-window tiling into four
    512x512 tiles, stitched back into a full-image prediction before
    mIoU is computed — same evaluate_sliding_window() harness the
    ResNet/ViT/RAMEN/UniverSat baselines use, so results are directly
    comparable across models. (An earlier version of this script tested
    on a single deterministic 512x512 building-biased crop per image,
    which both scored on ~1/4 of each image's area and systematically
    over-represented damage classes relative to the true distribution —
    not comparable to the other baselines' full-image numbers. Fixed
    here.)

    PerceiverSeg needs no adapter for this: img_size is not baked into
    any fixed-size weights (positional encoding is computed fresh from
    the actual input H, W on every forward call), so the same checkpoint
    trained on 512x512 crops runs directly on 512x512 sliding-window
    tiles.

Examples:
    python script_train_xview_perceiver.py --xp_name perceiver_xview \
        --batch_size 2 --grad_accum 2 --lr 1e-4 --epochs 80

    # Test-only mode
    python script_train_xview_perceiver.py --xp_name perceiver_xview \
        --test_only ./checkpoints/xview_perceiver/bl_perceiver_xview-best.ckpt
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_xview_baseline import (
    XView2BaselineDataset,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CHANNELS = 3                # RGB (BGR ordering, kept channels-first)
MODALITY_KEY = "s2"             # baseline convention; collate keys output here
NUM_FRAMES   = 2                # pre + post

# Synthetic ordinal DOY for the two frames.
# pre  = day  1  -> Fourier-normalized to ~-1.0
# post = day 183 -> Fourier-normalized to ~ 0.0  (half-year separation)
PRE_DOY  = 1
POST_DOY = 183


# =============================================================================
# COLLATE (train/val — fixed 512x512 crops)
# =============================================================================

def xview_collate(batch):
    """
    Collate xView2 samples and synthesize per-sample DOY.

    The dataset returns:
        {"image": [T=2, C=3, H, W], "target": [H, W], "metadata": {...}}

    Output (to BaselineTrainer with temporal=True, modality='s2'):
        {
            "image":    {"s2": [B, 2, 3, H, W]},
            "dates":    {"s2": [B, 2]},          # ordinal [PRE_DOY, POST_DOY]
            "target":   [B, H, W],
            "metadata": [list],
        }
    """
    images   = torch.stack([s["image"]  for s in batch], dim=0)   # [B, 2, 3, H, W]
    targets  = torch.stack([s["target"] for s in batch], dim=0)
    metadata = [s["metadata"] for s in batch]

    B = images.shape[0]
    dates = torch.tensor([[PRE_DOY, POST_DOY]] * B, dtype=torch.long)

    return {
        "image":    {MODALITY_KEY: images},
        "dates":    {MODALITY_KEY: dates},
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# COLLATE (test — full 1024x1024 images, batch_size=1)
# =============================================================================

def xview_test_collate(batch):
    """
    Same as xview_collate, but for full_image=True samples: images arrive
    as [T=2, C=3, 1024, 1024]. Kept as a separate function (rather than
    reusing xview_collate) so the distinction between train/val's fixed
    512x512 crops and test's full-resolution sliding-window input is
    explicit at the call site, matching evaluate_sliding_window()'s
    expectation of {"image": {"s2": [1, T, C, H, W]}, "target": [1, H, W]}
    per batch (test_loader is batch_size=1 — see DATALOADERS section).
    """
    images   = torch.stack([s["image"]  for s in batch], dim=0)   # [B, 2, 3, 1024, 1024]
    targets  = torch.stack([s["target"] for s in batch], dim=0)
    metadata = [s["metadata"] for s in batch]

    B = images.shape[0]
    dates = torch.tensor([[PRE_DOY, POST_DOY]] * B, dtype=torch.long)

    return {
        "image":    {MODALITY_KEY: images},
        "dates":    {MODALITY_KEY: dates},
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# SLIDING-WINDOW TEST EVALUATION
# =============================================================================
# Reuses the exact tiling/stitching/metrics protocol from
# script_train_xview_baselines.py's evaluate_sliding_window(), so Perceiver
# numbers are directly comparable to the ResNet/ViT/RAMEN/UniverSat
# baselines. PerceiverSeg takes doy as a second positional-ish kwarg
# (forward(image, doy=...)), so the tile-forward closure below passes the
# synthesized [PRE_DOY, POST_DOY] pair for every tile — the same DOY
# values used at train time, since doy encodes ordinal pre/post identity,
# not spatial position, and is invariant to how the image is tiled.

def _build_tile_batch(image: torch.Tensor, tile_size: int, stride: int):
    """Split one [1, T, C, H, W] full image into its sliding-window tile
    batch [N_tiles, T, C, tile, tile] and the (top, left) coords used to
    stitch it back. Mirrors script_train_xview_baselines.py exactly."""
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


def evaluate_sliding_window_perceiver(
    model: torch.nn.Module,
    test_loader,
    num_classes: int,
    tile_size: int = 512,
    stride: int = 512,
    device: str = "cuda",
    wandb_logger=None,
    class_names=None,
    ignore_index: int = 255,
    pre_doy: int = PRE_DOY,
    post_doy: int = POST_DOY,
):
    """
    Evaluate PerceiverSeg on full-resolution xView2 test images using
    non-overlapping sliding window. Each 1024x1024 test image is tiled
    into 512x512 patches, forwarded through the model (with the
    synthesized [pre_doy, post_doy] pair) in a single batch, then
    stitched back into a full-image prediction. mIoU is computed over
    the full test set.

    Same metrics/stitching logic as script_train_xview_baselines.py's
    evaluate_sliding_window(), minus the GFLOPs measurement (not needed
    here; add a FlopCounterMode pass analogous to that script's if a
    GFLOPs figure is later required for this model too).

    Args:
        model:        Lightning module wrapping PerceiverSeg. We call
                      model.model(tile_batch, doy=...) to bypass
                      Lightning's step hooks and get raw logits.
        test_loader:  DataLoader returning
                      {"image": {"s2": [1, T, C, H, W]}, "target": [1, H, W]}
                      per batch (batch_size=1 — see DATALOADERS section).
        num_classes:  Total classes including background.
        tile_size:    Spatial tile size. Default 512.
        stride:       Tile stride. Default 512 (non-overlapping).
        device:       Where to run the model.
        wandb_logger: Optional WandbLogger to record results.
        class_names:  Optional list of class names for per-class logging.
        pre_doy, post_doy: Synthetic DOY pair, broadcast to every tile
                      (DOY encodes pre/post identity, not spatial
                      position, so it doesn't change across tiles).
    Returns:
        Dict with 'mIoU', 'damage_mIoU', 'IoU_per_class', 'mean_acc' /
        'macro_acc', 'overall_acc'.
    """
    from torchmetrics.classification import (
        MulticlassJaccardIndex, MulticlassAccuracy,
    )
    from tqdm import tqdm

    model = model.to(device).eval()

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
          f"({tile_size}x{tile_size} tiles, stride={stride})...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            image  = batch["image"][MODALITY_KEY].to(device)   # [1, T, C, H, W]
            target = batch["target"].to(device)                # [1, H, W]

            B, T, C, H, W = image.shape
            assert B == 1, f"Test loader must have batch_size=1, got {B}"

            tile_batch, coords = _build_tile_batch(image, tile_size, stride)
            n_tiles = tile_batch.shape[0]

            doy_tile = torch.tensor(
                [[pre_doy, post_doy]] * n_tiles, dtype=torch.long, device=device,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_tiles = model.model(tile_batch, doy=doy_tile)  # [N_tiles, num_classes, t, t]
            logits_tiles = logits_tiles.float()

            if logits_tiles.shape[-1] != tile_size:
                logits_tiles = torch.nn.functional.interpolate(
                    logits_tiles, size=(tile_size, tile_size),
                    mode="bilinear", align_corners=False,
                )

            full_logits = torch.zeros((1, num_classes, H, W), device=device)
            count = torch.zeros((1, 1, H, W), device=device)
            for i, (top, left) in enumerate(coords):
                full_logits[:, :, top:top+tile_size, left:left+tile_size] += logits_tiles[i:i+1]
                count[:, :, top:top+tile_size, left:left+tile_size] += 1.0
            full_logits = full_logits / count.clamp(min=1.0)

            preds = full_logits.argmax(dim=1)  # [1, H, W]

            iou_metric.update(preds, target)
            macro_acc_metric.update(preds, target)
            overall_acc_metric.update(preds, target)

    iou_per_class = iou_metric.compute().cpu().tolist()
    mean_iou = sum(iou_per_class) / len(iou_per_class)
    macro_acc = macro_acc_metric.compute().item()
    overall_acc = overall_acc_metric.compute().item()

    damage_iou = iou_per_class[1:]
    damage_mIoU = sum(damage_iou) / len(damage_iou) if damage_iou else 0.0

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"\n{'='*60}")
    print(f"  SLIDING-WINDOW TEST RESULTS — Perceiver-IO")
    print(f"{'='*60}")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"  Macro accuracy:     {macro_acc:.4f}")
    print(f"  mIoU (5-class):     {mean_iou:.4f}")
    print(f"  mIoU (damage):      {damage_mIoU:.4f}  (excludes background)")
    print(f"  Per-class IoU:")
    for name, iou in zip(class_names, iou_per_class):
        print(f"    {name:>12s}: {iou:.4f}")
    print(f"{'='*60}\n")

    if wandb_logger is not None:
        try:
            import wandb
            wandb.log({
                "test_mIoU":        mean_iou,
                "test_damage_mIoU": damage_mIoU,
                "test_macro_acc":   macro_acc,
                "test_overall_acc": overall_acc,
                **{f"test_IoU_{name}": iou
                   for name, iou in zip(class_names, iou_per_class)},
            })
        except Exception as e:
            print(f"[sliding-window] wandb log failed: {e}")

    return {
        "mIoU":          mean_iou,
        "damage_mIoU":   damage_mIoU,
        "IoU_per_class": iou_per_class,
        "macro_acc":     macro_acc,
        "overall_acc":   overall_acc,
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="xView2 Perceiver-IO Baseline")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--data_dir",  type=str, default="./data/xview")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=2)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=2)

# Crop / image size — TRAINING is fixed at 512x512 tiles (native xView2 is
# 1024x1024; 512^2 * T=2 = 524k tokens/sample, already large — 1024x1024
# would be ~2M tokens and is not used for training). TEST always runs at
# full 1024x1024 resolution via sliding-window tiling, independent of
# this value (see DATASETS / evaluate_sliding_window_perceiver).
parser.add_argument("--crop_size", type=int, default=512,
                    help="Training crop size (random, building-biased) and "
                         "validation crop size (deterministic, building-"
                         "biased center crop). Does NOT affect test, which "
                         "always uses full 1024x1024 images tiled into "
                         "512x512 sliding-window tiles.")
parser.add_argument("--img_size", type=int, default=512,
                    help="Spatial size for token construction at train/val "
                         "time. Must match crop_size. (PerceiverSeg computes "
                         "its positional encoding fresh from the actual "
                         "input H, W on every forward call — this value is "
                         "not baked into any fixed-size weights, which is "
                         "also why test can run at a different tile "
                         "geometry, i.e. sliding-window 512x512 tiles over "
                         "a 1024x1024 image, without any adapter.)")

# Sliding-window test tiling (independent of --crop_size / --img_size)
parser.add_argument("--test_tile_size", type=int, default=512,
                    help="Tile size for sliding-window test evaluation on "
                         "full 1024x1024 images.")
parser.add_argument("--test_stride", type=int, default=512,
                    help="Tile stride for sliding-window test evaluation "
                         "(default 512 = non-overlapping, 2x2 tiles).")

# Disable damage-class oversampling during training (matches baseline default behavior).
parser.add_argument("--no_oversample", action="store_true",
                    help="Disable building-damage oversampling at train time.")

# Perceiver-IO config (matches the PASTIS/MADOS/BurnScars runs for parameter parity)
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=512)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=8)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=6)
parser.add_argument("--no_weight_tie",      action="store_true",
                    help="Disable weight-tying across encoder blocks.")
parser.add_argument("--num_freq_bands",     type=int, default=16)
parser.add_argument("--max_freq",           type=float, default=16.0)
parser.add_argument("--attn_dropout",       type=float, default=0.0)
parser.add_argument("--ff_dropout",         type=float, default=0.0)

args = parser.parse_args()


# =============================================================================
# SANITY
# =============================================================================

if args.crop_size != args.img_size:
    raise ValueError(
        f"--crop_size ({args.crop_size}) must equal --img_size ({args.img_size}) "
        f"for train/val. (This constraint does not apply to test, which "
        f"always runs sliding-window at --test_tile_size over full "
        f"1024x1024 images — PerceiverSeg's positional encoding is "
        f"recomputed per forward call from the actual tile size.)"
    )


# =============================================================================
# DATASETS
# =============================================================================

train_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
    oversample_building_damage=(not args.no_oversample),
)
val_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
    oversample_building_damage=False,
)
# Test: FULL 1024x1024 images (full_image=True), evaluated via sliding-
# window tiling in evaluate_sliding_window_perceiver — matches
# script_train_xview_baselines.py's test protocol so results are
# comparable across all baselines + Perceiver-IO.
test_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size,        # informational; full_image overrides
    augment=False,
    oversample_building_damage=False,
    full_image=True,
)


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  xView2 Perceiver-IO Baseline")
print(f"  Channels:     {NUM_CHANNELS} (RGB, BGR-ordered)")
print(f"  Frames:       T={NUM_FRAMES} (pre + post)")
print(f"  Train tiles:  {args.img_size}x{args.img_size} (random, building-biased crop)")
print(f"  Train tokens: {NUM_FRAMES * args.img_size ** 2:,} per sample "
      f"(2 frames x {args.img_size}^2)")
print(f"  Test tiles:   {args.test_tile_size}x{args.test_tile_size} "
      f"sliding-window over full 1024x1024, stride={args.test_stride}")
print(f"  Queries:      time-agnostic (last-frame tokens only)")
print(f"  Synthetic DOY: pre={PRE_DOY}, post={POST_DOY} "
      f"(half-year separation)")
print(f"  Latents:      {args.num_latents} x {args.latent_dim}")
print(f"  Depth:        {args.depth}")
print(f"  Classes:      {NUM_CLASSES}")
print(f"  Ignore index: {IGNORE_INDEX}")
print(f"  Epochs:       {args.epochs}")
print(f"  Batch size:   {args.batch_size}")
print(f"  Grad accum:   {args.grad_accum}")
print(f"  LR:           {args.lr}")
print(f"  GPUs:         {torch.cuda.device_count()}")
print(f"{'='*60}\n")

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples (full-image, sliding-window)")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=xview_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, **loader_kwargs)

# Test loader: batch_size=1, full 1024x1024xT=2 samples, own collate (no
# crop bookkeeping needed — full_image=True already returns uncropped
# images). Sliding-window tiling happens inside
# evaluate_sliding_window_perceiver, same as the baseline script.
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers,
    collate_fn=xview_test_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)


# =============================================================================
# MODEL
# =============================================================================

model = PerceiverSeg(
    in_channels=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    img_size=args.img_size,
    num_latents=args.num_latents,
    latent_dim=args.latent_dim,
    depth=args.depth,
    cross_heads=args.cross_heads,
    latent_heads=args.latent_heads,
    cross_dim_head=args.cross_dim_head,
    latent_dim_head=args.latent_dim_head,
    self_per_cross_attn=args.self_per_cross_attn,
    weight_tie_layers=(not args.no_weight_tie),
    num_freq_bands=args.num_freq_bands,
    max_freq=args.max_freq,
    attn_dropout=args.attn_dropout,
    ff_dropout=args.ff_dropout,
)


# =============================================================================
# TRAINER MODULE
# =============================================================================

# temporal=True so BaselineTrainer reads batch["dates"]["s2"] and passes
# it to model.forward(image, doy=...). PerceiverSeg.forward extracts the
# DOY into per-frame Fourier features; the synthesized [1, 183] gives
# the model a clean half-year separation between pre and post.
trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=True,
    task="xview2",
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
        run_name = f"BL_{args.xp_name}_perceiver"
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

ckpt_dir = "./checkpoints/xview_perceiver/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_perceiver-{{epoch:02d}}-{{val_mIoU:.4f}}",
            monitor="val_mIoU",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_perceiver-last",
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

    # find_unused_parameters=True: PerceiverSeg has params that don't
    # always enter the backward graph on a given step -- most notably
    # no_time_vector, which is only used when doy is None, but xView2's
    # collate always supplies a synthesized doy, so that parameter is
    # permanently unused here. weight_tie_layers=True (the default) can
    # also confuse DDP's bucket-rebuild since the same module is reused
    # across depth. Same fix the RAMEN/UniverSat integrations in the
    # PASTIS/xView2 baseline scripts apply for analogous reasons.
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        # FIXED: was hardcoded to 2, silently ignoring --grad_accum for
        # any other value.
        accumulate_grad_batches=args.grad_accum,
    )

    print(f"\n{'='*60}")
    print(f"  Starting: perceiver on xView2")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

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
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SLIDING-WINDOW TEST (mIoU, full 1024x1024 images)
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

# Load checkpoint into the trainer module directly (single GPU, no
# distributed test trainer needed — mirrors how the sliding-window eval
# is invoked in script_train_xview_baselines.py).
ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = trainer_module.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys.")
if missing:
    print(f"[load_state_dict] {len(missing)} missing keys (likely OK if these "
          f"are runtime caches): {missing[:5]}{'...' if len(missing) > 5 else ''}")

class_names = ["BG", "NoDamage", "Minor", "Major", "Destroyed"]

results = evaluate_sliding_window_perceiver(
    model=trainer_module,
    test_loader=test_loader,
    num_classes=NUM_CLASSES,
    tile_size=args.test_tile_size,
    stride=args.test_stride,
    device="cuda",
    wandb_logger=wandb_logger,
    class_names=class_names,
    ignore_index=IGNORE_INDEX,
    pre_doy=PRE_DOY,
    post_doy=POST_DOY,
)

print(f"[xView2-Perceiver] Test mIoU:        {results['mIoU']:.4f}")
print(f"[xView2-Perceiver] Test damage mIoU: {results['damage_mIoU']:.4f}")

if wandb_logger:
    import wandb
    wandb.finish()
