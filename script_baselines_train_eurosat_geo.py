"""
EuroSAT Baseline Training Script (Classification)
=====================================================

Train image-classification baselines on geo-bench m-eurosat (10-class
land-cover classification, single-frame S2, 13 optical bands incl. B10).

Unlike the segmentation baseline scripts (Sen1Floods11/Cashew/MADOS),
this is meaningfully simpler: no UPerNet decoder, no sliding-window
inference, no per-pixel IGNORE_INDEX, no D4-on-label (labels are scalar).
Models just need to map [B, C, H, W] -> [B, num_classes] logits.

ASSUMPTION FLAGGED: this script assumes classifier-head builder functions
`build_resnet_classifier` and `ViTClassifier` exist somewhere in the
codebase (distinct from the segmentation `build_resnet_upernet` /
`ViTUPerNet` you've shown me, which include a UPerNet decoder that
classification doesn't need). The import paths below
(training.ResNet.model_resnet_classifier,
training.VIT.model_vit_classifier) are GUESSES at plausible locations,
not confirmed — fix them to match wherever these actually live, or tell
me and I'll write minimal classifier heads (e.g. backbone + global-
average-pool + linear) if they don't exist yet.

Metric: macro F1 (torchmetrics.MulticlassF1Score), via
ClassificationBaselineTrainer (training/trainer_baselines_classification.py).

Examples:
    python script_train_eurosat_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 32 --lr 1e-4 --epochs 80

    python script_train_eurosat_baselines.py --xp_name vit \
        --model vit --batch_size 32 --lr 1e-4 --epochs 80
"""

import os
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_eurosat import (
    EurosatBaselineDataset,
)
# ResNet: training/ResNet/resnet.py — plain classification ResNet
# (ends in avgpool + fc(num_classes), no UPerNet decoder needed). Variant
# functions take (num_classes, channels), NOT a builder-with-variant-arg
# pattern like build_resnet_upernet.
from training.ResNet.resnet import (
    ResNetSuperSmall, ResNetSmall, ResNet50, ResNet101, ResNet152,
)
# ViT: training/VIT/simple_vit.py — SimpleViT (2D sin-cos pos embed,
# mean-pool, linear head). Keyword-only constructor with different arg
# names (dim/heads/mlp_dim, not embed_dim/num_heads/decoder_channels).
from training.VIT.simple_vit import SimpleViT
# RAMEN classifier: training/RAMEN/ramen_classifier.py (confirmed —
# every kwarg used below matches its real __init__ signature).
from training.RAMEN.ramen_classifier import build_ramen_classifier
# UniverSat classifier — real code, from training/Universat/universat_augmenter.py
from training.Universat.universat_augmenter import build_universat_classifier
from training.trainer_baselines_classification import ClassificationBaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = EurosatBaselineDataset.NUM_CLASSES   # 10
NUM_CHANNELS = EurosatBaselineDataset.NUM_CHANNELS  # 13
MODALITY_KEY = "s2"

# EuroSAT native patch size
NATIVE_SIZE = EurosatBaselineDataset.PATCH_SIZE  # 64
EUROSAT_GSD_M = 10.0  # single scalar used for RAMEN/UniverSat input_res,
                      # same simplification the other baseline scripts
                      # make for their S2 bands (which also natively vary
                      # 10/20/60m).


# =============================================================================
# RAMEN / UniverSat band metadata — optical only, single "optical"
# modality (EuroSAT has no SAR), in EurosatBaselineDataset.BAND_PREFIXES
# order. INCLUDES B10 (cirrus) — unlike Cashew, which dropped it; EuroSAT
# keeps the full standard 13-band Sentinel-2 L1C set.
# =============================================================================

S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B08A", "B09", "B10", "B11", "B12",
]

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B08A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}

RAMEN_INPUT_BANDS      = {"optical": S2_BAND_NAMES}
RAMEN_WAVELENGTHS      = {"optical": S2_WAVELENGTHS_NM}
UNIVERSAT_INPUT_BANDS  = {"optical": S2_BAND_NAMES}
UNIVERSAT_WAVELENGTHS  = {"optical": S2_WAVELENGTHS_NM}


# =============================================================================
# COLLATE
# =============================================================================

def eurosat_collate(batch):
    """Stack image tensors and scalar targets; keep metadata as a list."""
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets  = torch.stack([s["target"] for s in batch])  # [B] long
    metadata = [s["metadata"] for s in batch]

    return {"image": images, "target": targets, "metadata": metadata}


def eurosat_collate_ramen(batch):
    """
    RAMEN / UniverSat collate for EurosatBaselineDataset.

    EuroSAT has only one modality (optical, no SAR) — this just wraps
    the existing image["s2"] tensor as {"optical": ...} so it matches
    the dict interface RAMEN's classifier / UniverSatClassifier expect,
    same pattern as the Cashew/MADOS baseline scripts' *_collate_ramen.
    """
    merged = torch.stack([s["image"]["s2"] for s in batch])  # [B, 13, H, W]
    images = {"optical": merged}
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

_RESNET_VARIANTS = {
    "resnet_super_small": ResNetSuperSmall,
    "resnet_small":       ResNetSmall,
    "resnet50":           ResNet50,
    "resnet101":          ResNet101,
    "resnet152":          ResNet152,
}


def build_model(model_name: str, in_channels: int, num_classes: int, args):
    if model_name == "resnet":
        if args.resnet_variant not in _RESNET_VARIANTS:
            raise ValueError(
                f"Unknown --resnet_variant: {args.resnet_variant}. "
                f"Available: {sorted(_RESNET_VARIANTS)}"
            )
        # Real signature: fn(num_classes, channels=3) — note the arg
        # order (num_classes first), different from the *_upernet
        # builders' keyword-style calls elsewhere in this project.
        return _RESNET_VARIANTS[args.resnet_variant](num_classes, in_channels)
    elif model_name == "vit":
        # SimpleViT (training/VIT/simple_vit.py): keyword-only, uses
        # dim/heads/mlp_dim naming, no expects_full_image_dict — plain
        # [B, C, H, W] -> [B, num_classes], mean-pooled CLS-free output.
        mlp_dim = int(args.vit_embed_dim * args.vit_mlp_ratio)
        return SimpleViT(
            image_size=args.img_size,
            patch_size=args.vit_patch_size,
            num_classes=num_classes,
            dim=args.vit_embed_dim,
            depth=args.vit_depth,
            heads=args.vit_num_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
        )
    elif model_name == "ramen":
        # Confirmed against the real RAMENClassifier
        # (training/RAMEN/ramen_classifier.py) — every kwarg name below
        # matches its __init__ signature exactly.
        return build_ramen_classifier(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=num_classes,
            input_size=args.ramen_window_size,
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
        )
    elif model_name == "universat":
        # Real code (training/Universat/universat_augmenter.py). Classifier
        # never uses the sub-patch skip (see UniverSatClassifier's
        # docstring) — output_stride is irrelevant here and not exposed
        # as a CLI arg, unlike the segmentation scripts.
        return build_universat_classifier(
            input_bands=UNIVERSAT_INPUT_BANDS,
            wavelengths=UNIVERSAT_WAVELENGTHS,
            num_classes=num_classes,
            input_res={"optical": EUROSAT_GSD_M},
            patch_size_m=args.universat_patch_m,
            size=args.universat_size,
            pooling=args.universat_pooling,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'resnet', 'vit', "
            f"'ramen', 'universat'. (No 'unet' here — classification "
            f"doesn't need a decoder.)"
        )


# =============================================================================
# GFLOPs MEASUREMENT — FlopCounterMode, consistent with the segmentation
# baseline scripts (script_train_cashew_baselines.py /
# script_train_mados_baselines.py), NOT the Perceiver family's
# torch.profiler harness. See those scripts' GFLOPs section for why the
# two methods shouldn't be mixed when reporting.
# =============================================================================

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
    for b in batches[:n_warmup]:
        _ = forward_fn(b)
    if device == "cuda":
        torch.cuda.synchronize()

    flops_list = []
    for b in batches[n_warmup:]:
        fc = FlopCounterMode(display=False)
        with fc:
            _ = forward_fn(b)
        flops_list.append(fc.get_total_flops())

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="EuroSAT Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["resnet", "vit", "ramen", "universat"])
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-eurosat")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")
parser.add_argument("--resume",    type=str, default=None,
                    help="Path to a .ckpt file to resume training from "
                         "(full trainer state). Ignored if --test_only "
                         "is set.")

# Training
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

parser.add_argument("--crop_size", type=int, default=None,
                    help="Crop size (None = full 64x64 native).")
parser.add_argument("--img_size",  type=int, default=NATIVE_SIZE,
                    help="ViT positional embedding size.")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# ViT
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=8,
                    help="EuroSAT's native size (64) is much smaller than "
                         "Cashew/MADOS/Sen1Floods11 — patch_size=16 would "
                         "give only a 4x4=16-token grid. Default 8 gives "
                         "an 8x8=64-token grid instead. Must evenly "
                         "divide --img_size.")
parser.add_argument("--vit_mlp_ratio", type=float, default=4.0,
                    help="SimpleViT's mlp_dim = vit_embed_dim * this "
                         "ratio (standard transformer FFN expansion; "
                         "SimpleViT takes mlp_dim directly rather than a "
                         "ratio, so this is computed here).")

# GFLOPs
parser.add_argument("--flops", action="store_true", default=True)
parser.add_argument("--no_flops", dest="flops", action="store_false")
parser.add_argument("--flops_n", type=int, default=3)

# RAMEN
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=10.0,
                    help="Native GSD (m/px) of the input imagery.")
parser.add_argument("--ramen_res",       type=float, default=10.0,
                    help="Common working resolution (m/px). Left equal "
                         "to --ramen_input_res (no resampling).")
parser.add_argument("--ramen_window_size", type=int, default=64,
                    help="EuroSAT's native size (64) is small enough "
                         "that RAMEN can likely run at full resolution "
                         "directly, unlike the segmentation scripts' "
                         "windowed-training + sliding-window-eval split "
                         "(no sliding window needed here — classification "
                         "produces one label per image, not a per-pixel "
                         "map to stitch).")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"])
parser.add_argument("--universat_patch_m", type=float, default=40.0,
                    help="Patch size in METRES. 20 m = 2 px at EuroSAT's "
                         "10 m GSD, giving a 32x32 trunk grid on the "
                         "64x64 image. Must be an integer number of "
                         "pixels, and evenly divide the input side.")
parser.add_argument("--universat_pooling", type=str, default="max",
                    choices=["mean", "max", "cls"],
                    help="How UniverSatClassifier pools patch tokens "
                         "before the LN+Linear head. 'mean' matches the "
                         "repo's own LP_eval.py protocol (default here).")

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")


# =============================================================================
# SANITY CHECK FOR VIT
# =============================================================================

if args.model == "vit":
    eff_size = args.crop_size if args.crop_size is not None else NATIVE_SIZE
    if eff_size != args.img_size:
        raise ValueError(
            f"For ViT: input size ({eff_size}) must equal --img_size "
            f"({args.img_size})."
        )
    if args.img_size % args.vit_patch_size != 0:
        raise ValueError(
            f"--img_size ({args.img_size}) must be divisible by "
            f"--vit_patch_size ({args.vit_patch_size})."
        )

if args.model == "universat":
    eff_size = args.crop_size if args.crop_size is not None else NATIVE_SIZE
    universat_patch_px = args.universat_patch_m / EUROSAT_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {EUROSAT_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px)."
        )
    universat_patch_px = int(round(universat_patch_px))
    if eff_size % universat_patch_px:
        raise ValueError(
            f"Input side ({eff_size}) not divisible by "
            f"--universat_patch_m's pixel size ({universat_patch_px} px "
            f"@ {args.universat_patch_m} m). EuroSAT's native 64x64 "
            f"divides cleanly by patch sizes of 10/20/40/80m at 10m GSD "
            f"(1/2/4/8 px)."
        )


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  EuroSAT Baseline Training (classification)")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
print(f"  Channels:    {NUM_CHANNELS} (S2, 13 bands incl. B10)")
print(f"  Classes:     {NUM_CLASSES}")
crop_str = f"{args.crop_size}x{args.crop_size}" if args.crop_size else f"{NATIVE_SIZE}x{NATIVE_SIZE} (full)"
print(f"  Input size:  {crop_str}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  GPUs:        {torch.cuda.device_count()}")
if args.resume is not None:
    print(f"  Resuming from: {args.resume}")
if args.test_only is not None:
    print(f"  TEST-ONLY mode, loading: {args.test_only}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = EurosatBaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = EurosatBaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
)
test_ds = EurosatBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

collate_fn = (eurosat_collate_ramen if args.model in ("ramen", "universat")
              else eurosat_collate)

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

if args.test_only is not None:
    model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)
    trainer_module = ClassificationBaselineTrainer.load_from_checkpoint(
        args.test_only, strict=False, model=model,
        modality=MODALITY_KEY if args.model not in ("ramen", "universat")
                 else "optical",
        num_classes=NUM_CLASSES,
    )
    trainer_module.eval()
else:
    model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)
    trainer_module = ClassificationBaselineTrainer(
        model=model,
        modality=MODALITY_KEY if args.model not in ("ramen", "universat")
                 else "optical",
        num_classes=NUM_CLASSES,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and args.test_only is None:
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_{args.model}"
        if args.model == "resnet":
            run_name += f"_{args.resnet_variant}"
        wandb.init(
            name=run_name,
            project="Atomizer_EuroSAT_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_EuroSAT_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/eurosat_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_macro_f1:.4f}}",
            monitor="val_macro_f1",
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
            monitor="val_macro_f1",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

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
        accumulate_grad_batches=args.grad_accum,
    )

    print(f"\n{'='*60}")
    print(f"  Starting: {args.model} on EuroSAT")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=args.resume)

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
# GFLOPs (rank-zero only)
# =============================================================================

if args.flops and os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"\n{'='*60}")
    print(f"  GFLOPs measurement — {args.model}")
    print(f"{'='*60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = trainer_module.model.to(device).eval()

    flops_raw = []
    for b in test_loader:
        flops_raw.append(_to_device(b, device))
        if len(flops_raw) >= args.flops_n + 1:
            break

    if not flops_raw:
        print("[FLOPs] No test batches available; skipping GFLOPs measurement.")
    else:
        if args.model in ("ramen", "universat"):
            def fwd(b, m=eval_model):
                return m(b["image"])
        else:
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])

        eff_size = args.crop_size if args.crop_size is not None else NATIVE_SIZE
        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        print(f"  GFLOPs/forward (bs=1, {eff_size}x{eff_size}): {gflops:.4f}"
              f"  (mean of {len(flops_raw) - 1} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops})

if wandb_logger:
    import wandb
    wandb.finish()
