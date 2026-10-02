"""
ForestNet Baseline Training Script
====================================

Train classification baselines on geo-bench m-forestnet (12-class
deforestation driver classification, single-frame Landsat-style imagery).

Single-temporal classification — uses ClassificationBaselineTrainer (CE loss,
top-1/top-5 accuracy, macro-F1).

Supported models:
  - resnet : ResNet (avgpool + fc head). Variant via --resnet_variant.
  - vit    : ViT (CLS token + linear head).
  - ramen  : RAMENClassifier — multi-modal spectral tokenization + CLS
             token classification head. ForestNet has a single modality
             (Landsat optical, no SAR), so unlike the EuroSAT-SAR script's
             adapter (which splits a fused tensor into optical + sar),
             RAMENInputAdapter here just renames image["landsat"] into
             RAMEN's {"optical": ...} input dict — see RAMENInputAdapter
             below.
  - universat : UniverSat encoder (gastruc/UniverSat, AnySat v2) trained
             FROM SCRATCH, via UniverSatClassifier in
             training.Universat.universat_augmenter — the repo's OWN
             classification protocol (mirrors src/LP_eval.py): encoder
             tokens -> strip the prepended registers -> mean/max-pool the
             remaining patch tokens -> LayerNorm + Linear head. The
             alternative register-as-CLS convention (pooling="cls") is
             also exposed but is EXPLICITLY the stale/non-maintained path
             per that class's docstring (mean-pool is the maintained
             convention and this script's default).

             Classification never exercises the CA_Sub sub-patch skip:
             output_grid is set to the trunk's own latent grid (identity
             Bilinear_out resample), so --universat_output_stride is
             functionally irrelevant to the classifier's forward pass —
             this script always passes output_stride=1 when building it
             (divides every crop size cleanly, sidesteps a spurious
             divisibility check UniverSatClassifier inherits from
             UniverSatSegmenter's _grids() but doesn't actually need for
             classification).

             Unlike RAMEN, no rename adapter is needed: input_bands /
             wavelengths / input_res are keyed directly to "landsat",
             matching the dataset's own modality key (same convention as
             the BurnScars segmentation integration's "hls" key, not
             RAMEN's rename-to-"optical" pattern here).

Bands: 6 Landsat (Blue, Green, Red, NIR, SWIR1, SWIR2) at 15 m/px.
Input size: center-cropped 320×320 from native 332×332 (divisible by 16
            for ViT, retains 92% of the area).

Examples:
    # ResNet50
    python script_train_forestnet_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 32 --lr 1e-4 --epochs 80

    # ViT-S
    python script_train_forestnet_baselines.py --xp_name vit \
        --model vit \
        --batch_size 16 --lr 1e-4 --epochs 80

    # RAMEN
    python script_train_forestnet_baselines.py --xp_name ramen \
        --model ramen \
        --batch_size 32 --lr 1e-4 --epochs 80

    # UniverSat-S from scratch, mean-pooled patch tokens (default,
    # matches the repo's own LP_eval.py protocol)
    python script_train_forestnet_baselines.py --xp_name universat \
        --model universat \
        --batch_size 32 --lr 1e-4 --epochs 80

    # UniverSat, max-pooling instead of mean
    python script_train_forestnet_baselines.py --xp_name universat_max \
        --model universat --universat_pooling max \
        --batch_size 32 --lr 1e-4 --epochs 80
"""

import os
import argparse
from collections import Counter

import torch
import torch.nn as nn
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

from training.utils.datasets_baselines.utils_dataset_forestnet_baselines import (
    ForestNetBaselineDataset,
)
from training.VIT.model_vit_upernet import ViTClassifier
from training.ResNet.model_resnet_upernet import build_resnet_classifier
from training.RAMEN.ramen_classifier import build_ramen_classifier  # adjust import path
from training.Universat.universat_augmenter import build_universat_classifier
from training.trainer_baselines_classification import (
    ClassificationBaselineTrainer,
)


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = ForestNetBaselineDataset.NUM_CLASSES   # 12
NUM_CHANNELS = ForestNetBaselineDataset.NUM_CHANNELS  # 6
MODALITY_KEY = "landsat"
FORESTNET_GSD_M = 15.0   # Landsat native GSD — shared by RAMEN
                         # (--ramen_input_res default) and UniverSat
                         # (patch_px = patch_m / GSD)


# =============================================================================
# BAND METADATA — shared by RAMEN and UniverSat
# =============================================================================
# Central wavelengths (nm) for the 6 Landsat-style bands used by ForestNet,
# in the same order the docstring/dataset describes them: Blue, Green, Red,
# NIR, SWIR1, SWIR2. These are the standard Landsat 8 OLI band centers.
# NOTE: unlike the EuroSAT-SAR script, ForestNetBaselineDataset doesn't
# expose a NAME_TO_S2CODE-style mapping to derive these from, so they are
# hardcoded here — adjust if the underlying imagery uses different band
# passes (e.g. Landsat 7 ETM+ instead of Landsat 8 OLI).
FORESTNET_BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

RAMEN_BAND_WAVELENGTHS_NM = {
    "Blue":  482.0,
    "Green": 561.5,
    "Red":   654.5,
    "NIR":   865.0,
    "SWIR1": 1608.5,
    "SWIR2": 2200.5,
}

RAMEN_INPUT_BANDS = {
    "optical": FORESTNET_BAND_NAMES,
}
RAMEN_WAVELENGTHS = {
    "optical": RAMEN_BAND_WAVELENGTHS_NM,
}

# UniverSat: keyed to the dataset's own modality name ("landsat") directly
# — no rename adapter needed (contrast with RAMEN's "optical" rename
# above). Wavelengths reused verbatim; UniverSatClassifier converts
# nm -> µm internally.
UNIVERSAT_INPUT_BANDS = {
    MODALITY_KEY: FORESTNET_BAND_NAMES,
}
UNIVERSAT_WAVELENGTHS = {
    MODALITY_KEY: RAMEN_BAND_WAVELENGTHS_NM,
}


# =============================================================================
# RAMEN INPUT ADAPTER
# =============================================================================

class RAMENInputAdapter(nn.Module):
    """
    Wraps the dataset's single-sensor image["landsat"] : [B,6,H,W] tensor
    into RAMEN's expected {"optical": [B,6,H,W]} dict.

    ForestNet has no SAR modality, so — in contrast to the EuroSAT-SAR
    script's adapter, which splits a fused tensor into {"optical","sar"} —
    this is a straight rename/passthrough, not a split.
    """
    expects_full_image_dict = True

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: dict, **kwargs):
        image = x[MODALITY_KEY]  # [B, 6, H, W]
        return self.model({"optical": image}, **kwargs)


# =============================================================================
# COLLATE — stack images, stack scalar labels
# =============================================================================

def forestnet_collate(batch):
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets = torch.tensor([s["target"] for s in batch], dtype=torch.long)
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "target": targets,
        "metadata": metadata,
    }


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name: str, in_channels: int, num_classes: int, args):
    if model_name == "resnet":
        return build_resnet_classifier(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    elif model_name == "vit":
        return ViTClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            dropout=args.dropout,
        )
    elif model_name == "ramen":
        base = build_ramen_classifier(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=num_classes,
            input_size=args.img_size,  # must equal --crop_size (see sanity check)
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
            dropout=args.dropout,
        )
        return RAMENInputAdapter(base)
    elif model_name == "universat":
        # Random init, no pretrained weights. No rename adapter needed —
        # UNIVERSAT_INPUT_BANDS is keyed to "landsat" directly, matching
        # the dataset's own image dict key. output_stride is always 1
        # here: UniverSatClassifier's forward() ignores it functionally
        # (output_grid is set to the trunk's own latent grid), but the
        # inherited _grids() still divisibility-checks H % output_stride,
        # so 1 (which divides everything) sidesteps a spurious error for
        # a parameter that has no effect on classification. --dropout
        # applies to build_ramen_classifier/ViTClassifier's own dropout
        # arg elsewhere in this file, NOT to UniverSatClassifier, which
        # has no dropout knob in its LayerNorm+Linear head — silently
        # ignored here, not wired through.
        return build_universat_classifier(
            input_bands=UNIVERSAT_INPUT_BANDS,
            wavelengths=UNIVERSAT_WAVELENGTHS,
            num_classes=num_classes,
            input_res={MODALITY_KEY: FORESTNET_GSD_M},
            patch_size_m=args.universat_patch_m,
            output_stride=1,
            size=args.universat_size,
            subpatch_px=args.universat_subpatch_px,
            pooling=args.universat_pooling,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'resnet', 'vit', "
            f"'ramen', 'universat'"
        )


# =============================================================================
# CLASS WEIGHTS (for imbalanced training)
# =============================================================================

def compute_class_weights(dataset, num_classes):
    """Inverse-frequency class weights, normalized so weights sum to num_classes."""
    counts = Counter(dataset.name_to_label[n] for n in dataset.sample_names)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)
    weights = weights / weights.sum() * num_classes
    return weights


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="ForestNet Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["resnet", "vit", "ramen", "universat"])
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-forestnet")
parser.add_argument("--resume",    type=str, default=None,
                    help="Path to a .ckpt file to resume training from "
                         "(full trainer state: model, optimizer, LR "
                         "scheduler, epoch/step counters, callback state "
                         "e.g. EarlyStopping/ModelCheckpoint's best-score "
                         "tracking). Passed to Trainer.fit(ckpt_path=...). "
                         "If omitted, training starts from scratch. Same "
                         "convention as the Sen1Floods11/PASTIS scripts' "
                         "--resume flag.")
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training entirely and "
                         "test directly with this checkpoint's weights. "
                         "You must still pass the same --model / "
                         "--resnet_variant / --vit_* / --ramen_* / "
                         "--universat_* flags used when the checkpoint was "
                         "trained, since the architecture is rebuilt from "
                         "these flags, not from the checkpoint. Mutually "
                         "exclusive with --resume.")

# Training
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)
parser.add_argument("--dropout",      type=float, default=0.0)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--use_class_weights", action="store_true",
                    help="Use inverse-frequency class weighting in CE loss")

# Spatial
parser.add_argument("--crop_size", type=int, default=320,
                    help="Center-crop size (native 332).")
parser.add_argument("--img_size",  type=int, default=320,
                    help="ViT/RAMEN positional embedding size; must equal --crop_size.")

# ViT
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN
parser.add_argument("--ramen_embed_dim", type=int, default=384,
                    help="Matches --vit_embed_dim's default for a fair "
                         "parameter-count comparison; independent knob.")
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=6)
parser.add_argument("--ramen_input_res", type=float, default=15.0,
                    help="Native GSD (m/px) of ForestNet Landsat imagery.")
parser.add_argument("--ramen_res",       type=float, default=60.0,
                    help="Working resolution (m/px). Default equals "
                         "--ramen_input_res (no resampling, full native "
                         "detail). Increase to trade detail for speed.")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="'small' (384-d, 12 SA blocks, heads=6) is ~36.1M "
                         "total -- parameter-matched to the ViT-S / RAMEN "
                         "~34M budget. 'tiny' ~6.2M; 'base' ~201M.")
parser.add_argument("--universat_patch_m", type=float, default=60,
                    help="Patch size in METRES. 120 m = 8 px at ForestNet's "
                         "15 m GSD. Must be an integer number of pixels, "
                         "and --crop_size must be divisible by it "
                         "(320 / 8 = 40, satisfied at the defaults).")
parser.add_argument("--universat_subpatch_px", type=int, default=4,
                    help="Sub-patch size in PIXELS for the UPE's S1-axis "
                         "sub-patch skip. Classification doesn't exercise "
                         "the CA_Sub sub-patch decode at all (see "
                         "UniverSatClassifier docstring), so this has "
                         "little effect here; kept for API consistency "
                         "with the segmentation integrations. patch_px "
                         "must be a multiple of this.")
parser.add_argument("--universat_pooling", type=str, default="mean",
                    choices=["mean", "max", "cls"],
                    help="How UniverSatClassifier turns encoder patch "
                         "tokens into a single vector for the LN+Linear "
                         "head. 'mean' (default) matches the repo's own "
                         "LP_eval.py classification protocol. 'max' is "
                         "the same class's alternate pooling. 'cls' takes "
                         "the first register token instead -- flagged in "
                         "UniverSatClassifier's own docstring as the "
                         "STALE, non-maintained convention (the reference "
                         "implementation that used it mishandles the "
                         "encoder's (tokens, out) return); avoid unless "
                         "you have a specific reason to reproduce it.")

args = parser.parse_args()

if args.resume is not None and args.test_only is not None:
    raise ValueError(
        "--resume and --test_only are mutually exclusive: --test_only "
        "skips training entirely (loads weights, runs test), --resume "
        "continues training from a checkpoint. Pick one."
    )

if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

if args.test_only is not None and not os.path.isfile(args.test_only):
    raise FileNotFoundError(f"--test_only checkpoint not found: {args.test_only}")


# =============================================================================
# SANITY CHECKS
# =============================================================================

if args.model in ("vit", "ramen") and args.crop_size != args.img_size:
    raise ValueError(
        f"For ViT/RAMEN: --crop_size ({args.crop_size}) must equal "
        f"--img_size ({args.img_size}). Default: both 320."
    )

if args.model == "universat":
    universat_patch_px = args.universat_patch_m / FORESTNET_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {FORESTNET_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of "
            f"{FORESTNET_GSD_M}."
        )
    universat_patch_px = int(round(universat_patch_px))

    if universat_patch_px % args.universat_subpatch_px:
        raise ValueError(
            f"patch ({universat_patch_px} px) must be a multiple of "
            f"--universat_subpatch_px ({args.universat_subpatch_px})."
        )

    if args.crop_size % universat_patch_px:
        raise ValueError(
            f"--crop_size ({args.crop_size}) must be divisible by the "
            f"patch size in px ({universat_patch_px}, from "
            f"--universat_patch_m={args.universat_patch_m} @ "
            f"{FORESTNET_GSD_M} m GSD)."
        )
    # output_stride is always passed as 1 in build_model (see comment
    # there) — it's functionally inert for classification, and 1 divides
    # every --crop_size, so no separate divisibility check is needed here.

    if args.universat_pooling == "cls":
        print(f"[WARNING] --universat_pooling cls uses "
              f"UniverSatClassifier's explicitly-flagged STALE convention "
              f"(see that class's docstring) -- prefer 'mean' (default) "
              f"unless you specifically need to reproduce the old "
              f"register-as-CLS behavior.")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  ForestNet Baseline Classification")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Embed dim:   {args.ramen_embed_dim}, depth={args.ramen_depth}, "
          f"heads={args.ramen_num_heads}")
    print(f"  Resolution:  input_res={args.ramen_input_res}, res={args.ramen_res} "
          f"({'no resampling' if args.ramen_input_res == args.ramen_res else 'resampled'})")
if args.model == "universat":
    print(f"  UniverSat:   {args.universat_size} (from scratch, random init)")
    print(f"  Patch:       {args.universat_patch_m:.0f} m "
          f"({int(args.universat_patch_m / FORESTNET_GSD_M)} px @ "
          f"{FORESTNET_GSD_M:.0f} m)")
    print(f"  Sub-patch:   {args.universat_subpatch_px} px "
          f"(inert for classification -- no CA_Sub decode used)")
    print(f"  Pooling:     {args.universat_pooling} "
          f"({'repo-maintained LP_eval.py convention' if args.universat_pooling in ('mean', 'max') else 'STALE convention, see docstring'})")
print(f"  Channels:    {NUM_CHANNELS} (Landsat optical)")
print(f"  Classes:     {NUM_CLASSES}")
print(f"  Crop:        {args.crop_size}×{args.crop_size}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Class weights: {'ON' if args.use_class_weights else 'OFF'}")
print(f"  GPUs:        {torch.cuda.device_count()}")
if args.resume is not None:
    print(f"  Resuming from: {args.resume}")
if args.test_only is not None:
    print(f"  Mode:        TEST ONLY ({args.test_only})")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

if args.test_only is None:
    train_ds = ForestNetBaselineDataset(
        root_path=args.data_dir, mode="train",
        crop_size=args.crop_size, augment=True,
    )
    val_ds = ForestNetBaselineDataset(
        root_path=args.data_dir, mode="validation",
        crop_size=args.crop_size, augment=False,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
else:
    print(f"  [test-only mode] Skipping train/val dataset construction.")

test_ds = ForestNetBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
)
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=forestnet_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

test_loader = DataLoader(
    test_ds, batch_size=args.batch_size,
    shuffle=False, **loader_kwargs,
)

if args.test_only is None:
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

class_weights = None
if args.use_class_weights:
    if args.test_only is not None:
        print(f"[WARNING] --use_class_weights has no effect in "
              f"--test_only mode (no training loss is computed); ignoring.")
    else:
        class_weights = compute_class_weights(train_ds, NUM_CLASSES)
        print(f"  Class weights: {class_weights.tolist()}")

trainer_module = ClassificationBaselineTrainer(
    model=model,
    # RAMENInputAdapter / UniverSatClassifier both consume the full image
    # dict (see ClassificationBaselineTrainer._get_image); `modality` is
    # unused in that case, only shown in logs/hparams.
    modality=MODALITY_KEY,
    num_classes=NUM_CLASSES,
    lr=args.lr,
    weight_decay=args.weight_decay,
    class_weights=class_weights,
    label_smoothing=args.label_smoothing,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_{args.model}"
        if args.model == "resnet":
            run_name += f"_{args.resnet_variant}"
        if args.model == "universat":
            run_name += f"_{args.universat_size}_pool-{args.universat_pooling}"
        wandb.init(
            name=run_name,
            project="Atomizer_ForestNet_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_ForestNet_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/forestnet_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = []
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
        EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


# =============================================================================
# TRAINER
# =============================================================================

# RAMEN's RadarProjector (reused verbatim from RAMEN/ramen_encoder.py)
# registers polarization parameters that are never used in the forward
# graph when no "sar" modality is passed in (as here, ForestNet is
# optical-only) — DDP's default unused-parameter check rejects that.
# UniverSat similarly has structurally inert params in this single-
# modality, single-frame setting: the Bi_ACA_in fusion attention (one
# modality), the T-axis block (T=1), SAR/DEM channel codes, and (specific
# to UniverSatClassifier) the CA_Sub sub-patch skip's parameters are
# barely exercised since output_grid=latent_grid makes Bilinear_out an
# identity resample — find_unused_parameters=True is required for the
# same reason as every other UniverSat integration in this codebase.
strategy = (
    DDPStrategy(find_unused_parameters=True) if args.model in ("ramen", "universat")
    else DDPStrategy(find_unused_parameters=False)
)

trainer = Trainer(
    strategy=strategy,
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


# =============================================================================
# TRAIN (skipped in --test_only mode) + TEST
# =============================================================================

if args.test_only is None:
    print(f"\n{'='*60}")
    if args.resume is not None:
        print(f"  Resuming: {args.model} on ForestNet from {args.resume}")
    else:
        print(f"  Starting: {args.model} on ForestNet")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=args.resume)

    test_ckpt_path = "best"
else:
    print(f"\n{'='*60}")
    print(f"  [test-only mode] Skipping training, testing checkpoint:")
    print(f"  {args.test_only}")
    print(f"{'='*60}\n")

    test_ckpt_path = args.test_only

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {test_ckpt_path}")
print(f"{'='*60}\n")

trainer.test(trainer_module, test_loader, ckpt_path=test_ckpt_path)


if wandb_logger:
    import wandb
    wandb.finish()
