z"""
PASTIS Two-Stage Training: Frozen Atomizer + L-TAE
====================================================

Stage 2 of the two-stage temporal protocol:
    Stage 1 (done): Train Atomizer on T=1
    Stage 2 (this): Freeze encoder, train L-TAE to aggregate T frames

Protocol (matches PANGAEA two-stage):
    - Load T=1 checkpoint → freeze all encoder parameters
    - For each batch: run frozen encoder T times (once per frame)
    - L-TAE aggregates T sets of per-query predictions
    - Train only the L-TAE parameters

Usage:
    python script_train_pastis_ltae.py \\
        --xp_name pastis_ltae_run1 \\
        --ckpt_t1 ./checkpoints/pastis/pastis_t1_run1-best.ckpt \\
        --config_model config_pastis_fourier.yaml \\
        --multi_temporal 6 \\
        --use_s1

    # Test only
    python script_train_pastis_ltae.py \\
        --xp_name pastis_ltae_run1 \\
        --ckpt_t1 ./checkpoints/pastis/pastis_t1_run1-best.ckpt \\
        --config_model config_pastis_fourier.yaml \\
        --multi_temporal 6 \\
        --test --ckpt_path ./checkpoints/pastis_ltae/pastis_ltae_run1-best.ckpt
"""

import os
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_PASTIS import PASTISTrainer
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder

NUM_CLASSES  = 20
IGNORE_INDEX = 255

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
# COLLATE
# =============================================================================

def pastis_collate(samples):
    batch = collate_multitask(samples)
    if "queries" not in batch and "tasks" in batch:
        task_data             = next(iter(batch["tasks"].values()))
        batch["queries"]      = task_data["queries"]
        batch["queries_mask"] = task_data["queries_mask"]
    return batch


# =============================================================================
# L-TAE MODULE
# =============================================================================

import math


class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder.
    Operates on [B, T, D] → [B, D] via multi-head master-query attention.
    """

    def __init__(
        self,
        in_channels: int = 20,
        n_head: int = 4,
        d_k: int = 8,
        d_model: int = 64,
        dropout: float = 0.1,
        T: int = 1000,
    ):
        super().__init__()
        self.n_head  = n_head
        self.d_k     = d_k
        self.d_model = d_model

        self.fc_in   = nn.Linear(in_channels, d_model)
        self.key_proj = nn.Linear(d_model, n_head * d_k)
        self.fc_out  = nn.Linear(d_model, in_channels)

        # Master query — learned, not input-dependent
        self.master_query = nn.Parameter(
            torch.zeros(n_head, d_k), requires_grad=True)
        nn.init.normal_(self.master_query)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

        # Sinusoidal positional encoding on DOY
        pe  = torch.zeros(T, d_model)
        pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # [1, T, d_model]

        print(f"[LTAE] in={in_channels}, d_model={d_model}, "
              f"n_head={n_head}, d_k={d_k}, out={in_channels}")

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None):
        """
        Args:
            x:   [B, T, D]
            doy: [B, T] day-of-year

        Returns:
            out: [B, D]
        """
        B, T, D = x.shape

        h = self.fc_in(x)   # [B, T, d_model]

        # Positional encoding
        if doy is not None:
            idx = doy.long().clamp(0, self.pos_enc.shape[1] - 1)
            pe  = self.pos_enc.expand(B, -1, -1)
            # gather per-sample per-timestep
            pe_t = pe[torch.arange(B).unsqueeze(1), idx]  # [B, T, d_model]
            h = h + pe_t

        h = self.norm(h)

        # Keys: [B, T, n_head, d_k]
        keys = self.key_proj(h).view(B, T, self.n_head, self.d_k)
        keys = keys.permute(0, 2, 1, 3)  # [B, n_head, T, d_k]

        # Master query: [1, n_head, 1, d_k]
        q = self.master_query.unsqueeze(0).unsqueeze(2)

        # Attention: [B, n_head, T]
        scores = (q * keys).sum(dim=-1) / math.sqrt(self.d_k)
        att    = torch.softmax(scores, dim=-1)
        att    = self.dropout(att)

        # Weighted sum: [B, n_head, d_model] → mean → [B, d_model]
        h_exp  = h.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        out    = (att.unsqueeze(-1) * h_exp).sum(dim=2).mean(dim=1)

        return self.fc_out(out)   # [B, D]


# =============================================================================
# LIGHTNING TRAINER — FROZEN ENCODER + LTAE
# =============================================================================

class AtomizerLTAETrainer(pl.LightningModule):
    """
    Stage 2: Frozen Atomizer encoder + L-TAE temporal aggregation.

    For each batch:
        1. For each of T frames: run frozen encoder → per-query logits [B, N, C]
        2. Stack → [B, T, N, C]
        3. Permute → [B*N, T, C]
        4. L-TAE → [B*N, C]
        5. Reshape → [B, N, C]
        6. Supervise against labels
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        n_frames: int = 6,
        ltae_d_model: int = 64,
        ltae_n_head: int = 4,
        ltae_d_k: int = 8,
        ltae_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        grad_accum: int = 1,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.num_classes  = num_classes
        self.n_frames     = n_frames
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_accum   = grad_accum
        self.ignore_index = ignore_index

        # Frozen encoder
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # L-TAE — only trainable part
        self.ltae = LTAE(
            in_channels=num_classes,
            n_head=ltae_n_head,
            d_k=ltae_d_k,
            d_model=ltae_d_model,
            dropout=ltae_dropout,
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        for split in ("train", "val", "test"):
            setattr(self, f"{split}_mIoU", torchmetrics.JaccardIndex(
                task="multiclass", num_classes=num_classes,
                average="macro", ignore_index=ignore_index,
            ))

        enc_params  = sum(p.numel() for p in self.encoder.parameters())
        ltae_params = sum(p.numel() for p in self.ltae.parameters())
        print(f"[AtomizerLTAETrainer] Encoder (frozen): {enc_params:,}")
        print(f"[AtomizerLTAETrainer] L-TAE (trainable): {ltae_params:,}")

    def _split_batch_into_frames(self, batch):
        """
        Split a multi-temporal batch into T single-frame batches.

        Tokens in groups are ordered as [T*C*H*W, 8] — we need to split
        along the temporal dimension.

        Strategy: the tokens are built as T frames stacked vertically.
        We split into T equal chunks along dim 0.
        """
        frames = []
        groups = batch["groups"]

        for t in range(self.n_frames):
            frame_groups = {}
            for res, group_data in groups.items():
                tokens = group_data["tokens"]   # [B, N_total, 8]
                mask   = group_data["mask"]     # [B, N_total]
                shape  = group_data["shape"]

                # Tokens are stacked: T frames × C × H × W
                # Split evenly along token dim
                N_total = tokens.shape[1]
                N_per_frame = N_total // self.n_frames
                start = t * N_per_frame
                end   = start + N_per_frame

                frame_groups[res] = {
                    "tokens": tokens[:, start:end],
                    "mask":   mask[:, start:end],
                    "shape":  shape,
                }

            frame_batch = {
                "groups":           frame_groups,
                "queries":          batch["queries"],
                "queries_mask":     batch["queries_mask"],
                "label":            batch["label"],
                "target_resolution": batch["target_resolution"],
            }
            if "tasks" in batch:
                frame_batch["tasks"] = batch["tasks"]

            frames.append(frame_batch)

        return frames

    def _get_logits(self, batch):
        """
        Run frozen encoder T times, aggregate with L-TAE.

        Returns:
            logits: [B, N, C] final predictions
            target: [B, H, W] labels
        """
        frames = self._split_batch_into_frames(batch)
        target = batch["label"]

        # Per-frame encoder predictions
        frame_logits = []
        with torch.no_grad():
            for frame_batch in frames:
                result = self.encoder(frame_batch)
                if isinstance(result, dict):
                    logits_t = result.get("predictions",
                               result.get("logits", None))
                else:
                    logits_t = result
                frame_logits.append(logits_t)  # [B, N, C]

        # Stack: [B, T, N, C]
        stacked = torch.stack(frame_logits, dim=1)
        B, T, N, C = stacked.shape

        # Reshape for L-TAE: [B*N, T, C]
        x = stacked.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # DOY — use day indices 0..T-1 as proxy (no actual DOY in batch)
        doy = torch.arange(T, device=x.device).unsqueeze(0).expand(B * N, -1)

        # L-TAE: [B*N, C]
        out = self.ltae(x, doy)

        # Reshape: [B, N, C]
        logits = out.reshape(B, N, C)

        return logits, target

    def _shared_step(self, batch, split: str):
        logits, target = self._get_logits(batch)

        # logits: [B, N, C] — need to convert to spatial [B, C, H, W] for loss
        # Queries carry spatial positions, but for simplicity treat N as flat
        B, N, C = logits.shape
        H, W = target.shape[1], target.shape[2]

        # Reshape predictions to spatial grid
        # Assumes queries cover H*W pixels in raster order
        if N == H * W:
            logits_2d = logits.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            # Subsample case — interpolate to full resolution
            # Reshape to approximate spatial layout
            logits_2d = logits.permute(0, 2, 1).reshape(B, C, N, 1)
            logits_2d = nn.functional.interpolate(
                logits_2d, size=(H * W, 1),
                mode="bilinear", align_corners=False,
            ).reshape(B, C, H, W)

        loss  = self.loss_fn(logits_2d, target.long())
        preds = logits_2d.argmax(dim=1)

        getattr(self, f"{split}_mIoU").update(preds, target)
        self.log(f"{split}_loss", loss,
                 on_step=(split == "train"), on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        # Keep encoder in eval mode — we don't want BN/dropout to update
        self.encoder.eval()
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test")

    def _on_epoch_end(self, split: str):
        miou = getattr(self, f"{split}_mIoU").compute()
        self.log(f"{split}_mIoU", miou,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        if split in ("val", "test"):
            print(f"\n[{split.upper()}] mIoU: {miou:.4f}\n")
        getattr(self, f"{split}_mIoU").reset()

    def on_train_epoch_end(self):      self._on_epoch_end("train")
    def on_validation_epoch_end(self): self._on_epoch_end("val")
    def on_test_epoch_end(self):       self._on_epoch_end("test")

    def configure_optimizers(self):
        # Only optimize L-TAE parameters
        optimizer = torch.optim.AdamW(
            self.ltae.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps  = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.05 * total_steps))
        print(f"[AtomizerLTAETrainer] total_steps={total_steps}, warmup={warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PASTIS Two-Stage: Frozen Atomizer + L-TAE")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--ckpt_t1",        type=str, default=None,
                    help="T=1 Atomizer checkpoint (overrides config trainer.checkpoint_path)")
parser.add_argument("--config_model",   type=str,
                    default="config_pastis_fourier.yaml")
parser.add_argument("--data_dir",       type=str, default="./data/PASTIS-HD")
parser.add_argument("--multi_temporal", type=int, default=6)
parser.add_argument("--use_s1",         action="store_true")
parser.add_argument("--use_spot",       action="store_true")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--grad_accum",     type=int, default=1)
parser.add_argument("--epochs",         type=int, default=50)
parser.add_argument("--lr",             type=float, default=1e-3)
parser.add_argument("--patience",       type=int, default=15)
parser.add_argument("--ltae_d_model",   type=int, default=64)
parser.add_argument("--ltae_n_head",    type=int, default=4)
parser.add_argument("--ltae_d_k",       type=int, default=8)
# Test mode
parser.add_argument("--test",           action="store_true")
parser.add_argument("--ckpt_path",      type=str, default=None)
args = parser.parse_args()

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================

config_model         = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path      = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# Force T=1 for encoder (it was trained on T=1)
config_t1 = dict(config_model)
config_t1.setdefault("dataset", {})["multi_temporal"] = 1

# Multi-temporal config for L-TAE training
config_mt = dict(config_model)
config_mt.setdefault("dataset", {})["multi_temporal"] = args.multi_temporal

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

if args.use_s1:
    lookup_table.register_abstract_channel("VV_VH")

# Resolve T=1 checkpoint — arg overrides config
ckpt_t1 = args.ckpt_t1 or config_model.get("trainer", {}).get("checkpoint_path", None)
if ckpt_t1 is None:
    raise ValueError(
        "T=1 checkpoint not found. Either pass --ckpt_t1 or set "
        "trainer.checkpoint_path in the config YAML."
    )
args.ckpt_t1 = ckpt_t1

full_xp_name = f"PASTIS_LTAE_{args.xp_name}_T{args.multi_temporal}"
print(f"\n[PASTIS-LTAE] Experiment: {full_xp_name}")
print(f"[PASTIS-LTAE] Encoder: {args.ckpt_t1}")
print(f"[PASTIS-LTAE] Temporal: T={args.multi_temporal}")

# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        wandb.init(
            name=full_xp_name,
            project="PASTIS",
            config={
                **config_model,
                "stage": "ltae",
                "ckpt_t1": args.ckpt_t1,
                "multi_temporal": args.multi_temporal,
                "ltae_d_model": args.ltae_d_model,
            },
        )
        wandb_logger = WandbLogger(project="PASTIS")
    except Exception:
        pass

# =============================================================================
# DATASETS
# =============================================================================

def make_dataset(mode):
    return PastisHDDataset(
        root_path=args.data_dir,
        mode=mode,
        config_model=config_mt,   # uses multi_temporal=T
        look_up=lookup_table,
        use_s1=args.use_s1,
        use_spot=args.use_spot,
    )

if not args.test:
    train_ds = make_dataset("train")
    val_ds   = make_dataset("validation")
else:
    train_ds = val_ds = None

test_ds = make_dataset("test")

def make_loader(dataset, shuffle=False):
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=pastis_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

if not args.test:
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
else:
    train_loader = val_loader = None

test_loader = make_loader(test_ds, shuffle=False)

# =============================================================================
# LOAD FROZEN ENCODER
# =============================================================================

print(f"\n[PASTIS-LTAE] Loading T=1 encoder from: {args.ckpt_t1}")

# Build Atomizer with T=1 config
encoder_module = PASTISTrainer(
    config=config_t1, wand=False, name="encoder_t1",
    transform=None, lookup_table=lookup_table,
)

# Load checkpoint weights
ckpt = torch.load(args.ckpt_t1, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
result = encoder_module.load_state_dict(state, strict=False)
print(f"[PASTIS-LTAE] Encoder loaded — "
      f"missing: {len(result.missing_keys)}, "
      f"unexpected: {len(result.unexpected_keys)}")

# =============================================================================
# TRAINER MODULE
# =============================================================================

trainer_module = AtomizerLTAETrainer(
    encoder=encoder_module,
    num_classes=NUM_CLASSES,
    n_frames=args.multi_temporal,
    ltae_d_model=args.ltae_d_model,
    ltae_n_head=args.ltae_n_head,
    ltae_d_k=args.ltae_d_k,
    lr=args.lr,
    weight_decay=0.05,
    grad_accum=args.grad_accum,
)

# =============================================================================
# LIGHTNING TRAINER
# =============================================================================

ckpt_dir = "./checkpoints/pastis_ltae/"
os.makedirs(ckpt_dir, exist_ok=True)

if not args.test:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{full_xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
            monitor="val_mIoU", mode="max",
            save_top_k=1, verbose=True,
        ),
        EarlyStopping(
            monitor="val_mIoU", mode="max",
            patience=args.patience, verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
else:
    callbacks = []

trainer = Trainer(
    devices=1,           # single device — L-TAE is tiny, no need for DDP
    accelerator="gpu",
    max_epochs=args.epochs,
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# TRAIN & TEST
# =============================================================================

if not args.test:
    print(f"\n{'='*60}")
    print(f"  Stage 2: {full_xp_name}")
    print(f"  Frozen encoder + L-TAE (T={args.multi_temporal})")
    print(f"{'='*60}\n")
    trainer.fit(trainer_module, train_loader, val_loader)
    test_ckpt = "best"
else:
    if args.ckpt_path is None:
        raise ValueError("--test requires --ckpt_path")
    test_ckpt = args.ckpt_path
    print(f"\n[PASTIS-LTAE] Test-only: {test_ckpt}")

print(f"\n{'='*60}")
print(f"  Testing: {full_xp_name}")
print(f"{'='*60}\n")

results = trainer.test(trainer_module, test_loader,
                       ckpt_path="none" if test_ckpt is None else test_ckpt)

if results:
    r = results[0]
    print(f"\n{'='*60}")
    print(f"  FINAL — {full_xp_name}")
    print(f"  mIoU: {r.get('test_mIoU', 'N/A'):.4f}")
    print(f"{'='*60}\n")
