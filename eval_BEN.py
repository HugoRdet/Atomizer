"""
BigEarthNet Test Evaluation
=============================

Evaluates a trained BEN checkpoint on the TEST split, reporting
the same metrics as the reBEN paper (Table I):

    AP_M  — macro-averaged Average Precision
    AP_µ  — micro-averaged Average Precision
    F1_M  — macro-averaged F1 score
    F1_µ  — micro-averaged F1 score

Also reports per-class AP for analysis.

Usage:
    python eval_ben.py --ckpt_path ./checkpoints/ben/ben-39-0.7200.ckpt

    # Or evaluate on validation split
    python eval_ben.py --ckpt_path ./checkpoints/ben/best.ckpt --split val
"""

import os
import argparse
import json
import time

import torch
import numpy as np
import torchmetrics
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_Ben import BENPretrainTrainer
from training.utils.datasets.utils_dataset_Ben import (
    BigEarthNetAtomizer, collate_ben, register_ben_bands,
)
from training.utils.datasets.token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

CONFIG_MODEL_PATH = "./training/configs/config_test-Atomiser_Atos_One.yaml"
BANDS_YAML_PATH = "./data/bands_info/bands.yaml"
CONFIGS_DATASET_PATH = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# reBEN 19-class names
CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Agriculture w/ natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland & sparse veg.",
    "Moors, heathland & sclerophyllous",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048, 2.5: 2048, 4.78: 2048, 5.0: 2048,
    10.0: 2048, 20.0: 2048, 30.0: 2048, 60.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# ═══════════════════════════════════════════════════════════════════════
# BEN OVERRIDES (must match training)
# ═══════════════════════════════════════════════════════════════════════

BEN_OVERRIDES = {
    "trainer": {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "max_epochs": 40,
        "batch_size": 48,
        "grad_accum": 1,
        "num_workers": 8,
        "precision": "bf16-mixed",
        "num_classes": 19,
    },
    "data": {
        "images_lmdb": "data/Encoded-BigEarthNet",
        "metadata_parquet": "data/Encoded-BigEarthNet/metadata.parquet",
        "metadata_snow_cloud_parquet": "data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    },
    "ben_pretrain": {
        "tpl_min": 768,
        "tpl_max": 768,
        "tpl_val": 768,
    },
}


def load_config():
    config = read_yaml(CONFIG_MODEL_PATH)
    for key, value in BEN_OVERRIDES.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BigEarthNet Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .ckpt checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results/ben")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ──
    config = load_config()

    # ── Lookup table ──
    lookup_table = Lookup_encoding(
        read_yaml(CONFIGS_DATASET_PATH),
        read_yaml(BANDS_YAML_PATH),
        config,
    )
    register_all_resolutions(lookup_table)
    register_ben_bands(lookup_table)

    # ── Dataset ──
    data_dirs = config["data"]
    stats_path = config["data"].get(
        "norm_stats", "data/Encoded-BigEarthNet/ben_norm_stats.json"
    )
    dataset = BigEarthNetAtomizer(
        data_dirs=data_dirs,
        split=args.split,
        look_up=lookup_table,
        stats_path=stats_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ben,
        pin_memory=True,
    )

    print(f"\n{'='*60}")
    print(f"  BigEarthNet Evaluation")
    print(f"  Split:    {args.split} ({len(dataset)} samples)")
    print(f"  Ckpt:     {args.ckpt_path}")
    print(f"{'='*60}\n")

    # ── Model ──
    model = BENPretrainTrainer(
        config=config,
        wand=None,
        name="eval",
        transform=None,
        lookup_table=lookup_table,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[Eval] Loaded state_dict from checkpoint")
    else:
        model.load_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ── Metrics (matching reBEN paper exactly) ──
    nc = 19

    # AP: macro and micro
    ap_macro = torchmetrics.AveragePrecision(
        task="multilabel", num_labels=nc, average="macro",
    ).to(device)
    ap_micro = torchmetrics.AveragePrecision(
        task="multilabel", num_labels=nc, average="micro",
    ).to(device)

    # F1: macro and micro
    f1_macro = torchmetrics.F1Score(
        task="multilabel", num_labels=nc, average="macro",
    ).to(device)
    f1_micro = torchmetrics.F1Score(
        task="multilabel", num_labels=nc, average="micro",
    ).to(device)

    # Per-class AP
    ap_per_class = torchmetrics.AveragePrecision(
        task="multilabel", num_labels=nc, average=None,
    ).to(device)

    # ── Inference ──
    t_start = time.perf_counter()
    n_batches = len(loader)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (n_batches - i - 1)
                print(f"  Batch {i+1}/{n_batches} "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")

            # Move to device
            batch = _batch_to_device(batch, device)

            logits = model.forward(batch, training=False)  # [B, 19]
            labels = batch["label"]  # [B, 19]

            probs = torch.sigmoid(logits)
            labels_int = labels.long()

            ap_macro.update(probs, labels_int)
            ap_micro.update(probs, labels_int)
            f1_macro.update(probs, labels_int)
            f1_micro.update(probs, labels_int)
            ap_per_class.update(probs, labels_int)

    elapsed = time.perf_counter() - t_start

    # ── Compute ──
    ap_M = ap_macro.compute().item() * 100
    ap_mu = ap_micro.compute().item() * 100
    f1_M = f1_macro.compute().item() * 100
    f1_mu = f1_micro.compute().item() * 100
    ap_classes = ap_per_class.compute().cpu().numpy() * 100

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"  Results — {args.split} split ({len(dataset)} samples)")
    print(f"  Inference time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    print(f"  ┌────────────────────────────────────────────┐")
    print(f"  │  AP_M  (macro mAP):    {ap_M:>6.2f}%              │")
    print(f"  │  AP_µ  (micro mAP):    {ap_mu:>6.2f}%              │")
    print(f"  │  F1_M  (macro F1):     {f1_M:>6.2f}%              │")
    print(f"  │  F1_µ  (micro F1):     {f1_mu:>6.2f}%              │")
    print(f"  └────────────────────────────────────────────┘")

    # ── reBEN comparison ──
    print(f"\n  reBEN Baselines (S2 only, for comparison):")
    print(f"  {'Model':<25s} {'AP_M':>6s} {'AP_µ':>6s} {'F1_M':>6s} {'F1_µ':>6s}")
    print(f"  {'-'*51}")
    print(f"  {'ResNet-50':<25s} {'70.72':>6s} {'85.86':>6s} {'64.74':>6s} {'76.34':>6s}")
    print(f"  {'ResNet-101':<25s} {'70.63':>6s} {'85.92':>6s} {'64.19':>6s} {'76.13':>6s}")
    print(f"  {'MobileViT S':<25s} {'69.84':>6s} {'86.20':>6s} {'62.10':>6s} {'75.99':>6s}")
    print(f"  {'ConvNeXt V2 Base':<25s} {'68.61':>6s} {'85.13':>6s} {'62.64':>6s} {'75.43':>6s}")
    print(f"  {'MLP-Mixer Base':<25s} {'67.77':>6s} {'84.32':>6s} {'62.49':>6s} {'74.59':>6s}")
    print(f"  {'-'*51}")
    print(f"  {'Atomizer-IO (ours)':<25s} {ap_M:>6.2f} {ap_mu:>6.2f} {f1_M:>6.2f} {f1_mu:>6.2f}")
    note = "12 bands" if True else "10 bands"
    print(f"  (ours: {note}, others: 10 bands)")

    # ── Per-class AP ──
    print(f"\n  Per-Class AP:")
    print(f"  {'Class':<40s} {'AP (%)':>8s}")
    print(f"  {'-'*50}")
    for i in range(nc):
        print(f"  {CLASS_NAMES[i]:<40s} {ap_classes[i]:>8.2f}")

    # ── Save ──
    results = {
        "split": args.split,
        "n_samples": len(dataset),
        "ckpt_path": args.ckpt_path,
        "inference_time_s": elapsed,
        "AP_M": ap_M,
        "AP_mu": ap_mu,
        "F1_M": f1_M,
        "F1_mu": f1_mu,
        "per_class_AP": {CLASS_NAMES[i]: float(ap_classes[i]) for i in range(nc)},
        "reben_baselines": {
            "ResNet-50_S2": {"AP_M": 70.72, "AP_mu": 85.86, "F1_M": 64.74, "F1_mu": 76.34},
            "ResNet-101_S2": {"AP_M": 70.63, "AP_mu": 85.92, "F1_M": 64.19, "F1_mu": 76.13},
        },
    }

    results_path = os.path.join(args.output_dir, f"ben_{args.split}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {results_path}")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()