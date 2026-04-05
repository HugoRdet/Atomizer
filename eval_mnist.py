"""
MNIST Evaluation Script
========================
Load a checkpoint, run on test set, report accuracy.

Usage:
    python eval_mnist.py --ckpt_path ./checkpoints/mnist/best.ckpt
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MNIST import Model_MNIST
from training.utils.datasets.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.datasets.token_builder import TokenBuilder


def mnist_collate(samples):
    all_res = sorted(samples[0]["groups"].keys())
    groups = {}
    for res in all_res:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
            "mask": torch.stack([s["groups"][res]["mask"] for s in samples]),
            "shape": samples[0]["groups"][res]["shape"],
        }
    return {
        "groups": groups,
        "queries": torch.stack([s["queries"] for s in samples]),
        "queries_mask": torch.stack([s["queries_mask"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
    }


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_model", type=str,
                        default="config_test-Atomiser_Atos_One.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────
    config_model = read_yaml("./training/configs/" + args.config_model)
    config_model["trainer"]["max_tokens"] = 784
    config_model["trainer"]["max_tokens_reconstruction"] = 784

    if "latent_grid" not in config_model:
        config_model["latent_grid"] = {}
    config_model["latent_grid"]["tokens_per_latent"] =  784
    config_model["latent_grid"]["sigma_factor"] = 1.5
    config_model["latent_grid"]["max_k"] =  784

    # ── Lookup table ────────────────────────────────────────
    bands_yaml = "./data/bands_info/bands.yaml"
    configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset), read_yaml(bands_yaml), config_model
    )
    TokenBuilder.REFERENCE_SIZES[0.2] = 28
    lookup_table.get_or_register_modality(0.2, 28)
    lookup_table.get_resolution_idx(0.2)

    # ── Dataset ─────────────────────────────────────────────
    mode = "train" if args.split == "train" else "val" if args.split == "val" else "test"
    # MNIST only has train/test — map "test" and "val" to test split
    mnist_mode = "train" if args.split == "train" else "test"

    dataset = MNISTSparseCanvas(
        mode=mnist_mode, config_model=config_model, look_up=lookup_table,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=mnist_collate, pin_memory=True,
    )

    print(f"\n{'='*60}")
    print(f"  MNIST Evaluation — {args.split} split ({len(dataset)} samples)")
    print(f"{'='*60}")

    # ── Load model ──────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    model = Model_MNIST(
        config=config_model, wand=False, name="eval",
        transform=None, lookup_table=lookup_table,
    )

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.encoder.load_state_dict(ckpt, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[Eval] Model loaded on {device}")

    # ── Inference ───────────────────────────────────────────
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            batch_gpu = {
                "groups": {
                    res: {
                        "tokens": g["tokens"].to(device),
                        "mask": g["mask"].to(device),
                        "shape": g["shape"],
                    }
                    for res, g in batch["groups"].items()
                },
                "queries": batch["queries"].to(device),
                "queries_mask": batch["queries_mask"].to(device),
                "label": batch["label"].to(device),
            }

            logits = model(batch_gpu, training=False)
            labels = batch_gpu["label"]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")

    # ── Metrics ─────────────────────────────────────────────
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    avg_loss = total_loss / max(n_batches, 1)

    # Per-class accuracy
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Avg Loss:  {avg_loss:.4f}")
    print(f"  Samples:   {len(all_labels)}")

    print(f"\n  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*36}")
    for digit in range(10):
        mask = all_labels == digit
        total = mask.sum()
        correct = ((all_preds == digit) & mask).sum()
        acc = correct / max(total, 1) * 100
        print(f"  {digit:<8} {correct:>8} {total:>8} {acc:>7.1f}%")

    # Confusion matrix (compact)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':>4}", end="")
    for j in range(10):
        print(f"{j:>5}", end="")
    print()
    for i in range(10):
        print(f"  {i:>3}:", end="")
        for j in range(10):
            count = ((all_labels == i) & (all_preds == j)).sum()
            print(f"{count:>5}", end="")
        print()

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()