"""
MNIST Token-Removal Sweep — Atomizer
=====================================
Evaluates a SINGLE trained checkpoint (full-token training, subsample_keep_rate
forced to 1.0 -- see PATCH_mnist_val_split.py) across a grid of inference-time
token-removal configurations, matching the ViT-side ablation protocol exactly:

    rates:      [1.00, 0.75, 0.50, 0.25, 0.10]
    modes:      "uniform"          (== "random drop" / "rnd")
                "background_only"  (== "dark-pixel drop" / "drk", threshold=0.5)

For each (rate, mode) pair, we re-instantiate the TEST dataset with that
config (subsample_keep_rate and subsample_mode are dataset-construction-time
parameters, not per-batch overrides -- see MNISTSparseCanvas docstring), then
run trainer.test() with the same checkpoint throughout.

NOTE: batch_size is forced to 1 whenever rate < 1.0, per MNISTSparseCanvas's
own docstring ("batch_size MUST be 1 since token count varies per sample").
At rate == 1.0 every sample has exactly 784 tokens, so a larger batch size
is safe and faster.

Usage:
    python test_mnist_sweep.py --ckpt path/to/best_checkpoint.ckpt
"""

import argparse
import json
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MNIST import Model_MNIST
from training.utils.datasets.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MNIST token-removal sweep — Atomizer")
    p.add_argument("--ckpt", type=str, required=True,
                    help="Path to the trained Lightning checkpoint (.ckpt).")
    p.add_argument("--config", type=str,
                    default="./training/configs/config_test-MNIST.yaml",
                    help="Base config (architecture / lookup settings). "
                         "trainer.subsample_keep_rate / subsample_mode in this "
                         "file are IGNORED -- the sweep overrides them per cell.")
    p.add_argument("--out", type=str, default="./atomizer_mnist_sweep_results.json",
                    help="Where to write the JSON results.")
    return p.parse_args()


RATES = [1.00, 0.75, 0.50, 0.25, 0.10]
MODES = ["uniform","background_only"]  #"uniform",  "rnd", "drk"


# =============================================================================
# COLLATE (identical to train_mnist.py's mnist_collate)
# =============================================================================

def mnist_collate(samples):
    all_res = sorted(samples[0]["groups"].keys())
    groups = {}
    for res in all_res:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
            "mask":   torch.stack([s["groups"][res]["mask"]   for s in samples]),
            "shape":  samples[0]["groups"][res]["shape"],
        }
    return {
        "groups":            groups,
        "queries":           torch.stack([s["queries"]      for s in samples]),
        "queries_mask":      torch.stack([s["queries_mask"] for s in samples]),
        "target_resolution": samples[0]["target_resolution"],
        "latent_layout":     samples[0].get("latent_layout", "grid"),
        "label":             torch.stack([s["label"]        for s in samples]),
    }


def main():
    args = parse_args()
    seed_everything(42, workers=True)

    config_model = read_yaml(args.config)
    config_model["trainer"]["num_classes"] = 10

    bands_yaml = "./data/bands_info/bands.yaml"
    configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
    lookup_table = Lookup_encoding(
        None, read_yaml(bands_yaml), config_model
    )
    TokenBuilder.REFERENCE_SIZES[0.2] = 28
    lookup_table.get_or_register_modality(0.2, 28)
    lookup_table.get_resolution_idx(0.2)

    model = Model_MNIST(
        config=config_model, wand=False, name="mnist_sweep",
        transform=None, lookup_table=lookup_table,
    )

    results = {"rates": RATES, "uniform": {}, "background_only": {}}

    for mode in MODES:
        for rate in RATES:
            print(f"\n{'='*60}")
            print(f"Evaluating: mode={mode}  rate={rate:.2f}")
            print(f"{'='*60}")

            cell_config = dict(config_model)
            cell_config["trainer"] = dict(config_model["trainer"])
            cell_config["trainer"]["subsample_keep_rate"] = rate
            cell_config["trainer"]["subsample_mode"] = mode

            test_dataset = MNISTSparseCanvas(
                mode="test", config_model=cell_config, look_up=lookup_table,
            )

            # batch_size is forced to 1 for EVERY cell in the sweep, including
            # rate=1.00. This isn't strictly required at rate=1.00 (every
            # sample has exactly 784 tokens there, so a larger batch would be
            # safe token-count-wise), but keeping it fixed at 1 across the
            # whole sweep removes batch size as a possible confound between
            # the reference row and the ablation rows.
            eval_batch_size = 1

            test_loader = DataLoader(
                test_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=4, collate_fn=mnist_collate, pin_memory=True,
            )

            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision="16-mixed" if torch.cuda.is_available() else "32-true",
                logger=False,
                enable_checkpointing=False,
            )

            metrics = trainer.test(model, dataloaders=test_loader, ckpt_path=args.ckpt)
            # trainer.test returns a list of dicts (one per dataloader); we have one.
            test_acc = float(metrics[0]["test_accuracy"])
            test_f1 = float(metrics[0]["test_f1"])

            results[mode][rate] = {"acc": test_acc, "f1": test_f1}
            print(f"-> mode={mode} rate={rate:.2f}: acc={test_acc:.4f} f1={test_f1:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {args.out}")

    print("\n--- Table (accuracy %) ---")
    print(f"{'rate':<8}{'MNIST (rnd)':<14}{'MNIST (drk)':<14}")
    for rate in RATES:
        u = results["uniform"][rate]["acc"] * 100
        b = results["background_only"][rate]["acc"] * 100
        print(f"{rate:<8.2f}{u:<14.2f}{b:<14.2f}")


if __name__ == "__main__":
    main()
