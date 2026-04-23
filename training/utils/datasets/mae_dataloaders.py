"""
MAE Pre-Training DataModule
============================

Handles one or multiple datasets for MAE pre-training.

Single dataset:
    Returns standard DataLoader with collate_mae.

Multiple datasets (CombinedLoader):
    Returns one DataLoader per dataset, combined via Lightning's
    CombinedLoader with mode="max_size_cycle". Each training step
    receives one batch from EACH dataset simultaneously.

Usage:
    # Single dataset (MMEarth only)
    dm = MAEDataModule(
        datasets={"mmearth": {"class": MMEarthMAEDataset, "root": "./data/MM-Earth"}},
        config_model=config_model,
        dataset_config=dataset_config,
        look_up=look_up,
    )

    # Multi-dataset (add WorldStrat later)
    dm = MAEDataModule(
        datasets={
            "mmearth":    {"class": MMEarthMAEDataset,    "root": "./data/MM-Earth"},
            "worldstrat": {"class": WorldStratMAEDataset, "root": "./data/WorldStrat"},
        },
        ...
    )
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .token_grouping import collate_mae

try:
    from lightning.pytorch.utilities import CombinedLoader
    HAS_COMBINED_LOADER = True
except ImportError:
    try:
        from pytorch_lightning.utilities import CombinedLoader
        HAS_COMBINED_LOADER = True
    except ImportError:
        HAS_COMBINED_LOADER = False

try:
    import torch.distributed as dist
except ImportError:
    dist = None


class MAEDataModule(pl.LightningDataModule):
    """
    DataModule for MAE pre-training across one or multiple datasets.

    Each dataset entry in the `datasets` dict specifies:
        - "class": Dataset class (e.g., MMEarthMAEDataset)
        - "root":  Root path for the dataset
        - Any extra kwargs passed to the dataset constructor

    All datasets share the same config_model, dataset_config, and look_up.
    Each gets its own DataLoader with collate_mae.

    When multiple datasets are provided, train_dataloader() returns a
    CombinedLoader(mode="max_size_cycle") — each step yields one batch
    from every dataset. Shorter datasets cycle back to the start.
    """

    def __init__(
        self,
        datasets: Dict[str, Dict[str, Any]],
        config_model: dict,
        dataset_config: dict,
        look_up=None,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        """
        Args:
            datasets:       {name: {"class": DatasetClass, "root": path, ...extra_kwargs}}
            config_model:   Model config dict (reads masking_MAE and trainer sections)
            dataset_config: YAML band config dict
            look_up:        Lookup_encoding instance
            batch_size:     Batch size per dataset (fallback if not in config)
            num_workers:    Workers per DataLoader
        """
        super().__init__()
        self.dataset_specs = datasets
        self.config_model = config_model
        self.dataset_config = dataset_config
        self.look_up = look_up

        # ── Masking params from config ──────────────────────
        mae_cfg = config_model.get("masking_MAE", {})
        self.mask_ratio = mae_cfg.get("mask_ratio", 0.75)
        self.block_size = mae_cfg.get("block_size", 8)
        

        # ── Batch sizes from config ─────────────────────────
        trainer_cfg = config_model.get("trainer", {})
        self.max_queries = trainer_cfg.get("max_tokens_reconstruction", 100_000)
        self.batch_size_train = trainer_cfg.get("train_batch_size", batch_size)
        self.batch_size_val = trainer_cfg.get("val_batch_size", self.batch_size_train)
        self.num_workers = num_workers

        # Will hold {name: dataset} after setup()
        self.train_datasets = {}
        self.val_datasets = {}

    # =========================================================================
    # SETUP
    # =========================================================================

    def setup(self, stage=None):
        """Create train and val datasets for each entry."""
        for name, spec in self.dataset_specs.items():
            ds_class = spec["class"]
            root = spec["root"]
            extra_kwargs = {
                k: v for k, v in spec.items()
                if k not in ("class", "root")
            }

            common_kwargs = dict(
                root_path=root,
                dataset_config=self.dataset_config,
                config_model=self.config_model,
                look_up=self.look_up,
                mask_ratio=self.mask_ratio,
                block_size=self.block_size,
                max_queries=self.max_queries,
            )
            common_kwargs.update(extra_kwargs)

            self.train_datasets[name] = ds_class(mode="train", **common_kwargs)
            self.val_datasets[name] = ds_class(mode="validation", **common_kwargs)

            rank = dist.get_rank() if (dist and dist.is_initialized()) else 0
            if rank == 0:
                print(f"[MAEDataModule] {name}: "
                      f"train={len(self.train_datasets[name])}, "
                      f"val={len(self.val_datasets[name])}")

    # =========================================================================
    # DATALOADER HELPERS
    # =========================================================================

    def _make_loader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create a single DataLoader with collate_mae."""
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_mae,
            pin_memory=True,
            drop_last=True if shuffle else False,
        )
        # prefetch_factor and persistent_workers require num_workers > 0
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["persistent_workers"] = True

        return DataLoader(**loader_kwargs)

    def _make_combined_or_single(
        self,
        datasets: Dict[str, Any],
        batch_size: int,
        shuffle: bool,
    ):
        """
        If single dataset: return its DataLoader directly.
        If multiple: return CombinedLoader(mode="max_size_cycle").
        """
        loaders = {
            name: self._make_loader(ds, batch_size, shuffle)
            for name, ds in datasets.items()
        }

        if len(loaders) == 1:
            # Single dataset — return plain DataLoader
            return next(iter(loaders.values()))

        if not HAS_COMBINED_LOADER:
            raise ImportError(
                "CombinedLoader not available. Upgrade to lightning>=2.0 "
                "or use a single dataset."
            )

        return CombinedLoader(loaders, mode="max_size_cycle")

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    def train_dataloader(self):
        rank = dist.get_rank() if (dist and dist.is_initialized()) else 0
        if rank == 0:
            names = list(self.train_datasets.keys())
            print(f"[MAEDataModule] Train loaders: {names}, "
                  f"batch_size={self.batch_size_train}")

        return self._make_combined_or_single(
            self.train_datasets,
            self.batch_size_train,
            shuffle=True,
        )

    def val_dataloader(self):
        rank = dist.get_rank() if (dist and dist.is_initialized()) else 0
        if rank == 0:
            names = list(self.val_datasets.keys())
            print(f"[MAEDataModule] Val loaders: {names}, "
                  f"batch_size={self.batch_size_val}")

        return self._make_combined_or_single(
            self.val_datasets,
            self.batch_size_val,
            shuffle=False,
        )