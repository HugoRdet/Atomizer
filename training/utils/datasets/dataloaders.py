"""
Unified DataModule with factory pattern for different dataset types.

Supports:
- Tiny_BigEarthNet (HDF5-based)
- FLAIR_MAE (HDF5-based)
- MNISTSparseCanvas (in-memory MNIST)
- Sen1Floods11Dataset (grouped token format)
- HLSBurnScarsDataset (grouped token format)
- MADOSDataset (grouped token format, with sliding window)

Sliding window:
    When config_model["trainer"]["slide"] is True, val/test batch_size
    is forced to 1 (required by the collate for variable-length crop lists).
    Train batch_size is unaffected.
"""

import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset, DataLoader, Sampler
from .utils_dataset_SENFLOOD import Sen1Floods11Dataset
from .token_grouping import collate_grouped
import h5py
from tqdm import tqdm
import random
from datetime import datetime, timezone
import torch.distributed as dist
import time


def _init_worker_h5(worker_id):
    """Worker init function for HDF5-based datasets."""
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset
    if hasattr(ds, 'file_path') and ds.file_path is not None:
        ds.h5 = h5py.File(ds.file_path, 'r')


def _init_worker_simple(worker_id):
    """Worker init function for simple datasets (no special initialization needed)."""
    pass


class DistributedShapeBasedBatchSampler(Sampler):
    """
    A distributed batch sampler that groups samples by shape and partitions the batches
    across GPUs. Each process only sees a subset of the batches based on its rank.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, rank=None, world_size=None, mode="train"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.mode = mode

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        self.rank = rank
        self.world_size = world_size

        self.shape_to_indices = {}
        for idx in tqdm(range(len(dataset)), desc="Sampler initialization"):
            image_shape = dataset.Sampler_building(idx, mode=self.mode)
            shape_key = image_shape
            self.shape_to_indices.setdefault(shape_key, []).append(idx)
        
        self.batches = []
        for indices in tqdm(self.shape_to_indices.values(), desc="Batch creation"):
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    self.batches.append(batch)
        
        if self.shuffle:
            random.shuffle(self.batches)
        
        total_batches = len(self.batches)
        remainder = total_batches % self.world_size
        if remainder != 0:
            if not self.drop_last:
                pad_size = self.world_size - remainder
                self.batches.extend(self.batches[:pad_size])
                total_batches = len(self.batches)
            else:
                total_batches = total_batches - remainder
                self.batches = self.batches[:total_batches]
        self.total_batches = total_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for i in range(self.rank, self.total_batches, self.world_size):
            batch = self.batches[i]
            yield batch

    def __len__(self):
        return self.total_batches // self.world_size


class UnifiedDataModule(pl.LightningDataModule):
    """
    Unified DataModule that works with multiple dataset types.
    
    Dataset types:
    - 'h5':      HDF5-based (Tiny_BigEarthNet, FLAIR_MAE) — h5 worker init
    - 'simple':  In-memory (MNISTSparseCanvas) — no special init
    - 'grouped': Grouped token format (Sen1Floods11, HLSBurnScars, MADOS) — 
                 custom collate_fn for nested dict output

    Batch sizes (from config_model["trainer"]):
    - train_batch_size: training batch size (default: falls back to batch_size param)
    - val_batch_size:   validation batch size (default: same as train)
    - test_batch_size:  test batch size (default: same as val)
    - When slide=True in config, val/test batch_size is forced to 1.
    """
    
    H5_DATASET_CLASSES = {'Tiny_BigEarthNet', 'FLAIR_MAE', 'FLAIR_2'}
    SIMPLE_DATASET_CLASSES = {'MNISTSparseCanvas'}
    GROUPED_DATASET_CLASSES = {'Sen1Floods11Dataset', 
                                'HLSBurnScarsDataset', 
                                'MADOSDataset', 
                                'Sen1Floods11MultiResDataset',
                                'PastisHDDataset',
                                'MMEarthMAEDataset',
                                "PastisSpotReconDataset",
                                'PastisS2ReconDataset',
                                'MMEarthReconstruction',
                                'MMEarthSegDW',
                                'MMEarthSegESA',
                                'ForestNetDataset',
                                "pv4ger",
                                "BurnScarsDataset"
                                }
    
    def __init__(
        self,
        path=None,
        trans_modalities=None,
        trans_tokens=None,
        model="None",
        batch_size=32,
        num_workers=8,
        modality=None,
        ds=None,
        dataset_config=None,
        config_model=None,
        look_up=None,
        dataset_class=None,
        # MNISTSparseCanvas specific params
        canvas_size=64,
        min_digit_size=59,
        max_digit_size=59,
        num_samples_train=None,
        num_samples_val=None,
        fixed_position=False,
    ):
        super().__init__()
        
        # Common params
        self.num_workers = num_workers
        self.trans_modalities = trans_modalities
        self.model = model
        self.modality = modality
        self.dataset_config = dataset_config
        self.config_model = config_model
        self.look_up = look_up
        self.dataset_class = dataset_class
        
        # Determine dataset type early (needed for path setup)
        self._dataset_type = self._get_dataset_type()

        # ── Batch sizes (from config, with fallbacks) ──────
        trainer_cfg = (config_model or {}).get("trainer", {})
        self.use_sliding = trainer_cfg.get("slide", False)

        self.batch_size_train = trainer_cfg.get("train_batch_size", batch_size)
        self.batch_size_val = trainer_cfg.get("val_batch_size", self.batch_size_train)
        self.batch_size_test = trainer_cfg.get("test_batch_size", self.batch_size_val)

        if self.use_sliding:
            # Sliding window requires batch_size=1 for val/test
            self.batch_size_val = 1
            self.batch_size_test = 1
        
        # Path setup depends on dataset type
        self.path = path
        if path is not None:
            if self._dataset_type == 'grouped':
                # Grouped datasets: single root path, splits handled internally
                self.train_file = path
                self.val_file = path
                self.test_file = path
            else:
                # HDF5 datasets: separate files per split
                self.train_file = path + "_train.h5"
                self.val_file = path + "_val.h5"
                self.test_file = path + "_test.h5"
        
        # MNISTSparseCanvas specific params
        self.canvas_size = canvas_size
        self.min_digit_size = min_digit_size
        self.max_digit_size = max_digit_size
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.fixed_position = fixed_position
        
    def _get_dataset_type(self) -> str:
        """Determine the type of dataset for configuration purposes."""
        if self.dataset_class is None:
            return 'unknown'
        
        class_name = self.dataset_class.__name__
        
        if class_name in self.H5_DATASET_CLASSES:
            return 'h5'
        elif class_name in self.SIMPLE_DATASET_CLASSES:
            return 'simple'
        elif class_name in self.GROUPED_DATASET_CLASSES:
            return 'grouped'
        else:
            return 'h5'  # default for backward compat
    
    def _get_worker_init_fn(self):
        """Get the appropriate worker init function based on dataset type."""
        if self._dataset_type == 'h5':
            return _init_worker_h5
        else:
            return _init_worker_simple
    
    def _get_collate_fn(self):
        """Get the appropriate collate function based on dataset type."""
        if self._dataset_type == 'grouped':
            return collate_grouped
        else:
            return None  # PyTorch default
    
    # =========================================================================
    # DATASET CREATION
    # =========================================================================
    
    def _create_h5_dataset(self, file_path: str, mode: str):
        """Create an HDF5-based dataset."""
        return self.dataset_class(
            file_path,
            transform=self.trans_modalities,
            model=self.model,
            mode=mode,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )
    
    def _create_grouped_dataset(self, mode: str):
        """Create a grouped token format dataset."""
        return self.dataset_class(
            root_path=self.path,
            transform=self.trans_modalities,
            model=self.model,
            mode=mode,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up,
        )
    
    def _create_mnist_dataset(self, mode: str):
        """Create an MNISTSparseCanvas dataset."""
        num_samples = self.num_samples_train if mode == "train" else self.num_samples_val
        
        return self.dataset_class(
            canvas_size=self.canvas_size,
            min_digit_size=self.min_digit_size,
            max_digit_size=self.max_digit_size,
            num_bands=1,
            mode=mode,
            config_model=self.config_model,
            look_up=self.look_up,
            num_samples=num_samples,
            fixed_position=self.fixed_position,
        )
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    def setup(self, stage=None):
        """Setup datasets based on dataset type."""
        if self._dataset_type == 'h5':
            self._setup_h5_datasets()
        elif self._dataset_type == 'simple':
            self._setup_simple_datasets()
        elif self._dataset_type == 'grouped':
            self._setup_grouped_datasets()
        else:
            raise ValueError(f"Unknown dataset type: {self._dataset_type}")
    
    def _setup_h5_datasets(self):
        """Setup HDF5-based datasets."""
        self.train_dataset = self._create_h5_dataset(self.train_file, "train")
        self.val_dataset = self._create_h5_dataset(self.val_file, "validation")
        
        if self.modality is not None:
            self.val_dataset.modality_mode = self.modality
        
        self.test_dataset = self.dataset_class(
            self.test_file,
            transform=self.trans_modalities,
            model=self.model,
            mode="test",
            modality_mode=self.modality,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )
        
        if self.modality is not None:
            self.test_dataset.modality_mode = self.modality
    
    def _setup_grouped_datasets(self):
        """Setup grouped token format datasets."""
        self.train_dataset = self._create_grouped_dataset("train")
        self.val_dataset = self._create_grouped_dataset("validation")
        self.test_dataset = self._create_grouped_dataset("test")
    
    def _setup_simple_datasets(self):
        """Setup simple in-memory datasets."""
        self.train_dataset = self._create_mnist_dataset("train")
        self.val_dataset = self._create_mnist_dataset("validation")
        self.test_dataset = self._create_mnist_dataset("test")
    
    # =========================================================================
    # DATALOADERS
    # =========================================================================
    
    def _make_dataloader(self, dataset, batch_size, shuffle=False, prefetch_factor=2):
        """Create a DataLoader with the right collate/worker config."""
        worker_init_fn = self._get_worker_init_fn()
        collate_fn = self._get_collate_fn()
        
        num_workers = self.num_workers
        if self._dataset_type == 'simple' and num_workers > 4:
            num_workers = 4

        # pin_memory doesn't work well with nested dicts in sliding window
        pin_memory = not (self.use_sliding and batch_size == 1)
        
        return DataLoader(
            dataset,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    def train_dataloader(self):
        if self.modality is None:
            self.modality = "train"
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}, batch_size={self.batch_size_train}")
        
        pf = 8 if self._dataset_type == 'h5' else 2
        return self._make_dataloader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            prefetch_factor=pf,
        )

    def val_dataloader(self):
        if self.modality is None:
            self.modality = "validation"
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}, batch_size={self.batch_size_val}"
              f"{' (sliding window)' if self.use_sliding else ''}")
        
        pf = 8 if self._dataset_type == 'h5' else 2
        return self._make_dataloader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            prefetch_factor=pf,
        )

    def test_dataloader(self):
        if self.modality is None:
            self.modality = "test"
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}, batch_size={self.batch_size_test}"
              f"{' (sliding window)' if self.use_sliding else ''}")
        
        pf = 4 if self._dataset_type == 'h5' else 2
        return self._make_dataloader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            prefetch_factor=pf,
        )


# Backward compatibility alias
Tiny_BigEarthNetDataModule = UnifiedDataModule


# Factory function for easy creation
def create_datamodule(
    dataset_type: str,
    config_model: dict,
    look_up=None,
    **kwargs
) -> UnifiedDataModule:
    """
    Factory function to create the appropriate DataModule.
    """
    if dataset_type.lower() == 'mnist':
        from .mnist_sparse_canvas import MNISTSparseCanvas
        return UnifiedDataModule(
            dataset_class=MNISTSparseCanvas,
            config_model=config_model,
            look_up=look_up,
            path=None,
            **kwargs
        )
    elif dataset_type.lower() == 'flair':
        from .FLAIR_2 import FLAIR_MAE
        return UnifiedDataModule(
            dataset_class=FLAIR_MAE,
            config_model=config_model,
            look_up=look_up,
            **kwargs
        )
    elif dataset_type.lower() == 'bigearthnet':
        from .FLAIR_2 import Tiny_BigEarthNet
        return UnifiedDataModule(
            dataset_class=Tiny_BigEarthNet,
            config_model=config_model,
            look_up=look_up,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                        f"Expected one of: 'mnist', 'flair', 'bigearthnet'")