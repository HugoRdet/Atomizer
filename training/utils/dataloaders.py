import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from .FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
from .lookup_positional import*
from .utils_dataset_h5 import*


import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist
import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist

def _init_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset
    # open the file once per worker, and keep it around on `ds.h5`
    ds.h5 = h5py.File(ds.file_path, 'r')

class DistributedShapeBasedBatchSampler(Sampler):
    """
    A distributed batch sampler that groups samples by shape and partitions the batches
    across GPUs. Each process only sees a subset of the batches based on its rank.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, rank=None, world_size=None,mode="train"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.mode=mode


        # Set up distributed parameters.
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank=0
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        self.rank = rank
        self.world_size = world_size

        # Group indices by image shape.
        self.shape_to_indices = {}
        # Use a temporary DataLoader to iterate over the dataset (batch_size=1).
        #loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
        for idx in tqdm(range(len(dataset)), desc="Sampler initialization"):
            # Assuming each sample returns (image, label, ...); adjust as needed.
            image_shape = dataset.Sampler_building(idx,mode=self.mode)
            # Convert image.shape (a torch.Size) to a tuple so it can be used as a key.
            shape_key = image_shape
            self.shape_to_indices.setdefault(shape_key, []).append(idx)
        
        # Create batches from the groups.
        self.batches = []
        for indices in tqdm(self.shape_to_indices.values(), desc="Batch creation"):
            random.shuffle(indices)
            # Create batches for this shape group.
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    self.batches.append(batch)
        
        if self.shuffle:
            random.shuffle(self.batches)
        
        # Make sure total number of batches is divisible by the number of processes.
        total_batches = len(self.batches)
        remainder = total_batches % self.world_size
        if remainder != 0:
            if not self.drop_last:
                # Pad with extra batches (repeating from the beginning) so each process has equal work.
                pad_size = self.world_size - remainder
                self.batches.extend(self.batches[:pad_size])
                total_batches = len(self.batches)
            else:
                # If dropping last incomplete batches, remove the excess.
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
        # Number of batches that this process will iterate over.
        return self.total_batches // self.world_size



import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist


class Tiny_BigEarthNetDataModule(pl.LightningDataModule):
    def __init__(self, path,
                trans_modalities,
                trans_tokens=None, 
                model="None", 
                batch_size=32, 
                num_workers=8,
                modality=None,
                ds=None,
                dataset_config=None,
                config_model=None,
                look_up=None,
                dataset_class=Tiny_BigEarthNet):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_train.h5"#"_validation.h5"
        self.test_file = path + "_test.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_modalities = trans_modalities
        self.model = model
        self.modality=modality
        self.trans_tokens=trans_tokens
        self.dataset_config=dataset_config
        self.config_model=config_model
        self.look_up=look_up
        self.dataset_class=dataset_class
        
 

    def setup(self, stage=None):
        
        
        
        
        self.train_dataset = self.dataset_class(
            self.train_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="train",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

            
        self.val_dataset = self.dataset_class(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        self.val_dataset_mode_train = self.dataset_class(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        self.val_dataset_mode_train.modality_mode="train"

        if self.modality!=None:
            self.val_dataset.modality_mode=self.modality
            
        self.test_dataset = self.dataset_class(
            self.test_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="test",
            modality_mode=self.modality,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        if self.modality!=None:
            self.test_dataset.modality_mode=self.modality


    def train_dataloader(self):
        # Create the custom distributed sampler inside the DataLoader call.
        
        
        
        if self.modality==None:
      
            self.modality="train"

    
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )


    def val_dataloader(self):

        if self.modality==None:
            self.modality="validation"

        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")

        val_mod_val=DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )

        if self.dataset_class!=Tiny_BigEarthNet:
            return val_mod_val
            
        val_mod_train=DataLoader(
            self.val_dataset_mode_train,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )
        
        return [val_mod_val,val_mod_train]


    def test_dataloader(self):


        if self.modality==None:
            self.modality="test"

        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            worker_init_fn=_init_worker,
            drop_last=False,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            prefetch_factor=4,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )

