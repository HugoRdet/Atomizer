import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class RoundRobinDistributedBatchSampler(Sampler):
    def __init__(self, dataset_lengths, batch_size, num_replicas=None, rank=None, shuffle=True, seed=42):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset_lengths = dataset_lengths
        self.min_len = min(dataset_lengths)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.num_datasets = len(dataset_lengths)

        # How many full distributed batches we can get per dataset
        self.num_batches_per_dataset = self.min_len // (self.batch_size * self.num_replicas)
        self.total_batches = self.num_batches_per_dataset * self.num_datasets

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        dataset_indices = []
        current_offset = 0
        
        for d_len in self.dataset_lengths:
            # 1. Generate shuffled indices for this specific dataset
            if self.shuffle:
                indices = torch.randperm(d_len, generator=g).tolist()
            else:
                indices = list(range(d_len))
                
            # 2. Downsample to min_len so all datasets match size
            indices = indices[:self.min_len]
            
            # 3. Offset indices to match ConcatDataset mapping
            indices = [i + current_offset for i in indices]
            dataset_indices.append(indices)
            
            current_offset += d_len

        # 4. Yield batches alternating between datasets (Round Robin)
        for b in range(self.num_batches_per_dataset):
            for d_idx in range(self.num_datasets):
                start = b * self.batch_size * self.num_replicas
                end = start + self.batch_size * self.num_replicas
                
                # The global batch for ALL GPUs
                global_batch = dataset_indices[d_idx][start:end]
                
                # The local batch for THIS GPU
                local_batch = global_batch[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                
                yield local_batch

    def __len__(self):
        return self.total_batches

    def set_epoch(self, epoch):
        self.epoch = epoch