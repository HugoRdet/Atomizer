"""
PretrainDataModule — Multi-Task Round-Robin Training (DDP-safe)
================================================================

Creates datasets from MMEarth and FLAIR-HUB and interleaves
them in chunks within a single DataLoader.

Supported tasks:
    - esa_worldcover:   MMEarth ESA WorldCover segmentation (11 classes)
    - dynamic_world:    MMEarth Dynamic World segmentation (9 classes)
    - reconstruction:   MMEarth reconstruction (MSE)
    - flairhub_cosia:   FLAIR-HUB COSIA land cover segmentation (18 classes)
    - flairhub_lpis:    FLAIR-HUB LPIS crop type segmentation (23 classes)
    - flairhub_recon:   FLAIR-HUB reconstruction (MSE, shares "reconstruction" head)

DDP handling:
    chunk_size = batch_size * world_size, so each chunk contains
    enough samples for ALL ranks. Both ranks always process the
    same task at the same step → no gradient sync deadlock.

    CRITICAL DDP safety:
        1. task_names sorted alphabetically → deterministic order across ranks
        2. _build_indices uses fixed seed → identical within-task orderings
        3. DistributedChunkSampler uses fixed seed → identical chunk order
        4. set_epoch propagates: sampler → dataset → _build_indices

IMPORTANT: Use with Trainer(use_distributed_sampler=False).

Usage:
    dm = PretrainDataModule(
        mmearth_path="./data/MM-Earth",
        flairhub_path="./data/FLAIR-HUB/toy/FLAIR-HUB_TOY",
        enable_mmearth=True,
        enable_flairhub=True,
        ...
    )
    trainer = Trainer(use_distributed_sampler=False, ...)
    trainer.fit(model, datamodule=dm)
"""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl

from training.utils.datasets.utils_dataset_MM_Earth_pretrain import (
    MMEarthSegESA, MMEarthSegDW, MMEarthReconstruction,
)
from training.utils.datasets.utils_dataset_FLAIRHUB import (
    FlairHubSegCOSIA, FlairHubSegLPIS, FlairHubReconstruction,
)
from training.utils.datasets.token_grouping import collate_grouped
from training.utils import read_yaml


# =============================================================================
# WORKER INIT (deterministic across ranks)
# =============================================================================

def _worker_init_fn(worker_id):
    """
    Ensure DataLoader workers have deterministic but distinct seeds.
    Each worker gets: base_seed + worker_id.
    base_seed is the same across ranks (from PyTorch's default seeding).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =============================================================================
# DISTRIBUTED CHUNK SAMPLER
# =============================================================================

class DistributedChunkSampler(Sampler):
    """
    Both ranks iterate through the SAME chunks in the SAME order.
    Each rank takes its slice of batch_size samples from each chunk.

    chunk_size = batch_size * world_size

    For chunk i, rank r yields indices:
        [i * chunk_size + r * batch_size,
         i * chunk_size + (r+1) * batch_size)

    This guarantees all ranks process the same task at every step.

    DDP safety: chunk ordering uses a deterministic seed (epoch + 42)
    so all ranks produce identical chunk permutations.
    """

    def __init__(self, dataset, chunk_size, batch_size,
                 rank=None, world_size=None, shuffle_chunks=True):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle_chunks = shuffle_chunks
        self.epoch = 0

        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = rank
        self.world_size = world_size

        # Number of complete chunks
        self.num_chunks = len(dataset) // chunk_size

        print(f"[DistributedChunkSampler] rank={rank}/{world_size}, "
              f"chunk_size={chunk_size}, batch_size={batch_size}, "
              f"num_chunks={self.num_chunks}, "
              f"samples_per_rank={self.num_chunks * batch_size}")

    def __iter__(self):
        chunk_order = list(range(self.num_chunks))

        if self.shuffle_chunks:
            # CRITICAL: Same seed on ALL ranks → identical chunk order
            g = torch.Generator()
            g.manual_seed(self.epoch + 42)
            perm = torch.randperm(len(chunk_order), generator=g).tolist()
            chunk_order = [chunk_order[i] for i in perm]

        # All ranks iterate the SAME chunk order
        # Each rank takes its own slice within each chunk
        for chunk_idx in chunk_order:
            start = chunk_idx * self.chunk_size + self.rank * self.batch_size
            for i in range(start, start + self.batch_size):
                yield i

    def __len__(self):
        return self.num_chunks * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
        # Propagate to dataset for reshuffling within-task indices
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)


# =============================================================================
# CHUNKED INTERLEAVED DATASET
# =============================================================================

class ChunkedInterleavedDataset(Dataset):
    """
    Yields samples from multiple tasks in chunks.

    Layout (chunk_size=8, 3 tasks):
        chunk 0 → ESA   samples [0, 8)
        chunk 1 → DW    samples [0, 8)
        chunk 2 → REC   samples [0, 8)
        chunk 3 → ESA   samples [8, 16)
        chunk 4 → DW    samples [8, 16)
        chunk 5 → REC   samples [8, 16)
        ...

    All tasks truncated to equal length. Within-task order randomized.

    DDP safety: task_names are SORTED so all ranks have identical
    ordering. _build_indices uses a FIXED SEED so all ranks produce
    identical within-task sample orderings.
    """

    def __init__(self, datasets: dict, chunk_size: int):
        self.datasets = datasets
        # CRITICAL: Sort task names for deterministic ordering across ranks
        self.task_names = sorted(datasets.keys())
        self.num_tasks = len(self.task_names)
        self.chunk_size = chunk_size
        self.epoch = 0

        self.min_len = min(len(d) for d in datasets.values())
        self.chunks_per_task = self.min_len // chunk_size
        self.samples_per_task = self.chunks_per_task * chunk_size
        self.total_len = self.samples_per_task * self.num_tasks

        self._build_indices()

        print(f"[InterleavedDataset] {self.num_tasks} tasks: {self.task_names}")
        print(f"[InterleavedDataset] chunk_size={chunk_size}, "
              f"min_len={self.min_len}, "
              f"chunks_per_task={self.chunks_per_task}, "
              f"samples_per_task={self.samples_per_task}, "
              f"total={self.total_len}")

    def _build_indices(self):
        """
        Build shuffled sample indices per task.

        CRITICAL for DDP: Uses a FIXED SEED so all ranks produce
        identical index orderings. Different epochs get different
        orderings via the epoch-based seed.
        """
        self.task_indices = {}
        for i, task_name in enumerate(self.task_names):
            dataset = self.datasets[task_name]
            # Deterministic seed: unique per task + epoch, same across ranks
            g = torch.Generator()
            g.manual_seed(self.epoch * 1000 + i + 7)
            indices = torch.randperm(len(dataset), generator=g)[:self.samples_per_task].tolist()
            self.task_indices[task_name] = indices

    def set_epoch(self, epoch):
        """Update epoch and rebuild indices with new seed."""
        self.epoch = epoch
        self._build_indices()

    def reshuffle(self):
        """Legacy alias — use set_epoch instead."""
        self._build_indices()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        chunk_idx = index // self.chunk_size
        pos_in_chunk = index % self.chunk_size

        task_idx = chunk_idx % self.num_tasks
        round_idx = chunk_idx // self.num_tasks

        task_name = self.task_names[task_idx]
        sample_idx = self.task_indices[task_name][round_idx * self.chunk_size + pos_in_chunk]

        return self.datasets[task_name][sample_idx]


# =============================================================================
# DATAMODULE
# =============================================================================

class PretrainDataModule(pl.LightningDataModule):
    """
    Multi-task datamodule for Atomizer pre-training.

    Supports MMEarth and FLAIR-HUB datasets, individually or combined.

    chunk_size is automatically set to batch_size * world_size
    so that each chunk can be split across DDP ranks while
    keeping all ranks on the same task.

    IMPORTANT: Use with Trainer(use_distributed_sampler=False).

    Args:
        mmearth_path: Path to MMEarth data directory.
        flairhub_path: Path to FLAIR-HUB data directory.
        enable_mmearth: Whether to include MMEarth tasks.
        enable_flairhub: Whether to include FLAIR-HUB tasks.
        flairhub_tasks: Which FLAIR-HUB tasks to include.
            Options: "cosia", "lpis", "recon", or list of these.
            Default: ["cosia", "lpis", "recon"]
    """

    def __init__(
        self,
        # Paths
        mmearth_path: str = "./data/MM-Earth",
        flairhub_path: str = "./data/FLAIR-HUB/FLAIR-HUB_TOY",
        bands_yaml_path: str = "./data/bands_info/bands.yaml",
        config_model: dict = None,
        look_up=None,
        # Toggles
        enable_mmearth: bool = True,
        enable_flairhub: bool = False,
        flairhub_tasks: list = None,
        # Common
        batch_size: int = 4,
        num_workers: int = 4,
        subset: str = "MMEarth",
        max_queries_recon: int = 200_000,
        # FLAIR-HUB specific
        flairhub_temporal_dropout: float = 0.3,
        flairhub_max_timestamps: int = 40,
        flairhub_csv_dir: str = None,
        # Legacy compat
        root_path: str = None,
    ):
        super().__init__()


        # Legacy: root_path maps to mmearth_path
        if root_path is not None:
            mmearth_path = root_path

        self.mmearth_path = mmearth_path
        self.flairhub_path = flairhub_path
        self.bands_yaml_path = bands_yaml_path
        self.config_model = config_model
        self.look_up = look_up
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset = subset
        self.max_queries_recon = max_queries_recon
        self.enable_mmearth = enable_mmearth
        self.enable_flairhub = enable_flairhub
        self.flairhub_temporal_dropout = flairhub_temporal_dropout
        self.flairhub_max_timestamps = flairhub_max_timestamps
        # CSV dir defaults to flairhub_path (CSVs alongside data)
        self.flairhub_csv_dir = flairhub_csv_dir or flairhub_path

        # Default FLAIR-HUB tasks
        if flairhub_tasks is None:
            self.flairhub_tasks = ["cosia", "lpis", "recon"]
        elif isinstance(flairhub_tasks, str):
            self.flairhub_tasks = [flairhub_tasks]
        else:
            self.flairhub_tasks = flairhub_tasks

        # chunk_size = batch_size * world_size (set in setup)
        self.chunk_size = None
        self.dataset_config = read_yaml(bands_yaml_path)

        if not enable_mmearth and not enable_flairhub:
            raise ValueError("At least one of enable_mmearth or enable_flairhub must be True")

    def _get_world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    # =========================================================================
    # MMEARTH DATASET CREATION
    # =========================================================================

    def _make_mmearth_dataset(self, cls, mode, **extra_kwargs):
        return cls(
            root_path=self.mmearth_path,
            mode=mode,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up,
            subset=self.subset,
            **extra_kwargs,
        )

    # =========================================================================
    # FLAIR-HUB DATASET CREATION
    # =========================================================================

    def _make_flairhub_dataset(self, cls, mode, **extra_kwargs):
        return cls(
            root_path=self.flairhub_path,
            mode=mode,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up,
            temporal_dropout=self.flairhub_temporal_dropout if mode == "train" else 0.0,
            max_timestamps=self.flairhub_max_timestamps,
            csv_dir=self.flairhub_csv_dir,
            **extra_kwargs,
        )

    # =========================================================================
    # SETUP
    # =========================================================================

    def setup(self, stage=None):
        world_size = self._get_world_size()
        self.chunk_size = self.batch_size * world_size

        print(f"[PretrainDM] batch_size={self.batch_size}, "
              f"world_size={world_size}, "
              f"chunk_size={self.chunk_size}")
        print(f"[PretrainDM] MMEarth={self.enable_mmearth}, "
              f"FLAIR-HUB={self.enable_flairhub} "
              f"(tasks={self.flairhub_tasks})")

        # =====================================================================
        # MMEARTH DATASETS
        # =====================================================================
        train_datasets = {}
        val_datasets = {}

        if self.enable_mmearth:
            self.train_dataset_esa = self._make_mmearth_dataset(MMEarthSegESA, "train")
            self.train_dataset_dw = self._make_mmearth_dataset(MMEarthSegDW, "train")
            self.train_dataset_recon = self._make_mmearth_dataset(
                MMEarthReconstruction, "train",
                max_queries=self.max_queries_recon,
            )

            # Split 1% of MMEarth train as val
            val_fraction = 0.01
            full_len = len(self.train_dataset_esa)
            val_len = max(self.chunk_size, int(full_len * val_fraction))
            train_len = full_len - val_len

            generator = torch.Generator().manual_seed(42)
            all_indices = torch.randperm(full_len, generator=generator).tolist()
            train_idx = all_indices[:train_len]
            val_idx = all_indices[train_len:]

            # Override tile_indices on train datasets
            for ds in [self.train_dataset_esa, self.train_dataset_dw,
                       self.train_dataset_recon]:
                original_tiles = ds.tile_indices
                ds.tile_indices = [original_tiles[i] for i in train_idx]

            # Create val datasets with val indices
            self.val_dataset_esa = self._make_mmearth_dataset(MMEarthSegESA, "train")
            self.val_dataset_dw = self._make_mmearth_dataset(MMEarthSegDW, "train")
            self.val_dataset_recon = self._make_mmearth_dataset(
                MMEarthReconstruction, "train",
                max_queries=self.max_queries_recon,
            )
            for ds in [self.val_dataset_esa, self.val_dataset_dw,
                       self.val_dataset_recon]:
                original_tiles = ds.tile_indices
                ds.tile_indices = [original_tiles[i] for i in val_idx]

            train_datasets["esa_worldcover"] = self.train_dataset_esa
            train_datasets["dynamic_world"] = self.train_dataset_dw
            train_datasets["reconstruction"] = self.train_dataset_recon

            val_datasets["esa_worldcover"] = self.val_dataset_esa
            val_datasets["dynamic_world"] = self.val_dataset_dw
            val_datasets["reconstruction"] = self.val_dataset_recon

            print(f"[PretrainDM] MMEarth split: train={train_len}, val={val_len}")
            print(f"[PretrainDM]   ESA={len(self.train_dataset_esa)}, "
                  f"DW={len(self.train_dataset_dw)}, "
                  f"Recon={len(self.train_dataset_recon)}")

        # =====================================================================
        # FLAIR-HUB DATASETS
        # =====================================================================
        if self.enable_flairhub:
            # FLAIR-HUB handles train/val split internally via CSVs

            if "cosia" in self.flairhub_tasks:
                self.train_dataset_cosia = self._make_flairhub_dataset(
                    FlairHubSegCOSIA, "train"
                )
                self.val_dataset_cosia = self._make_flairhub_dataset(
                    FlairHubSegCOSIA, "validation"
                )
                train_datasets["flairhub_cosia"] = self.train_dataset_cosia
                val_datasets["flairhub_cosia"] = self.val_dataset_cosia

            if "lpis" in self.flairhub_tasks:
                self.train_dataset_lpis = self._make_flairhub_dataset(
                    FlairHubSegLPIS, "train"
                )
                self.val_dataset_lpis = self._make_flairhub_dataset(
                    FlairHubSegLPIS, "validation"
                )
                train_datasets["flairhub_lpis"] = self.train_dataset_lpis
                val_datasets["flairhub_lpis"] = self.val_dataset_lpis

            if "recon" in self.flairhub_tasks:
                self.train_dataset_flair_recon = self._make_flairhub_dataset(
                    FlairHubReconstruction, "train",
                    max_queries=self.max_queries_recon,
                )
                self.val_dataset_flair_recon = self._make_flairhub_dataset(
                    FlairHubReconstruction, "validation",
                    max_queries=self.max_queries_recon,
                )
                # FLAIR-HUB recon uses same head as MMEarth recon.
                # If MMEarth recon is also enabled, merge via ConcatDataset.
                if "reconstruction" in train_datasets:
                    from torch.utils.data import ConcatDataset
                    train_datasets["reconstruction"] = ConcatDataset([
                        train_datasets["reconstruction"],
                        self.train_dataset_flair_recon,
                    ])
                    val_datasets["reconstruction"] = ConcatDataset([
                        val_datasets["reconstruction"],
                        self.val_dataset_flair_recon,
                    ])
                    print(f"[PretrainDM] Merged FLAIR-HUB recon into MMEarth recon "
                          f"(train={len(train_datasets['reconstruction'])})")
                else:
                    train_datasets["reconstruction"] = self.train_dataset_flair_recon
                    val_datasets["reconstruction"] = self.val_dataset_flair_recon

            # Log FLAIR-HUB dataset sizes
            for key in ["flairhub_cosia", "flairhub_lpis"]:
                if key in train_datasets:
                    print(f"[PretrainDM]   {key}: train={len(train_datasets[key])}, "
                          f"val={len(val_datasets[key])}")

        # =====================================================================
        # INTERLEAVED WRAPPERS
        # =====================================================================
        if not train_datasets:
            raise RuntimeError("No datasets were created. Check enable flags and paths.")

        self.train_dataset = ChunkedInterleavedDataset(
            datasets=train_datasets,
            chunk_size=self.chunk_size,
        )

        self.val_dataset = ChunkedInterleavedDataset(
            datasets=val_datasets,
            chunk_size=self.chunk_size,
        )

        print(f"[PretrainDM] Total tasks: {self.train_dataset.task_names}")
        print(f"[PretrainDM] Interleaved train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}")

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    def _make_loader(self, dataset, shuffle_chunks=False):
        sampler = None
        if isinstance(dataset, ChunkedInterleavedDataset):
            sampler = DistributedChunkSampler(
                dataset,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                shuffle_chunks=shuffle_chunks,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # sampler handles ordering
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_grouped,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            worker_init_fn=_worker_init_fn,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle_chunks=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle_chunks=False)

    def test_dataloader(self):
        return self.val_dataloader()
    

