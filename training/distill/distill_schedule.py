"""
distill_schedule.py -- DUNE's training schedule (from main_dune.py), numpy only.

Where it goes: training/distill/distill_schedule.py

DUNE (main_dune.py):
  * an "epoch" is epoch_size IMAGES (default 1,281,167, the size of ImageNet-1K); 100 epochs for pre-training;
  * batch_size = images per step summed over teachers; iters = ceil(epochs * epoch_size / batch_size), made even;
  * peak lr = lr * batch_size / 256, linear warmup over 20% of the iterations, cosine decay to min_lr (1e-6);
  * AdamW(betas=(0.9, 0.99), eps=1e-15), weight decay 0.03, gradient clipping 1.0, bf16 autocast;
  * teacher-feature statistics EMA momentum: cosine schedule from 1.0 to 0.001 (no warmup).
The cosine scheduler below is written from the behaviour of DUNE's utils/exp.py cosine_scheduler (checked against it:
warmup_iters = floor(warmup_frac * iters), linspace(0, base, warmup_iters), then cosine over the remaining iterations).
"""
import math

import numpy as np

DUNE_DEFAULTS = dict(lr=3e-4, min_lr=1e-6, wd=0.03, clip=1.0, betas=(0.9, 0.99), eps=1e-15,
                     warmup_frac=0.2, tnorm_momentum_start=1.0, tnorm_momentum_end=0.001)


def num_iters(epochs: float, epoch_size: int, batch_size: int) -> int:
    it = math.ceil(epochs * epoch_size / batch_size)
    return it + (it % 2)                       # DUNE rounds up to an even number


def cosine_scheduler(base: float, final: float, iters: int, warmup_iters: int = 0, start_warmup: float = 0.0):
    """Array of length `iters`: linear warmup start_warmup -> base over warmup_iters, then cosine decay base -> final."""
    warm = np.linspace(start_warmup, base, warmup_iters) if warmup_iters > 0 else np.array([])
    n = iters - warmup_iters
    t = np.arange(n)
    decay = final + 0.5 * (base - final) * (1 + np.cos(np.pi * t / n))
    return np.concatenate([warm, decay])


def dune_lr_schedule(lr: float, batch_size: int, epochs: float, epoch_size: int,
                     min_lr: float = 1e-6, warmup_frac: float = 0.2):
    """Returns (schedule array of per-iteration lr, iters). Peak lr = lr * batch_size / 256."""
    iters = num_iters(epochs, epoch_size, batch_size)
    return cosine_scheduler(lr * batch_size / 256.0, min_lr, iters, int(warmup_frac * iters)), iters


def tnorm_momentum_schedule(iters: int, start: float = 1.0, end: float = 0.001):
    return cosine_scheduler(start, end, iters, 0)
