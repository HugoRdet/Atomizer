"""
distill_optim.py -- optimizer helpers matching DUNE's training (utils/optim.py + main_dune.py behaviour).

Where it goes: training/distill/distill_optim.py

Behaviour (read from DUNE's utils/optim.py and main_dune.py; written independently, not copied -- the DUNE repo has its
own license file, check it before copying code verbatim):
  * two parameter groups: weight decay on everything except biases and 1-D parameters (norm scales, seeds, ...);
    DUNE sets the decay to `wd` on the FIRST group only each step; the second group keeps 0;
  * AdamW(betas=(0.9, 0.99), eps=1e-15), lr set every iteration from the schedule (the optimizer is built with lr=0);
  * gradient clipping is PER PARAMETER (each parameter's gradient norm is clipped to `clip`, 1.0), not the global
    norm of torch.nn.utils.clip_grad_norm_. It returns the per-parameter norms for logging (mean/std/max/min).
"""
import torch


def param_groups(model):
    """[{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]"""
    reg, noreg = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (noreg if (name.endswith(".bias") or p.ndim == 1) else reg).append(p)
    return [{"params": reg}, {"params": noreg, "weight_decay": 0.0}]


def make_optimizer(model, betas=(0.9, 0.99), eps=1e-15, fused=None):
    if fused is None:
        fused = torch.cuda.is_available()
    return torch.optim.AdamW(param_groups(model), lr=0.0, betas=betas, eps=eps, fused=fused)


def set_lr_wd(optimizer, lr, wd):
    """Per-iteration update as in DUNE: lr on every group, weight decay `wd` on group 0 only."""
    for i, g in enumerate(optimizer.param_groups):
        g["lr"] = lr
        if i == 0:
            g["weight_decay"] = wd


@torch.no_grad()
def clip_per_parameter(parameters, clip=1.0):
    """Clip each parameter's gradient to L2 norm `clip` separately. Returns the stacked per-parameter norms
    (before clipping)."""
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        n = p.grad.norm(p=2)
        norms.append(n)
        coef = clip / (n + 1e-6)
        if coef < 1:
            p.grad.mul_(coef)
    return torch.stack(norms) if norms else torch.zeros(1)
