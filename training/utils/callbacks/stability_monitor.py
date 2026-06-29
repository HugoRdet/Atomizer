"""
Numerical Stability Monitor — Lightning Callback
================================================

Per-step diagnostics to catch and localize training collapse, logged to wandb.
No model edits required: reads gradients (after backward, before optimizer step)
and parameters directly.

What it logs every step (prefix "diag/"):
  grad_norm_total          : global L2 grad norm (PRE-clip) — explosion detector
  grad_norm_<group>        : per-group grad norm (encoder/decoder/skip/head/
                             error_predictor/other) — localizes the bad gradient
  grad_nonfinite           : 1 if ANY grad is nan/inf else 0 — the smoking gun
  grad_max_abs             : largest single grad element (saturation precursor)
  weight_norm_total        : global L2 param norm — weight blow-up detector
  weight_nonfinite         : 1 if any param is nan/inf (poison has entered weights)
  loss_value               : the step loss (mirror, for alignment)
  loss_nonfinite           : 1 if loss is nan/inf

Why pre-clip grad norm: clipping caps magnitude but not direction. A clipped-
but-wrong gradient can still push the model into the constant-prediction
attractor. The PRE-clip norm reveals the true spike that clipping may be hiding.

Reading the result in wandb:
  - grad_norm_total spikes at the collapse step  -> gradient-driven explosion;
    look at grad_norm_<group> to see WHERE it originates.
  - grad_norm_total FLAT but grad_nonfinite -> 1 -> a nan appeared in the
    backward (forward-pass numerical event), NOT an explosion. The group with
    the first nonfinite grad localizes it.
  - weight_nonfinite flips 1 and stays -> the exact step poison entered weights
    (collapse becomes irreversible right after).
  - everything finite, norms bounded, but loss/preds still collapse -> it's the
    loss-landscape attractor (constant prediction), not numerics -> the fix is
    LR / regularization / warmup, not precision.

Usage (in the launch script):
    from training.utils.callbacks.stability_monitor import StabilityMonitor
    callbacks = [..., StabilityMonitor(log_every_n_steps=1)]
"""

import torch
from pytorch_lightning.callbacks import Callback


# group a parameter by a coarse module bucket from its name
def _group_of(name: str) -> str:
    n = name.lower()
    if "error_predictor" in n:
        return "error_predictor"
    if "pixel_cross_attn" in n or "pixel_query" in n or "pixel_q_norm" in n:
        return "skip"
    if "reconstruction_head" in n or "to_logits" in n:
        return "head"
    if "decoder" in n or "global_query" in n or "dec_q_norm" in n:
        return "decoder"
    if "encoder_layers" in n or "spatial_latent" in n or "global_latents" in n \
            or "input_processor" in n or "mask_token" in n:
        return "encoder"
    return "other"


class StabilityMonitor(Callback):
    def __init__(self, log_every_n_steps: int = 1, only_rank0: bool = True):
        super().__init__()
        self.every = max(1, int(log_every_n_steps))
        self.only_rank0 = only_rank0

    # -- gradients: available after backward, before optimizer step --------
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.every != 0:
            return
        if self.only_rank0 and getattr(trainer, "global_rank", 0) != 0:
            return

        logs = {}
        total_sq = 0.0
        group_sq = {}
        max_abs = 0.0
        any_nonfinite = 0

        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            finite = torch.isfinite(g)
            if not bool(finite.all()):
                any_nonfinite = 1
                # use nan-safe values for the norm so one nan doesn't blank everything
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            gn = g.float().norm(2)
            gn2 = float(gn.item() ** 2)
            total_sq += gn2
            grp = _group_of(name)
            group_sq[grp] = group_sq.get(grp, 0.0) + gn2
            m = float(g.abs().max().item()) if g.numel() else 0.0
            if m > max_abs:
                max_abs = m

        logs["diag/grad_norm_total"] = total_sq ** 0.5
        logs["diag/grad_max_abs"]    = max_abs
        logs["diag/grad_nonfinite"]  = float(any_nonfinite)
        for grp, s in group_sq.items():
            logs[f"diag/grad_norm_{grp}"] = s ** 0.5

        # -- weights (total + per-group) ------------------------------------
        # Per-group weight norm names WHICH component's weights are climbing
        # (encoder/decoder/skip/head/error_predictor/other). The total alone
        # can't localize a runaway; the per-group split does.
        w_sq = 0.0
        w_nonfinite = 0
        w_group_sq = {}
        w_group_max = {}   # max |weight| per group — catches a single blowing-up matrix
        for name, p in pl_module.named_parameters():
            pf = p.detach()
            if not bool(torch.isfinite(pf).all()):
                w_nonfinite = 1
                pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
            pn2 = float(pf.float().norm(2).item() ** 2)
            w_sq += pn2
            grp = _group_of(name)
            w_group_sq[grp] = w_group_sq.get(grp, 0.0) + pn2
            pmax = float(pf.abs().max().item()) if pf.numel() else 0.0
            if pmax > w_group_max.get(grp, 0.0):
                w_group_max[grp] = pmax
        logs["diag/weight_norm_total"] = w_sq ** 0.5
        logs["diag/weight_nonfinite"]  = float(w_nonfinite)
        for grp, s in w_group_sq.items():
            logs[f"diag/weight_norm_{grp}"] = s ** 0.5
        for grp, mx in w_group_max.items():
            logs[f"diag/weight_max_{grp}"] = mx

        # log to whatever logger is attached (wandb)
        for k, v in logs.items():
            pl_module.log(k, v, on_step=True, on_epoch=False,
                          prog_bar=False, sync_dist=False, rank_zero_only=True)

    # -- loss finiteness: read the step output ----------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every != 0:
            return
        if self.only_rank0 and getattr(trainer, "global_rank", 0) != 0:
            return

        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif torch.is_tensor(outputs):
            loss = outputs
        if loss is None:
            return
        lv = loss.detach()
        finite = bool(torch.isfinite(lv).all())
        pl_module.log("diag/loss_value",
                      float(torch.nan_to_num(lv).item()),
                      on_step=True, on_epoch=False, rank_zero_only=True)
        pl_module.log("diag/loss_nonfinite", 0.0 if finite else 1.0,
                      on_step=True, on_epoch=False, rank_zero_only=True)
