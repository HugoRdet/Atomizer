"""
distill_losses.py -- DUNE's teacher-feature standardization and distillation loss, for the Atomiser distillation.

Where it goes: training/distill/distill_losses.py

Re-implemented from the behaviour of DUNE's model/teacher_norm.py, model/losses.py and model/teacher_dropping.py
(read from the files you attached; written independently, not copied -- check the DUNE license before copying code).

What DUNE does (and this file reproduces):
  * TeacherNorm (patch tokens): per-channel standardization over the batch AND the tokens, eps 1e-3.
      - training with momentum m > 0: normalize with the CURRENT batch statistics (gathered across GPUs), then update
        the buffers  mean <- (1-m) mean + m batch_mean,  std <- (1-m) std + m batch_std;
      - evaluation, or m == 0: normalize with the stored (EMA) statistics.
    m follows a cosine schedule from 1.0 to 0.001 over training (distill_schedule.tnorm_momentum_schedule).
  * Loss per image and teacher:  mean over tokens of ( 0.5 * (1 - cos) + 0.5 * smoothL1(beta=1, mean over channels) ).
    (the code uses 0.5 / 0.5; the paper's equation is written with weights 1 / 1).
  * Aggregation over (image, teacher) pairs: coefficients are divided by their SUM, so the loss is an average over the
    active pairs. 'own_data' = a teacher only counts on its own dataset's images (our no-sharing case);
    'lowest_loss' = UNIC teacher dropping: per image the highest-loss active teacher always stays, the others are
    dropped with probability p.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Concatenate x over all ranks (no gradient). Identity when not distributed. Needs equal shapes on all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    out = [torch.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, 0)


class PatchStandardizer(nn.Module):
    """DUNE's StandardNormalizer for patch tokens x [B, M, D] (statistics over B and M, per channel)."""

    def __init__(self, dim: int, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, ema_momentum: float = 0.0) -> torch.Tensor:
        assert x.dim() == 3, f"expected [B, M, D], got {tuple(x.shape)}"
        x = x.float()
        if (not self.training) or ema_momentum == 0:
            return (x - self.mean) / self.std.clamp(min=self.eps)
        xa = concat_all_gather(x)
        mean, std = xa.mean(dim=(0, 1)), xa.std(dim=(0, 1))
        out = (x - mean) / std.clamp(min=self.eps)
        self.mean.copy_(self.mean * (1 - ema_momentum) + mean * ema_momentum)
        self.std.copy_(self.std * (1 - ema_momentum) + std * ema_momentum)
        return out


def distill_loss_per_image(pred: torch.Tensor, target: torch.Tensor, lam_cos: float = 0.5, lam_sl1: float = 0.5):
    """pred, target [B, M, D] -> (loss per image [B], metrics). Same maths as DUNE's loss_per_teacher (patch tokens)."""
    pred, target = pred.float(), target.float()
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)                                # [B, M]
    sl1 = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0).mean(dim=-1)         # [B, M]
    loss = (lam_cos * cos + lam_sl1 * sl1).mean(dim=1)                                    # [B]
    var = target.var(dim=(0, 1), unbiased=False).mean().clamp_min(1e-12)
    fid = 1.0 - ((pred - target) ** 2).mean() / var                                       # 1 = perfect, ~0 = predicts the mean
    return loss, {"cos": cos.mean().detach(), "sl1": sl1.mean().detach(), "fidelity": fid.detach()}


def aggregate_teacher_losses(losses: dict, active: dict = None, method: str = "none", p: float = 0.5):
    """losses: {teacher: [B] per-image loss}. active: {teacher: bool [B]} images each teacher may use (own data);
    default all. Returns (scalar loss, {teacher: summed coefficient}) with DUNE's normalization (coeffs / their sum)."""
    names = sorted(losses)
    L = torch.stack([losses[n] for n in names])                                           # [T, B]
    A = torch.ones_like(L) if active is None else torch.stack([active[n].float() for n in names])
    coeff = A.clone()
    if method == "lowest_loss":
        with torch.no_grad():
            top = L.detach().masked_fill(A == 0, float("-inf")).argmax(0)                  # highest-loss active teacher
            keep_top = F.one_hot(top, len(names)).t().bool()
            coeff = A * (keep_top | (torch.rand_like(L) >= p)).float()
    elif method != "none":
        raise ValueError(f"unknown teacher-dropping method {method!r}")
    assert bool((coeff.sum(0) >= 1).all()), "an image is not assigned to any teacher"
    coeff = (coeff / coeff.sum()).detach()
    return (coeff * L).sum(), {n: float(c) for n, c in zip(names, coeff.sum(1))}


def resid_corr(a, b, k: int = 20) -> float:
    """Proxy for DUNE's 'correlation of loss updates' between two teachers: Pearson correlation of their loss series after
    removing a k-step moving average. Positive = the two losses rise and fall together. NaN while the series are short."""
    import numpy as np
    a, b = np.asarray(list(a), dtype=np.float64), np.asarray(list(b), dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    if n < 3 * k:
        return float("nan")
    ker = np.ones(k) / k
    ra = a[k - 1:] - np.convolve(a, ker, mode="valid")
    rb = b[k - 1:] - np.convolve(b, ker, mode="valid")
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    torch.manual_seed(0)
    # TeacherNorm: batch stats in training, EMA stats in eval
    n = PatchStandardizer(8)
    n.train()
    x = torch.randn(4, 100, 8) * 3 + 5
    y = n(x, ema_momentum=1.0)
    assert abs(float(y.mean())) < 1e-4 and abs(float(y.std()) - 1) < 0.02, "training output is batch-standardized"
    assert torch.allclose(n.mean, x.mean((0, 1)), atol=1e-5), "momentum 1.0 copies the batch statistics"
    n.eval()
    y2 = n(x, ema_momentum=0.5)
    assert torch.allclose(y, y2, atol=1e-3), "eval uses the stored statistics (== batch stats here)"
    # loss
    t = torch.randn(2, 10, 16)
    l, m = distill_loss_per_image(t.clone(), t)
    assert float(l.abs().max()) < 1e-6 and abs(float(m["fidelity"]) - 1) < 1e-5
    l, m = distill_loss_per_image(torch.randn_like(t), t)
    assert l.shape == (2,) and float(l.min()) > 0.1
    # aggregation: own_data -> plain average of own losses
    la, lb = torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0])
    both = {"a": torch.cat([la, torch.zeros(2)]), "b": torch.cat([torch.zeros(2), lb])}
    act = {"a": torch.tensor([1, 1, 0, 0]).bool(), "b": torch.tensor([0, 0, 1, 1]).bool()}
    agg, coef = aggregate_teacher_losses(both, act, "none")
    assert abs(float(agg) - 4.0) < 1e-6 and abs(coef["a"] - 0.5) < 1e-6, (float(agg), coef)
    # lowest_loss: the highest-loss teacher of each image is never dropped
    for _ in range(50):
        _, coef = aggregate_teacher_losses({"a": torch.tensor([1.0, 9.0]), "b": torch.tensor([5.0, 2.0])}, None, "lowest_loss", p=1.0)
        assert abs(sum(coef.values()) - 1.0) < 1e-6
    print("distill_losses self-test OK")
