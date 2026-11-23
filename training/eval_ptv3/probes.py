"""Heads trained on top of frozen dense features [B, D, H, W]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSegProbe(nn.Module):
    """OlmoEarth-style linear probe: one Linear per pixel (1x1 conv). Measures feature quality directly."""
    def __init__(self, d, k):
        super().__init__(); self.norm = nn.BatchNorm2d(d, affine=False); self.fc = nn.Conv2d(d, k, 1)
    def forward(self, f): return self.fc(self.norm(f))


class LightSegDecoder(nn.Module):
    """PANGAEA-like decoder stand-in. PANGAEA uses UPerNet on multi-scale ViT features; our dec0 features are
    already full-resolution, so a small multi-dilation conv head plays the same role (context aggregation)."""
    def __init__(self, d, k, w=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(d, w, 1), nn.BatchNorm2d(w), nn.GELU())
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(w, w, 3, padding=r, dilation=r),
                                                     nn.BatchNorm2d(w), nn.GELU()) for r in (1, 4, 12)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(w, w, 1), nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(5 * w, w, 3, padding=1), nn.BatchNorm2d(w), nn.GELU(), nn.Conv2d(w, k, 1))
    def forward(self, f):
        x = self.inp(f)
        g = self.pool(x).expand_as(x)
        return self.out(torch.cat([x, g] + [b(x) for b in self.branches], 1))


class TemporalAggregator(nn.Module):
    """Per-pixel aggregation of T frozen frames (PANGAEA multi-temporal: L-TAE or linear mapping).
    mode: "mean" | "attn" (light L-TAE: learned query, day-of-year encoding, padded frames masked)."""
    def __init__(self, d, mode="attn", heads=4):
        super().__init__(); self.mode = mode
        if mode == "attn":
            self.q = nn.Parameter(torch.zeros(1, 1, d)); self.norm = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
    @staticmethod
    def doy_enc(doy, d):
        i = torch.arange(d // 2, device=doy.device)
        a = doy[..., None].float() / (1000 ** (2 * i / d))
        return torch.cat([a.sin(), a.cos()], -1)
    def forward(self, f, doy=None, mask=None):   # f [B, T, D, H, W], mask [B, T] True = padded
        B, T, D, H, W = f.shape
        if self.mode == "mean":
            if mask is None: return f.mean(1)
            w = (~mask).float()[..., None, None, None]
            return (f * w).sum(1) / w.sum(1).clamp(min=1)
        x = f.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        if doy is not None:
            x = x + self.doy_enc(doy, D).repeat_interleave(H * W, 0)
        kpm = None if mask is None else mask.repeat_interleave(H * W, 0)
        o, _ = self.attn(self.q.expand(x.shape[0], -1, -1), self.norm(x), self.norm(x), key_padding_mask=kpm)
        return o.reshape(B, H, W, D).permute(0, 3, 1, 2)


class LinearClsProbe(nn.Module):
    def __init__(self, d, k):
        super().__init__(); self.norm = nn.BatchNorm1d(d, affine=False); self.fc = nn.Linear(d, k)
    def forward(self, f): return self.fc(self.norm(f.mean((2, 3)) if f.dim() == 4 else f))


@torch.no_grad()
def knn_classify(train_f, train_y, test_f, k=20, T=0.07, n_cls=None):
    """Weighted kNN on L2-normalized pooled features (DINO / OlmoEarth convention)."""
    train_f, test_f = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    n_cls = n_cls or int(train_y.max()) + 1
    preds = []
    for chunk in test_f.split(1024):
        sim, idx = (chunk @ train_f.T).topk(k, dim=1)
        votes = torch.zeros(chunk.shape[0], n_cls, device=chunk.device)
        votes.scatter_add_(1, train_y[idx], (sim / T).exp())
        preds.append(votes.argmax(1))
    return torch.cat(preds)
