"""
distill_heads.py (v2) -- teacher-specific heads + student wrapper for DUNE-style distillation of Atomiser_Unet.

Where it goes: training/distill/distill_heads.py   (replaces the first version)

Changes vs v1, from reading DUNE's projector / options / forward files and your TokenBuilder:
  * DUNE's transformer projector is ONE ViT block with LayerScale (init 1e-4, so it starts near the identity) followed by
    a Linear. Our head now has the same two stages: (1) `n_pre` blocks of local self-attention among the level-0 latents
    (the existing LocalBlock, RoPE on cell anchors) behind a per-channel gate initialised at 1e-4, then (2) the query
    read: cross-attention from each query position to its k nearest latents + FFN + Linear to the teacher dimension.
    The raw-atom QueryDecoder is still not used (it could bypass the encoder).
  * token_centers_from_tokens: teacher token centres from the TokenBuilder layout (channel-major tokens; x = column,
    y = row; positions are pixel indices on the centred 512 reference grid + a per-resolution offset).
  * The loss lives in distill_losses.py (DUNE: 0.5 cos + 0.5 smooth-L1, per image).

Conventions (checked against Atomiser_unet.py and token_builder.py): tokens [B, N, 8] =
[value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]; mask True = invalid atom.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

try:
    from training.atomiser.Atomiser_unet import Atomiser_Unet, FFN, LocalBlock, fourier_2d
except ImportError:  # local testing
    from Atomiser_unet import Atomiser_Unet, FFN, LocalBlock, fourier_2d


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
def token_centers_from_tokens(tokens: torch.Tensor, H: int, W: int, patch: int) -> torch.Tensor:
    """Centres of the teacher's tokens, in the SAME units as the atoms' x, y columns.

    tokens [B, N, 8] built by TokenBuilder.build_tokens for an image [C, H, W]: channel-major, so the first H*W tokens
    are band 0 with x = column index (+ offsets) and y = row index (+ offsets). The centre of patch (a, b) is the mean of
    the x, y over its patch x patch pixels. Returns [B, (H/patch) * (W/patch), 2] = (x, y), row-major over the teacher's
    token grid (same order as the teacher's [g, g, D] map flattened to [g*g, D])."""
    assert H % patch == 0 and W % patch == 0, f"{H}x{W} not divisible by patch {patch}"
    B = tokens.shape[0]
    x = tokens[:, : H * W, 1].reshape(B, H, W).float()
    y = tokens[:, : H * W, 2].reshape(B, H, W).float()
    pool = lambda a: a.reshape(B, H // patch, patch, W // patch, patch).mean(dim=(2, 4))  # noqa: E731
    return torch.stack([pool(x), pool(y)], dim=-1).reshape(B, -1, 2)


class _GatedLocalBlock(nn.Module):
    """LocalBlock (self-attention among neighbouring cells + FFN) added through a per-channel gate initialised at 1e-4:
    x + gamma * (block(x) - x). DUNE's LayerScale (init 1e-4) makes its projector start near the identity."""

    def __init__(self, dim: int, heads: int, init: float = 1e-4):
        super().__init__()
        self.blk = LocalBlock(dim, heads)
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x, idx, coords):
        return x + self.gamma * (self.blk(x, idx, coords) - x)


# ---------------------------------------------------------------------------------------------------------
# teacher head (DUNE projector analogue): sees encoder latents only
# ---------------------------------------------------------------------------------------------------------
class LatentQueryHead(nn.Module):
    def __init__(self, dim: int, qdim: int, out_dim: int, heads: int = 4, k_nn: int = 9,
                 n_pre: int = 1, pre_heads: int | None = None):
        super().__init__()
        assert qdim % heads == 0, f"query width {qdim} not divisible by {heads} heads"
        self.h, self.dh, self.k_nn = heads, qdim // heads, k_nn
        ph = pre_heads or max(1, dim // 64)
        self.pre = nn.ModuleList(_GatedLocalBlock(dim, ph) for _ in range(n_pre))
        self.q_init = nn.Parameter(torch.randn(qdim) * 0.02)
        self.lat_ln = nn.LayerNorm(dim)
        self.lat_kv = nn.Linear(dim, 2 * qdim)                 # K/V projected once per latent
        self.pe_kv = nn.Linear(2 * (2 * 16 + 1), 2 * qdim)     # relative offset (cell units) added per edge
        self.q_ln = nn.LayerNorm(qdim)
        self.lat_q = nn.Linear(qdim, qdim)
        self.lat_o = nn.Linear(qdim, qdim)
        self.ffn = FFN(qdim)
        self.head = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, out_dim))

    def forward(self, d0, centers, scales, pq, local_idx=None, chunk: int = 4096):
        """d0 [B, L, D] latents; centers [L, 2], scales [L] of the level-0 cells; pq [B, M, 2] query positions in the same
        units as `centers`; local_idx [L, K] = geom.local_idx[0] (needed for the pre-blocks). Returns [B, M, out_dim]."""
        B, M, _ = pq.shape
        if len(self.pre) > 0:
            assert local_idx is not None, "pass geom.local_idx[0] (the head has pre-blocks)"
            coords = centers / scales.mean()                    # = AtomizerUNet.cell_coords(geom, 0, 0)
            for blk in self.pre:
                d0 = blk(d0, local_idx, coords)
        lat_k, lat_v = self.lat_kv(self.lat_ln(d0)).chunk(2, -1)
        bidx = torch.arange(B, device=pq.device)[:, None, None]
        outs = []
        for s in range(0, M, chunk):
            p = pq[:, s:s + chunk]
            m = p.shape[1]
            dist = torch.cdist(p.float(), centers.float().unsqueeze(0).expand(B, -1, -1))
            knn = dist.topk(min(self.k_nn, centers.shape[0]), dim=-1, largest=False).indices      # [B, m, k]
            off = (centers[knn] - p.unsqueeze(2)) / scales[knn].unsqueeze(-1)                       # [B, m, k, 2]
            pk, pv = self.pe_kv(fourier_2d(off.to(p.dtype))).chunk(2, -1)
            kk = (lat_k[bidx, knn] + pk).view(B, m, -1, self.h, self.dh)
            vv = (lat_v[bidx, knn] + pv).view(B, m, -1, self.h, self.dh)
            x = self.q_init.expand(B, m, -1)
            q = self.lat_q(self.q_ln(x)).view(B, m, self.h, self.dh)
            a = (torch.einsum("bmhd,bmkhd->bmkh", q, kk) / math.sqrt(self.dh)).softmax(dim=2)
            x = x + self.lat_o(torch.einsum("bmkh,bmkhd->bmhd", a, vv).reshape(B, m, -1))
            x = x + self.ffn(x)
            outs.append(self.head(x))
        return torch.cat(outs, 1)


# ---------------------------------------------------------------------------------------------------------
# student wrapper
# ---------------------------------------------------------------------------------------------------------
class DistillStudent(nn.Module):
    """Atomiser_Unet encoder (leaf read + U-Net) + one LatentQueryHead per teacher.

    teacher_dims: {"olmo": 768, "dino": 1024}. Build the config with unet_task: segmentation (so the decoder-side
    pieces exist), any positive trainer.num_classes (unused), unet_atom_features: legacy, unet_decoder: overlap."""

    def __init__(self, config, lookup_table, teacher_dims: dict, k_nn: int = 9, n_pre: int = 1):
        super().__init__()
        self.net = Atomiser_Unet(config, lookup_table)
        self.net.decoder = None   # the raw-atom QueryDecoder is not used: it could bypass the encoder
        a = config.get("Atomiser", {})
        qdim = int(a.get("unet_query_dim", 256))
        head_dim = int(a.get("unet_head_dim", 64))
        dim = self.net.unet.out_dim
        self.heads = nn.ModuleDict({name: LatentQueryHead(dim, qdim, int(d), qdim // head_dim, k_nn, n_pre)
                                    for name, d in teacher_dims.items()})
        n_enc = sum(p.numel() for p in self.net.parameters()) / 1e6
        n_head = sum(p.numel() for p in self.heads.parameters()) / 1e6
        print(f"[DistillStudent] encoder {n_enc:.1f}M params (kept) + heads {n_head:.1f}M (discarded)")

    def encode(self, batch, k: int | None = None, m: int | None = None):
        """Mirrors Atomiser_Unet.forward up to the decoded level-0 latents. Returns (d0 [B, L0, D], geom).
        k, m = atoms per leaf and max atoms read per leaf; any k works (the pruning geometry is cached per (N, k)).
        None: random choice from unet_km_train in training, unet_km_eval otherwise."""
        net = self.net
        tokens, mask = net._gather_inputs(batch)
        if k is None:
            k, m = random.choice(net.km_train) if self.training else net.km_eval
        tok_sel, _, bias = net.pruning(tokens, mask, None, geo_k=m, atoms_per_latent=k)
        geom = net.pruning.last_geometry
        z0 = net.leaf(tok_sel, bias, geom)
        z0 = z0 + net.empty_emb * net.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
        return net.unet(z0, geom), geom

    def predict(self, teacher: str, d0, geom, pq):
        """Student features projected to `teacher`'s space at query positions pq [B, M, 2]."""
        return self.heads[teacher](d0, geom.centers[0], geom.scales[0], pq, geom.local_idx[0])

    def encoder_state_dict(self):
        """What to keep after distillation: the encoder only (heads excluded)."""
        return self.net.state_dict()


if __name__ == "__main__":      # head-only smoke test (no data, no geometry module)
    torch.manual_seed(0)
    from distill_losses import distill_loss_per_image
    B, L, D, M, T = 2, 50, 64, 300, 96
    centers = torch.rand(L, 2) * 100
    scales = torch.full((L,), 8.0)
    local_idx = torch.cdist(centers, centers).topk(7, dim=-1, largest=False).indices      # self first
    d0 = torch.randn(B, L, D, requires_grad=True)
    pq = torch.rand(B, M, 2) * 100
    head = LatentQueryHead(D, 64, T, heads=4, k_nn=9, n_pre=1, pre_heads=1)
    out = head(d0, centers, scales, pq, local_idx, chunk=128)
    ref = head(d0, centers, scales, pq, local_idx, chunk=4096)
    assert out.shape == (B, M, T) and torch.allclose(out, ref, atol=1e-5)
    loss, _ = distill_loss_per_image(out, torch.randn(B, M, T))
    loss.mean().backward()
    assert torch.isfinite(d0.grad).all()
    # token centres from the TokenBuilder layout: channel-major, x = column, y = row
    H = W = 8
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    one = torch.stack([torch.zeros(H, W), xx + 100.0, yy + 200.0, torch.zeros(H, W), torch.zeros(H, W),
                       torch.zeros(H, W), torch.zeros(H, W), torch.zeros(H, W)], -1).reshape(H * W, 8)
    tok = torch.cat([one, one], 0)[None].expand(2, -1, -1)                                   # 2 bands, batch of 2
    tc = token_centers_from_tokens(tok, H, W, 4)
    assert tc.shape == (2, 4, 2) and torch.allclose(tc[0, 0], torch.tensor([101.5, 201.5])) \
        and torch.allclose(tc[0, 1], torch.tensor([105.5, 201.5])) and torch.allclose(tc[0, 2], torch.tensor([101.5, 205.5])), tc
    print("distill_heads v2 smoke test OK")
