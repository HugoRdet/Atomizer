"""
distill_heads_ptv3.py -- AtomizerPTv3 student for the DUNE-style distillation.

Where it goes: training/distill/distill_heads_ptv3.py
Selected by Model_Distill when the config has PTv3.enabled: true (see trainer_distill.py).

Same interface as DistillStudent (distill_heads.py), so Model_Distill, the data module, teachers, losses, optimizer and
launcher are unchanged:
    d0, geom = student.encode(batch, k, m)          # k, m (Atomiser_Unet leaf sizes) are ignored
    pred     = student.predict(teacher, d0, geom, q) # [B, g*g, teacher_dim], row-major over the teacher grid
    student.net, student.heads, student.encoder_state_dict()

Where the student is read
-------------------------
The FULL U-Net runs (tokenizer, stem, encoder, decoder): the student feature of a pixel is the decoder's level-0 output,
i.e. what the segmentation head sees at fine-tuning. A teacher token covers patch x patch pixels, so its student feature
is the mean of the level-0 features over those pixels (OlmoEarth: 2 x 2 pixels, DINOv3 on FLAIR: 16 x 16), taken
row-major over the teacher grid, the same order as token_centers_from_tokens and the teacher's [g, g, D] map. Every part
of the network, decoder included, is trained by both teachers.

Projector (DUNE: one ViT block with LayerScale 1e-4, then a Linear): LayerNorm, a residual MLP behind a per-channel gate
initialised at 1e-4 (starts near the identity), LayerNorm, Linear to the teacher dimension. Heads are discarded after
distillation; encoder_state_dict() returns the AtomizerPTv3 weights (tokenizer + PTv3 encoder and decoder), without its
segmentation head, ready to load into the Cashew model with strict=False.

Coordinates: points use the config's PTv3.gsd for every dataset, i.e. positions are in pixels of the tile (the real
ground resolution, 10 m or 0.2 m, reaches the model through the atoms' resolution encoding). This keeps a RoPE, if
enabled, in pixel units for both datasets.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.atomiser.atomizer_ptv3 import AtomizerPTv3


class PTv3Projector(nn.Module):
    """LN -> x + gamma * MLP(x) (gamma init 1e-4) -> LN -> Linear(dim, out_dim)."""

    def __init__(self, dim: int, out_dim: int, mlp_ratio: float = 4.0, init: float = 1e-4):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.ln_in = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.gamma = nn.Parameter(torch.full((dim,), init))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, x):
        x = self.ln_in(x.float())
        x = x + self.gamma * self.mlp(x)
        return self.out(x)


class PTv3DistillStudent(nn.Module):

    def __init__(self, config, lookup_table, teacher_dims: dict, teacher_patches: dict, dataset_teacher: dict):
        """teacher_dims {"olmo": 768, "dino": 1024}; teacher_patches {"olmo": 2, "dino": 16} (pixels per teacher token
        side); dataset_teacher {"olmo": "olmo", "flair": "dino"} (which teacher supervises which dataset)."""
        super().__init__()
        self.net = AtomizerPTv3(config=config, lookup_table=lookup_table)
        if self.net.mode != "pixel":
            raise NotImplementedError("PTv3 distillation supports PTv3.mode: pixel only")
        if len(self.net.backbone.dec) == 0:
            raise ValueError("the PTv3 student needs its decoder (non-empty dec_depths)")
        self.net.head = None                      # segmentation head unused here (re-created at fine-tuning)
        self.dataset_teacher = dict(dataset_teacher)
        self.patches = {t: int(p) for t, p in teacher_patches.items()}
        self.last_dense = None                   # [<=dense_keep, H, W, C] level-0 features of the last eval batch
        self.dense_keep = 4                      # tiles kept for the figures (viz_distill_pca.py raises it)

        dim = self.net.backbone.cfg.dec_channels[0]                       # level-0 decoder output width
        self.heads = nn.ModuleDict({t: PTv3Projector(dim, int(d)) for t, d in teacher_dims.items()})

        n_enc = sum(p.numel() for p in self.net.parameters()) / 1e6
        n_head = sum(p.numel() for p in self.heads.parameters()) / 1e6
        pools = ", ".join(f"{t}: mean over {p}x{p} px" for t, p in self.patches.items())
        print(f"[PTv3DistillStudent] student {n_enc:.1f}M params (kept, decoder included) + heads {n_head:.1f}M "
              f"(discarded) | read: decoder level-0 output ({dim} ch), {pools}")

    # ------------------------------------------------------------------ full U-Net, pooled per teacher token
    def _features(self, tokens, mask, chw):
        """tokens [B, N, 8] (dense, channel-major) -> per-pixel decoder level-0 features [B, H, W, C]."""
        net = self.net
        C, H, W = chw
        B, N, Dt = tokens.shape
        HW = H * W
        if N != C * HW:
            raise RuntimeError(f"[PTv3DistillStudent] {N} tokens != C*H*W = {C}*{H}*{W}")
        tok = tokens.view(B, C, HW, Dt).transpose(1, 2)                                 # [B, HW, C, 8]
        msk = (mask if mask.dtype == torch.bool else mask > 0.5).view(B, C, HW).transpose(1, 2)
        feat = net._embed_pixels(tok, msk, self.training)                               # [B, HW, pixel_dim]

        g = net._grid(H, W, tokens.device)
        out = net.backbone(net._point(g, B, HW, feat.reshape(B * HW, -1).float()))      # full U-Net
        if not torch.equal(out["grid_coord"].int(), g.repeat(B, 1)):
            raise RuntimeError("PTv3 output rows are not in input order")
        if not self.training:                    # for the validation figure: per-pixel features, before pooling
            self.last_dense = out["feat"].detach().view(B, H, W, -1)[:self.dense_keep]
        return out["feat"].view(B, H, W, -1)

    @staticmethod
    def _pool(dense, patch):
        """[B, H, W, C] -> [B, (H/patch) * (W/patch), C], mean over each patch, row-major over the teacher grid."""
        B, H, W, C = dense.shape
        if H % patch or W % patch:
            raise RuntimeError(f"{H}x{W} is not divisible by the teacher patch {patch}")
        x = dense.view(B, H // patch, patch, W // patch, patch, C).mean(dim=(2, 4))
        return x.reshape(B, (H // patch) * (W // patch), C)

    # ------------------------------------------------------------------ DistillStudent interface
    def encode(self, batch, k=None, m=None):
        """k, m: Atomiser_Unet leaf sizes, unused. Runs the U-Net ONCE per batch; returns (dense [B, H, W, C], geom).
        Pooling happens in predict, per teacher (teachers of one batch may have different grids)."""
        res = next(iter(batch["groups"]))
        grp = batch["groups"][res]
        dense = self._features(grp["tokens"], grp["mask"], self.net._chw(grp["shape"]))
        return dense, {"dataset": batch["dataset_name"]}

    def predict(self, teacher: str, d0, geom, pq):
        """Pools the dense features over this teacher's patch: g = sqrt(number of teacher tokens) (pq [B, g*g, 2],
        from token_centers_from_tokens), patch = H / g; without pq, the teacher's default patch."""
        dense = d0
        H = dense.shape[1]
        if pq is not None:
            g = int(round(pq.shape[1] ** 0.5))
            if g * g != pq.shape[1] or H % g:
                raise RuntimeError(f"{teacher}: {pq.shape[1]} teacher tokens do not tile a {H}x{H} crop")
            patch = H // g
        else:
            patch = self.patches[teacher]
        return self.heads[teacher](self._pool(dense, patch))

    def encoder_state_dict(self):
        """The student to keep: AtomizerPTv3 without its segmentation head (projector heads excluded)."""
        return self.net.state_dict()
