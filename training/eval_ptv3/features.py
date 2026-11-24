"""Build a distilled AtomizerPTv3 and read its dense decoder level-0 features.

The read is PTv3DistillStudent._features without the teacher-patch pooling: tokenizer -> per-pixel cross-attention ->
full PTv3 U-Net -> level-0 rows (input order, checked) -> [B, D, H, W]. This is exactly the representation the
projectors were trained on, at full resolution.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from training.atomiser.atomizer_ptv3 import AtomizerPTv3
from training.utils import Lookup_encoding, read_yaml


def build_lookup(distill_cfg: dict, bands_yaml: str):
    """Same construction as train_distill_atomiser.py. The first argument of Lookup_encoding is ignored
    (init_config hard-codes the resolutions 10/15/0.1/0.5/30/0.2/1.6 m), so resolution indices always match.
    table_wave depends on the ORDER of bands.yaml: use the file the distillation ran with, or only append to it."""
    return Lookup_encoding(None, read_yaml(bands_yaml), distill_cfg)


def build_net(distill_cfg: dict, lookup, num_classes: int, ckpt: str | None):
    cfg = copy.deepcopy(distill_cfg)
    cfg.setdefault("trainer", {})["num_classes"] = int(num_classes)
    if cfg.get("PTv3", {}).get("mode", "pixel") != "pixel":
        raise NotImplementedError("distilled students are pixel mode only")
    net = AtomizerPTv3(config=cfg, lookup_table=lookup)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in sd:                     # full Lightning checkpoint (step*.ckpt, last.ckpt, timelimit.ckpt)
            prefix = "student.net."
            sd = {k[len(prefix):]: v for k, v in sd["state_dict"].items() if k.startswith(prefix)}
            if not sd:
                raise RuntimeError(f"no '{prefix}*' weights in {ckpt}")
            print(f"[eval] Lightning checkpoint: kept {len(sd)} '{prefix}*' tensors (projectors / optimizer dropped)")
        # Lazily built, non-learned buffers derived from the lookup table: rebuilt from the current lookup on first use.
        lazy = [k for k in sd if k.endswith("time_encoder.time_values")]
        for k in lazy:
            sd.pop(k)
        if lazy:
            print(f"[eval]   dropped lazy buffers (rebuilt from the lookup table): {lazy}")
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        print(f"[eval] loaded {ckpt}\n[eval]   missing (excl. head.*): {missing}\n[eval]   unexpected: {unexpected}")
        if missing or unexpected:
            raise RuntimeError("checkpoint / config mismatch: pass the distillation config of this run")
    return net


def check_lookup(lookup, ref_table_wave: dict | None):
    """Optional: compare against a table_wave dumped at distillation time (torch.save(lookup.table_wave, ...))."""
    if ref_table_wave is None:
        return
    bad = {k: (v, lookup.table_wave.get(k)) for k, v in ref_table_wave.items() if lookup.table_wave.get(k) != v}
    if bad:
        raise RuntimeError(f"table_wave differs from distillation for {len(bad)} bands, e.g. {list(bad.items())[:3]}")
    print("[eval] table_wave matches the distillation one")


class DenseExtractor(nn.Module):
    """bn_mode: 'batch' (per-batch statistics, running stats untouched; what distillation validation used),
    'eval' (running statistics), 'recal' (running stats re-estimated on downstream train data, then eval).
    With PTv3.norm: ln the three are identical."""

    def __init__(self, net: AtomizerPTv3, bn_mode: str = "batch", time_idx_override: float | None = -1.0,
                 feature: str = "dec0", mask_channels=None, value_affine=None):
        """feature: 'dec0'     decoder level-0 per-pixel features [B, dec_channels[0], H, W] (what distillation trained)
                    'enc_last' last encoder stage (coarsest level), mean over each sample's points [B, enc_channels[-1]]"""
        super().__init__()
        self.net, self.bn_mode, self.time_override, self.feature = net, bn_mode, time_idx_override, feature
        self.mask_channels = sorted(int(c) for c in (mask_channels or []))   # channels whose atoms are masked
        self.value_affine = value_affine                                     # (A [C], B [C]): value -> A + B * value
        self.net.head = None
        cfg = self.net.backbone.cfg
        if feature == "dec0":
            self.dim = cfg.dec_channels[0]
        elif feature == "enc_last":
            self.dim = cfg.enc_channels[-1]
            self._grab = {}
            self.net.backbone.enc[-1].register_forward_hook(self._hook)
        else:
            raise ValueError(feature)

    def _hook(self, module, args, output):
        x = output[0] if isinstance(output, (tuple, list)) else output
        x = getattr(x, "feat", x)
        grid = args[1] if len(args) > 1 else None
        self._grab = {"feat": x, "seq_lens": None if grid is None else getattr(grid, "seq_lens", None)}

    def _pool_enc_last(self, B):
        x, lens = self._grab["feat"], self._grab["seq_lens"]
        if lens is None:
            raise RuntimeError("enc_last: could not read per-sample point counts (grid.seq_lens) from the last stage")
        lens = [int(v) for v in (lens.tolist() if torch.is_tensor(lens) else lens)]
        if len(lens) != B or sum(lens) != x.shape[0]:
            raise RuntimeError(f"enc_last: seq_lens {lens[:4]}... (sum {sum(lens)}) vs {x.shape[0]} rows for B={B}")
        return torch.stack([c.float().mean(0) for c in x.split(lens, 0)])      # [B, enc_channels[-1]]

    def _bns(self):
        return [m for m in self.net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def set_mode(self, trainable: bool):
        self.net.train(trainable)
        for m in self._bns():
            if self.bn_mode == "batch":
                m.train(); m.momentum = 0.0
            elif not trainable:
                m.eval()

    def dense(self, tokens, mask, C, H, W, training=False):
        """tokens [B, C*H*W, 8] channel-major -> [B, D, H, W]."""
        net = self.net
        B, N, Dt = tokens.shape
        HW = H * W
        if N != C * HW:
            raise RuntimeError(f"{N} tokens != C*H*W = {C}*{H}*{W} (token subsampling must be off)")
        if self.value_affine is not None:                                    # e.g. the distillation input pipeline
            A, B_ = (t.to(tokens.device, tokens.dtype) for t in self.value_affine)
            if A.numel() != C:
                raise ValueError(f"value_affine has {A.numel()} channels, data has {C}")
            tokens = tokens.clone()
            v = tokens[..., 0].view(B, C, HW)
            tokens[..., 0] = (A.view(1, C, 1) + B_.view(1, C, 1) * v).clamp(-10, 10).view(B, N)
        if self.mask_channels:                                               # e.g. B10, never seen in distillation
            if max(self.mask_channels) >= C:
                raise ValueError(f"mask_channels {self.mask_channels} out of range for {C} channels")
            mask = (mask if mask.dtype == torch.bool else mask > 0.5).clone().view(B, C, HW)
            mask[:, self.mask_channels] = True
            mask = mask.view(B, N)
        if self.time_override is not None:
            tokens = tokens.clone(); tokens[..., 7] = self.time_override     # distillation used time_idx = -1
        tok = tokens.view(B, C, HW, Dt).transpose(1, 2)
        msk = (mask if mask.dtype == torch.bool else mask > 0.5).view(B, C, HW).transpose(1, 2)
        feat = net._embed_pixels(tok, msk, training)
        g = net._grid(H, W, tokens.device)
        out = net.backbone(net._point(g, B, HW, feat.reshape(B * HW, -1).float()))
        if not torch.equal(out["grid_coord"].int(), g.repeat(B, 1)):
            raise RuntimeError("PTv3 output rows are not in input order")
        if self.feature == "enc_last":
            return self._pool_enc_last(B)
        return out["feat"].view(B, H, W, -1).permute(0, 3, 1, 2)

    def forward(self, batch, training=False):
        """-> [B, D, H, W], or [B, T, D, H, W] for temporal batches (tokens [B, T, N, 8])."""
        res = next(iter(batch["groups"]))
        grp = batch["groups"][res]
        tokens, mask = grp["tokens"], grp["mask"]
        shape = [int(s) for s in grp["shape"]]                                  # plain tuple (collate_eval)
        if tokens.dim() == 4:                                                   # temporal: fold T into the batch
            B, T, N, Dt = tokens.shape
            H, W = shape[-2:]
            C = N // (H * W)
            f = self.dense(tokens.reshape(B * T, N, Dt), mask.reshape(B * T, N), C, H, W, training)
            return f.view(B, T, *f.shape[1:])
        C, H, W = shape
        return self.dense(tokens, mask, C, H, W, training)

    @torch.no_grad()
    def recalibrate_bn(self, loader, device, n_batches=100):
        bns = self._bns()
        if not bns:
            return
        self.net.eval()
        for m in bns:
            m.reset_running_stats(); m.momentum = None; m.train()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            self(to_device(batch, device))
        for m in bns:
            m.eval()
        self.bn_mode = "eval"
        print(f"[eval] BatchNorm running stats re-estimated on {min(i + 1, n_batches)} train batches")


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
