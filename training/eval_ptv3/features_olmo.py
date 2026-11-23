"""Frozen OlmoEarth features for the same downstream protocol (same splits, heads, metrics, seeds).

OlmoEarth gets its OWN input convention, as in distillation (teachers.py / pca_olmo.py): raw Sentinel values in its band
order, normalized by olmoearth's Normalizer(std_multiplier=2.0), plus timestamps. Raw values are recovered by inverting
the dataset's per-band z-score (the Atomizer datasets z-score raw DN; the inversion is exact before clamping).

Output: [B, 768, H/patch, W/patch] patch-token grid (mean over band-set tokens, as OlmoEarthTeacher). Classification
heads mean-pool it; segmentation heads are upsampled bilinearly to the label size by the runner.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.distill.teachers import find_tokens

# OlmoEarth S2 L2A channel order (= OLMO_S2 in utils_dataset_distill.py)
OLMO_S2 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]

# Per dataset: S2 channel order of the dataset's `image` tensor, and its S1 channels (None = no S1).
DATASET_S2 = {
    "eurosat": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
}


class OlmoEarthExtractor(nn.Module):
    def __init__(self, ckpt, dataset_name, train_ds, patch=2, timestamp=(15, 6, 2018), dim=768):
        super().__init__()
        if dataset_name not in DATASET_S2:
            raise NotImplementedError(f"OlmoEarth input mapping not written yet for {dataset_name}")
        import olmoearth_pretrain_minimal as oep
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
        self.Sample, self.Modality, self.norm = MaskedOlmoEarthSample, Modality, Normalizer(std_multiplier=2.0)
        self.model = oep.load_model_from_path(ckpt).eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.patch, self.dim, self.ts = int(patch), dim, list(timestamp)
        names = DATASET_S2[dataset_name]
        self.s2_idx = [names.index(b) for b in OLMO_S2]                      # dataset channel -> OlmoEarth order
        self.mean = train_ds.norm_mean.reshape(1, -1, 1, 1).float()          # dataset z-score stats (raw DN units)
        self.std = train_ds.norm_std.reshape(1, -1, 1, 1).float()
        self._printed = False
        print(f"[olmo] {ckpt} | patch {patch} | S2 from {dataset_name} channels {self.s2_idx} | no S1 | ts {self.ts}")

    def set_mode(self, trainable):
        self.model.eval()

    @torch.no_grad()
    def forward(self, batch, training=False):
        img = batch["image"].float()                                         # [B, C, H, W] z-scored
        raw = img.cpu() * self.std + self.mean
        s2 = raw[:, self.s2_idx].permute(0, 2, 3, 1)[:, :, :, None, :]       # [B, H, W, 1, 12] raw DN
        s2n = self.norm.normalize(self.Modality.SENTINEL2_L2A, s2.numpy().astype(np.float32))
        if not self._printed:
            print("[olmo] normalized S2 percentiles 1/50/99:", np.percentile(s2n, [1, 50, 99]).round(2),
                  "(distillation data was roughly in [0, 1])")
            self._printed = True
        dev = img.device
        B, H, W = s2.shape[:3]
        if H % self.patch or W % self.patch:
            raise RuntimeError(f"{H}x{W} not divisible by OlmoEarth patch {self.patch}")
        g = H // self.patch
        sample = self.Sample(
            timestamps=torch.tensor(self.ts, dtype=torch.long, device=dev).view(1, 1, 3).expand(B, 1, 3).contiguous(),
            sentinel2_l2a=torch.from_numpy(s2n).float().to(dev),
            sentinel2_l2a_mask=torch.zeros(B, H, W, 1, dtype=torch.long, device=dev),
        )
        out = self.model.encoder(sample, patch_size=self.patch, input_res=10, fast_pass=True)
        toks = [x for pth, x in find_tokens(out, g, self.dim) if "location" not in pth]
        assert toks, "no OlmoEarth token tensor found (inspect with radio/pca_olmo.py dump())"
        red = [(x.float().flatten(3, -2).mean(3) if x.ndim > 4 else x.float()) for x in toks]
        return torch.stack(red).mean(0).permute(0, 3, 1, 2)                 # [B, 768, g, g]
