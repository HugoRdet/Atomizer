"""
teachers.py -- the frozen teachers of the distillation.

Where it goes: training/distill/teachers.py
    OlmoEarthTeacher : Sentinel tiles -> [B, g, g, 768]  (mean of the S2 and S1 token grids; location tokens excluded)
    DinoV3Teacher    : aerial RGB uint8 [B, 3, H, W] -> [B, g, g, 1024]  (patch tokens; CLS and register tokens dropped);
                       t["up"] (optional, float) overrides the input upsampling for one batch
    PanopticonTeacher: aerial R, G, B, NIR -> [B, S, S, 768] (Panopticon ViT-B/14, AnyUp to t["out_side"] if larger)
All take the `teacher` dict of a batch (see utils_dataset_distill.py) already on the GPU, and return the teacher's token
map in the order of the student's queries (row-major over the grid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_tokens(out, g, dim):
    """All tensors shaped (B, g, g, ..., dim) in an OlmoEarth encoder output, with their path."""
    found = []

    def walk(o, path):
        if torch.is_tensor(o):
            if o.ndim >= 4 and o.shape[-1] == dim and o.shape[1] == g and o.shape[2] == g:
                found.append((path, o))
        elif hasattr(o, "_asdict"):
            walk(o._asdict(), path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif hasattr(o, "__dict__"):
            walk(vars(o), path)

    walk(out, "out")
    return found


class OlmoEarthTeacher(nn.Module):
    def __init__(self, ckpt, patch, dim=768):
        super().__init__()
        import olmoearth_pretrain_minimal as oep
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
        self.Sample, self.patch, self.dim = MaskedOlmoEarthSample, int(patch), dim
        self.model = oep.load_model_from_path(ckpt)
        self.model.eval()
        for q in self.model.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def forward(self, t):
        s2n, s1n = t["s2n"].float(), t["s1n"].float()                           # [B, H, W, 1, C]
        B, H, W = s2n.shape[:3]
        g = H // self.patch
        z = lambda: torch.zeros(B, H, W, 1, dtype=torch.long, device=s2n.device)  # noqa: E731
        sample = self.Sample(timestamps=t["ts"], sentinel2_l2a=s2n, sentinel2_l2a_mask=z(), sentinel1=s1n, sentinel1_mask=z())
        out = self.model.encoder(sample, patch_size=self.patch, input_res=10, fast_pass=True)
        toks = [x for pth, x in find_tokens(out, g, self.dim) if "location" not in pth]
        assert toks, "no OlmoEarth token tensor found; run pca_teachers.py to inspect the output structure"
        red = [(x.float().flatten(3, -2).mean(3) if x.ndim > 4 else x.float()) for x in toks]
        return torch.stack(red).mean(0)                                         # [B, g, g, dim]


class DinoV3Teacher(nn.Module):
    IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    SAT = ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143])   # values used in public DINOv3 demos, not from the model card

    def __init__(self, ckpt, norm="auto", student_patch=None):
        """student_patch: pixels covered by one teacher token in the student's image (None: the teacher's own patch size).
        DINOv3's patch embedding is fixed (16), so a smaller footprint is obtained by upsampling the input by
        teacher_patch / student_patch (bicubic, align_corners=True, as DUNE's get_teacher_outputs)."""
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.n_reg = int(getattr(self.model.config, "num_register_tokens", 0))
        self.patch = int(getattr(self.model.config, "patch_size", 16))
        self.up = 1.0 if not student_patch else self.patch / float(student_patch)
        if norm == "auto":
            norm = "sat" if "sat493m" in str(ckpt).lower() else "imagenet"
        mean, std = self.SAT if norm == "sat" else self.IMAGENET
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, t):
        x = (t["rgb"].float() / 255.0 - self.mean) / self.std                    # RGB only
        up = float(t.get("up", self.up))                                         # per-batch view (flair_views)
        if up != 1.0:
            x = F.interpolate(x, scale_factor=up, mode="bicubic", align_corners=True)
        assert x.shape[-1] % self.patch == 0 and x.shape[-2] % self.patch == 0, \
            f"input {tuple(x.shape[-2:])} after resizing is not a multiple of the teacher patch {self.patch}"
        h = self.model(pixel_values=x).last_hidden_state                         # [B, 1 + registers + patches, D]
        tok = h[:, 1 + self.n_reg:]
        g = int(round(tok.shape[1] ** 0.5))
        assert g * g == tok.shape[1], f"unexpected DINOv3 token count {tok.shape[1]}"
        return tok.reshape(tok.shape[0], g, g, tok.shape[2]).float()


# ---------------------------------------------------------------------------------------------------------
def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PanopticonTeacher(nn.Module):
    """Panopticon ViT-B/14 (any-sensor, DINOv2-style) on aerial R, G, B, NIR -> [B, S, S, 768], upsampled by AnyUp.

    Input: t["rgb"] uint8 [B, 3, H, W], t["nir"] uint8 [B, 1, H, W]; z-scored with the FLAIR student statistics (first 4
    channels of flair_stats: R, G, B, NIR on [0, 1] values; ~standard normal per channel, as Panopticon expects), resized
    to H * up * 14 / 16 (Panopticon grid = H * up / 16, the same as DINOv3 at that view), channel ids = wavelengths (nm).
    t["out_side"] (optional): side of the returned grid; if larger than Panopticon's grid, AnyUp (Wimmer et al. 2026)
    upsamples the tokens, guided by the RGB crop. Everything is loaded offline from the torch.hub caches:
        <pano_dir>/Panopticon-FM_panopticon_main + <pano_dir>/panopticon_vitb14_teacher.pth
        <anyup_dir>/wimmerth_anyup_main (+ its weights in <anyup_dir>/checkpoints)
    """
    IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, pano_dir, stats_path, wavelengths=(664, 559, 493, 842), anyup_dir=None, q_chunk=256):
        super().__init__()
        import os
        import sys
        pano_dir = os.path.abspath(pano_dir)
        repo = os.path.join(pano_dir, "Panopticon-FM_panopticon_main")
        ckpt = os.path.join(pano_dir, "panopticon_vitb14_teacher.pth")
        if not (os.path.isdir(repo) and os.path.isfile(ckpt)):
            raise FileNotFoundError(f"Panopticon cache missing ({repo}, {ckpt}): run once on a login node\n"
                                    f"  python -c \"import torch; torch.hub.set_dir('{pano_dir}'); "
                                    f"torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14')\"")
        if repo not in sys.path:
            sys.path.insert(0, repo)                     # its hubconf imports dinov2.* from the repo root
        hub = _load_module("panopticon_hubconf", os.path.join(repo, "hubconf.py"))
        self.model = hub._panopticon_vitb14()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model.eval()
        st = torch.load(stats_path, weights_only=True)
        self.register_buffer("mean", st["mean"].reshape(-1)[:4].view(1, 4, 1, 1).float(), persistent=False)
        self.register_buffer("std", st["std"].reshape(-1)[:4].view(1, 4, 1, 1).float(), persistent=False)
        self.register_buffer("ids", torch.tensor([int(w) for w in wavelengths]).view(1, -1), persistent=False)
        self.register_buffer("im_mean", torch.tensor(self.IMAGENET[0]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor(self.IMAGENET[1]).view(1, 3, 1, 1), persistent=False)
        self.q_chunk = int(q_chunk)
        self.anyup = None
        if anyup_dir:
            anyup_dir = os.path.abspath(anyup_dir)
            arepo = os.path.join(anyup_dir, "wimmerth_anyup_main")
            if not os.path.isdir(arepo):
                raise FileNotFoundError(f"AnyUp cache missing ({arepo}): run once on a login node\n"
                                        f"  python -c \"import torch; torch.hub.set_dir('{anyup_dir}'); "
                                        f"torch.hub.load('wimmerth/anyup', 'anyup')\"")
            torch.hub.set_dir(anyup_dir)                 # its weights are cached in <anyup_dir>/checkpoints
            self.anyup = torch.hub.load(arepo, "anyup", source="local", trust_repo=True).eval()
        for q in self.parameters():
            q.requires_grad_(False)
        n = sum(q.numel() for q in self.model.parameters()) / 1e6
        print(f"[PanopticonTeacher] {n:.0f}M params, wavelengths {list(wavelengths)}, stats {stats_path}, "
              f"AnyUp {'on' if self.anyup is not None else 'off'}")

    @torch.no_grad()
    def forward(self, t):
        rgb = t["rgb"]
        x = torch.cat([rgb, t["nir"]], 1).float() / 255.0
        x = (x - self.mean) / self.std
        B, _, H, W = x.shape
        up = float(t.get("up", 1.0))
        g = H * up / 16.0
        if g != int(g) or H != W:
            raise ValueError(f"Panopticon: tile {H}x{W} with up {up} does not give an integer 16-px grid")
        g = int(g)
        x = F.interpolate(x, size=(g * 14, g * 14), mode="bilinear", align_corners=False)
        ids = self.ids.expand(B, -1)
        out = self.model.get_intermediate_layers({"imgs": x, "chn_ids": ids}, n=1, return_class_token=False)
        tok = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(tok, (list, tuple)):
            tok = tok[0]
        assert tok.shape[1] == g * g, f"Panopticon returned {tok.shape[1]} tokens, expected {g}x{g}"
        tok = tok.float().reshape(B, g, g, -1)
        S = int(t.get("out_side", g))
        if S == g:
            return tok
        if self.anyup is None or S < g:
            raise ValueError(f"Panopticon grid {g} -> {S} needs AnyUp (pano_anyup_dir) and S >= {g}")
        guide = F.interpolate(rgb.float() / 255.0, size=(S, S), mode="bilinear", antialias=True, align_corners=False)
        guide = (guide - self.im_mean) / self.im_std
        lr = tok.permute(0, 3, 1, 2).contiguous()
        with torch.autocast(x.device.type, enabled=False):
            try:
                hr = self.anyup(guide, lr, output_size=(S, S), q_chunk_size=self.q_chunk)
            except TypeError:
                hr = self.anyup(guide, lr, output_size=(S, S))
        if hasattr(hr, "last_hidden_state"):
            hr = hr.last_hidden_state
        return hr.float().permute(0, 2, 3, 1)                                    # [B, S, S, 768]
