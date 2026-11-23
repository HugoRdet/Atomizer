"""
teachers.py -- the two frozen teachers of the distillation.

Where it goes: training/distill/teachers.py
    OlmoEarthTeacher : Sentinel tiles -> [B, g, g, 768]  (mean of the S2 and S1 token grids; location tokens excluded)
    DinoV3Teacher    : aerial RGB uint8 [B, 3, H, W] -> [B, g, g, 1024]  (patch tokens; CLS and register tokens dropped);
                       t["up"] (optional, float) overrides the input upsampling for one batch
Both take the `teacher` dict of a batch (see utils_dataset_distill.py) already on the GPU, and return the teacher's token
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
