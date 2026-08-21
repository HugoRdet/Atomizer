"""
UniverSat from-scratch segmentation adapter
============================================

Wraps the UniverSat encoder (gastruc/UniverSat) as a BaselineTrainer-
compatible segmentation model, trained FROM SCRATCH (random init — no
HuggingFace download anywhere on this path).

Consumes the full modality dict (`expects_full_image_dict = True`), same
duck-typing as RAMENUPerNet, so batch["image"] = {"hls": [B, 6, H, W]}
(BurnScars) or {"optical": [B, 13, H, W], "sar": [B, 2, H, W]}
(Sen1Floods11) passes straight through — including through
sliding_window_inference, which crops dict inputs generically.

Output: logits [B, num_classes, G, G] on the model's output grid.
BaselineTrainer already bilinearly upsamples logits to the target's
spatial size when they differ, so G < H is fine (and is the default:
G = H // output_stride, with a pixel-level G = H available via
output_stride=1 thanks to the subpatch factor of 1).
"""

from functools import partial
from types import MethodType

import torch
import torch.nn as nn

from .UniverSat import UniverSat as _UniverSatEncoder
from .UniversalPatchEncoder import UniversalPatchEncoder


# ── Size presets ─────────────────────────────────────────────────────────
# "base" / "tiny" replicate the repo's configs (hubconf.MODEL_CONFIGS and
# configs/model/network/encoder/UniverSat_Tiny.yaml). "small" is our
# parameter-matched variant targeting the ~34M budget of the other
# baselines (ViT-S 384/12, RAMEN 384/12) — verify the printed param count.
UNIVERSAT_CONFIGS = {
    "base": dict(embed_dim=768, num_heads=12, sa_depth=12, spatial_encoder_div=8,
                 expand_dim=[2, 2, 2, 2]),
    "small": dict(embed_dim=384, num_heads=6, sa_depth=12, spatial_encoder_div=8,
                  expand_dim=[2, 2, 2, 2]),
    "tiny": dict(embed_dim=192, num_heads=8, sa_depth=6, spatial_encoder_div=4,
                 expand_dim=[1, 2, 2, 2]),
}

_UPE_ORDER = ["S1", "C", "T", "S"]
_UPE_N_QUERIES = [1, 1, 1, 1]


def _set_compile_mode(model: _UniverSatEncoder, compile_model: bool) -> None:
    """UPE_forward / ViT_forward may carry @torch.compile decorators at
    source level (upstream does; harmless no-op if you deleted them from
    your local UniverSat.py). Rebind the eager __wrapped__ implementations
    (same trick as the repo's hubconf._configure_compilation), optionally
    re-compiling. Default is EAGER: comparable with the other (eager)
    baselines, plays nicely with the GFLOPs profiler, and sidesteps
    inductor edge cases on the large per-pixel CA_Sub graphs."""
    for name in ("UPE_forward", "ViT_forward"):
        fn = getattr(type(model), name)
        fn = MethodType(getattr(fn, "__wrapped__", fn), model)
        if compile_model:
            fn = torch.compile(fn)
        object.__setattr__(model, name, fn)


def _build_encoder(size: str, gating: bool = True, compile_model: bool = False,
                   **overrides) -> _UniverSatEncoder:
    """Random-init UniverSat encoder — mirrors hubconf._build_model but
    with no hub / compile machinery and no SSL projector heads."""
    if size not in UNIVERSAT_CONFIGS:
        raise ValueError(f"Unknown size {size!r}. Available: {sorted(UNIVERSAT_CONFIGS)}")
    cfg = {**UNIVERSAT_CONFIGS[size], **overrides}

    embed_dim = cfg["embed_dim"]
    spatial_encoder = partial(
        UniversalPatchEncoder,
        embed_dim=embed_dim // cfg["spatial_encoder_div"],
        final_dim=embed_dim,
        n_queries=_UPE_N_QUERIES,
        expand_dim=cfg["expand_dim"],
        order=_UPE_ORDER,
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        attn_drop_rate=0.0,
        gating=gating,
    )
    model = _UniverSatEncoder(
        spatial_encoder=spatial_encoder,
        block_type=["Bi_ACA_in", f"SAx{cfg['sa_depth']}", "Bilinear_out", "CA_Sub"],
        embed_dim=embed_dim,
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        qkv_bias=False,
        n_registers=4,
        pre_norm=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        gating=gating,
        proba_drop_modalities=0.0,   # supervised training — never drop
        modalities_dict={},          # no SSL projector heads (dataset="" path)
    )
    _set_compile_mode(model, compile_model)
    return model


class UniverSatSegmenter(nn.Module):
    """
    UniverSat encoder + linear (1x1 conv) per-token segmentation head.

    forward(x_dict) -> logits [B, num_classes, G, G] where
    G = H // output_stride. BaselineTrainer upsamples logits to [H, W].

    Geometry (per modality m):
        patch_px    = patch_size_m / input_res[m]   (pixels per patch side)
        latent side = H / patch_px                  (trunk grid)
        output side = H / output_stride             (CA_Sub grid)
    Both divisions must be exact for every H the model will see
    (train crop AND eval window) — asserted at forward time. Grids are
    derived from the FIRST modality in input_bands; with mixed-GSD
    modalities UniverSat aligns them internally via input_res.
    """

    expects_full_image_dict = True

    def __init__(
        self,
        input_bands: dict,          # {modality: [band names]}   (len -> C check)
        wavelengths: dict,          # {modality: {band: nm/µm}} or {modality: [values/codes]}
        num_classes: int,
        input_res: dict,            # {modality: GSD in m}, e.g. {"hls": 30.0}
        patch_size_m: float = 240.0,   # 8 px @ 30 m (use 80.0 for 8 px @ 10 m)
        output_stride: int = 4,        # logits at H/4; 1 = per-pixel
        size: str = "small",
        subpatch_px: int = 1,          # sub-patch size in PIXELS (1 = finest skip)
        gating: bool = True,
        compile_model: bool = False,
        **encoder_overrides,
    ):
        super().__init__()
        self.modalities = list(input_bands.keys())
        self.input_res = {m: float(r) for m, r in input_res.items()}
        self.scale = patch_size_m / 10.0          # encoder works in 10 m units
        self.patch_size_m = patch_size_m
        self.output_stride = int(output_stride)
        self.subpatches = {m: int(subpatch_px) for m in self.modalities}

        # Wavelengths: UniverSat's registry uses MICROMETRES for optical
        # bands and STRING channel codes for SAR/elevation ("VV", "VH",
        # "HH", "HV", "Ratio_VV_VH", "Ratio_HH_HV", "DSM", "nDEM" — looked
        # up as learned Encoding_<code> embeddings in the UPE, so codes
        # must match those attribute names exactly; NOT RAMEN's
        # "asc_vv"-style pol_map keys). Accept either a {band: value} dict
        # (your RAMEN convention, nm) or a plain list; numeric values that
        # look like nanometres are converted to µm, strings pass through
        # untouched.
        self.wavelengths = {}
        for m in self.modalities:
            wl = wavelengths[m]
            vals = [wl[b] for b in input_bands[m]] if isinstance(wl, dict) else list(wl)
            numeric = [float(v) for v in vals if not isinstance(v, str)]
            nm_to_um = bool(numeric) and max(numeric) > 100.0   # clearly nm
            self.wavelengths[m] = [
                v if isinstance(v, str)
                else (float(v) / 1000.0 if nm_to_um else float(v))
                for v in vals
            ]

        # sanity: patch size must be an integer pixel count per modality
        for m in self.modalities:
            patch_px = patch_size_m / self.input_res[m]
            if abs(patch_px - round(patch_px)) > 1e-6:
                raise ValueError(
                    f"patch_size_m={patch_size_m} is not an integer number of "
                    f"pixels at {m}'s GSD {self.input_res[m]} m."
                )

        self.encoder = _build_encoder(size, gating=gating,
                                      compile_model=compile_model,
                                      **encoder_overrides)
        self.head = nn.Conv2d(self.encoder.embed_dim, num_classes, kernel_size=1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[UniverSatSeg] size={size}, embed_dim={self.encoder.embed_dim}, "
              f"params={n_params/1e6:.2f}M")
        print(f"[UniverSatSeg] modalities={self.modalities}")
        print(f"[UniverSatSeg] patch={patch_size_m} m "
              f"({patch_size_m / self.input_res[self.modalities[0]]:.0f} px @ "
              f"{self.input_res[self.modalities[0]]:g} m), "
              f"output_stride={output_stride}, subpatch_px={subpatch_px}")

    # ------------------------------------------------------------------
    def _grids(self, H: int, W: int, modality: str):
        if H != W:
            raise ValueError(f"UniverSatSegmenter expects square inputs, got {H}x{W}")
        patch_px = int(round(self.patch_size_m / self.input_res[modality]))
        if H % patch_px:
            raise ValueError(
                f"Input side {H} not divisible by patch size {patch_px} px "
                f"({self.patch_size_m} m @ {self.input_res[modality]} m). "
                f"Pick crop/window sizes divisible by {patch_px}."
            )
        latent_side = H // patch_px
        if H % self.output_stride:
            raise ValueError(f"Input side {H} not divisible by output_stride "
                             f"{self.output_stride}.")
        out_side = H // self.output_stride
        return latent_side ** 2, out_side

    # ------------------------------------------------------------------
    def forward(self, x: dict):
        ref = self.modalities[0]
        # Snapshot modalities: (B, C, H, W). Time-series modalities:
        # (B, T, C, H, W) with a companion x[f"{mod}_dates"] (B, T) key —
        # the caller injects the dates (see e.g. the xView2 adapter);
        # UniverSat's UPE_forward handles the temporal axis natively when
        # the _dates key is present.
        B = x[ref].shape[0]
        H, W = x[ref].shape[-2], x[ref].shape[-1]
        latent_grid, out_side = self._grids(H, W, ref)

        tokens, _ = self.encoder(
            x,
            wavelengths=self.wavelengths,
            input_res=self.input_res,
            scale=self.scale,
            latent_grid=latent_grid,
            output_grid=out_side ** 2,
            subpatches=self.subpatches,
            dataset="",                 # skip SSL projector heads
        )
        tokens = tokens[:, self.encoder.n_registers:]           # registers are PREPENDED
        feats = tokens.view(B, out_side, out_side, -1).permute(0, 3, 1, 2)
        return self.head(feats)                                  # [B, K, G, G]


def build_universat_segmenter(**kwargs) -> UniverSatSegmenter:
    return UniverSatSegmenter(**kwargs)


# ── Smoke test ───────────────────────────────────────────────────────────
# Run from the repo root as a module (relative imports):
#   python -m training.Universat.universat_augmenter
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) BurnScars-style single modality (numeric nm wavelengths)
    model = build_universat_segmenter(
        input_bands={"hls": ["B02", "B03", "B04", "B8A", "B11", "B12"]},
        wavelengths={"hls": {"B02": 492.4, "B03": 559.8, "B04": 664.6,
                             "B8A": 864.7, "B11": 1613.7, "B12": 2202.4}},
        num_classes=2,
        input_res={"hls": 30.0},
        patch_size_m=240.0,     # 8 px
        output_stride=4,
        size="small",
    )
    x = {"hls": torch.randn(2, 6, 128, 128)}
    with torch.no_grad():
        out = model(x)
    print("logits:", tuple(out.shape))     # expect (2, 2, 32, 32)

    # different input size, same weights (sliding-window / full-image eval)
    with torch.no_grad():
        out2 = model({"hls": torch.randn(1, 6, 256, 256)})
    print("logits @256:", tuple(out2.shape))  # expect (1, 2, 64, 64)

    # 2) Sen1Floods11-style multimodal (optical nm + SAR string codes)
    S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
          "B08", "B8A", "B09", "B10", "B11", "B12"]
    S2_NM = {"B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
             "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
             "B8A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
             "B12": 2202.4}
    mm = build_universat_segmenter(
        input_bands={"optical": S2, "sar": ["VV", "VH"]},
        wavelengths={"optical": S2_NM, "sar": ["VV", "VH"]},
        num_classes=2,
        input_res={"optical": 10.0, "sar": 10.0},
        patch_size_m=80.0,      # 8 px @ 10 m
        output_stride=4,
        size="tiny",            # tiny to keep the smoke test light
    )
    print("wl optical[:3]:", mm.wavelengths["optical"][:3],
          "| sar:", mm.wavelengths["sar"])
    xm = {"optical": torch.randn(1, 13, 64, 64),
          "sar": torch.randn(1, 2, 64, 64)}
    logits = mm(xm)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.randint(0, 2, (1, 16, 16)))
    loss.backward()
    vv = dict(mm.named_parameters())["encoder.spatial_encoder.Encoding_VV"]
    print(f"multimodal logits {tuple(logits.shape)}, loss={loss.item():.3f}, "
          f"Encoding_VV grad: {'yes' if vv.grad is not None else 'NO'}")
