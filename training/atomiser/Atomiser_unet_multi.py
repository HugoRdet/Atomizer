"""
Atomiser_UnetMulti — one U-Net shared by several datasets on a physical scale chain
===================================================================================

Every dataset uses the SAME chain of levels. Level 0 has the same ground size
for every dataset (k per dataset = atoms per pixel x pixels per leaf, chosen
from its GSD), and each level doubles the spacing of the previous one, so
level l is the same ground scale everywhere. A dataset with S levels runs
levels 0 .. S-1 of the chain (S = its own number of levels, from N_min).

Weights
    shared by every dataset   atom embedding, leaf reader, global self-attention
                              (applied after each dataset's LAST level), query
                              decoder (without head)
    per level of the chain    local self-attention blocks, merge l -> l+1,
                              temporal transformer, decoder stage
    per dataset               MLP head; hierarchy geometry (cached)

Datasets (config["Atomiser"]["unet_datasets"]):
    unet_datasets:
        planteur:      {levels: 4, km_train: [[52, 52]],   km_eval: [52, 52],   num_classes: 8, temporal: true}
        agripotential: {levels: 3, km_train: [[160, 160]], km_eval: [160, 160], num_classes: 5, temporal: true}
        cashew:        {levels: 5, km_train: [[48, 48]],   km_eval: [48, 48],   num_classes: 7, temporal: false}
    The chain has max(levels) levels. Per-date datasets (temporal: true) give
    tokens [B, T, N, 8] + batch["time_positions"]; single-date ones [B, N, 8].

The dataset of a batch: batch["dataset"] (a name), or forward(..., dataset=name).

Shared keys (config["Atomiser"]): the U-Net keys of Atomiser_unet.py (unet_dims or
unet_dim — ONE width for every level, since global and query modules are shared —,
unet_head_dim, unet_depth, unet_dropout, unet_fusion, unet_dec_refine (one entry per
level except the last, finest first), unet_dec_ffn, unet_region_size_mlp,
unet_atom_features / unet_atom_dim, unet_leaf_heads / unet_leaf_head_dim,
unet_query_dim, unet_decoder_neighbors, unet_leaf_chunk, unet_query_chunk,
unet_temporal_transformer) plus:
    unet_global_depth (2)    blocks of the shared global self-attention
    unet_head_hidden (256)   hidden width of each dataset's MLP head

Frozen-encoder evaluation: model.freeze_encoder() freezes (and keeps in eval mode)
the embedding, leaf reader, local blocks, merges, global module and temporal
transformers; the decoder stages, query decoder and heads stay trainable.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

try:
    from training.atomiser.Atomiser_unet import (
        AtomEmbedding, LegacyAtomEmbedding, LeafEncoder, LocalBlock, GlobalBlock, GatherAttention,
        OverlapTransportDecoder, PerLevelTemporalAggregation, QueryDecoder, overlap_weights)
    from training.atomiser.GeographicPruningUNet import GeographicPruningUNet
except ImportError:  # local testing
    from Atomiser_unet import (
        AtomEmbedding, LegacyAtomEmbedding, LeafEncoder, LocalBlock, GlobalBlock, GatherAttention,
        OverlapTransportDecoder, PerLevelTemporalAggregation, QueryDecoder, overlap_weights)
    from GeographicPruningUNet import GeographicPruningUNet


# =============================================================================
# The chain: local blocks at every level, one shared global module
# =============================================================================
class ChainUNet(nn.Module):
    """n_levels chain levels of width `dim`. encode() runs levels 0 .. S-1 of the
    chain, S = geom.num_levels: at every level, `depth` local blocks; between
    levels, a cross-attention merge; after level S-1, the shared global module.
    decode() runs the per-level decoder stages from S-1 back to 0."""

    def __init__(self, n_levels, dim, head_dim=64, depth=2, global_depth=2, dropout=0.0,
                 fusion="gate", dec_refine=0, dec_ffn=False, region_size_mlp=False):
        super().__init__()
        L = self.n_levels = int(n_levels)
        H = max(1, dim // head_dim)
        assert dim % H == 0 and (dim // H) % 4 == 0, f"width {dim} incompatible with head_dim {head_dim}"
        self.local = nn.ModuleList(nn.ModuleList(LocalBlock(dim, H, dropout) for _ in range(depth))
                                   for _ in range(L))
        self.merge = nn.ModuleList(GatherAttention(dim, H, dropout, kv_dim=dim) for _ in range(L - 1))
        self.glob = nn.ModuleList(GlobalBlock(dim, H, dropout) for _ in range(global_depth))
        # seeds[l]: initial coarse queries of level l (index 0 unused: the leaf reader
        # initialises level 0)
        self.seeds = nn.ParameterList(nn.Parameter(torch.randn(dim) * 0.02) for _ in range(L))
        self.region = (nn.ModuleList(nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, dim))
                                     for _ in range(L)) if region_size_mlp else None)
        rf = [int(dec_refine)] * (L - 1) if isinstance(dec_refine, (int, float)) else [int(x) for x in dec_refine]
        assert len(rf) == L - 1, f"unet_dec_refine needs {L - 1} entries (finest first)"
        self.odec = OverlapTransportDecoder([dim] * L, [dim] * L, rf, head_dim, dropout, fusion, ffn=dec_ffn)
        self.fuse = self.odec.fuse

    @staticmethod
    def cell_coords(geom, lvl, frame):
        return geom.centers[lvl] / geom.scales[frame].mean()

    def region_queries(self, geom, lvl):
        if self.region is None:
            return self.seeds[lvl].expand(geom.sizes[lvl], -1)
        rel = torch.log(geom.scales[lvl] / geom.scales[lvl].mean()).unsqueeze(-1)
        return self.seeds[lvl] + self.region[lvl](rel)

    def encode(self, z0, geom):
        S, B = geom.num_levels, z0.shape[0]
        if S > self.n_levels:
            raise ValueError(f"geometry has {S} levels, the chain only {self.n_levels}")
        x, skips = z0, []
        for l in range(S):
            c = self.cell_coords(geom, l, l)
            for blk in self.local[l]:
                x = blk(x, geom.local_idx[l], c)
            if l == S - 1:                                    # the dataset's last level
                for blk in self.glob:
                    x = blk(x, c)
            skips.append(x)
            if l < S - 1:
                q = self.region_queries(geom, l + 1).expand(B, -1, -1)
                x = q + self.merge[l](q, x, geom.merge_idx[l], geom.merge_valid[l],
                                      self.cell_coords(geom, l + 1, l + 1),
                                      self.cell_coords(geom, l, l + 1))
        return skips

    def decode(self, skips, geom, return_levels=False):
        """Overlap-transport decoder over the dataset's S levels (stage l = level l)."""
        od, S = self.odec, len(skips)
        d = od.proj[S - 1](skips[S - 1])
        levels = {S - 1: d}
        for l in reversed(range(S - 1)):
            s = od.proj[l](skips[l])
            t = od.transport[l](d)
            w = overlap_weights(geom, l).to(t.dtype)
            m = (t[:, geom.up_idx[l]] * w[None, :, :, None]).sum(2)
            x = od.fuse[l](s, m)
            if od.ffn is not None:
                x = x + od.ffn[l](x)
            c = self.cell_coords(geom, l, l)
            for blk in od.refine[l]:
                x = blk(x, geom.local_idx[l], c)
            d = x
            levels[l] = d
        return (d, [levels[l] for l in range(S)]) if return_levels else d


# =============================================================================
# The model
# =============================================================================
class Atomiser_UnetMulti(nn.Module):
    def __init__(self, config, lookup_table=None):
        super().__init__()
        a = config.get("Atomiser", {})
        ds_cfg = a.get("unet_datasets")
        if not ds_cfg:
            raise ValueError("[Atomiser_UnetMulti] Atomiser.unet_datasets is required")
        self.datasets = {}
        for name, d in ds_cfg.items():
            self.datasets[name] = {
                "levels": int(d["levels"]),
                "km_train": [tuple(x) for x in d["km_train"]],
                "km_eval": tuple(d["km_eval"]),
                "num_classes": int(d["num_classes"]),
                "temporal": bool(d.get("temporal", False)),
            }
        L = max(d["levels"] for d in self.datasets.values())
        any_temporal = any(d["temporal"] for d in self.datasets.values())

        dims = a.get("unet_dims")
        dims = [int(x) for x in dims] if dims is not None else [int(a.get("unet_dim", 384))]
        if len(set(dims)) != 1:
            raise ValueError("[Atomiser_UnetMulti] one width for every level (shared global and "
                             f"query modules), got unet_dims={dims}")
        dim = dims[0]
        head_dim = int(a.get("unet_head_dim", 64))
        dropout = float(a.get("unet_dropout", 0.0))

        # ---- atom embedding (shared) -----------------------------------------
        legacy = a.get("unet_atom_features", "atomizer") == "legacy"
        if legacy:
            if lookup_table is not None and hasattr(lookup_table, "get_or_register_time_idx"):
                for doy in range(1, 367):                 # one day -> index map in every worker
                    lookup_table.get_or_register_time_idx(doy)
            self.embed = LegacyAtomEmbedding(int(a.get("unet_atom_dim", 128)),
                                             int(a.get("unet_num_spectral", 512)),
                                             int(a.get("unet_num_resolution", 64)),
                                             int(a.get("unet_num_time", 512)))
        else:
            self.embed = AtomEmbedding(config, lookup_table, a.get("unet_encodings_package"))
        atom_dim = self.embed.out_dim

        # ---- one hierarchy geometry per dataset (no parameters; own cache) ----
        self.pruning = {name: GeographicPruningUNet(
                            self.embed.sensor_geometry, levels=d["levels"],
                            atoms_per_latent_choices=sorted({k for k, _ in d["km_train"]} | {d["km_eval"][0]}))
                        for name, d in self.datasets.items()}

        # ---- leaf reader (shared) ------------------------------------------------
        if legacy:
            lh = int(a.get("unet_leaf_heads", max(1, atom_dim // 32)))
            lhd = int(a.get("unet_leaf_head_dim", atom_dim // lh))
        else:
            lh = int(a.get("unet_leaf_heads", a.get("cross_heads", 8)))
            lhd = int(a.get("unet_leaf_head_dim", max(1, dim // lh)))
        self.leaf = LeafEncoder(self.embed, atom_dim, dim, heads=lh, head_dim=lhd,
                                chunk=int(a.get("unet_leaf_chunk", 256)), legacy_offsets=legacy)
        self.empty_emb = nn.Parameter(torch.zeros(dim))

        # ---- the chain -------------------------------------------------------------
        refine = a.get("unet_dec_refine", [1] + [0] * (L - 2))
        if not isinstance(refine, (int, float)):
            # one entry per decoder stage of THIS chain (L - 1), finest first: missing entries
            # are 0, entries beyond the chain are unused (one list fits any chain length)
            refine = (list(refine) + [0] * (L - 1))[:L - 1]
        self.unet = ChainUNet(L, dim, head_dim, int(a.get("unet_depth", 2)),
                              int(a.get("unet_global_depth", 2)), dropout,
                              a.get("unet_fusion", "gate"), refine,
                              bool(a.get("unet_dec_ffn", False)),
                              bool(a.get("unet_region_size_mlp", False)))
        self.temporal = (PerLevelTemporalAggregation([dim] * L, a.get("unet_temporal_transformer", {}))
                         if any_temporal else None)

        # ---- query decoder (shared, without head) + one MLP head per dataset --------
        qdim = int(a.get("unet_query_dim", 256))
        if qdim % head_dim != 0:
            raise ValueError(f"unet_query_dim={qdim} must be divisible by unet_head_dim={head_dim}")
        self.decoder = QueryDecoder(self.embed, atom_dim, dim, qdim, 1, qdim // head_dim,
                                    int(a.get("unet_decoder_neighbors", 9)),
                                    int(a.get("unet_query_chunk", 16384)))
        self.decoder.head = nn.Identity()                  # features out; heads are per dataset
        hid = int(a.get("unet_head_hidden", 256))
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, hid), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hid, d["num_classes"]))
            for name, d in self.datasets.items()})

        self._frozen = False
        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[Atomiser_UnetMulti] {n:.1f}M params | chain of {L} levels, width {dim}, "
              f"depth {int(a.get('unet_depth', 2))} local + {int(a.get('unet_global_depth', 2))} global "
              f"| atoms={'legacy' if legacy else 'atomizer'} | temporal={'per_level' if any_temporal else 'none'}")
        for name, d in self.datasets.items():
            print(f"[Atomiser_UnetMulti]   {name}: levels 0-{d['levels'] - 1}, k,m train={d['km_train']} "
                  f"eval={d['km_eval']}, {d['num_classes']} classes, "
                  f"{'per-date' if d['temporal'] else 'single date'}")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _gather_inputs(batch):
        toks, masks = [], []
        for res in sorted(batch["groups"].keys(), key=float):
            g = batch["groups"][res]
            toks.append(g["tokens"])
            masks.append(g["mask"].bool())
        dim = toks[0].dim() - 2                                    # atom axis
        return torch.cat(toks, dim), torch.cat(masks, dim)

    def _dataset(self, batch, dataset):
        name = dataset or batch.get("dataset")
        if isinstance(name, (list, tuple)):
            if len(set(name)) != 1:
                raise ValueError(f"[Atomiser_UnetMulti] one dataset per batch, got {set(name)}")
            name = name[0]
        if name is None and len(self.datasets) == 1:
            name = next(iter(self.datasets))
        if name not in self.datasets:
            raise ValueError(f"[Atomiser_UnetMulti] unknown dataset {name!r}; "
                             f"known: {list(self.datasets)}")
        return name

    def gate_stats(self):
        return {f"gate_L{l}": (f.last_gate_mean.item() if f.last_gate_mean is not None else None)
                for l, f in enumerate(self.unet.fuse)}

    # ------------------------------------------------------------------ encoder
    def encode(self, batch, dataset=None):
        """-> (name, geom, maps, tokens_for_decoder, mask_for_decoder, n_dates).
        maps: the per-level encoder maps the decoder reads (temporally aggregated
        for per-date datasets)."""
        name = self._dataset(batch, dataset)
        cfg = self.datasets[name]
        k, m = random.choice(cfg["km_train"]) if self.training else cfg["km_eval"]
        pruning = self.pruning[name]
        tokens, mask = self._gather_inputs(batch)
        if tokens.dim() == 4:                                      # [B, T, N, 8]
            if not cfg["temporal"]:
                raise ValueError(f"[Atomiser_UnetMulti] {name}: per-date tokens but temporal: false")
            B, T, N, C = tokens.shape
            tok_sel, _, bias = pruning(tokens.reshape(B * T, N, C), mask.reshape(B * T, N), None,
                                       geo_k=m, atoms_per_latent=k)
        else:
            if cfg["temporal"]:
                raise ValueError(f"[Atomiser_UnetMulti] {name}: temporal: true expects per-date "
                                 "tokens [B, T, N, 8]")
            B, T = tokens.shape[0], 1
            tok_sel, _, bias = pruning(tokens, mask, None, geo_k=m, atoms_per_latent=k)
        geom = pruning.last_geometry
        if geom.num_levels != cfg["levels"]:
            raise RuntimeError(f"{name}: geometry has {geom.num_levels} levels, expected {cfg['levels']}")
        z0 = self.leaf(tok_sel, bias, geom)
        z0 = z0 + self.empty_emb * pruning.last_empty.unsqueeze(-1).to(z0.dtype)
        skips = self.unet.encode(z0, geom)
        if tokens.dim() == 4:
            skips_t = [s.view(B, T, s.shape[1], s.shape[2]) for s in skips]
            tp = batch.get("time_positions", batch.get("time_indices", None))
            tp = None if tp is None else tp.to(z0.device).float()
            maps = self.temporal(skips_t, tp, mask.all(-1))      # levels 0 .. S-1 only
            return name, geom, maps, tokens.reshape(B, T * N, C), mask.reshape(B, T * N), T
        return name, geom, skips, tokens, mask, 1

    # ------------------------------------------------------------------ forward
    def forward(self, batch, training=False, return_for_error=False, task=None, dataset=None,
                return_levels=False):
        name, geom, maps, tok, msk, T = self.encode(batch, dataset)
        if return_levels:
            d0, dec_levels = self.unet.decode(maps, geom, return_levels=True)
        else:
            d0 = self.unet.decode(maps, geom)
        feats = self.decoder(batch["queries"], tok, msk, d0, geom, n_dates=T)
        out = {"predictions": self.heads[name](feats), "num_latents": geom.sizes[0], "dataset": name}
        if return_levels:
            out.update(geom=geom, enc_levels=maps, dec_levels=dec_levels)
        return out

    # ------------------------------------------------------------------ frozen-encoder protocol
    def encoder_modules(self):
        mods = [self.embed, self.leaf, self.unet.local, self.unet.merge, self.unet.glob]
        if self.unet.region is not None:
            mods.append(self.unet.region)
        if self.temporal is not None:
            mods.append(self.temporal)
        return mods

    def encoder_parameters(self):
        params = [p for mod in self.encoder_modules() for p in mod.parameters()]
        return params + [self.empty_emb] + list(self.unet.seeds)

    def freeze_encoder(self, frozen=True):
        """Freeze the encoder side (and keep it in eval mode, so its dropout is off);
        the decoder stages, query decoder and heads stay trainable."""
        self._frozen = frozen
        for p in self.encoder_parameters():
            p.requires_grad_(not frozen)
        self.train(self.training)
        n_f = sum(p.numel() for p in self.encoder_parameters())
        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Atomiser_UnetMulti] encoder {'FROZEN' if frozen else 'trainable'}: "
              f"{n_f / 1e6:.1f}M encoder params, {n_t / 1e6:.2f}M trainable")
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_frozen", False):
            for mod in self.encoder_modules():
                mod.eval()
        return self
