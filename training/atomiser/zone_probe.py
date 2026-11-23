"""
Zone Probing for Adaptive Decode
================================

Parallel to GeographicPruning, but for the DECODER side. Where GeographicPruning
groups input TOKENS by their nearest latent (for encoding), ZoneProbe groups
query PIXELS by their nearest latent (for decoding), and selects a small probe
set per zone.

This is the geometry-only assignment that drives probe-and-broadcast decoding:
  - zone_id  [B, M]      : each query pixel's nearest latent (Voronoi cell)
  - probe_idx [B, L, k]  : k seeded-random pixel indices per zone (-1 = pad)
  - probe_cnt [B, L]     : number of valid probe pixels per zone (<= k)

Both outputs depend ONLY on query-pixel coords and latent coords (not image
content), exactly like GeographicPruning's Voronoi assignment — so they are
computed once per input geometry and are amortizable in the same way.

From these, the decode path derives everything:
  - probe pixels to decode  = probe_idx[probe_idx >= 0]
  - hard pixels to decode   = all pixels whose zone is flagged hard
  - easy pixels             = broadcast their zone's probe class

Usage (mirrors self.geo_pruning):
    self.zone_probe = ZoneProbe(geometry=self.input_processor.geometry)
    zone_id, probe_idx, probe_cnt = self.zone_probe(
        query_tokens, latent_coords, probe_k=10, seed=42)
"""

import torch
import torch.nn as nn


class ZoneProbe(nn.Module):
    """
    Assigns query pixels to their nearest latent (Voronoi zones) and selects a
    seeded-random probe set per zone. Geometry-only, content-independent.

    Mirrors GeographicPruning's role on the decoder side.
    """

    def __init__(self, geometry, cdist_chunk: int = 20000):
        """
        Args:
            geometry      : the input_processor.geometry object (provides
                            get_token_centers to map query tokens -> coords),
                            same object GeographicPruning uses.
            cdist_chunk   : chunk size for the pixel->latent distance computation
                            (bounds peak memory for large M).
        """
        super().__init__()
        self.geometry = geometry
        self.cdist_chunk = cdist_chunk

    @torch.no_grad()
    def forward(self, query_tokens, latent_coords, probe_k: int = 10,
                seed: int = 42):
        """
        Args:
            query_tokens  : [B, M, 8]  decoder query tokens (one per pixel)
            latent_coords : [B, L, 2]  spatial latent positions (decoder's
                                       all_coords, in the same frame as queries)
            probe_k       : probe pixels per zone
            seed          : RNG seed for the per-zone random probe selection
                            (deterministic / reproducible, matches the test)

        Returns:
            zone_id   : [B, M]      nearest-latent index per query pixel
            probe_idx : [B, L, k]   pixel indices of probe set per zone (-1 pad)
            probe_cnt : [B, L]      valid probe count per zone (<= k)
        """
        B, M, _ = query_tokens.shape
        L = latent_coords.shape[1]
        device = query_tokens.device

        # query-pixel coords in the same frame as latents (same call the
        # decoder's reconstruct() uses internally)
        qcoords = self.geometry.get_token_centers(query_tokens)        # [B, M, 2]

        zone_id   = torch.empty(B, M, dtype=torch.long, device=device)
        probe_idx = torch.full((B, L, probe_k), -1, dtype=torch.long, device=device)
        probe_cnt = torch.zeros(B, L, dtype=torch.long, device=device)

        for b in range(B):
            lc = latent_coords[b]                                       # [L, 2]
            qc = qcoords[b]                                             # [M, 2]

            # ── nearest-latent assignment (chunked Voronoi) ───────────
            zid = torch.empty(M, dtype=torch.long, device=device)
            cs = self.cdist_chunk
            for i in range(0, M, cs):
                d2 = torch.cdist(qc[i:i+cs], lc)                        # [chunk, L]
                zid[i:i+cs] = d2.argmin(dim=1)
            zone_id[b] = zid

            # ── per-zone members via sort (CSR-style), then seeded probe ─
            order = torch.argsort(zid)                                  # pixels grouped by zone
            counts = torch.bincount(zid, minlength=L)                   # [L]
            offsets = torch.zeros(L + 1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(counts, dim=0)

            g = torch.Generator(device="cpu"); g.manual_seed(seed + b)
            order_cpu = order.cpu()
            for z in range(L):
                s = int(offsets[z].item()); e = int(offsets[z + 1].item())
                n = e - s
                if n == 0:
                    continue
                kk = min(probe_k, n)
                perm = torch.randperm(n, generator=g)[:kk]
                probe_idx[b, z, :kk] = order_cpu[s:e][perm].to(device)
                probe_cnt[b, z] = kk

        return zone_id, probe_idx, probe_cnt
