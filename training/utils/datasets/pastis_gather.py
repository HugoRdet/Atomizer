"""
PASTIS-HD SKIP: per-query gather index into a pixel's own atoms
================================================================

Drop-in addition for PastisHDDataset to emit `query_token_idx` /
`query_token_valid`, exactly like Sen1Floods11SkipDataset — but the per-pixel
atom set spans BANDS x TIMESTEPS (and both sensors), not just bands.

POOL LAYOUT (verified against TokenBuilder.build_tokens + the dataset's cats):
    sat_tokens = cat([ S2_f0(c h w), S2_f1, ..., S2_f(T-1),
                       S1_f0(c h w), S1_f1, ..., S1_f(T-1) ])
  - within a frame, build_tokens flattens channel-major `(c h w)->row`
    => pixel p = h*W + w sits at {p + c*HW} per (frame, sensor) sub-block.
  - frames are FRAME-MAJOR (torch.cat over t).
  - S2 block precedes S1 block (torch.cat([s2_tokens, s1_tokens])).

  So pixel p's atom rows are:
    S2:  { t*C2*HW + c*HW + p : t in 0..T-1, c in 0..C2-1 }
    S1:  (T*C2*HW) + { t*C1*HW + c*HW + p : t in 0..T-1, c in 0..C1-1 }

  Total atoms/pixel = T*(C2+C1).  No pixel filtering in build_tokens, so the
  stride is exact and data-independent (numerically verified).

IMPORTANT: indices are RELATIVE TO THIS SAMPLE's sat_tokens pool (the 10m
SAT group). They do NOT include the SPOT group (1m, separate group/shape).
The skip operates on the SAT group only — consistent with queries being built
at SAT_RESOLUTION.

USAGE — three edits to PastisHDDataset:
  1. add the two methods below to the class.
  2. in __getitem__, capture kept_indices from subsample_queries:
        queries, kept_indices = self.token_builder.subsample_queries(
            queries, max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX, prioritize_valid=True,
            return_indices=True)                       # <-- add return_indices
  3. after queries are built, compute + emit the gather index:
        T = s2_data.shape[0]              # actual sampled frame count
        C2 = self.NUM_S2_BANDS
        C1 = self.NUM_S1_BANDS if self.use_s1 else 0
        qti, qtv = self._build_query_token_index(T, C2, C1, H, W,
                                                  kept_indices=kept_indices)
        result["query_token_idx"]   = qti
        result["query_token_valid"] = qtv
  (validation/test path: kept_indices=None -> full pixel grid in order, since
   queries == build_queries order; but note PASTIS subsamples on ALL splits via
   max_queries, so capture kept_indices on every split — see note at bottom.)
"""

import torch


def _build_full_pixel_index(T, C2, C1, H, W):
    """
    Closed-form gather index for ALL pixels, pixel order p = h*W + w.

    Returns [H*W, T*(C2+C1)] long, columns ordered:
        [ S2: (t,c) frame-major then channel-major ] ++
        [ S1: (t,c) frame-major then channel-major ]
    Verified numerically against build_tokens' einops flatten.
    """
    HW = H * W
    p = torch.arange(HW)                                       # [HW]

    blocks = []
    # ── S2 sub-block: t*C2*HW + c*HW + p ──
    t2 = torch.arange(T).view(T, 1, 1)
    c2 = torch.arange(C2).view(1, C2, 1)
    s2 = (t2 * C2 * HW + c2 * HW).reshape(-1, 1) + p.view(1, -1)   # [T*C2, HW]
    blocks.append(s2)

    # ── S1 sub-block (offset by full S2 block): off + t*C1*HW + c*HW + p ──
    if C1 > 0:
        off = T * C2 * HW
        t1 = torch.arange(T).view(T, 1, 1)
        c1 = torch.arange(C1).view(1, C1, 1)
        s1 = (off + t1 * C1 * HW + c1 * HW).reshape(-1, 1) + p.view(1, -1)  # [T*C1, HW]
        blocks.append(s1)

    full = torch.cat(blocks, dim=0).t().contiguous()           # [HW, T*(C2+C1)]
    return full


def _build_query_token_index(self, T, C2, C1, H, W, kept_indices=None):
    """
    Per-query gather index into the pixel's own atoms (bands x timesteps).

    Args:
        T          : actual number of sampled frames in THIS sample.
        C2, C1     : S2 band count, S1 band count (C1=0 if use_s1 False).
        H, W       : SAT-resolution patch dims used to build sat_tokens.
        kept_indices: [N_q] long or None. The row positions (into the full
                      pixel grid) that subsample_queries kept, in the SAME
                      order as the returned queries. None -> full grid in order.

    Returns:
        idx   : [N_q, T*(C2+C1)] long  -- rows into sat_tokens
        valid : [N_q] bool             -- all True (closed form always resolves)
    """
    full = _build_full_pixel_index(T, C2, C1, H, W)            # [H*W, T*(C2+C1)]
    if kept_indices is None:
        idx = full
    else:
        idx = full[kept_indices]                               # [N_q, T*(C2+C1)]
    valid = torch.ones(idx.shape[0], dtype=torch.bool)
    return idx, valid


# Bind as methods (or paste the two functions into the class body directly).
# _build_full_pixel_index is static-like; expose it on the instance too.
def attach_to_dataset(cls):
    cls._build_full_pixel_index = staticmethod(_build_full_pixel_index)
    cls._build_query_token_index = _build_query_token_index
    return cls
