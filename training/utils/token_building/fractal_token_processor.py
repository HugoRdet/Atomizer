"""
FractalTokenProcessor — TokenProcessor subclass with echo-aware routing
========================================================================

Overrides `process_data_for_encoder` to route col 7 of each token through
either the time encoder (for non-LIDAR tokens) or an echo encoder (for
LIDAR tokens identified by spectral_idx == ELEVATION's idx).

Rationale
---------
TokenBuilder.build_sparse_tokens writes the echo lookup index into col 7
of LIDAR tokens. The parent TokenProcessor reads col 7 as a time index
unconditionally and runs it through the time encoder. For LIDAR tokens
this produces meaningless temporal embeddings rather than usable echo
context (return number / total returns).

This subclass fixes that by:
  1. Detecting LIDAR tokens via their spectral_idx
  2. Reading col 7 as echo_idx for those tokens
  3. Gathering the continuous (a, b) encoding from the lookup table's
     pre-computed echo LUT, where:
         a = (return_number - 1) / total_returns      (proportion above)
         b = (total_returns - return_number) / total_returns  (proportion below)
  4. Running (a, b) through a small MLP to produce features that match
     the time encoder's output dimension, so the encoder MLP downstream
     doesn't change shape
  5. Using torch.where to select per-token between time features (non-LIDAR)
     and echo features (LIDAR)

The dispatch happens at the tensor level (torch.where), so it's GPU-friendly
and adds negligible overhead.

Echo MLP design
---------------
The MLP is intentionally small: 2 → 64 → time_encoder.out_dim, 2 layers
with GELU and LayerNorm. The input (a, b) is a pair of scalars in [0, 1]
encoding integer ratios, so there's no value in Fourier-expanding them.
The MLP gives the model nonlinear capacity to learn an embedding for each
discrete (a, b) combination.

Final layer is zero-initialized so echo features start at exactly zero,
matching the parent's behavior for time_idx = -1 (zeros). This means the
LIDAR tokens behave identically to the legacy buggy code at init and the
model gradually learns to use the echo information.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .processor import (
    TokenProcessor,
    build_mlp,
    TOKEN_DIM,
    TOKEN_VALUE_IDX,
    TOKEN_X_IDX,
    TOKEN_Y_IDX,
    TOKEN_SPECTRAL_IDX,
    TOKEN_LABEL_IDX,
    TOKEN_QUERY_IDX,
    TOKEN_RESOLUTION_IDX,
    TOKEN_TIME_IDX,
)


class FractalTokenProcessor(TokenProcessor):
    """
    TokenProcessor with FRACTAL-specific echo encoding for LIDAR tokens.

    Public API matches the parent — `process_data_for_encoder` has the
    same signature and return shape. Only the internal temporal feature
    computation differs.

    Args:
        config:       Atomizer config dict
        lookup_table: Lookup_encoding instance. Must have:
                        - get_abstract_channel_idx("ELEVATION") returning
                          the spectral index assigned to LIDAR elevation
                        - build_echo_continuous_lut() returning the
                          [num_echo_indices, 2] table of (a, b) values
    """

    def __init__(self, config, lookup_table):
        super().__init__(config, lookup_table)

        # ── 1. Resolve ELEVATION spectral index ───────────────────────
        # LIDAR tokens are tagged with this spectral index by TokenBuilder.
        # If ELEVATION wasn't registered, this raises with a clear message.
        try:
            elev_idx = lookup_table.get_abstract_channel_idx("ELEVATION")
        except KeyError as e:
            raise KeyError(
                "FractalTokenProcessor requires 'ELEVATION' to be registered "
                "as an abstract channel in the lookup table. Call "
                "lookup_table.register_abstract_channel('ELEVATION') before "
                "instantiating the model."
            ) from e

        # Stored as a buffer so it moves with the model and shows up in
        # state_dict (consistent treatment across DDP ranks).
        self.register_buffer(
            "elevation_spectral_idx",
            torch.tensor(elev_idx, dtype=torch.long),
        )

        # ── 2. Build the echo continuous LUT as a buffer ──────────────
        # echo_lut: [num_echo_indices, 2]
        # Row 0 is (0, 0) for LEARNED_ECHO_IDX (non-LIDAR fallback).
        # Rows ≥ 1 are the (a, b) encodings for specific (r, t) pairs.
        echo_lut = lookup_table.build_echo_continuous_lut()
        self.register_buffer("echo_lut", echo_lut)
        self.num_echo_indices = echo_lut.shape[0]

        # ── 3. Echo encoder MLP ───────────────────────────────────────
        # Maps (a, b) ∈ ℝ² → time_encoder.out_dim features so the
        # downstream encoder_mlp input dim stays unchanged.
        echo_hidden = 64
        self.echo_encoder = build_mlp(
            in_dim=2,
            hidden_dim=echo_hidden,
            out_dim=self.time_encoder.out_dim,
            num_layers=2,
        )

        # Zero-init the final linear so echo features start at 0.
        # Matches the parent's behavior for time_idx = -1 (which the
        # time encoder maps to zeros), giving identical behavior at
        # init. The model then learns to use the echo signal.
        last_linear = None
        for module in self.echo_encoder:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        print(f"[FractalTokenProcessor] Echo routing enabled.")
        print(f"[FractalTokenProcessor]   ELEVATION spectral_idx = {elev_idx}")
        print(f"[FractalTokenProcessor]   echo_lut shape = {tuple(echo_lut.shape)} "
              f"(idx 0 = no-echo, rows 1..{self.num_echo_indices - 1} = (r,t) pairs)")
        print(f"[FractalTokenProcessor]   echo MLP: 2 → {echo_hidden} → "
              f"{self.time_encoder.out_dim} [final layer zero-initialized]")

    # =========================================================================
    # OVERRIDE: process_data_for_encoder
    # =========================================================================

    def process_data_for_encoder(
        self,
        token_data: torch.Tensor,
        mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        FRACTAL-aware version of the parent method.

        Identical to TokenProcessor.process_data_for_encoder except that
        the temporal features for LIDAR tokens are computed from echo
        metadata rather than from the time encoder.

        Args:
            token_data:       [B, L, m, 8] or [B, L, m, 8+2K] (with temporal profile)
            mask:             [B, L, m] bool mask (True = masked out)
            latent_positions: [B, L, 2] latent grid coordinates in meters

        Returns:
            features: [B, L, m, tokenizer_out_dim]
        """
        B, L, m, C = token_data.shape

        # ── Detect temporal profile (parent compatibility) ────────────
        # FRACTAL doesn't use temporal profiles in practice, but we keep
        # the branch so the override stays drop-in for any future use.
        has_profile = self.has_temporal_profile and C > TOKEN_DIM
        if has_profile:
            K           = self.n_temporal_centers
            base_tokens = token_data[..., :TOKEN_DIM]
            profiles    = token_data[..., TOKEN_DIM:TOKEN_DIM + K]
            supports    = token_data[..., TOKEN_DIM + K:TOKEN_DIM + 2 * K]
        else:
            base_tokens = token_data

        # ── Step 1: Relative positions (meters) ───────────────────────
        token_coords = self.geometry.get_token_centers(base_tokens)

        if latent_positions is not None:
            latent_coords = latent_positions.unsqueeze(2).expand(-1, -1, m, -1)
        else:
            raise ValueError(
                "latent_positions must be provided to process_data_for_encoder"
            )

        delta_x = token_coords[..., 0] - latent_coords[..., 0]
        delta_y = token_coords[..., 1] - latent_coords[..., 1]

        gsd = (self._constant_gsd if self.use_constant_gsd
               else self.geometry.get_token_gsd(base_tokens))

        # ── Step 2: Sub-encodings ──────────────────────────────────────

        # Positional (relative, compressed)
        compression_scale = self.compression_alpha * gsd
        pos_features = self.pos_encoder(
            delta_x, delta_y, compression_scale=compression_scale
        )
        if pos_features.dim() < 4:
            pos_features = pos_features.unsqueeze(-2)

        # Spectral
        channel_indices  = base_tokens[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)

        # Reflectance
        b_values             = base_tokens[..., TOKEN_VALUE_IDX]
        reflectance_features = self.reflectance_encoder(b_values)
        if reflectance_features.dim() < 4:
            reflectance_features = reflectance_features.unsqueeze(-2)

        # Resolution
        resolution_indices  = base_tokens[..., TOKEN_RESOLUTION_IDX].long()
        resolution_features = self.resolution_encoder(resolution_indices)

        # ── Temporal / echo (the only block that differs from parent) ─
        if has_profile:
            # Temporal profile path: keep parent behavior, no echo routing.
            # (Would only fire if a future FRACTAL variant added profiles.)
            temporal_features = torch.cat([profiles, supports], dim=-1)
        else:
            temporal_features = self._compute_temporal_or_echo(
                base_tokens, channel_indices
            )

        # ── Step 3: Concatenate + project (unchanged) ─────────────────
        raw_features = torch.cat([
            pos_features,
            spectral_features,
            reflectance_features,
            resolution_features,
            temporal_features,
        ], dim=-1)

        return self.encoder_mlp(raw_features)

    # =========================================================================
    # Echo routing helper
    # =========================================================================

    def _compute_temporal_or_echo(
        self,
        base_tokens:      torch.Tensor,
        channel_indices:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute col-7 features with modality-aware routing.

        For LIDAR tokens (spectral_idx == ELEVATION's idx):
            col 7 holds echo_idx → gather (a, b) from echo_lut →
            echo_encoder → time_encoder.out_dim features.

        For all other tokens:
            col 7 holds time_idx → time_encoder → time_encoder.out_dim
            features (parent behavior).

        Returns:
            features: [B, L, m, time_encoder.out_dim]
        """
        col_7 = base_tokens[..., TOKEN_TIME_IDX].long()         # [B, L, m]

        # ── Time branch: parent behavior ──────────────────────────────
        # Time encoder is responsible for handling -1 → zeros internally.
        time_features = self.time_encoder(col_7)                # [B, L, m, T]

        # ── Echo branch: lookup → MLP ─────────────────────────────────
        # Clamp col_7 to a valid echo_lut range before indexing. This is
        # only needed because we compute echo features for ALL tokens
        # (including VHR with time_idx=-1) and select per-token via
        # torch.where afterwards. For non-LIDAR tokens the clamped value
        # leads to a garbage lookup, but those rows are discarded by
        # the torch.where below.
        echo_idx = col_7.clamp(min=0, max=self.num_echo_indices - 1)
        echo_ab = self.echo_lut[echo_idx]                       # [B, L, m, 2]
        echo_features = self.echo_encoder(echo_ab)              # [B, L, m, T]

        # ── Per-token selection ───────────────────────────────────────
        is_lidar = (channel_indices == self.elevation_spectral_idx)  # [B, L, m]
        features = torch.where(
            is_lidar.unsqueeze(-1),
            echo_features,
            time_features,
        )
        return features
