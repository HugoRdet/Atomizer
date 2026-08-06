"""
DalesTokenProcessor — TokenProcessor subclass with:
  1. Echo-aware routing for LIDAR tokens (col 7: echo_idx vs time_idx)
  2. Intensity-aware routing for LIDAR tokens (col 6: intensity vs
     resolution_idx)
==============================================================================

Adapted from FractalTokenProcessor. DALES points are now single-channel
LIDAR tokens (elevation in col 0, via the generic reflectance encoder —
no separate intensity channel/token). Instead, intensity rides in column 6
(normally resolution_idx), since GSD is constant/uninformative for LIDAR
anyway — repurposing that column rather than doubling token count.

Both column 6 and column 7 need per-token routing between "real" (for
non-LIDAR tokens, e.g. any VHR/raster tokens sharing this processor) and
"repurposed" (for LIDAR tokens) interpretations:

    col 6: resolution_idx (categorical, embedded via resolution_encoder)
           for non-LIDAR tokens; raw intensity value (continuous, via a
           small intensity_encoder) for LIDAR tokens.
    col 7: time_idx (categorical, via time_encoder) for non-LIDAR tokens;
           echo_idx (looked up + encoded via echo_encoder) for LIDAR
           tokens — this part is unchanged from FractalTokenProcessor.

Both branches are computed for ALL tokens then selected per-token via
torch.where — same pattern as FractalTokenProcessor's echo routing, just
applied to two columns instead of one.

REQUIRES: TokenBuilder.build_sparse_tokens called with intensity_override
set for LIDAR points (see token_builder_PATCH_intensity.txt) — otherwise
column 6 for LIDAR tokens still holds a real resolution_idx, and this
processor's is_lidar routing would incorrectly discard that column's
input in favor of treating it as an intensity encoder's raw input.
"""

import torch
import torch.nn as nn
from typing import Optional

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


class DalesTokenProcessor(TokenProcessor):
    """
    TokenProcessor with DALES-specific echo (col 7) AND intensity (col 6)
    routing for LIDAR tokens — single channel (elevation), no separate
    intensity token.

    Args:
        config:       Atomizer config dict
        lookup_table: Lookup_encoding instance. Must have:
                        - get_abstract_channel_idx("ELEVATION")
                        - build_echo_continuous_lut() returning the
                          [num_echo_indices, 2] table of (a, b) values
    """

    def __init__(self, config, lookup_table):
        super().__init__(config, lookup_table)

        # ── 1. Resolve ELEVATION spectral index ─────────────────────────
        # Single LIDAR channel now — both echo AND intensity routing key
        # off this same mask (a token is either "the LIDAR point" or not).
        try:
            elev_idx = lookup_table.get_abstract_channel_idx("ELEVATION")
        except KeyError as e:
            raise KeyError(
                "DalesTokenProcessor requires 'ELEVATION' to be registered "
                "as an abstract channel in the lookup table. Call "
                "lookup_table.register_abstract_channel('ELEVATION') before "
                "instantiating the model."
            ) from e

        self.register_buffer(
            "elevation_spectral_idx",
            torch.tensor(elev_idx, dtype=torch.long),
        )

        # ── 2. Echo continuous LUT (unchanged from FractalTokenProcessor) ─
        echo_lut = lookup_table.build_echo_continuous_lut()
        self.register_buffer("echo_lut", echo_lut)
        self.num_echo_indices = echo_lut.shape[0]

        echo_hidden = 64
        self.echo_encoder = build_mlp(
            in_dim=2,
            hidden_dim=echo_hidden,
            out_dim=self.time_encoder.out_dim,
            num_layers=2,
        )
        self._zero_init_last_linear(self.echo_encoder)

        # ── 3. NEW: intensity encoder ────────────────────────────────────
        # Maps a single scalar (normalized intensity, col 6 for LIDAR
        # tokens) -> resolution_encoder.out_dim features, so the encoder
        # MLP's input dim stays unchanged whether a token is LIDAR or not.
        intensity_hidden = 64
        self.intensity_encoder = build_mlp(
            in_dim=1,
            hidden_dim=intensity_hidden,
            out_dim=self.resolution_encoder.out_dim,
            num_layers=2,
        )
        # Zero-init so intensity features start at exactly zero — matches
        # the same "start neutral, let the model learn to use it" pattern
        # as the echo encoder, rather than injecting untrained noise into
        # the resolution slot from step 0.
        self._zero_init_last_linear(self.intensity_encoder)

        print(f"[DalesTokenProcessor] Echo routing (col 7) + intensity "
              f"routing (col 6) enabled for LIDAR tokens.")
        print(f"[DalesTokenProcessor]   ELEVATION spectral_idx = {elev_idx}")
        print(f"[DalesTokenProcessor]   echo_lut shape = {tuple(echo_lut.shape)} "
              f"(idx 0 = no-echo, rows 1..{self.num_echo_indices - 1} = (r,t) pairs)")
        print(f"[DalesTokenProcessor]   echo MLP: 2 → {echo_hidden} → "
              f"{self.time_encoder.out_dim} [zero-initialized]")
        print(f"[DalesTokenProcessor]   intensity MLP: 1 → {intensity_hidden} → "
              f"{self.resolution_encoder.out_dim} [zero-initialized]")

    @staticmethod
    def _zero_init_last_linear(mlp: nn.Sequential):
        last_linear = None
        for module in mlp:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    # =========================================================================
    # OVERRIDE: process_data_for_encoder
    # =========================================================================

    def process_data_for_encoder(
        self,
        token_data: torch.Tensor,
        mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, m, C = token_data.shape

        has_profile = self.has_temporal_profile and C > TOKEN_DIM
        if has_profile:
            K           = self.n_temporal_centers
            base_tokens = token_data[..., :TOKEN_DIM]
            profiles    = token_data[..., TOKEN_DIM:TOKEN_DIM + K]
            supports    = token_data[..., TOKEN_DIM + K:TOKEN_DIM + 2 * K]
        else:
            base_tokens = token_data

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

        compression_scale = self.compression_alpha * gsd
        pos_features = self.pos_encoder(
            delta_x, delta_y, compression_scale=compression_scale
        )
        if pos_features.dim() < 4:
            pos_features = pos_features.unsqueeze(-2)

        channel_indices  = base_tokens[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)

        # Reflectance encoder handles elevation (col 0) generically —
        # no special-casing needed, same as before.
        b_values             = base_tokens[..., TOKEN_VALUE_IDX]
        reflectance_features = self.reflectance_encoder(b_values)
        if reflectance_features.dim() < 4:
            reflectance_features = reflectance_features.unsqueeze(-2)

        is_lidar = (channel_indices == self.elevation_spectral_idx)  # [B, L, m]

        # ── NEW: col 6 routing — intensity (LIDAR) vs resolution_idx (else) ─
        resolution_or_intensity_features = self._compute_resolution_or_intensity(
            base_tokens, is_lidar
        )

        # ── col 7 routing — echo (LIDAR) vs time (else), unchanged pattern ──
        if has_profile:
            temporal_features = torch.cat([profiles, supports], dim=-1)
        else:
            temporal_features = self._compute_temporal_or_echo(
                base_tokens, is_lidar
            )

        raw_features = torch.cat([
            pos_features,
            spectral_features,
            reflectance_features,
            resolution_or_intensity_features,
            temporal_features,
        ], dim=-1)

        return self.encoder_mlp(raw_features)

    # =========================================================================
    # NEW: intensity/resolution routing helper (col 6)
    # =========================================================================

    def _compute_resolution_or_intensity(
        self,
        base_tokens: torch.Tensor,
        is_lidar:    torch.Tensor,  # [B, L, m] bool
    ) -> torch.Tensor:
        """
        For LIDAR tokens: col 6 holds a raw, continuous, normalized
        intensity value -> intensity_encoder -> resolution_encoder.out_dim
        features.

        For all other tokens: col 6 holds resolution_idx (categorical) ->
        resolution_encoder -> resolution_encoder.out_dim features (parent
        behavior).

        Returns:
            features: [B, L, m, resolution_encoder.out_dim]
        """
        col_6 = base_tokens[..., TOKEN_RESOLUTION_IDX]  # [B, L, m], float

        # ── Resolution branch (parent behavior) ─────────────────────────
        # .long() truncation is safe here even for LIDAR tokens (whose
        # col_6 holds a value in ~[0, 1], truncating to index 0): those
        # rows are discarded by torch.where below, exactly the same
        # "compute both, select after" pattern as the echo routing.
        resolution_indices = col_6.long()
        resolution_features = self.resolution_encoder(resolution_indices)

        # ── Intensity branch (NEW) ───────────────────────────────────────
        intensity_value = col_6.unsqueeze(-1)  # [B, L, m, 1]
        intensity_features = self.intensity_encoder(intensity_value)

        # ── Per-token selection ──────────────────────────────────────────
        features = torch.where(
            is_lidar.unsqueeze(-1),
            intensity_features,
            resolution_features,
        )
        return features

    # =========================================================================
    # Echo routing helper (col 7) — unchanged from before, single-channel mask
    # =========================================================================

    def _compute_temporal_or_echo(
        self,
        base_tokens: torch.Tensor,
        is_lidar:    torch.Tensor,  # [B, L, m] bool — reused from col-6 routing
    ) -> torch.Tensor:
        """
        For LIDAR tokens: col 7 holds echo_idx -> gather (a, b) from
        echo_lut -> echo_encoder -> time_encoder.out_dim features.

        For all other tokens: col 7 holds time_idx -> time_encoder ->
        time_encoder.out_dim features (parent behavior).
        """
        col_7 = base_tokens[..., TOKEN_TIME_IDX].long()

        time_features = self.time_encoder(col_7)

        echo_idx = col_7.clamp(min=0, max=self.num_echo_indices - 1)
        echo_ab = self.echo_lut[echo_idx]
        echo_features = self.echo_encoder(echo_ab)

        features = torch.where(
            is_lidar.unsqueeze(-1),
            echo_features,
            time_features,
        )
        return features
