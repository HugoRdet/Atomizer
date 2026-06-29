import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .geometry import SensorGeometry
from .positional_encodings import build_position_encoder
from .spectral_encodings import build_spectral_encoder
from .reflectance_encodings import build_reflectance_encoder
from .resolution_encodings import build_resolution_encoder
from .time_encodings import build_time_encoder


# Token column indices (must match lookup_encoding.py)
TOKEN_VALUE_IDX      = 0
TOKEN_X_IDX          = 1
TOKEN_Y_IDX          = 2
TOKEN_SPECTRAL_IDX   = 3
TOKEN_LABEL_IDX      = 4
TOKEN_QUERY_IDX      = 5
TOKEN_RESOLUTION_IDX = 6
TOKEN_TIME_IDX       = 7
TOKEN_DIM            = 8


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int,
              norm_output: bool = True) -> nn.Sequential:
    """
    MLP with GELU activations and post-LayerNorm on hidden layers, plus an
    optional LayerNorm on the OUTPUT.

    num_layers=1 → single linear (no activation)
    num_layers=2 → Linear → GELU → LN → Linear
    num_layers=N → adds (N-2) hidden blocks between first and last linear

    norm_output=True appends a LayerNorm(out_dim) after the final Linear.
    """
    if num_layers == 1:
        layers = [nn.Linear(in_dim, out_dim)]
        if norm_output:
            layers.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*layers)

    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
    layers.append(nn.Linear(hidden_dim, out_dim))
    if norm_output:
        layers.append(nn.LayerNorm(out_dim))

    return nn.Sequential(*layers)


class TokenProcessor(nn.Module):
    """
    Orchestrates all sub-encoders and projects tokens into the latent space.

    Token format (8 columns):
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]

    With temporal profile (8 + 2K columns):
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx,
         profile_1..K, support_1..K]
        Profile + support are fed DIRECTLY into the encoder MLP (no projection
        bottleneck). The temporal slot expands from time_encoder.out_dim to 2K.

    Encoder pipeline:
        pos + spectral + reflectance + resolution + temporal → MLP → D

    Decoder pipeline:
        spectral + resolution + learned_embedding → MLP → D

    Encoder dim breakdown (with default config):
        pos:          514   (pos_num_freq_bands=128 → (1+2×128)×2)
        spectral:      19   (19 Gaussians from wavelengths_encoding)
        reflectance:   17   (bandvalue_num_freq_bands=8 → 1+2×8)
        resolution:    33   (resolution_num_bands=16 → 1+2×16)
        time:          24   (time_num_centers=24, zeros for time_idx=-1)
        ─────────────────
        total:        607   → MLP(hidden=768, layers=2) → 128

    Decoder dim breakdown:
        spectral:  19
        resolution: 33
        ──────────────
        total:     52   → MLP(hidden=768, layers=2) → 128
    """

    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config       = config
        self.lookup_table = lookup_table

        self.use_constant_gsd = config["Atomiser"].get("use_constant_gsd", True)

        # ── 1. Physics engine ──────────────────────────────────────────
        self.geometry = SensorGeometry(config, lookup_table)

        # ── 2. Sub-encoders ────────────────────────────────────────────
        self.pos_encoder         = build_position_encoder(config)
        self.spectral_encoder    = build_spectral_encoder(config, lookup_table)
        self.reflectance_encoder = build_reflectance_encoder(config)
        self.resolution_encoder  = build_resolution_encoder(config, lookup_table)
        self.time_encoder        = build_time_encoder(config, lookup_table)
        self.compression_alpha   = config["Atomiser"].get("compression_alpha", 10.0)

        # ── 3. Temporal profile (direct feed, no projection) ───────────
        temporal_cfg = config.get("temporal_profile", {})
        self.has_temporal_profile = temporal_cfg.get("enabled", False)
        if self.has_temporal_profile:
            self.n_temporal_centers = temporal_cfg.get("n_centers", 24)
            self.temporal_dim       = self.n_temporal_centers * 2  # profile + support
            print(f"[TokenProcessor] Temporal profile: {self.temporal_dim} dims direct feed "
                  f"(replaces time_encoder {self.time_encoder.out_dim} dims)")
        else:
            self.n_temporal_centers = 0
            self.temporal_dim       = self.time_encoder.out_dim

        # ── 4. Cache constant GSD ──────────────────────────────────────
        if self.use_constant_gsd:
            self._constant_gsd = float(self.geometry.default_gsd)
        else:
            self._constant_gsd = None

        # ── 5. Encoder dim ─────────────────────────────────────────────
        self._raw_encoder_dim = (
            self.pos_encoder.out_dim
            + self.spectral_encoder.out_dim
            + self.reflectance_encoder.out_dim
            + self.resolution_encoder.out_dim
            + self.temporal_dim
        )

        # ── 6. Decoder dim ─────────────────────────────────────────────
        # No learned embedding — a constant vector broadcast to all queries
        # adds zero information beyond the MLP's own bias terms.
        self._raw_decoder_dim = (
            self.spectral_encoder.out_dim
            + self.resolution_encoder.out_dim
        )

        # ── 7. Projection MLPs ─────────────────────────────────────────
        tokenizer_hidden = config["Atomiser"]["tokenizer_hidden_size"]
        tokenizer_layers = config["Atomiser"]["tokenizer_nb_layers"]
        tokenizer_out    = config["Atomiser"]["tokenizer_out_dim"]

        self.encoder_mlp = build_mlp(
            in_dim=self._raw_encoder_dim,
            hidden_dim=tokenizer_hidden,
            out_dim=tokenizer_out,
            num_layers=tokenizer_layers,
        )

        self.decoder_mlp = build_mlp(
            in_dim=self._raw_decoder_dim,
            hidden_dim=tokenizer_hidden,
            out_dim=tokenizer_out,
            num_layers=tokenizer_layers,
        )

        # ── 8. Output dims ─────────────────────────────────────────────
        self._encoder_out_dim = tokenizer_out
        self._decoder_out_dim = tokenizer_out

        print(f"[TokenProcessor] Encoder: {self._raw_encoder_dim} → "
              f"MLP({tokenizer_layers}L, h={tokenizer_hidden}) → {tokenizer_out}")
        print(f"[TokenProcessor]   pos={self.pos_encoder.out_dim} + "
              f"spectral={self.spectral_encoder.out_dim} + "
              f"reflectance={self.reflectance_encoder.out_dim} + "
              f"resolution={self.resolution_encoder.out_dim} + "
              f"temporal={self.temporal_dim} = {self._raw_encoder_dim}")
        print(f"[TokenProcessor] Decoder: spectral({self.spectral_encoder.out_dim}) + "
              f"resolution({self.resolution_encoder.out_dim}) = {self._raw_decoder_dim} → "
              f"MLP({tokenizer_layers}L, h={tokenizer_hidden}) → {tokenizer_out}")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def encoder_output_dim(self) -> int:
        return self._encoder_out_dim

    @property
    def decoder_output_dim(self) -> int:
        return self._decoder_out_dim

    def get_encoder_output_dim(self) -> int:
        return self._encoder_out_dim

    def get_decoder_output_dim(self) -> int:
        return self._decoder_out_dim

    # =========================================================================
    # ENCODER PIPELINE
    # =========================================================================

    def process_data_for_encoder(
        self,
        token_data: torch.Tensor,
        mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokens into latent-space features.

        Args:
            token_data:       [B, L, m, 8] or [B, L, m, 8+2K] (with temporal profile)
            mask:             [B, L, m] bool mask (True = masked out)
            latent_positions: [B, L, 2] latent grid coordinates in meters

        Returns:
            features: [B, L, m, tokenizer_out_dim]
        """
        B, L, m, C = token_data.shape

        # ── Detect temporal profile ────────────────────────────────────
        has_profile = self.has_temporal_profile and C > TOKEN_DIM
        if has_profile:
            K           = self.n_temporal_centers
            base_tokens = token_data[..., :TOKEN_DIM]
            profiles    = token_data[..., TOKEN_DIM:TOKEN_DIM + K]
            supports    = token_data[..., TOKEN_DIM + K:TOKEN_DIM + 2 * K]
        else:
            base_tokens = token_data

        # ── Step 1: Relative positions (meters) ───────────────────────
        token_coords = self.geometry.get_token_centers(base_tokens)  # [B, L, m, 2]

        if latent_positions is not None:
            latent_coords = latent_positions.unsqueeze(2).expand(-1, -1, m, -1)
        else:
            # Should not reach here in normal operation
            raise ValueError("latent_positions must be provided to process_data_for_encoder")

        delta_x = token_coords[..., 0] - latent_coords[..., 0]  # [B, L, m]
        delta_y = token_coords[..., 1] - latent_coords[..., 1]  # [B, L, m]

        gsd = (self._constant_gsd if self.use_constant_gsd
               else self.geometry.get_token_gsd(base_tokens))

        # ── Step 2: Sub-encodings ──────────────────────────────────────

        # Positional (relative, compressed)
        compression_scale = self.compression_alpha * gsd
        pos_features = self.pos_encoder(delta_x, delta_y, compression_scale=compression_scale)
        if pos_features.dim() < 4:
            pos_features = pos_features.unsqueeze(-2)

        # Spectral
        channel_indices  = base_tokens[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)

        # Reflectance
        b_values              = base_tokens[..., TOKEN_VALUE_IDX]
        reflectance_features  = self.reflectance_encoder(b_values)
        if reflectance_features.dim() < 4:
            reflectance_features = reflectance_features.unsqueeze(-2)

        # Resolution
        resolution_indices  = base_tokens[..., TOKEN_RESOLUTION_IDX].long()
        resolution_features = self.resolution_encoder(resolution_indices)

        # Temporal
        if has_profile:
            temporal_features = torch.cat([profiles, supports], dim=-1)
        else:
            time_indices      = base_tokens[..., TOKEN_TIME_IDX].long()
            temporal_features = self.time_encoder(time_indices)

        # ── Step 3: Concatenate + project ─────────────────────────────
        raw_features = torch.cat([
            pos_features,
            spectral_features,
            reflectance_features,
            resolution_features,
            temporal_features,
        ], dim=-1)

        return self.encoder_mlp(raw_features)

    # =========================================================================
    # DECODER PIPELINE
    # =========================================================================

    def process_data_for_decoder(
        self,
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        target_resolution: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[None, None]]:
        """
        Build query features for the decoder.

        Queries carry no reflectance value (to be predicted) and no
        positional encoding — spatial context reaches the decoder via
        relative position encoding computed inside _decode_single_grid.

        Args:
            query_tokens:      [B, N, 8]
            query_mask:        [B, N]
            target_resolution: optional GSD override for resolution encoding

        Returns:
            features:   [B, N, tokenizer_out_dim]
            query_mask: [B, N]  (unchanged)
            (None, None): placeholder, legacy bias is removed
        """
        assert query_tokens.shape[-1] == TOKEN_DIM, \
            f"Expected {TOKEN_DIM} columns, got {query_tokens.shape[-1]}"

        B, N, _ = query_tokens.shape
        device  = query_tokens.device

        # Spectral
        channel_indices   = query_tokens[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)

        # Resolution
        if target_resolution is not None:
            res_idx = self.lookup_table.get_resolution_idx(target_resolution)
            resolution_indices = torch.full((B, N), res_idx, dtype=torch.long, device=device)
        else:
            resolution_indices = query_tokens[..., TOKEN_RESOLUTION_IDX].long()
        resolution_features = self.resolution_encoder(resolution_indices)

        raw_features = torch.cat([spectral_features, resolution_features], dim=-1)
        features     = self.decoder_mlp(raw_features)

        # Legacy bias removed — was computed but never used downstream
        return features, query_mask, (None, None)

    # =========================================================================
    # ENCODER BIAS HELPER
    # =========================================================================

    def get_encoder_bias(
        self,
        token_data: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.geometry.get_encoder_bias(token_data, latent_positions)

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _zero_if_constant(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Zero out features that are constant across the batch (debug utility)."""
        flat = features.reshape(-1, features.shape[-1])
        if flat.var(dim=0).max().item() < eps:
            return torch.zeros_like(features)
        return features

    def extra_repr(self) -> str:
        temporal_str = (
            f"temporal_profile={self.n_temporal_centers}×2={self.temporal_dim} (direct)"
            if self.has_temporal_profile
            else f"time_encoder={self.time_encoder.out_dim}"
        )
        return (
            f"encoder: {self._raw_encoder_dim}→{self._encoder_out_dim}  "
            f"decoder: {self._raw_decoder_dim}→{self._decoder_out_dim}  "
            f"constant_gsd={self.use_constant_gsd}\n"
            f"  pos={self.pos_encoder.out_dim} + "
            f"spectral={self.spectral_encoder.out_dim} + "
            f"reflectance={self.reflectance_encoder.out_dim} + "
            f"resolution={self.resolution_encoder.out_dim} + "
            f"{temporal_str} = {self._raw_encoder_dim}"
        )
