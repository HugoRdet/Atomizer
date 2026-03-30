import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .geometry import SensorGeometry
from .positional_encodings import build_position_encoder
from .spectral_encodings import build_spectral_encoder
from .reflectance_encodings import build_reflectance_encoder
from .resolution_encodings import build_resolution_encoder
from .time_encodings import build_time_encoder


# Token column indices (must match lookup_encoding_v2.py)
TOKEN_VALUE_IDX      = 0
TOKEN_X_IDX          = 1
TOKEN_Y_IDX          = 2
TOKEN_SPECTRAL_IDX   = 3
TOKEN_LABEL_IDX      = 4
TOKEN_QUERY_IDX      = 5
TOKEN_RESOLUTION_IDX = 6
TOKEN_TIME_IDX       = 7
TOKEN_DIM            = 8


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """
    Build an MLP with GELU activations and Post-LayerNorm on hidden layers.
    
    The first Linear sees raw calibrated features directly (no input LN),
    preserving the per-encoder normalization schemes (RBF L2-norm, Fourier [-1,1], etc.).
    LayerNorm is applied only after activations on learned hidden representations.
    
    Architecture:
        num_layers=1: Linear(in → out)
        num_layers=2: Linear(in → hidden) → GELU → LN → Linear(hidden → out)
        num_layers=3: Linear(in → hidden) → GELU → LN → Linear(hidden → hidden) → GELU → LN → Linear(hidden → out)
    """
    if num_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
    layers.append(nn.Linear(hidden_dim, out_dim))
    
    return nn.Sequential(*layers)


class TokenProcessor(nn.Module):
    """
    The Orchestrator (Back-End Processor).
    
    Responsibilities:
    1. Coordinate Geometry (Physics) and Encodings (Math).
    2. Prepare feature tensors for the Transformer Encoder.
    3. Prepare query/bias tensors for the Transformer Decoder.
    4. Project raw features into model dimension via learned MLPs.
    
    Token Data Format (8 columns):
        [0]: Reflectance / band value
        [1]: X position index (global, includes modality offset)
        [2]: Y position index (global, includes modality offset)
        [3]: Channel/wavelength index (into table_wave)
        [4]: Label (unused by encoder, carried for loss)
        [5]: Query offset (identifies modality)
        [6]: Resolution index (-1 = non-optical/N/A, ≥0 = optical GSD)
        [7]: Time index (-1 = no temporal info, ≥0 = registered timestamp)
    
    Sentinel convention (project-wide):
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features
    
    Pipeline:
        Encoder: pos + spectral + reflectance + resolution + time → zero constants → MLP → [B, L, m, D]
        Decoder: spectral + resolution + learned_embedding → MLP → [B, N, D]
    
    Note on constant features in the encoder:
        On single-resolution/single-time datasets, resolution and/or time features
        are identical for every token. These are zeroed before the MLP because
        constant inputs cause coherent gradient accumulation across millions of
        tokens, destabilizing training. The decoder is unaffected (fewer tokens).
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table
        
        # Option: Use constant GSD (original behavior) or per-token GSD
        self.use_constant_gsd = config["Atomiser"].get("use_constant_gsd", True)
        
        # 1. The Physics Engine
        self.geometry = SensorGeometry(config, lookup_table)
        
        # 2. The Mathematical Encoders
        self.pos_encoder = build_position_encoder(config)
        self.spectral_encoder = build_spectral_encoder(config, lookup_table)
        self.reflectance_encoder = build_reflectance_encoder(config)
        self.resolution_encoder = build_resolution_encoder(config, lookup_table)
        self.time_encoder = build_time_encoder(config, lookup_table)
        self.compression_alpha = config["Atomiser"].get("compression_alpha", 10.0)
        # 3. Cache constant GSD
        if self.use_constant_gsd:
            self._constant_gsd = float(self.geometry.default_gsd)
        else:
            self._constant_gsd = None
        
        # 4. Raw feature dimensions (before MLP)
        self._raw_encoder_dim = (
            self.pos_encoder.out_dim
            + self.spectral_encoder.out_dim
            + self.reflectance_encoder.out_dim
            + self.resolution_encoder.out_dim
            + self.time_encoder.out_dim
        )
        # 5. Decoder learned embedding (task prior)
        self.decoder_learned_dim = config["Atomiser"].get("decoder_learned_dim", 32)
        self.decoder_learned_embedding = nn.Parameter(
            torch.randn(self.decoder_learned_dim) * 0.02
        )
        
        self._raw_decoder_dim = (
            self.spectral_encoder.out_dim
            + self.resolution_encoder.out_dim
            + self.decoder_learned_dim
        )
        
        # 6. Projection MLPs (Post-LN on hidden layers, raw features seen directly)
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
        
        # 7. Final output dimensions (after MLP)
        self._encoder_out_dim = tokenizer_out
        self._decoder_out_dim = tokenizer_out
        
        print(f"[TokenProcessor] Encoder: {self._raw_encoder_dim} → "
              f"MLP({tokenizer_layers} layers, hidden={tokenizer_hidden}) → "
              f"{tokenizer_out}")
        print(f"[TokenProcessor] Decoder: spectral({self.spectral_encoder.out_dim}) + "
              f"resolution({self.resolution_encoder.out_dim}) + "
              f"learned({self.decoder_learned_dim}) = {self._raw_decoder_dim} → "
              f"MLP({tokenizer_layers} layers, hidden={tokenizer_hidden}) → "
              f"{tokenizer_out}")

    @property
    def encoder_output_dim(self) -> int:
        """Total feature dimension for encoder input (after MLP)."""
        return self._encoder_out_dim
    
    @property
    def decoder_output_dim(self) -> int:
        """Feature dimension for decoder queries (after MLP)."""
        return self._decoder_out_dim
    
    # Backward compatibility
    def get_encoder_output_dim(self) -> int:
        return self._encoder_out_dim
    
    def get_decoder_output_dim(self) -> int:
        return self._decoder_out_dim

    def process_data_for_encoder(
        self, 
        token_data: torch.Tensor, 
        mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Main pipeline for the Encoder.
        
        Args:
            token_data: [B, L, m, 8] Raw token attributes
            mask: [B, L, m] Valid token mask
            latent_positions: [B, L, 2] Optional custom latent positions in meters.
        
        Returns:
            features: [B, L, m, tokenizer_out_dim] Projected features for the transformer.
        """
        B, L, m, C = token_data.shape
        device = token_data.device
        
        # =========================================================
        # STEP 1: PHYSICS (Get Coordinates)
        # =========================================================
        
        # A. Token Centers in Meters: [B, L, m, 2]
        token_coords = self.geometry.get_token_centers(token_data)
        
        # B. Latent Centers in Meters
        if latent_positions is not None:
            latent_coords = latent_positions.unsqueeze(2).expand(-1, -1, m, -1)
        else:
            grid = self.geometry.get_default_latent_grid(device)
            latent_coords = grid.view(1, L, 1, 2).expand(B, -1, m, -1)
        
        # C. Relative Displacement: [B, L, m]
        delta_x = token_coords[..., 0] - latent_coords[..., 0]
        delta_y = token_coords[..., 1] - latent_coords[..., 1]
        
        
        
        # E. GSD: scalar or [B, L, m]
        gsd = self._constant_gsd if self.use_constant_gsd else self.geometry.get_token_gsd(token_data)

        # =========================================================
        # STEP 2: ENCODINGS
        # =========================================================
        
        # A. Positional: [B, L, m, pos_dim]
        compression_scale = self.compression_alpha * gsd
        pos_features = self.pos_encoder(delta_x, delta_y, compression_scale=compression_scale)
        if len(pos_features.shape) < 4:
            pos_features = pos_features.unsqueeze(-2)
        
        # B. Spectral: [B, L, m, spec_dim]
        channel_indices = token_data[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)
        
        # C. Reflectance: [B, L, m, refl_dim]
        b_values = token_data[..., TOKEN_VALUE_IDX]
        reflectance_features = self.reflectance_encoder(b_values)
        if len(reflectance_features.shape) < 4:
            reflectance_features = reflectance_features.unsqueeze(-2)
        
        # D. Resolution: [B, L, m, res_dim]
        # Encoder handles -1 → zeros natively
        resolution_indices = token_data[..., TOKEN_RESOLUTION_IDX].long()
      
        resolution_features = self.resolution_encoder(resolution_indices)
        
        # E. Time: [B, L, m, time_dim]
        # Encoder handles -1 → zeros natively
        time_indices = token_data[..., TOKEN_TIME_IDX].long()
        time_features = self.time_encoder(time_indices)

        
        
        
        # =========================================================
        # STEP 3: ZERO CONSTANT METADATA (encoder only)
        # =========================================================
        # Constant features carry zero discriminative signal and cause
        # coherent gradient accumulation across all tokens, preventing
        # convergence. We detect and zero them before the MLP.
        # (The decoder path is unaffected — far fewer tokens.)
        
        #resolution_features = self._zero_if_constant(resolution_features)
        
        
        # =========================================================
        # STEP 4: ASSEMBLY + PROJECTION
        # =========================================================

        
        raw_features = torch.cat([
            pos_features,
            spectral_features,
            reflectance_features,
            resolution_features,
            time_features,
        ], dim=-1)  # [B, L, m, raw_encoder_dim]


        
        features = self.encoder_mlp(raw_features)  # [B, L, m, tokenizer_out_dim]
        
        return features

    def process_data_for_decoder(
        self, 
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        target_resolution: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pipeline for the Decoder (Query construction).
        
        Query features = spectral + resolution + learned_embedding → MLP.
        
        The learned embedding is a task prior that gets broadcast to all
        queries and mixed with spectral/resolution features through the MLP.
        
        Resolution comes from either:
          (a) target_resolution arg (float m/px) → looked up & broadcast to all queries
          (b) per-token resolution_idx already in query_tokens[:, :, 6]
        
        Args:
            query_tokens:      [B, N, 8] Raw query data
            query_mask:        [B, N] Valid query mask
            target_resolution: Optional float (m/px). If provided, overrides
                               the per-token resolution_idx for all queries.
            
        Returns:
            features: [B, N, tokenizer_out_dim] Projected query features
            mask:     [B, N] Passed through
            bias:     (token_bias, latent_bias) For relative attention
        """
        assert query_tokens.shape[-1] == TOKEN_DIM, \
            f"Expected {TOKEN_DIM} columns, got {query_tokens.shape[-1]}"
        
        B, N, _ = query_tokens.shape
        device = query_tokens.device
        
        # ── Spectral encoding ──────────────────────────────
        channel_indices = query_tokens[..., TOKEN_SPECTRAL_IDX].long()
        spectral_features = self.spectral_encoder(channel_indices)  # [B, N, spec_dim]
        
        # ── Resolution encoding ────────────────────────────
        # Encoder handles -1 → zeros natively
        if target_resolution is not None:
            # Uniform resolution for all queries in the batch
            res_idx = self.lookup_table.get_resolution_idx(target_resolution)
            resolution_indices = torch.full(
                (B, N), res_idx, dtype=torch.long, device=device
            )
        else:
            # Per-token resolution from column 6
            resolution_indices = query_tokens[..., TOKEN_RESOLUTION_IDX].long()
        
        resolution_features = self.resolution_encoder(resolution_indices)  # [B, N, res_dim]
        
        # ── Combine + Project ──────────────────────────────
        # Broadcast learned embedding: [decoder_learned_dim] → [B, N, decoder_learned_dim]
        learned = self.decoder_learned_embedding.expand(B, N, -1)
        
        raw_features = torch.cat([spectral_features, resolution_features, learned], dim=-1)
        features = self.decoder_mlp(raw_features)  # [B, N, tokenizer_out_dim]
        
        # ── Bias (legacy edge-based format) ────────────────
        bias_tokens, bias_latents = self.geometry.get_decoder_bias_legacy(query_tokens)
        
        return features, query_mask, (bias_tokens, bias_latents)

    def get_encoder_bias(
        self,
        token_data: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get bias data for encoder cross-attention.
        
        Args:
            token_data: [B, L, m, 8]
            latent_positions: [B, L, 2] or None
            
        Returns:
            token_bias: [B, L, m, 2, 2] edge bounds
            latent_bias: [B, L, 2] positions
        """
        return self.geometry.get_encoder_bias(token_data, latent_positions)

    @staticmethod
    def _zero_if_constant(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Zero out feature tensors that are constant across all tokens.
        
        Constant features carry zero mutual information with labels but cause
        coherent gradient accumulation: since every token contributes the same
        input to the MLP, gradients for the corresponding weight columns sum
        constructively instead of partially cancelling. With millions of tokens,
        this produces disproportionately large updates that destabilize training.
        
        This is NOT a normalization issue (LayerNorm doesn't help) — it's a 
        fundamental property of gradient accumulation with identical inputs.
        
        Detection: we check variance across the token dimensions. For 4D tensors
        [B, L, m, D], we flatten B×L×m and check if any feature dim varies.
        
        Args:
            features: [..., D] feature tensor
            eps: variance threshold below which features are considered constant
            
        Returns:
            features or zeros (same shape, same device, same dtype)
        """
        # Flatten all dims except the last (feature dim)
        flat = features.reshape(-1, features.shape[-1])
       
        # Check if ALL feature dims are constant across tokens
        if flat.var(dim=0).max().item() < eps:
            return torch.zeros_like(features)
        
        
        return features
    
    def extra_repr(self) -> str:
        return (
            f"encoder_dim={self._raw_encoder_dim}→{self._encoder_out_dim}, "
            f"decoder_dim={self._raw_decoder_dim}→{self._decoder_out_dim}, "
            f"constant_gsd={self.use_constant_gsd}\n"
            f"  raw encoder: pos={self.pos_encoder.out_dim} + "
            f"spectral={self.spectral_encoder.out_dim} + "
            f"reflectance={self.reflectance_encoder.out_dim} + "
            f"resolution={self.resolution_encoder.out_dim} + "
            f"time={self.time_encoder.out_dim} = {self._raw_encoder_dim}\n"
            f"  raw decoder: spectral={self.spectral_encoder.out_dim} + "
            f"resolution={self.resolution_encoder.out_dim} + "
            f"learned={self.decoder_learned_dim} = {self._raw_decoder_dim}"
        ) 