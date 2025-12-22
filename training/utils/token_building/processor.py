import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .geometry import SensorGeometry
from .positional_encodings import build_position_encoder
from .spectral_encodings import build_spectral_encoder
from .reflectance_encodings import build_reflectance_encoder


class TokenProcessor(nn.Module):
    """
    The Orchestrator (Back-End Processor).
    
    Responsibilities:
    1. Coordinate Geometry (Physics) and Encodings (Math).
    2. Prepare feature tensors for the Transformer Encoder.
    3. Prepare query/bias tensors for the Transformer Decoder.
    
    Token Data Format (6 columns):
        [0]: Reflectance / B-value (raw pixel intensity)
        [1]: X position index (global, includes modality offset)
        [2]: Y position index (global, includes modality offset)
        [3]: Channel/wavelength index (into table_wave)
        [4]: (unused or auxiliary)
        [5]: Query offset (identifies modality)
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        
        # Option: Use constant GSD (original behavior) or per-token GSD
        self.use_constant_gsd = config["Atomiser"].get("use_constant_gsd", True)
        
        # 1. The Physics Engine
        self.geometry = SensorGeometry(config, lookup_table)
        
        # 2. The Mathematical Encoders
        self.pos_encoder = build_position_encoder(config)
        self.spectral_encoder = build_spectral_encoder(config, lookup_table)
        self.reflectance_encoder = build_reflectance_encoder(config)
        
        # 3. Cache constant GSD to avoid per-call tensor allocation
        if self.use_constant_gsd:
            # Store as Python float - encoder will handle conversion efficiently
            self._constant_gsd = float(self.geometry.default_gsd)
        else:
            self._constant_gsd = None
        
        # 4. Cache output dimensions (computed once)
        self._encoder_out_dim = (
            self.pos_encoder.out_dim +
            self.spectral_encoder.out_dim +
            self.reflectance_encoder.out_dim
        )
        self._decoder_out_dim = self.spectral_encoder.out_dim
        
        # 5. Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required config keys exist."""
        required_keys = ["spatial_latents", "latent_surface"]
        for key in required_keys:
            if key not in self.config["Atomiser"]:
                raise ValueError(f"Missing required config key: Atomiser.{key}")

    @property
    def encoder_output_dim(self) -> int:
        """Total feature dimension for encoder input."""
        return self._encoder_out_dim
    
    @property
    def decoder_output_dim(self) -> int:
        """Feature dimension for decoder queries (spectral only)."""
        return self._decoder_out_dim
    
    # Keep methods for backward compatibility
    def get_encoder_output_dim(self) -> int:
        """Total feature dimension for encoder input."""
        return self._encoder_out_dim
    
    def get_decoder_output_dim(self) -> int:
        """Feature dimension for decoder queries (spectral only in legacy mode)."""
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
            token_data: [B, L, m, 6] Raw token attributes
            mask: [B, L, m] Valid token mask (used by attention, not here)
            latent_positions: [B, L, 2] Optional custom latent positions in meters.
        
        Returns:
            features: [B, L, m, D] Concatenated features for the transformer.
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
            # [B, L, 2] -> [B, L, m, 2]
            latent_coords = latent_positions.unsqueeze(2).expand(-1, -1, m, -1)
        else:
            # [L, 2] -> [B, L, m, 2]
            grid = self.geometry.get_default_latent_grid(device)
            latent_coords = grid.view(1, L, 1, 2).expand(B, -1, m, -1)
            
        # C. Relative Displacement: [B, L, m]
        delta_x = token_coords[..., 0] - latent_coords[..., 0]
        delta_y = token_coords[..., 1] - latent_coords[..., 1]

        # D. Normalization Scale: [B, 1, 1] (broadcasts)
        physical_scale = self.geometry.get_physical_scale(token_data)
        
        # E. GSD: scalar (cached) or [B, L, m] tensor
        gsd = self._constant_gsd if self.use_constant_gsd else self.geometry.get_token_gsd(token_data)

        # =========================================================
        # STEP 2: ENCODINGS
        # =========================================================
        
        # A. Positional: [B, L, m, pos_dim]
        pos_features = self.pos_encoder(delta_x, delta_y, physical_scale, gsd)
        
        # B. Spectral: [B, L, m, spec_dim]
        channel_indices = token_data[..., 3].long()
        spectral_features = self.spectral_encoder(channel_indices)
        
        # C. Reflectance: [B, L, m, refl_dim]
        b_values = token_data[..., 0]
        reflectance_features = self.reflectance_encoder(b_values)
        
        # =========================================================
        # STEP 3: ASSEMBLY
        # =========================================================
        
        features = torch.cat([
            pos_features, 
            spectral_features, 
            reflectance_features
        ], dim=-1)
        
        return features

    def process_data_for_decoder(
        self, 
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pipeline for the Decoder (Query construction).
        
        Matches legacy 'apply_transformations_optique' behavior:
        - Query content = spectral encoding only
        - Bias = edge-based coordinate bounds
        
        Args:
            query_tokens: [B, N, 6] Raw query data
            query_mask: [B, N] Valid query mask
            
        Returns:
            features: [B, N, spec_dim] Query content features
            mask: [B, N] Passed through
            bias: (token_bias, latent_bias) For relative attention
        """
        # Debug assertion
        assert query_tokens.shape[-1] == 6, f"Expected 6 columns, got {query_tokens.shape[-1]}"
        
        # =========================================================
        # 1. Query Features (Spectral Only - matches legacy)
        # =========================================================
        channel_indices = query_tokens[..., 3].long()
        features = self.spectral_encoder(channel_indices)
        
        # =========================================================
        # 2. Bias Calculation (Legacy edge-based format)
        # =========================================================
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
            token_data: [B, L, m, 6]
            latent_positions: [B, L, 2] or None
            
        Returns:
            token_bias: [B, L, m, 2, 2] edge bounds
            latent_bias: [B, L, 2] positions
        """
        return self.geometry.get_encoder_bias(token_data, latent_positions)
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"encoder_dim={self._encoder_out_dim}, "
            f"decoder_dim={self._decoder_out_dim}, "
            f"constant_gsd={self.use_constant_gsd}"
        )