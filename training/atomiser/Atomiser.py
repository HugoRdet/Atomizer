"""
Atomiser Model with Learnable Latent Positions

This module contains the Atomiser model for satellite image processing.
Position update strategies are imported from displacement.py.

Usage:
    from atomiser import Atomiser
    
    model = Atomiser(config=config, transform=transform)
    output = model(data, mask, mae_tokens, mae_tokens_mask, task="reconstruction")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from einops import repeat
from typing import Optional, Tuple, List, Dict, Any
import time

# Import from your codebase
from .nn_comp import (
    PreNorm, 
    SelfAttention, 
    FeedForward, 
    LatentAttentionPooling, 
    LocalCrossAttention,
    SelfAttentionWithRelativePosition
)

# Import displacement strategies
from .displacement import (
    PositionUpdateStrategy,
    NoPositionUpdate,
    MLPDisplacementUpdate,
    DeformableOffsetUpdate,
    create_position_updater,
    compute_displacement_stats,
)


# =============================================================================
# Utilities
# =============================================================================

def cache_fn(f):
    """Cache function results for weight sharing across layers."""
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


# =============================================================================
# Atomiser Model
# =============================================================================

class Atomiser(pl.LightningModule):
    """
    Atomizer model for satellite image processing with learnable latent positions.
    
    Spatial latents: Arranged on a grid, use geographic attention (local)
    Global latents: No spatial position, participate in self-attention only
    
    Position updates allow spatial latents to move through encoder layers.
    """
    
    def __init__(self, *, config, transform):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        
        self.config = config
        self.transform = transform
        
        # =====================================================================
        # Latent configuration
        # =====================================================================
        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"]
        self.num_spatial_latents = self.spatial_latents_per_row ** 2
        self.num_global_latents = config["Atomiser"]["global_latents"]
        self.num_latents = self.num_spatial_latents + self.num_global_latents
        
        # =====================================================================
        # Compute dimensions
        # =====================================================================
        self.input_dim = self._compute_input_dim()
        self.query_dim_recon = self._compute_query_dim_recon()
        
        # =====================================================================
        # Model architecture parameters
        # =====================================================================
        self.depth = config["Atomiser"]["depth"]
        self.latent_dim = config["Atomiser"].get("latent_dim", self.input_dim)
        self.pos_encoding_size = self._get_encoding_dim("pos") * 2
        self.cross_heads = config["Atomiser"]["cross_heads"]
        self.latent_heads = config["Atomiser"]["latent_heads"]
        self.cross_dim_head = config["Atomiser"]["cross_dim_head"]
        self.latent_dim_head = config["Atomiser"]["latent_dim_head"]
        self.num_classes = config["trainer"]["num_classes"]
        self.attn_dropout = config["Atomiser"]["attn_dropout"]
        self.ff_dropout = config["Atomiser"]["ff_dropout"]
        self.weight_tie_layers = config["Atomiser"]["weight_tie_layers"]
        self.self_per_cross_attn = config["Atomiser"]["self_per_cross_attn"]
        self.final_classifier_head = config["Atomiser"]["final_classifier_head"]
        
        # Token limits
        self.max_tokens_forward = config["trainer"]["max_tokens_forward"]
        self.max_tokens_val = config["trainer"]["max_tokens_val"]
        
        # Geographic attention parameters
        self.geo_k = config["Atomiser"].get("geo_k", 1500)
        self.geo_m_train = config["Atomiser"].get("geo_m_train", 500)
        self.geo_m_val = config["Atomiser"].get("geo_m_val", 500)
        self.k_latents = config["Atomiser"].get("k_latents", 4)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 1)
        
        # Initialize components
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
        self._init_position_updater()

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _compute_input_dim(self):
        """Compute total input dimension from all encodings."""
        return 343

    def _compute_query_dim_recon(self):
        """Compute query dimension for reconstruction (no band values)."""
        return self._get_encoding_dim("wavelength")
    
    def _get_encoding_dim(self, attribute):
        """Get encoding dimension for a specific attribute."""
        encoding_type = self.config["Atomiser"][f"{attribute}_encoding"]

        if encoding_type == "NOPE":
            return 0
        elif encoding_type == "NATURAL":
            return 1
        elif encoding_type == "FF":
            num_bands = self.config["Atomiser"][f"{attribute}_num_freq_bands"]
            max_freq = self.config["Atomiser"][f"{attribute}_max_freq"]
            if num_bands == -1:
                return int(max_freq) * 2 + 1
            else:
                return int(num_bands) * 2 + 1
        elif encoding_type == "GAUSSIANS":
            return len(self.config["wavelengths_encoding"])
        elif encoding_type == "GAUSSIANS_POS":
            return 420
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _init_latents(self):
        """Initialize learnable latent vectors (spatial + global)."""
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing."""
        
        get_cross_attn = cache_fn(lambda: 
            PreNorm(
                dim=self.latent_dim,
                fn=LocalCrossAttention(
                    dim_query=self.latent_dim,
                    dim_context=self.input_dim,
                    dim_out=self.latent_dim,
                    heads=self.cross_heads,
                    dim_head=self.cross_dim_head,
                    dropout=self.attn_dropout
                ),
                context_dim=self.input_dim
            )
        )
        
        get_cross_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))
        
        get_latent_attn = cache_fn(lambda: PreNorm(
            self.latent_dim,
            SelfAttention(
                dim=self.latent_dim,
                heads=self.latent_heads,
                dim_head=self.latent_dim_head,
                dropout=self.attn_dropout,
                use_flash=True
            )
        ))
        
        get_latent_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))

        self.encoder_layers = nn.ModuleList()
        for i in range(self.depth):
            cache_args = {'_cache': (i > 0 and self.weight_tie_layers)}

            cross_attn = get_cross_attn(**cache_args)
            cross_ff = get_cross_ff(**cache_args)

            self_attns = nn.ModuleList()
            for j in range(self.self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=j),
                    get_latent_ff(**cache_args, key=j)
                ]))
            
            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _init_decoder(self):
        """Initialize decoder cross-attention and output head."""
        D_pe = self.transform.get_polar_encoding_dimension()
        cross_attn_out_dim = self.latent_dim
        
        self.decoder_cross_attn = LocalCrossAttention(
            dim_query=self.query_dim_recon,
            dim_context=self.latent_dim + D_pe,
            dim_out=cross_attn_out_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout
        )
        
        hidden_dim = cross_attn_out_dim * 2
        mlp_input_dim = cross_attn_out_dim + self.query_dim_recon
        
        self.output_head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def _init_classifier(self):
        """Initialize classification head."""
        if self.final_classifier_head:
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(
                    self.latent_dim, 
                    heads=self.latent_heads, 
                    dim_head=self.latent_dim_head, 
                    dropout=self.attn_dropout
                ),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.num_classes)
            )
        else:
            self.to_logits = nn.Identity()

    def _init_position_updater(self):
        """Initialize the position update strategy."""
        self.use_displacement = bool(self.config["Atomiser"].get("use_displacement", False))
        
        if self.use_displacement:
            updater_config = {
                "use_displacement": True,
                "position_strategy": self.config["Atomiser"].get("position_strategy", "mlp"),
                "latent_dim": self.latent_dim,
                "depth": self.depth,
                "max_displacement": self.config["Atomiser"].get("max_displacement", 10.0),
                "displacement_hidden_dim": self.config["Atomiser"].get("displacement_hidden_dim", None),
                "share_displacement_weights": self.config["Atomiser"].get("share_displacement_weights", True),
                "deformable_heads": self.config["Atomiser"].get("deformable_heads", 4),
                "deformable_points": self.config["Atomiser"].get("deformable_points", 4),
            }
            
            self.position_updater = create_position_updater(updater_config)
            
            strategy = self.config["Atomiser"].get("position_strategy", "mlp")
            max_disp = self.config["Atomiser"].get("max_displacement",10.0)
            print(f"[Atomiser] Position updates ENABLED: strategy={strategy}, max_displacement={max_disp}")
        else:
            # Use NoPositionUpdate strategy (fixed grid)
            self.position_updater = NoPositionUpdate()
            print("[Atomiser] Position updates DISABLED (fixed grid)")

    # =========================================================================
    # Coordinate Utilities
    # =========================================================================

    def _get_default_latent_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get default grid coordinates for spatial latents.
        
        Uses transform._precompute_latent_physical_positions() which returns [L, 2]
        and expands to [B, L, 2].
        
        Returns:
            coords: [B, L_spatial, 2] grid coordinates in physical space (meters)
        """
        # Get [L, 2] positions from transform (cached)
        positions = self.transform._precompute_latent_physical_positions(device)
        
        # Expand to batch: [L, 2] -> [B, L, 2]
        return positions.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # =========================================================================
    # Geographic Pruning
    # =========================================================================

    def compute_token_latent_affinity(self, tokens, latents_coords=None, sigma=0.5):
        """Fast affinity computation using lookup tables."""
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        
        x_indices = tokens[:, :, 1].long()
        y_indices = tokens[:, :, 2].long()
        
        if latents_coords is not None:
            mu_x = latents_coords[:, :, 0]
            mu_y = latents_coords[:, :, 1]
        else:
            bias_data = self.transform.get_bias_data(tokens, latent_positions=None)
            _, latent_positions = bias_data
            mu_x = latent_positions[:, :, 0, 0]
            mu_y = latent_positions[:, :, 1, 0]
        
        integral_x_lut, integral_y_lut = self._precompute_integral_lut(
            x_indices, y_indices, mu_x, mu_y, sigma, device
        )
        
        x_indices_exp = x_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_x = torch.gather(integral_x_lut, dim=1, index=x_indices_exp)
        
        y_indices_exp = y_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_y = torch.gather(integral_y_lut, dim=1, index=y_indices_exp)
        
        affinity = integral_x * integral_y
        affinity = affinity.permute(0, 2, 1)
        
        return affinity

    def _precompute_integral_lut(self, x_indices, y_indices, mu_x, mu_y, sigma, device, normalize=True):
        """Precompute 1D Gaussian integrals."""
        B, L = mu_x.shape
        
        num_x_positions = x_indices.max().item() + 1
        num_y_positions = y_indices.max().item() + 1
        
        token_centers = self.transform._precompute_token_physical_centers(device)
        
        if len(token_centers) > 1:
            token_width = (token_centers[1] - token_centers[0]).abs()
        else:
            token_width = torch.tensor(1.0, device=device)
        
        half_width = token_width / 2
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
        
        # X integral LUT
        x_pos_indices = torch.arange(num_x_positions, device=device)
        x_centers = token_centers[x_pos_indices]
        x_min = (x_centers - half_width).view(1, -1, 1)
        x_max = (x_centers + half_width).view(1, -1, 1)
        mu_x_exp = mu_x.unsqueeze(1)
        
        z_x_min = (x_min - mu_x_exp) / (sigma * sqrt_2)
        z_x_max = (x_max - mu_x_exp) / (sigma * sqrt_2)
        integral_x_lut = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
        
        # Y integral LUT
        y_pos_indices = torch.arange(num_y_positions, device=device)
        y_centers = token_centers[y_pos_indices]
        y_min = (y_centers - half_width).view(1, -1, 1)
        y_max = (y_centers + half_width).view(1, -1, 1)
        mu_y_exp = mu_y.unsqueeze(1)
        
        z_y_min = (y_min - mu_y_exp) / (sigma * sqrt_2)
        z_y_max = (y_max - mu_y_exp) / (sigma * sqrt_2)
        integral_y_lut = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
        
        if normalize:
            integral_x_lut = integral_x_lut / (integral_x_lut.sum(dim=-1, keepdim=True) + 1e-8)
            integral_y_lut = integral_y_lut / (integral_y_lut.sum(dim=-1, keepdim=True) + 1e-8)
        
        return integral_x_lut, integral_y_lut

    def geographic_pruning(self, tokens, mask, latents_coords=None):
        """Chunked geographic pruning using pre-normalized LUTs."""
        k = self.geo_k
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        chunk_size = 50
        
        with torch.no_grad():
            x_indices = tokens[:, :, 1].long()
            y_indices = tokens[:, :, 2].long()
            
            if latents_coords is not None:
                mu_x = latents_coords[:, :, 0]
                mu_y = latents_coords[:, :, 1]
            else:
                bias_data = self.transform.get_bias_data(tokens, latent_positions=None)
                _, latent_positions = bias_data
                mu_x = latent_positions[:, :, 0, 0]
                mu_y = latent_positions[:, :, 1, 0]
            
            integral_x_lut, integral_y_lut = self._precompute_integral_lut(
                x_indices, y_indices, mu_x, mu_y, sigma=0.5, device=device, normalize=True
            )
            
            all_indices = []
            all_bias_values = []
            
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_L = chunk_end - chunk_start
                
                x_idx_exp = x_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                y_idx_exp = y_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                
                integral_x_chunk = torch.gather(
                    integral_x_lut[:, :, chunk_start:chunk_end], 
                    dim=1, 
                    index=x_idx_exp
                )
                
                integral_y_chunk = torch.gather(
                    integral_y_lut[:, :, chunk_start:chunk_end], 
                    dim=1, 
                    index=y_idx_exp
                )
                
                chunk_affinity = integral_x_chunk * integral_y_chunk
                chunk_affinity = chunk_affinity.permute(0, 2, 1)
                chunk_affinity = torch.log(chunk_affinity + 1e-8)
                
                topk_result = torch.topk(chunk_affinity, k=k, dim=-1, largest=True, sorted=False)
                all_indices.append(topk_result.indices)
                all_bias_values.append(topk_result.values)
                
                del integral_x_chunk, integral_y_chunk, chunk_affinity
            
            selected_indices = torch.cat(all_indices, dim=1)
            selected_bias = torch.cat(all_bias_values, dim=1)
        
        tokens_per_latent = self._gather_tokens_efficient(tokens, selected_indices)
        masks_per_latent = self._gather_masks_efficient(mask, selected_indices)
        
        return tokens_per_latent, masks_per_latent, selected_bias

    def _gather_tokens_efficient(self, tokens, indices):
        """Gather tokens without expanding the full tensor."""
        B, N, D = tokens.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(tokens, dim=1, index=flat_indices_exp)
        
        return gathered.reshape(B, L, k, D)

    def _gather_masks_efficient(self, mask, indices):
        """Gather masks without expanding."""
        B, N = mask.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        gathered = torch.gather(mask, dim=1, index=flat_indices)
        
        return gathered.reshape(B, L, k).bool()

    # =========================================================================
    # Encoder
    # =========================================================================

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latents_coords: torch.Tensor = None,
        training: bool = True,
        return_trajectory: bool = False
    ):
        """
        Encode tokens into latent representations.
        
        Position updates are controlled by the position_updater strategy:
        - NoPositionUpdate: fixed grid (baseline)
        - MLPDisplacementUpdate: learnable displacements
        - DeformableOffsetUpdate: multi-point sampling
        
        Args:
            tokens: [B, N, D]
            mask: [B, N]
            latents_coords: [B, L_spatial, 2] initial positions or None for grid
            training: bool
            return_trajectory: If True, return position history
            
        Returns:
            If return_trajectory:
                (latents, final_coords, trajectory)
            Else:
                (latents, final_coords)
        """
        B = tokens.shape[0]
        L_spatial = self.num_spatial_latents
        L_global = self.num_global_latents
        device = tokens.device
        
        # Initialize latents
        latents = repeat(self.latents, 'n d -> b n d', b=B)
        
        # Initialize coordinates
        if latents_coords is not None:
            current_coords = latents_coords.clone()
        else:
            current_coords = self._get_default_latent_coords(B, device)
        
        # Track trajectory
        trajectory = [current_coords.clone()] if return_trajectory else None
        
        # Initial geographic pruning
        geographic_tokens, geographic_masks, _ = self.geographic_pruning(
            tokens, mask, current_coords
        )
        k = geographic_tokens.shape[2]
        
        num_layers = len(self.encoder_layers)
        
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            
            # Sample tokens from geographic pool
            m = self.geo_m_train if training else self.geo_m_val
            m = min(m, k)
            
            perm = torch.randperm(k, device=device)[:m]
            sampled_tokens = geographic_tokens[:, :, perm, :]
            sampled_masks = geographic_masks[:, :, perm]
            
            # Process tokens with current positions
            processed_tokens = self.transform.process_data_for_encoder(
                sampled_tokens,
                sampled_masks,
                device=device,
                latent_positions=current_coords
            )
            
            # Cross attention (spatial latents only)
            latents_spatial = latents[:, :L_spatial, :]
            latents_global = latents[:, L_spatial:, :]
            
            spatial_out = cross_attn(
                latents_spatial,
                context=processed_tokens,
                mask=~sampled_masks,
            )
            
            latents_spatial = spatial_out + latents_spatial
            latents_spatial = cross_ff(latents_spatial) + latents_spatial
            
            # Recombine
            latents = torch.cat([latents_spatial, latents_global], dim=1)
            
            # Self-attention (all latents)
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents
            
            # Position update (NoPositionUpdate returns same coords if disabled)
            latents_spatial = latents[:, :L_spatial, :]
            current_coords, _ = self.position_updater(latents_spatial, current_coords, layer_idx)
            
            if return_trajectory:
                trajectory.append(current_coords.clone())
            
            # Re-compute geographic pruning for next layer
            if layer_idx < num_layers - 1:
                geographic_tokens, geographic_masks, _ = self.geographic_pruning(
                    tokens, mask, current_coords
                )
        
        if return_trajectory:
            return latents, current_coords, trajectory
        return latents, current_coords

    # =========================================================================
    # Decoder
    # =========================================================================

    def reconstruct(self, latents, latents_coords, query_tokens, query_mask):
        """Reconstruct query tokens using spatial latents."""
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        device = latents.device
        D = latents.shape[-1]
        k_spatial = self.decoder_k_spatial
        
        spatial_indices, _ = self.transform.get_topk_latents_for_decoder(
            query_tokens, k=k_spatial, device=device, latent_positions=latents_coords
        )
        
        spatial_latents = latents[:, :L_spatial, :]
        
        flat_indices = spatial_indices.reshape(B, N * k_spatial)
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        
        flat_gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_expanded)
        selected_spatial = flat_gathered.reshape(B, N, k_spatial, D)
        
        del flat_indices, flat_indices_expanded, flat_gathered
        
        relative_pe = self.transform.get_decoder_relative_pe(
            query_tokens, spatial_indices, device, latent_positions=latents_coords
        )
        
        spatial_context = torch.cat([selected_spatial, relative_pe], dim=-1)
        
        processed_queries, _, _ = self.transform.process_data(
            query_tokens, query_mask, query=True
        )
        
        output = self.decoder_cross_attn(processed_queries, spatial_context)
        
        output_with_query = torch.cat([output, processed_queries], dim=-1)
        result = self.output_head(output_with_query)
        
        return result
    
    def classify(self, latents):
        """Classify from latent representations."""
        return self.to_logits(latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self, 
        data, 
        mask, 
        mae_tokens=None, 
        mae_tokens_mask=None, 
        latents_coords=None,
        training=True, 
        task="reconstruction"
    ):
        """
        Forward pass of the Atomizer.
        
        Args:
            data: [B, N, 6] input tokens
            mask: [B, N] attention mask
            mae_tokens: [B, M, 6] query tokens for reconstruction
            mae_tokens_mask: [B, M] query mask
            latents_coords: [B, L, 2] initial positions or None for grid
            training: bool
            task: "reconstruction", "visualization", "encoder", or "classification"
            return_trajectory: If True, return position history
            
        Returns:
            For task="encoder":
                dict with 'latents', 'final_coords', and optionally 'trajectory'
            For task="reconstruction"/"vizualisation":
                If return_trajectory: (predictions, trajectory)
                Else: predictions
            For task="classification":
                logits
        """
        # Encode
        if task == "vizualisation":
            latents, final_coords, trajectory = self.encode(
                data, mask, latents_coords, training, return_trajectory=True
            )
        else:
            latents, final_coords = self.encode(
                data, mask, latents_coords, training, return_trajectory=False
            )
            trajectory = None
        
        if task == "encoder":
            return latents
        
        elif task == "reconstruction" or task == "vizualisation":
            chunk_size = 100000
            N = mae_tokens.shape[1]
            
            if N > chunk_size:
                preds_list = []
                for i in range(0, N, chunk_size):
                    chunk_tokens = mae_tokens[:, i:i + chunk_size]
                    chunk_mask = mae_tokens_mask[:, i:i + chunk_size]
                    p = self.reconstruct(latents, final_coords, chunk_tokens, chunk_mask)
                    preds_list.append(p)
                predictions = torch.cat(preds_list, dim=1)
            else:
                predictions = self.reconstruct(latents, final_coords, mae_tokens, mae_tokens_mask)
            
            if task == "vizualisation":
                return predictions, trajectory
            return predictions
        
        else:  # classification
            return self.classify(latents)

    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag
    
    def freeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, False)
        self.latents.requires_grad = False
        self._set_requires_grad(self.position_updater, False)
    
    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.latents.requires_grad = True
        self._set_requires_grad(self.position_updater, True)
    
    def freeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, False)
        self._set_requires_grad(self.output_head, False)
    
    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.output_head, True)
    
    def freeze_classifier(self):
        self._set_requires_grad(self.to_logits, False)
    
    def unfreeze_classifier(self):
        self._set_requires_grad(self.to_logits, True)
    
    def freeze_position_updater(self):
        self._set_requires_grad(self.position_updater, False)
    
    def unfreeze_position_updater(self):
        self._set_requires_grad(self.position_updater, True)
    
    def get_displacement_stats(self, trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """Compute statistics about latent movement from trajectory."""
        return compute_displacement_stats(trajectory)