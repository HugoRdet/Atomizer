import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from einops import repeat
from .nn_comp import PreNorm, CrossAttention, SelfAttention, FeedForward, LatentAttentionPooling, LocalCrossAttention
import einops as einops


def print_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")


def reset_memory_stats():
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def cache_fn(f):
    """Cache function results for weight sharing across layers"""
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


class Atomiser(pl.LightningModule):
    """
    Atomizer model for satellite image processing.
    
    Spatial latents: Arranged on a grid, use geographic attention (local)
    Global latents: No spatial position, participate in self-attention only
    """
    
    def __init__(self, *, config, transform):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        
        self.config = config
        self.transform = transform
        
        # =====================================================================
        # Latent configuration
        # =====================================================================
        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"]  # e.g., 20
        self.num_spatial_latents = self.spatial_latents_per_row ** 2          # e.g., 400
        self.num_global_latents = config["Atomiser"]["global_latents"]        # e.g., 16
        self.num_latents = self.num_spatial_latents + self.num_global_latents # e.g., 416
        
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
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 1)
        
        # Initialize components
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()

    def _compute_input_dim(self):
        """Compute total input dimension from all encodings"""
        pos_dim = self._get_encoding_dim("pos")
        wavelength_dim = self._get_encoding_dim("wavelength") 
        bandvalue_dim = self._get_encoding_dim("bandvalue")
        return 343  # 2 * pos_dim + wavelength_dim + bandvalue_dim
    
    def _compute_query_dim_recon(self):
        """Compute query dimension for reconstruction (no band values)"""
        pos_dim = self._get_encoding_dim("pos")
        wavelength_dim = self._get_encoding_dim("wavelength")
        return wavelength_dim
    
    def _get_encoding_dim(self, attribute):
        """Get encoding dimension for a specific attribute"""
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
        """Initialize learnable latent vectors (spatial + global)"""
        # All latents have same dimension, but different roles
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
        
        # Note: latents[:num_spatial_latents] are spatial (have positions)
        #       latents[num_spatial_latents:] are global (no positions)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing"""
        
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

        # Build encoder layers
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
        
        # MLP input = cross_attn output + query (wavelength)
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
        """Initialize classification head"""
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


    def geographic_pruning(self, tokens, mask, latents_coords=None):
        """
        Memory-efficient geographic pruning for spatial latents.
        Only applies to spatial latents (which have positions).
        
        Args:
            tokens: [B, N, D]
            mask: [B, N]
            latents_coords: [B, L, 2] or None for default grid
        """
        k = self.geo_k
        chunk_size = 50
        
        B, N, D = tokens.shape
        L = self.num_spatial_latents  
        
        SIGMA = 0.5
        
        with torch.no_grad():
            # Get bias data (pass latent positions)
            bias_data = self.transform.get_bias_data(tokens, latent_positions=latents_coords)
            tokens_positions, latent_positions = bias_data
            
            x_min = tokens_positions[:, :, 0, 0]
            x_max = tokens_positions[:, :, 0, 1]
            y_min = tokens_positions[:, :, 1, 0]
            y_max = tokens_positions[:, :, 1, 1]
            
            # Handle both formats:
            # - Custom positions: [B, L, 2]
            # - Default grid returns complex encoding from get_bias_latents_encoding
            if latents_coords is not None:
                # Simple format: [B, L, 2]
                mu_x_all = latent_positions[:, :, 0]  # [B, L]
                mu_y_all = latent_positions[:, :, 1]  # [B, L]
            else:
                # Old format from get_bias_latents_encoding
                mu_x_all = latent_positions[:, :, 0, 0]
                mu_y_all = latent_positions[:, :, 1, 0]
            
            sqrt_2 = torch.sqrt(torch.tensor(2.0, device=tokens.device))
            
            all_indices = []
            all_bias_values = []
            
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                
                mu_x = mu_x_all[:, chunk_start:chunk_end, None]  # [B, chunk, 1]
                mu_y = mu_y_all[:, chunk_start:chunk_end, None]  # [B, chunk, 1]
                
                z_x_min = (x_min[:, None, :] - mu_x) / (SIGMA * sqrt_2)
                z_x_max = (x_max[:, None, :] - mu_x) / (SIGMA * sqrt_2)
                integral_x = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
                
                z_y_min = (y_min[:, None, :] - mu_y) / (SIGMA * sqrt_2)
                z_y_max = (y_max[:, None, :] - mu_y) / (SIGMA * sqrt_2)
                integral_y = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
                
                chunk_bias = integral_x * integral_y
                chunk_bias = chunk_bias / (chunk_bias.max(dim=-1, keepdim=True)[0] + 1e-8)
                chunk_bias = torch.log(chunk_bias + 1e-8)
                
                topk_result = torch.topk(chunk_bias, k=k, dim=2, largest=True, sorted=False)
                all_indices.append(topk_result.indices)
                all_bias_values.append(topk_result.values)
            
            selected_indices = torch.cat(all_indices, dim=1)
            selected_bias = torch.cat(all_bias_values, dim=1)

        debug_overlap = False

        if debug_overlap:
            # 1. Find problematic latents
            stats = self.diagnose_problematic_latents(selected_indices, tokens)
    
            # 2. Visualize a normal latent (center)
            center = self.num_spatial_latents // 2
            self.visualize_latent_token_selection(selected_indices, tokens, center, 
                                                save_path='./figures/latent_center.png')
            
            # 3. Visualize the worst latent (highest avg distance)
            worst_latent = stats[0]['latent']
            self.visualize_latent_token_selection(selected_indices, tokens, worst_latent,
                                                save_path='./figures/latent_worst.png')
            
            # 4. Visualize one from the red bands (~600)
            self.visualize_latent_token_selection(selected_indices, tokens, 600,
                                                save_path='./figures/latent_600.png')
            
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

    def encode(self, tokens, mask, latents_coords=None, training=True):
        """
        Encode tokens into latent representations.
        Spatial latents: Use geographic cross-attention (local)
        Global latents: Skip cross-attention, only participate in self-attention
        All latents: Participate in self-attention together
        
        Args:
            tokens: [B, N, 6]
            mask: [B, N]
            latents_coords: [B, L, 2] or None for default grid
            training: bool
        """

        diagnose = False
        if diagnose:
            reset_memory_stats()
            print("=" * 80)
            print("ENCODER MEMORY DIAGNOSIS")
            print("=" * 80)
            print(f"[INPUT] tokens: {tokens.shape}, mask: {mask.shape}")
            print(f"[CONFIG] L_spatial={self.num_spatial_latents}, L_global={self.num_global_latents}")
            print(f"[CONFIG] geo_k={self.geo_k}, geo_m_train={self.geo_m_train}")
            print_memory("0. START")
        
        B = tokens.shape[0]
        L_spatial = self.num_spatial_latents
        L_global = self.num_global_latents
        
        # Initialize all latents
        latents = repeat(self.latents, 'n d -> b n d', b=B)
        
        if diagnose:
            print_memory("1. After latents init")
            print(f"   latents: {latents.shape} | {latents.numel() * 4 / 1024**2:.2f} MB")
        
        # Geographic pruning (pass latent positions)
        geographic_tokens, geographic_masks, geographic_bias = self.geographic_pruning(
            tokens, mask, latents_coords
        )
        k = geographic_tokens.shape[2]
        
        if diagnose:
            print_memory(f"2. After geographic_pruning (k={k})")
            print(f"   geographic_tokens: {geographic_tokens.shape} | {geographic_tokens.numel() * 4 / 1024**2:.2f} MB")
            print(f"   geographic_masks: {geographic_masks.shape} | {geographic_masks.numel() / 1024**2:.2f} MB")
            print(f"   geographic_bias: {geographic_bias.shape} | {geographic_bias.numel() * 4 / 1024**2:.2f} MB")
        
        num_layers = len(self.encoder_layers)
        
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            if diagnose:
                print(f"\n{'─' * 60}")
                print(f"LAYER {layer_idx}/{num_layers-1}")
                print(f"{'─' * 60}")
                print_memory(f"Layer {layer_idx} START")
            
            m = self.geo_m_train if training else self.geo_m_val
            m = min(m, k)
            
            # Sample tokens from geographic pool
            perm = torch.randperm(k, device=tokens.device)[:m]
            sampled_tokens = geographic_tokens[:, :, perm, :]
            sampled_masks = geographic_masks[:, :, perm]
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after sampling (m={m})")
                print(f"   sampled_tokens: {sampled_tokens.shape} | {sampled_tokens.numel() * 4 / 1024**2:.2f} MB")
            
            # Process tokens with polar positional encoding (pass latent positions)
            processed_tokens = self.transform.process_data_for_encoder(
                sampled_tokens,
                sampled_masks,
                device=tokens.device,
                latent_positions=latents_coords
            )
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after process_data_for_encoder")
                print(f"   processed_tokens: {processed_tokens.shape} | {processed_tokens.numel() * 4 / 1024**2:.2f} MB")
            
            # Split latents
            latents_spatial = latents[:, :L_spatial, :]
            latents_global = latents[:, L_spatial:, :]
            
            if diagnose:
                print_memory(f"Layer {layer_idx} before cross_attn")
                print(f"   latents_spatial: {latents_spatial.shape}")
                print(f"   context (processed_tokens): {processed_tokens.shape}")
                # Estimate attention matrix size
                heads = cross_attn.heads if hasattr(cross_attn, 'heads') else 8
                attn_size_mb = B * L_spatial * heads * m * 4 / 1024**2
                print(f"   [ESTIMATE] attention matrix: {B}×{L_spatial}×{heads}×{m} = {attn_size_mb:.2f} MB")
            
            # Cross attention: ONLY spatial latents attend to tokens
            spatial_out = cross_attn(
                latents_spatial,
                context=processed_tokens,
                mask=~sampled_masks,
            )
            
            if diagnose:
                print_memory(f"Layer {layer_idx} AFTER cross_attn ← CRITICAL")
                
            # Residual + FF for spatial latents
            latents_spatial = spatial_out + latents_spatial
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after residual")
            
            latents_spatial = cross_ff(latents_spatial) + latents_spatial
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after cross_ff")
            
            # Recombine all latents
            latents = torch.cat([latents_spatial, latents_global], dim=1)
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after concat")
            
            # Self-attention: ALL latents interact (spatial + global)
            for j, (self_attn, self_ff) in enumerate(self_attns):
                if diagnose:
                    L_total = L_spatial + L_global
                    heads = self_attn.heads if hasattr(self_attn, 'heads') else 8
                    self_attn_size_mb = B * L_total * heads * L_total * 4 / 1024**2
                    print(f"   [ESTIMATE] self_attn[{j}] matrix: {B}×{L_total}×{heads}×{L_total} = {self_attn_size_mb:.2f} MB")
                
                latents = self_attn(latents) + latents
                
                if diagnose:
                    print_memory(f"Layer {layer_idx} after self_attn[{j}]")
                
                latents = self_ff(latents) + latents
                
                if diagnose:
                    print_memory(f"Layer {layer_idx} after self_ff[{j}]")
            
            if diagnose:
                print_memory(f"Layer {layer_idx} END")
        
        if diagnose:
            print(f"\n{'=' * 80}")
            print("ENCODER COMPLETE")
            print(f"{'=' * 80}")
            print_memory("FINAL")
            print(f"   output latents: {latents.shape} | {latents.numel() * 4 / 1024**2:.2f} MB")
            
            # Summary
            final_mem = torch.cuda.memory_allocated() / 1024**3
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n{'─' * 40}")
            print(f"SUMMARY")
            print(f"{'─' * 40}")
            print(f"Final allocated: {final_mem:.2f} GB")
            print(f"Peak allocated:  {peak_mem:.2f} GB")
            print(f"{'=' * 80}\n")
        
        return latents

    def reconstruct(self, latents, latents_coords, query_tokens, query_mask):
        """
        Reconstruct query tokens using spatial latents only.
        Global latents are only used in encoder self-attention.
        
        Args:
            latents: [B, L_total, D]
            latents_coords: [B, L, 2] or None for default grid
            query_tokens: [B, N, 6]
            query_mask: [B, N]
        """

        diagnose = False
    
        if diagnose:
            reset_memory_stats()
            print("=" * 80)
            print("DECODER MEMORY DIAGNOSIS")
            print("=" * 80)
            print_memory("0. START")
        
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        device = latents.device
        D = latents.shape[-1]
        D_pe = self.transform.get_polar_encoding_dimension()
        k_spatial = self.decoder_k_spatial
        
        if diagnose:
            print(f"[SHAPES] B={B}, N={N}, L_spatial={L_spatial}")
            print(f"[SHAPES] D={D}, D_pe={D_pe}, k_spatial={k_spatial}")
            print(f"[SHAPES] latents: {latents.shape}")
        
        # =========================================================================
        # 1. Select top-k spatial latents per query (pass positions)
        # =========================================================================
        spatial_indices, _ = self.transform.get_topk_latents_for_decoder(
            query_tokens, k=k_spatial, device=device, latent_positions=latents_coords
        )
        
        if diagnose:
            print_memory("1. After get_topk_latents_for_decoder")
            print(f"   spatial_indices: {spatial_indices.shape} | {spatial_indices.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 2. Gather spatial latents (MEMORY OPTIMIZED)
        # =========================================================================
        spatial_latents = latents[:, :L_spatial, :]  # [B, L_spatial, D]
        
        flat_indices = spatial_indices.reshape(B, N * k_spatial)
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        
        if diagnose:
            print_memory("2a. After flat_indices_expanded")
            print(f"   flat_indices_expanded: {flat_indices_expanded.shape} | {flat_indices_expanded.numel() * 4 / 1024**2:.2f} MB")
        
        flat_gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_expanded)
        selected_spatial = flat_gathered.reshape(B, N, k_spatial, D)
        
        if diagnose:
            print_memory("2b. After gather")
            print(f"   selected_spatial: {selected_spatial.shape} | {selected_spatial.numel() * 4 / 1024**2:.2f} MB")
        
        del flat_indices, flat_indices_expanded, flat_gathered
        
        # =========================================================================
        # 3. Compute relative PE and concat (pass positions)
        # =========================================================================
        relative_pe = self.transform.get_decoder_relative_pe(
            query_tokens, spatial_indices, device, latent_positions=latents_coords
        )
        
        if diagnose:
            print_memory("3a. After get_decoder_relative_pe")
            print(f"   relative_pe: {relative_pe.shape} | {relative_pe.numel() * 4 / 1024**2:.2f} MB")
        
        spatial_context = torch.cat([selected_spatial, relative_pe], dim=-1)
        
        if diagnose:
            print_memory("3b. After concat spatial_context")
            print(f"   spatial_context: {spatial_context.shape} | {spatial_context.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 4. Process queries
        # =========================================================================
        processed_queries, _, _ = self.transform.process_data(
            query_tokens, query_mask, query=True
        )
        
        if diagnose:
            print_memory("4. After process_data (queries)")
            print(f"   processed_queries: {processed_queries.shape} | {processed_queries.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 5. Cross-attention (spatial only)
        # =========================================================================
        if diagnose:
            print_memory("5a. BEFORE decoder_cross_attn")
            print(f"   Query: {processed_queries.shape}")
            print(f"   Spatial context: {spatial_context.shape}")
            
            # Estimate attention sizes
            heads = self.decoder_cross_attn.heads if hasattr(self.decoder_cross_attn, 'heads') else 8
            
            # Spatial attention: [B, N, heads, k_spatial]
            spatial_attn_mb = B * N * heads * k_spatial * 4 / 1024**2
            print(f"   [ESTIMATE] attention matrix: {B}×{N}×{heads}×{k_spatial} = {spatial_attn_mb:.2f} MB")
            
            # K, V projections: [B, N, k_spatial, heads, dim_head]
            dim_head = self.decoder_cross_attn.dim_head if hasattr(self.decoder_cross_attn, 'dim_head') else 64
            kv_spatial_mb = 2 * B * N * k_spatial * heads * dim_head * 4 / 1024**2
            print(f"   [ESTIMATE] K+V: 2×{B}×{N}×{k_spatial}×{heads}×{dim_head} = {kv_spatial_mb:.2f} MB")
        
        output = self.decoder_cross_attn(processed_queries, spatial_context)
        
        if diagnose:
            print_memory("5b. AFTER decoder_cross_attn")
            print(f"   output: {output.shape} | {output.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 6. Output head
        # =========================================================================
        output_with_query = torch.cat([output, processed_queries], dim=-1)
        result = self.output_head(output_with_query)
        
        if diagnose:
            print_memory("6. After output_head")
            print(f"   result: {result.shape}")
            print("=" * 80)
        
        return result
    
    def classify(self, latents):
        """Classify from latent representations"""
        return self.to_logits(latents)
    
    
    def forward(self, data, mask, mae_tokens=None, mae_tokens_mask=None, latents_coords=None,
                training=True, task="reconstruction"):
        """
        Forward pass of the Atomizer
        
        Args:
            data: [B, N, 6] input tokens
            mask: [B, N] attention mask
            mae_tokens: [B, M, 6] query tokens for reconstruction
            latents_coords: [B, L, 2] custom latent positions or None for default grid
            mae_tokens_mask: [B, M] query mask
            training: bool
            task: "reconstruction", "vizualisation", "encoder", or "classification"
        """
        latents = self.encode(data, mask, latents_coords, training=training)
        
        if task == "encoder":
            return latents
        
        elif task == "reconstruction" or task == "vizualisation":
            # Chunked reconstruction
            chunk_size = 100000
            N = mae_tokens.shape[1]
            
            if N > chunk_size:
                preds_list = []
                for i in range(0, N, chunk_size):
                    chunk_tokens = mae_tokens[:, i:i + chunk_size]
                    chunk_mask = mae_tokens_mask[:, i:i + chunk_size]
                    
                    p = self.reconstruct(latents, latents_coords, chunk_tokens, chunk_mask)
                    preds_list.append(p)
                return torch.cat(preds_list, dim=1)
            else:
                return self.reconstruct(latents, latents_coords, mae_tokens, mae_tokens_mask)
        
        else:
            return self.classify(latents)
        
  
    def _set_requires_grad(self, module, flag):
        """Set requires_grad for all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = flag
    
    def freeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, False)
        self.latents.requires_grad = False
    
    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.latents.requires_grad = True
    
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