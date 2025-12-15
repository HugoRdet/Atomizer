###
# kernel trick
###

def reconstruct(self, latents, query_tokens, query_mask):
    """
    Reconstruct using kernel decoder.
    Latent is reshaped to 2D kernel, position samples from it.
    """
    B, N, _ = query_tokens.shape
    L_spatial = self.num_spatial_latents
    device = latents.device
    D = latents.shape[-1]
    
    # =========================================================================
    # 1. Get nearest latent for each query (k=1)
    # =========================================================================
    spatial_indices, _ = self.transform.get_topk_latents_for_decoder(
        query_tokens, k=1, device=device
    )  # [B, N, 1]
    spatial_indices = spatial_indices.squeeze(-1)  # [B, N]
    
    # =========================================================================
    # 2. Gather latents
    # =========================================================================
    spatial_latents = latents[:, :L_spatial, :]  # [B, L_spatial, D]
    
    idx_expanded = spatial_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
    selected_latents = torch.gather(spatial_latents, dim=1, index=idx_expanded)  # [B, N, D]
    
    # =========================================================================
    # 3. Get relative position normalized to [-1, 1]
    # =========================================================================
    rel_pos = self.transform.get_relative_position_normalized(
        query_tokens, spatial_indices, device
    )  # [B, N, 2]
    
    # =========================================================================
    # 4. Get wavelength encoding
    # =========================================================================
    wavelength_enc, _, _ = self.transform.process_data(
        query_tokens, query_mask, query=True
    )  # [B, N, 19]
    
    # =========================================================================
    # 5. Kernel decode
    # =========================================================================
    output = self.kernel_decoder(selected_latents, rel_pos, wavelength_enc)
    
    return output


#def _init_decoder(self):
    """Initialize kernel-based decoder."""
    
    self.kernel_decoder = KernelDecoder(
        latent_dim=self.latent_dim,
        query_dim=self.query_dim_recon,  # 19 (wavelength)
        kernel_size=8,
        num_channels=64,
    )


class KernelDecoder(nn.Module):
    """
    Decoder that interprets latents as 2D spatial kernels.
    Position directly indexes the kernel via bilinear sampling.
    """
    
    def __init__(self, latent_dim, query_dim, kernel_size=8, num_channels=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        
        # Latent → 2D kernel
        self.to_kernel = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_channels * kernel_size * kernel_size),
        )
        
        # Combine spatial features + wavelength → output
        hidden_dim = num_channels * 2
        self.output_head = nn.Sequential(
            nn.Linear(num_channels + query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, latent, rel_pos, wavelength_enc):
        """
        Args:
            latent: [B, N, latent_dim] - one latent per query
            rel_pos: [B, N, 2] - relative (x, y) in [-1, 1]
            wavelength_enc: [B, N, query_dim] - wavelength encoding
        Returns:
            [B, N, 1]
        """
        B, N, D = latent.shape
        K = self.kernel_size
        C = self.num_channels
        
        # 1. Latent → 2D kernel
        kernel = self.to_kernel(latent)  # [B, N, C * K * K]
        kernel = kernel.view(B * N, C, K, K)  # [B*N, C, K, K]
        
        # 2. Sample at relative position
        grid = rel_pos.view(B * N, 1, 1, 2)  # [B*N, 1, 1, 2]
        
        sampled = F.grid_sample(
            kernel,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B*N, C, 1, 1]
        
        spatial_features = sampled.view(B, N, C)  # [B, N, C]
        
        # 3. Combine with wavelength
        combined = torch.cat([spatial_features, wavelength_enc], dim=-1)
        
        # 4. Output
        output = self.output_head(combined)  # [B, N, 1]
        
        return output


#
def get_relative_position_normalized(self, query_tokens, latent_indices, device):
    """
    Get relative position of each query to its latent, normalized to [-1, 1].
    
    Args:
        query_tokens: [B, N, 6]
        latent_indices: [B, N] - which latent each query belongs to
    Returns:
        rel_pos: [B, N, 2] - (x, y) in [-1, 1]
    """
    B, N = latent_indices.shape
    
    # Token positions in meters
    token_centers = self._precompute_token_physical_centers(device)
    token_x = token_centers[query_tokens[..., 1].long()]  # [B, N]
    token_y = token_centers[query_tokens[..., 2].long()]  # [B, N]
    
    # Latent positions in meters
    latent_positions = self._precompute_latent_physical_positions(device)  # [L, 2]
    latent_x = latent_positions[latent_indices, 0]  # [B, N]
    latent_y = latent_positions[latent_indices, 1]  # [B, N]
    
    # Relative position in meters
    delta_x = token_x - latent_x  # [B, N]
    delta_y = token_y - latent_y  # [B, N]
    
    # Normalize to [-1, 1] based on latent spacing
    half_extent = self.physical_scale / 2.0
    
    rel_x = delta_x / half_extent
    rel_y = delta_y / half_extent
    
    # Clamp for tokens at edges
    rel_x = torch.clamp(rel_x, -1.0, 1.0)
    rel_y = torch.clamp(rel_y, -1.0, 1.0)
    
    rel_pos = torch.stack([rel_x, rel_y], dim=-1)  # [B, N, 2]
    
    return rel_pos


# def reconstruct(self, latents, query_tokens, query_mask):
    """Reconstruct with optional latent fusion before kernel."""
    B, N, _ = query_tokens.shape
    L_spatial = self.num_spatial_latents
    device = latents.device
    D = latents.shape[-1]
    k = self.decoder_k_spatial
    
    # 1. Get k nearest latents
    spatial_indices, distances = self.transform.get_topk_latents_for_decoder(
        query_tokens, k=k, device=device
    )  # [B, N, k]
    
    # 2. Gather latents
    spatial_latents = latents[:, :L_spatial, :]
    flat_indices = spatial_indices.reshape(B, N * k)
    flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
    flat_gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_expanded)
    selected_latents = flat_gathered.reshape(B, N, k, D)  # [B, N, k, D]
    
    # 3. Fuse latents (distance-weighted average)
    if k > 1:
        weights = F.softmax(-distances, dim=-1)  # [B, N, k]
        fused_latent = (weights.unsqueeze(-1) * selected_latents).sum(dim=2)  # [B, N, D]
    else:
        fused_latent = selected_latents.squeeze(2)  # [B, N, D]
    
    # 4. Get relative position (to nearest latent)
    nearest_idx = spatial_indices[:, :, 0]  # [B, N]
    rel_pos = self.transform.get_relative_position_normalized(
        query_tokens, nearest_idx, device
    )  # [B, N, 2]
    
    # 5. Wavelength
    wavelength_enc, _, _ = self.transform.process_data(
        query_tokens, query_mask, query=True
    )
    
    # 6. Kernel decode
    output = self.kernel_decoder(fused_latent, rel_pos, wavelength_enc)
    
    return output