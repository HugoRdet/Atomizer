"""
Geographic Attention Module

This module handles geographic pruning - selecting the most relevant tokens
for each latent based on spatial proximity using 2D Gaussian integrals.

The key idea: Instead of computing full N×L attention, we pre-select 
the k most relevant tokens for each latent based on spatial affinity,
reducing complexity from O(N×L) to O(L×k).

Classes:
    GeographicPruning: Main class for geographic token selection
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class GeographicPruning(nn.Module):
    """
    Geographic pruning module for efficient spatial attention.
    
    Uses 2D Gaussian integrals to compute affinity between tokens and latents,
    then selects the top-k most relevant tokens for each latent.
    
    This is memory-efficient through:
    1. Separable 1D Gaussian integrals (instead of full 2D)
    2. Lookup table precomputation
    3. Chunked processing to avoid large intermediate tensors
    
    Args:
        geometry: SensorGeometry instance for coordinate lookups
        num_spatial_latents: Number of spatial latents (L)
        geo_k: Number of tokens to select per latent
        default_sigma: Default Gaussian sigma for affinity computation
    
    Usage:
        pruner = GeographicPruning(geometry, num_spatial_latents=400, geo_k=1500)
        tokens_per_latent, masks_per_latent, bias = pruner(
            tokens, mask, latent_coords
        )
        # tokens_per_latent: [B, L, k, D]
        # masks_per_latent: [B, L, k]
        # bias: [B, L, k] (log-affinity values)
    """
    
    def __init__(
        self,
        geometry,  # SensorGeometry instance
        num_spatial_latents: int,
        geo_k: int = 1500,
        default_sigma: float = 0.5,
    ):
        super().__init__()
        self.geometry = geometry
        self.num_spatial_latents = num_spatial_latents
        self.geo_k = geo_k
        self.default_sigma = default_sigma
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        sigma: Optional[float] = None,
        chunk_size: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens for each latent based on geographic affinity.
        
        Args:
            tokens: [B, N, D] input tokens with position info in indices 1,2
            mask: [B, N] attention mask (0 = valid, 1 = masked)
            latent_coords: [B, L, 2] latent positions in meters
            sigma: Gaussian sigma for affinity (default: self.default_sigma)
            chunk_size: Process latents in chunks to save memory
        
        Returns:
            tokens_per_latent: [B, L, k, D] selected tokens for each latent
            masks_per_latent: [B, L, k] masks for selected tokens
            selected_bias: [B, L, k] log-affinity values (attention bias)
        """
        if sigma is None:
            sigma = self.default_sigma
        
        k = self.geo_k
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        
        with torch.no_grad():
            # Extract position indices from tokens
            x_indices = tokens[:, :, 1].long()
            y_indices = tokens[:, :, 2].long()
            
            # Extract latent positions
            mu_x = latent_coords[:, :, 0]
            mu_y = latent_coords[:, :, 1]
            
            # Precompute 1D Gaussian integral lookup tables
            integral_x_lut, integral_y_lut = self._precompute_integral_lut(
                x_indices, y_indices, mu_x, mu_y, sigma, device, normalize=True
            )
            
            # Process in chunks to save memory
            all_indices = []
            all_bias_values = []
            
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_L = chunk_end - chunk_start
                
                # Get LUT slices for this chunk of latents
                x_lut_chunk = integral_x_lut[:, :, chunk_start:chunk_end]
                y_lut_chunk = integral_y_lut[:, :, chunk_start:chunk_end]
                
                # Gather integral values for each token
                x_idx_exp = x_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                y_idx_exp = y_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                
                integral_x_chunk = torch.gather(x_lut_chunk, dim=1, index=x_idx_exp)
                integral_y_chunk = torch.gather(y_lut_chunk, dim=1, index=y_idx_exp)
                
                # 2D affinity = product of 1D integrals
                chunk_affinity = integral_x_chunk * integral_y_chunk
                chunk_affinity = chunk_affinity.permute(0, 2, 1)  # [B, chunk_L, N]
                chunk_affinity = torch.log(chunk_affinity + 1e-8)
                
                # Select top-k tokens for each latent
                topk_result = torch.topk(
                    chunk_affinity,
                    k=k,
                    dim=-1,
                    largest=True,
                    sorted=False
                )
                
                all_indices.append(topk_result.indices)
                all_bias_values.append(topk_result.values)
                
                # Clean up
                del integral_x_chunk, integral_y_chunk, chunk_affinity
                del x_lut_chunk, y_lut_chunk
            
            # Concatenate results from all chunks
            selected_indices = torch.cat(all_indices, dim=1)  # [B, L, k]
            selected_bias = torch.cat(all_bias_values, dim=1)  # [B, L, k]
            
            del integral_x_lut, integral_y_lut
        
        # Gather tokens and masks using selected indices
        tokens_per_latent = self._gather_tokens(tokens, selected_indices)
        masks_per_latent = self._gather_masks(mask, selected_indices)
        
        return tokens_per_latent, masks_per_latent, selected_bias
    
    def compute_affinity(
        self,
        tokens: torch.Tensor,
        latent_coords: torch.Tensor,
        sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute full affinity matrix between tokens and latents.
        
        This is the non-pruned version, useful for analysis/visualization.
        Warning: Creates [B, L, N] tensor which can be large!
        
        Args:
            tokens: [B, N, D] input tokens
            latent_coords: [B, L, 2] latent positions
            sigma: Gaussian sigma
        
        Returns:
            affinity: [B, L, N] affinity matrix
        """
        if sigma is None:
            sigma = self.default_sigma
        
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        
        x_indices = tokens[:, :, 1].long()
        y_indices = tokens[:, :, 2].long()
        
        mu_x = latent_coords[:, :, 0]
        mu_y = latent_coords[:, :, 1]
        
        integral_x_lut, integral_y_lut = self._precompute_integral_lut(
            x_indices, y_indices, mu_x, mu_y, sigma, device, normalize=False
        )
        
        x_indices_exp = x_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_x = torch.gather(integral_x_lut, dim=1, index=x_indices_exp)
        
        y_indices_exp = y_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_y = torch.gather(integral_y_lut, dim=1, index=y_indices_exp)
        
        affinity = integral_x * integral_y  # [B, N, L]
        
        return affinity.permute(0, 2, 1)  # [B, L, N]
    
    def _precompute_integral_lut(
        self,
        x_indices: torch.Tensor,
        y_indices: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        sigma: float,
        device: torch.device,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute 1D Gaussian integrals as lookup tables.
        
        For each latent position μ and each possible token position,
        computes the integral of a Gaussian over the token's spatial extent.
        
        This exploits separability: 2D Gaussian integral = product of 1D integrals.
        
        Args:
            x_indices: [B, N] x position indices of tokens
            y_indices: [B, N] y position indices of tokens
            mu_x: [B, L] x coordinates of latents (meters)
            mu_y: [B, L] y coordinates of latents (meters)
            sigma: Gaussian standard deviation
            device: torch device
            normalize: If True, normalize so integrals sum to 1 per latent
        
        Returns:
            integral_x_lut: [B, num_x_positions, L] lookup table for x
            integral_y_lut: [B, num_y_positions, L] lookup table for y
        """
        B, L = mu_x.shape
        
        # Determine number of unique positions
        num_x_positions = x_indices.max().item() + 1
        num_y_positions = y_indices.max().item() + 1
        
        # Get geometry constants
        sqrt_2, half_width, token_centers = self.geometry.get_integral_constants()
        
        # === X dimension ===
        x_pos_indices = torch.arange(num_x_positions, device=device)
        x_centers = token_centers[x_pos_indices]
        
        # Token bounds
        x_min = (x_centers - half_width).view(1, -1, 1)  # [1, num_x, 1]
        x_max = (x_centers + half_width).view(1, -1, 1)
        mu_x_exp = mu_x.unsqueeze(1)  # [B, 1, L]
        
        # Compute z-scores for erf
        sigma_sqrt_2 = sigma * sqrt_2
        z_x_min = (x_min - mu_x_exp) / sigma_sqrt_2
        z_x_max = (x_max - mu_x_exp) / sigma_sqrt_2
        
        # Gaussian integral via erf
        integral_x_lut = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
        
        # === Y dimension ===
        y_pos_indices = torch.arange(num_y_positions, device=device)
        y_centers = token_centers[y_pos_indices]
        
        y_min = (y_centers - half_width).view(1, -1, 1)
        y_max = (y_centers + half_width).view(1, -1, 1)
        mu_y_exp = mu_y.unsqueeze(1)
        
        z_y_min = (y_min - mu_y_exp) / sigma_sqrt_2
        z_y_max = (y_max - mu_y_exp) / sigma_sqrt_2
        
        integral_y_lut = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
        
        # Normalize if requested
        if normalize:
            integral_x_lut = integral_x_lut / (integral_x_lut.sum(dim=1, keepdim=True) + 1e-8)
            integral_y_lut = integral_y_lut / (integral_y_lut.sum(dim=1, keepdim=True) + 1e-8)
        
        return integral_x_lut, integral_y_lut
    
    def _gather_tokens(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather tokens according to selected indices.
        
        Args:
            tokens: [B, N, D] all tokens
            indices: [B, L, k] indices of selected tokens per latent
        
        Returns:
            gathered: [B, L, k, D] selected tokens per latent
        """
        B, N, D = tokens.shape
        L, k = indices.shape[1], indices.shape[2]
        
        # Flatten indices for gather
        flat_indices = indices.reshape(B, L * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        
        # Gather and reshape
        gathered = torch.gather(tokens, dim=1, index=flat_indices_exp)
        return gathered.reshape(B, L, k, D)
    
    def _gather_masks(
        self,
        mask: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather masks according to selected indices.
        
        Args:
            mask: [B, N] attention mask
            indices: [B, L, k] indices of selected tokens per latent
        
        Returns:
            gathered: [B, L, k] masks for selected tokens (as bool)
        """
        B, N = mask.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        gathered = torch.gather(mask, dim=1, index=flat_indices)
        
        return gathered.reshape(B, L, k).bool()


def create_geographic_pruning(config: dict, geometry) -> GeographicPruning:
    """
    Factory function to create GeographicPruning from config.
    
    Args:
        config: Model configuration dictionary
        geometry: SensorGeometry instance
    
    Returns:
        GeographicPruning instance
    """
    spatial_latents_per_row = config["Atomiser"]["spatial_latents"]
    num_spatial_latents = spatial_latents_per_row ** 2
    geo_k = config["Atomiser"].get("geo_k", 1500)
    default_sigma = config["Atomiser"].get("geo_sigma", 0.5)
    
    return GeographicPruning(
        geometry=geometry,
        num_spatial_latents=num_spatial_latents,
        geo_k=geo_k,
        default_sigma=default_sigma,
    )