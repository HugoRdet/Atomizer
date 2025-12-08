import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from einops import repeat
from .nn_comp import PreNorm, CrossAttention, SelfAttention, FeedForward, LatentAttentionPooling, CrossAttentionEinsum, DecoderCrossAttention
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
        self.geo_k = config["Atomiser"].get("geo_k", 500)
        self.geo_m_train = config["Atomiser"].get("geo_m_train", 500)
        self.geo_m_val = config["Atomiser"].get("geo_m_val", 500)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
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
        return 198  # 2 * pos_dim + wavelength_dim + bandvalue_dim
    
    def _compute_query_dim_recon(self):
        """Compute query dimension for reconstruction (no band values)"""
        pos_dim = self._get_encoding_dim("pos")
        wavelength_dim = self._get_encoding_dim("wavelength")
        return   wavelength_dim #2 * pos_dim
    
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
                fn=CrossAttentionEinsum(
                    query_dim=self.latent_dim,
                    context_dim=self.input_dim,
                    heads=self.cross_heads,
                    dim_head=self.cross_dim_head,
                    dropout=self.attn_dropout
                ),
                context_dim=self.input_dim
            ))
        
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
        """Initialize decoder for reconstruction"""
        D_pe = self.transform.get_polar_encoding_dimension()
        
        # Cross-attention with efficient global handling
        self.decoder_cross_attn = DecoderCrossAttention(
            dim_query=self.query_dim_recon,
            dim_spatial_context=self.latent_dim + D_pe,
            dim_global_context=self.latent_dim + D_pe,
            dim_out=self.query_dim_recon,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head
        )
        
        # Output head
        hidden_dim = self.query_dim_recon * 2
        self.output_head = nn.Sequential(
            nn.Linear(self.query_dim_recon, hidden_dim),
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


    def analyze_latent_token_overlap(self, selected_indices, latent_idx=None, top_n_neighbors=10, verbose=True):
        """
        Analyze how much tokens are shared between latents.
        
        Args:
            selected_indices: [B, L, k] tensor of token indices per latent
            latent_idx: Specific latent to analyze (None = analyze all and return summary)
            top_n_neighbors: Number of top overlapping neighbors to show
            verbose: Print detailed info
        
        Returns:
            Dictionary with overlap statistics
        """
        B, L, k = selected_indices.shape
        
        # Work with first batch item for simplicity
        indices = selected_indices[0]  # [L, k]
        
        # Convert each latent's tokens to a set for fast overlap computation
        latent_sets = [set(indices[l].tolist()) for l in range(L)]
        
        if latent_idx is not None:
            # Analyze single latent
            target_set = latent_sets[latent_idx]
            
            overlaps = []
            for other_idx in range(L):
                if other_idx == latent_idx:
                    continue
                other_set = latent_sets[other_idx]
                shared = len(target_set & other_set)
                overlap_pct = 100 * shared / k
                overlaps.append({
                    'latent': other_idx,
                    'shared_tokens': shared,
                    'overlap_pct': overlap_pct
                })
            
            # Sort by overlap
            overlaps.sort(key=lambda x: x['overlap_pct'], reverse=True)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TOKEN OVERLAP ANALYSIS FOR LATENT {latent_idx}")
                print(f"{'='*60}")
                print(f"Latent {latent_idx} has {k} tokens")
                print(f"\nTop {top_n_neighbors} overlapping neighbors:")
                print(f"{'Latent':<10} {'Shared':<10} {'Overlap %':<10}")
                print("-" * 30)
                for item in overlaps[:top_n_neighbors]:
                    print(f"{item['latent']:<10} {item['shared_tokens']:<10} {item['overlap_pct']:.1f}%")
                
                # Summary stats
                all_overlap_pcts = [x['overlap_pct'] for x in overlaps]
                print(f"\nSummary for latent {latent_idx}:")
                print(f"  Max overlap:  {max(all_overlap_pcts):.1f}%")
                print(f"  Mean overlap: {sum(all_overlap_pcts)/len(all_overlap_pcts):.1f}%")
                print(f"  Min overlap:  {min(all_overlap_pcts):.1f}%")
                print(f"  Latents with >50% overlap: {sum(1 for x in all_overlap_pcts if x > 50)}")
                print(f"  Latents with >25% overlap: {sum(1 for x in all_overlap_pcts if x > 25)}")
                print(f"  Latents with >10% overlap: {sum(1 for x in all_overlap_pcts if x > 10)}")
            
            return {'latent': latent_idx, 'overlaps': overlaps}
        
        else:
            # Global analysis: compute overlap matrix and summary
            if verbose:
                print(f"\n{'='*60}")
                print(f"GLOBAL TOKEN OVERLAP ANALYSIS")
                print(f"{'='*60}")
                print(f"Number of latents: {L}")
                print(f"Tokens per latent (k): {k}")
            
            # Sample latents for efficiency (full matrix is L×L)
            sample_latents = list(range(0, L, max(1, L // 20)))  # ~20 samples
            
            all_max_overlaps = []
            all_mean_overlaps = []
            all_nonzero_overlaps = []
            
            for l in range(L):
                target_set = latent_sets[l]
                overlaps_pct = []
                
                for other in range(L):
                    if other == l:
                        continue
                    shared = len(target_set & latent_sets[other])
                    overlaps_pct.append(100 * shared / k)
                
                all_max_overlaps.append(max(overlaps_pct))
                all_mean_overlaps.append(sum(overlaps_pct) / len(overlaps_pct))
                all_nonzero_overlaps.append(sum(1 for x in overlaps_pct if x > 0))
            
            # Compute overall statistics
            stats = {
                'k': k,
                'L': L,
                'max_overlap': {
                    'mean': sum(all_max_overlaps) / L,
                    'max': max(all_max_overlaps),
                    'min': min(all_max_overlaps),
                },
                'mean_overlap': {
                    'mean': sum(all_mean_overlaps) / L,
                    'max': max(all_mean_overlaps),
                    'min': min(all_mean_overlaps),
                },
                'neighbors_with_overlap': {
                    'mean': sum(all_nonzero_overlaps) / L,
                    'max': max(all_nonzero_overlaps),
                }
            }
            
            if verbose:
                print(f"\n--- PER-LATENT STATISTICS (averaged over {L} latents) ---")
                print(f"\nMax overlap with any neighbor:")
                print(f"  Average of max overlaps: {stats['max_overlap']['mean']:.1f}%")
                print(f"  Highest max overlap:     {stats['max_overlap']['max']:.1f}%")
                print(f"  Lowest max overlap:      {stats['max_overlap']['min']:.1f}%")
                
                print(f"\nMean overlap with all neighbors:")
                print(f"  Average: {stats['mean_overlap']['mean']:.2f}%")
                print(f"  Max:     {stats['mean_overlap']['max']:.2f}%")
                print(f"  Min:     {stats['mean_overlap']['min']:.2f}%")
                
                print(f"\nNeighbors with any overlap (>0 shared tokens):")
                print(f"  Average: {stats['neighbors_with_overlap']['mean']:.1f} / {L-1} latents")
                print(f"  Max:     {stats['neighbors_with_overlap']['max']} / {L-1} latents")
                
                # Distribution of max overlaps
                print(f"\n--- DISTRIBUTION OF MAX OVERLAPS ---")
                bins = [0, 10, 25, 50, 75, 100]
                for i in range(len(bins) - 1):
                    count = sum(1 for x in all_max_overlaps if bins[i] <= x < bins[i+1])
                    pct = 100 * count / L
                    bar = '█' * int(pct / 2)
                    print(f"  {bins[i]:>3}%-{bins[i+1]:<3}%: {count:>4} latents ({pct:>5.1f}%) {bar}")
                
                # Recommendation
                print(f"\n--- RECOMMENDATION ---")
                avg_max = stats['max_overlap']['mean']
                if avg_max < 10:
                    print(f"  ✅ Excellent! Average max overlap ({avg_max:.1f}%) is low.")
                    print(f"     Latents have distinct token pools.")
                elif avg_max < 25:
                    print(f"  ✅ Good. Average max overlap ({avg_max:.1f}%) is acceptable.")
                    print(f"     Consider reducing k slightly for more distinctness.")
                elif avg_max < 50:
                    print(f"  ⚠️  Moderate overlap ({avg_max:.1f}%).")
                    print(f"     Reduce k or increase sigma to make latents more local.")
                else:
                    print(f"  ❌ High overlap ({avg_max:.1f}%)!")
                    print(f"     Latents are sharing too many tokens.")
                    print(f"     Strongly recommend: reduce k, decrease sigma, or add more latents.")
            
            return stats


    def visualize_latent_token_selection(self, selected_indices, tokens, latent_idx, save_path=None):
        """
        Visualize which tokens a specific latent is selecting.
        Shows latent position and selected token positions.
        """
        import matplotlib.pyplot as plt
        
        B, L, k = selected_indices.shape
        indices = selected_indices[0]
        grid_size = int(L ** 0.5)
        
        with torch.no_grad():
            bias_data = self.transform.get_bias_data(tokens)
            tokens_positions, latent_gaussians = bias_data
            
            # All token centers (in meters)
            all_token_x = ((tokens_positions[0, :, 0, 0] + tokens_positions[0, :, 0, 1]) / 2).cpu().numpy()
            all_token_y = ((tokens_positions[0, :, 1, 0] + tokens_positions[0, :, 1, 1]) / 2).cpu().numpy()
            
            # All latent positions (in meters)
            all_latent_x = latent_gaussians[0, :, 0, 0].cpu().numpy()
            all_latent_y = latent_gaussians[0, :, 1, 0].cpu().numpy()
        
        # Selected tokens for this latent
        selected_token_indices = indices[latent_idx].tolist()
        selected_x = all_token_x[selected_token_indices]
        selected_y = all_token_y[selected_token_indices]
        
        # This latent's position
        latent_x = all_latent_x[latent_idx]
        latent_y = all_latent_y[latent_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot all tokens as small gray dots (subsample for speed)
        subsample = max(1, len(all_token_x) // 10000)
        ax.scatter(all_token_x[::subsample], all_token_y[::subsample], 
                c='lightgray', s=1, alpha=0.3, label='All tokens')
        
        # Plot selected tokens as blue dots
        ax.scatter(selected_x, selected_y, c='blue', s=20, alpha=0.8, 
                label=f'Selected tokens (k={k})')
        
        # Plot all latents as small red squares
        ax.scatter(all_latent_x, all_latent_y, c='red', s=10, marker='s', 
                alpha=0.3, label='All latents')
        
        # Highlight this latent
        ax.scatter([latent_x], [latent_y], c='green', s=200, marker='*', 
                edgecolors='black', linewidths=2, label=f'Latent {latent_idx}', zorder=10)
        
        # Draw lines from latent to selected tokens
        for tx, ty in zip(selected_x, selected_y):
            ax.plot([latent_x, tx], [latent_y, ty], 'g-', alpha=0.3, linewidth=0.5)
        
        # Compute stats
        avg_dist = ((selected_x - latent_x)**2 + (selected_y - latent_y)**2).mean()**0.5
        max_dist = ((selected_x - latent_x)**2 + (selected_y - latent_y)**2).max()**0.5
        
        ax.set_xlabel('X position (meters)')
        ax.set_ylabel('Y position (meters)')
        ax.set_title(f'Token Selection for Latent {latent_idx}\n'
                    f'(row={latent_idx//grid_size}, col={latent_idx%grid_size})\n'
                    f'Latent pos: ({latent_x:.1f}, {latent_y:.1f}) | Avg dist: {avg_dist:.1f}m | Max dist: {max_dist:.1f}m')
        ax.legend(loc='upper right')
        
        # Set limits to meters (-53 to 53)
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add reference circles around latent
        for radius in [5, 10, 20]:
            circle = plt.Circle((latent_x, latent_y), radius, fill=False, 
                                color='green', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()


    def diagnose_problematic_latents(self, selected_indices, tokens, grid_size=None):
        """
        Find latents that have unusually high overlap with distant latents.
        """
        B, L, k = selected_indices.shape
        indices = selected_indices[0]
        
        if grid_size is None:
            grid_size = int(L ** 0.5)
        
        print(f"\n{'='*70}")
        print(f"DIAGNOSING PROBLEMATIC LATENTS")
        print(f"{'='*70}")
        print(f"Grid: {grid_size}x{grid_size} = {L} latents, k={k} tokens each")
        
        # Get positions in meters
        with torch.no_grad():
            bias_data = self.transform.get_bias_data(tokens)
            tokens_positions, latent_gaussians = bias_data
            
            # Token centers (in meters)
            token_x = ((tokens_positions[0, :, 0, 0] + tokens_positions[0, :, 0, 1]) / 2).cpu().numpy()
            token_y = ((tokens_positions[0, :, 1, 0] + tokens_positions[0, :, 1, 1]) / 2).cpu().numpy()
            
            # Latent positions (in meters)
            latent_x = latent_gaussians[0, :, 0, 0].cpu().numpy()
            latent_y = latent_gaussians[0, :, 1, 0].cpu().numpy()
        
        print(f"\n--- POSITION RANGES (should be ~-53 to 53) ---")
        print(f"Token X: [{token_x.min():.1f}, {token_x.max():.1f}]")
        print(f"Token Y: [{token_y.min():.1f}, {token_y.max():.1f}]")
        print(f"Latent X: [{latent_x.min():.1f}, {latent_x.max():.1f}]")
        print(f"Latent Y: [{latent_y.min():.1f}, {latent_y.max():.1f}]")
        
        # For each latent, compute average distance to its selected tokens
        latent_stats = []
        
        for l in range(L):
            selected_token_indices = indices[l].tolist()
            sel_x = token_x[selected_token_indices]
            sel_y = token_y[selected_token_indices]
            
            lx, ly = latent_x[l], latent_y[l]
            
            distances = ((sel_x - lx)**2 + (sel_y - ly)**2)**0.5
            avg_dist = distances.mean()
            max_dist = distances.max()
            
            latent_stats.append({
                'latent': l,
                'row': l // grid_size,
                'col': l % grid_size,
                'latent_x': lx,
                'latent_y': ly,
                'avg_dist': avg_dist,
                'max_dist': max_dist,
                'tokens_x_range': (sel_x.min(), sel_x.max()),
                'tokens_y_range': (sel_y.min(), sel_y.max()),
            })
        
        # Sort by average distance (problematic = high distance)
        latent_stats.sort(key=lambda x: x['avg_dist'], reverse=True)
        
        print(f"\n--- TOP 20 LATENTS WITH DISTANT TOKEN SELECTION ---")
        print(f"(These select tokens far from their position - BAD!)")
        print(f"{'Latent':<8} {'Pos (x,y)':<16} {'Avg Dist':<10} {'Max Dist':<10} {'Token X Range':<20}")
        print("-" * 70)
        
        for item in latent_stats[:20]:
            pos_str = f"({item['latent_x']:.1f}, {item['latent_y']:.1f})"
            tx_range = f"[{item['tokens_x_range'][0]:.1f}, {item['tokens_x_range'][1]:.1f}]"
            print(f"{item['latent']:<8} {pos_str:<16} {item['avg_dist']:<10.1f} {item['max_dist']:<10.1f} {tx_range:<20}")
        
        print(f"\n--- BOTTOM 5 LATENTS (should be normal) ---")
        for item in latent_stats[-5:]:
            pos_str = f"({item['latent_x']:.1f}, {item['latent_y']:.1f})"
            tx_range = f"[{item['tokens_x_range'][0]:.1f}, {item['tokens_x_range'][1]:.1f}]"
            print(f"{item['latent']:<8} {pos_str:<16} {item['avg_dist']:<10.1f} {item['max_dist']:<10.1f} {tx_range:<20}")
        
        # Distribution
        avg_dists = [x['avg_dist'] for x in latent_stats]
        print(f"\n--- DISTANCE DISTRIBUTION ---")
        print(f"Mean of avg distances: {sum(avg_dists)/len(avg_dists):.1f}m")
        print(f"Max of avg distances:  {max(avg_dists):.1f}m")
        print(f"Min of avg distances:  {min(avg_dists):.1f}m")
        
        # Count problematic
        threshold = 20  # meters
        problematic = [x for x in latent_stats if x['avg_dist'] > threshold]
        print(f"\nLatents with avg_dist > {threshold}m: {len(problematic)} / {L} ({100*len(problematic)/L:.1f}%)")
        
        # Check if problematic latents are at edges
        if len(problematic) > 0:
            edge_count = 0
            corner_count = 0
            for item in problematic:
                r, c = item['row'], item['col']
                is_edge = (r == 0 or r == grid_size-1 or c == 0 or c == grid_size-1)
                is_corner = (r in [0, grid_size-1]) and (c in [0, grid_size-1])
                if is_corner:
                    corner_count += 1
                elif is_edge:
                    edge_count += 1
            
            print(f"  Of those: {corner_count} at corners, {edge_count} at edges, {len(problematic)-corner_count-edge_count} interior")
        
        return latent_stats


  

    def visualize_overlap_heatmap_fixed(self, selected_indices, sample_size=50):
        """
        Fixed heatmap with better color scale and actual values printed.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        B, L, k = selected_indices.shape
        indices = selected_indices[0]
        
        # Sample latents evenly
        step = max(1, L // sample_size)
        sampled_latents = list(range(0, L, step))[:sample_size]
        n = len(sampled_latents)
        
        print(f"Sampling {n} latents: {sampled_latents[:5]}...{sampled_latents[-5:]}")
        
        # Build overlap matrix
        overlap_matrix = np.zeros((n, n))
        
        latent_sets = {l: set(indices[l].tolist()) for l in sampled_latents}
        
        for i, l1 in enumerate(sampled_latents):
            for j, l2 in enumerate(sampled_latents):
                if i == j:
                    overlap_matrix[i, j] = 100
                else:
                    shared = len(latent_sets[l1] & latent_sets[l2])
                    overlap_matrix[i, j] = 100 * shared / k
        
        # Print some actual values
        print(f"\nSample overlap values (should see 0s for distant latents):")
        print(f"  overlap[0, 1] (neighbors): {overlap_matrix[0, 1]:.1f}%")
        print(f"  overlap[0, n//2] (distant): {overlap_matrix[0, n//2]:.1f}%")
        print(f"  overlap[0, n-1] (very distant): {overlap_matrix[0, n-1]:.1f}%")
        
        # Check for all-zero rows/columns (latents with no overlap)
        zero_rows = np.where(overlap_matrix.sum(axis=1) == 100)[0]  # Only self-overlap
        print(f"  Latents with NO overlap with sampled neighbors: {len(zero_rows)}")
        
        # Check for high overlap rows (problematic latents)
        mean_overlap = (overlap_matrix.sum(axis=1) - 100) / (n - 1)  # Exclude self
        high_overlap_idx = np.where(mean_overlap > 20)[0]
        print(f"  Latents with >20% mean overlap: {len(high_overlap_idx)}")
        
        if len(high_overlap_idx) > 0:
            print(f"    Indices: {[sampled_latents[i] for i in high_overlap_idx[:10]]}")
        
        # Plot with better colormap
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Original scale
        im1 = axes[0].imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=100)
        axes[0].set_title(f'Overlap % (0-100 scale)')
        plt.colorbar(im1, ax=axes[0])
        
        # Right: Log scale or adjusted scale for better visibility
        # Mask diagonal
        masked = overlap_matrix.copy()
        np.fill_diagonal(masked, np.nan)
        
        im2 = axes[1].imshow(masked, cmap='YlOrRd', vmin=0, vmax=50)  # Cap at 50%
        axes[1].set_title(f'Overlap % (0-50 scale, diagonal masked)')
        plt.colorbar(im2, ax=axes[1])
        
        for ax in axes:
            ax.set_xlabel('Latent Index')
            ax.set_ylabel('Latent Index')
            
            tick_positions = list(range(0, n, max(1, n // 10)))
            tick_labels = [sampled_latents[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
        
        plt.tight_layout()
        plt.savefig('latent_overlap_heatmap_fixed.png', dpi=150)
        plt.show()
        
        return overlap_matrix

    def analyze_spatial_overlap_pattern(self, selected_indices, grid_size=None):
        """
        Analyze overlap based on spatial distance between latents.
        Shows if nearby latents share more tokens (expected behavior).
        
        Args:
            selected_indices: [B, L, k] tensor
            grid_size: Size of latent grid (e.g., 30 for 30x30). Auto-detected if None.
        """
        B, L, k = selected_indices.shape
        indices = selected_indices[0]
        
        # Detect grid size
        if grid_size is None:
            grid_size = int(L ** 0.5)
            if grid_size ** 2 != L:
                print(f"Warning: L={L} is not a perfect square. Using grid_size={grid_size}")
        
        print(f"\n{'='*60}")
        print(f"SPATIAL OVERLAP PATTERN ANALYSIS")
        print(f"{'='*60}")
        print(f"Grid: {grid_size}x{grid_size} = {L} latents")
        print(f"Tokens per latent: {k}")
        
        latent_sets = [set(indices[l].tolist()) for l in range(L)]
        
        # Compute overlap by Manhattan distance
        distance_overlaps = {}  # distance -> list of overlap percentages
        
        for l1 in range(L):
            r1, c1 = l1 // grid_size, l1 % grid_size
            
            for l2 in range(l1 + 1, L):
                r2, c2 = l2 // grid_size, l2 % grid_size
                
                # Manhattan distance
                dist = abs(r1 - r2) + abs(c1 - c2)
                
                # Overlap
                shared = len(latent_sets[l1] & latent_sets[l2])
                overlap_pct = 100 * shared / k
                
                if dist not in distance_overlaps:
                    distance_overlaps[dist] = []
                distance_overlaps[dist].append(overlap_pct)
        
        # Print results
        print(f"\n{'Distance':<10} {'Avg Overlap':<12} {'Max Overlap':<12} {'Pairs':<10}")
        print("-" * 50)
        
        for dist in sorted(distance_overlaps.keys())[:15]:  # Show first 15 distances
            overlaps = distance_overlaps[dist]
            avg = sum(overlaps) / len(overlaps)
            max_ov = max(overlaps)
            print(f"{dist:<10} {avg:<12.1f}% {max_ov:<12.1f}% {len(overlaps):<10}")
        
        # Expected pattern check
        dist_1_avg = sum(distance_overlaps.get(1, [0])) / max(1, len(distance_overlaps.get(1, [1])))
        dist_5_avg = sum(distance_overlaps.get(5, [0])) / max(1, len(distance_overlaps.get(5, [1])))
        
        print(f"\n--- PATTERN CHECK ---")
        if dist_1_avg > dist_5_avg:
            print(f"  ✅ Nearby latents share more tokens (dist=1: {dist_1_avg:.1f}%, dist=5: {dist_5_avg:.1f}%)")
            print(f"     This is expected behavior for geographic attention.")
        else:
            print(f"  ⚠️  Overlap pattern seems random (dist=1: {dist_1_avg:.1f}%, dist=5: {dist_5_avg:.1f}%)")
            print(f"     Check your SIGMA parameter.")
        
        return distance_overlaps

    def geographic_pruning(self, tokens, mask):
        """
        Memory-efficient geographic pruning for spatial latents.
        Only applies to spatial latents (which have positions).
        """
        k = self.geo_k
        chunk_size = 50
        
        B, N, D = tokens.shape
        L = self.num_spatial_latents  # Only spatial latents participate
        
        # Fixed sigma based on latent grid spacing
        SIGMA = 0.5
        
        with torch.no_grad():
            bias_data = self.transform.get_bias_data(tokens)
            tokens_positions, latent_gaussians = bias_data
            
            x_min = tokens_positions[:, :, 0, 0]
            x_max = tokens_positions[:, :, 0, 1]
            y_min = tokens_positions[:, :, 1, 0]
            y_max = tokens_positions[:, :, 1, 1]
            
            mu_x_all = latent_gaussians[:, :, 0, 0]
            mu_y_all = latent_gaussians[:, :, 1, 0]
            
            sqrt_2 = torch.sqrt(torch.tensor(2.0, device=tokens.device))
            
            all_indices = []
            all_bias_values = []
            
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                
                mu_x = mu_x_all[:, chunk_start:chunk_end, None]
                mu_y = mu_y_all[:, chunk_start:chunk_end, None]
                
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

        debug_overlap=False

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

    def encode(self, tokens, mask, training=True, viz=False, diagnose=False):
        """
        Encode tokens into latent representations.
        Spatial latents: Use geographic cross-attention (local)
        Global latents: Skip cross-attention, only participate in self-attention
        All latents: Participate in self-attention together
        """
        diagnose=False
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
        
        # Geographic pruning (only for spatial latents)
        geographic_tokens, geographic_masks, geographic_bias = self.geographic_pruning(tokens, mask)
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
            if training:
                perm = torch.randperm(k, device=tokens.device)[:m]
                sampled_tokens = geographic_tokens[:, :, perm, :]
                sampled_masks = geographic_masks[:, :, perm]
                sampled_bias = geographic_bias[:, :, perm]
            else:
                sampled_tokens = geographic_tokens[:, :, :m, :]
                sampled_masks = geographic_masks[:, :, :m]
                sampled_bias = geographic_bias[:, :, :m]
            
            if diagnose:
                print_memory(f"Layer {layer_idx} after sampling (m={m})")
                print(f"   sampled_tokens: {sampled_tokens.shape} | {sampled_tokens.numel() * 4 / 1024**2:.2f} MB")
            
            # Process tokens with polar positional encoding
            processed_tokens = self.transform.process_data_for_encoder(
                sampled_tokens,
                sampled_masks,
                device=tokens.device
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
                bias=sampled_bias
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
            num_self_attns = len(self_attns)
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

    def reconstruct(self, latents, query_tokens, query_mask, training=True, diagnose=False):
        """
        Reconstruct with memory profiling.
        """
        diagnose=False
        if diagnose:
            reset_memory_stats()
            print("=" * 80)
            print("DECODER MEMORY DIAGNOSIS")
            print("=" * 80)
            print_memory("0. START")
        
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        L_global = self.num_global_latents
        device = latents.device
        D = latents.shape[-1]
        D_pe = self.transform.get_polar_encoding_dimension()
        k_spatial = self.decoder_k_spatial
        
        if diagnose:
            print(f"[SHAPES] B={B}, N={N}, L_spatial={L_spatial}, L_global={L_global}")
            print(f"[SHAPES] D={D}, D_pe={D_pe}, k_spatial={k_spatial}")
            print(f"[SHAPES] latents: {latents.shape}")
        
        # =========================================================================
        # 1. Select top-k spatial latents per query
        # =========================================================================
        spatial_indices, _ = self.transform.get_topk_latents_for_decoder(
            query_tokens, k=k_spatial, device=device
        )
        if diagnose:
            print_memory("1. After get_topk_latents_for_decoder")
            print(f"   spatial_indices: {spatial_indices.shape} | {spatial_indices.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 2. Gather spatial latents (MEMORY OPTIMIZED)
        # =========================================================================
        spatial_latents = latents[:, :L_spatial, :]  # [B, L_spatial, D]
        
        # Optimized gather: flatten and gather
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
        # 3. Compute relative PE and concat
        # =========================================================================
        relative_pe = self.transform.get_decoder_relative_pe(
            query_tokens, spatial_indices, device
        )
        if diagnose:
            print_memory("3a. After get_decoder_relative_pe")
            print(f"   relative_pe: {relative_pe.shape} | {relative_pe.numel() * 4 / 1024**2:.2f} MB")
        
        spatial_context = torch.cat([selected_spatial, relative_pe], dim=-1)
        if diagnose:
            print_memory("3b. After concat spatial_context")
            print(f"   spatial_context: {spatial_context.shape} | {spatial_context.numel() * 4 / 1024**2:.2f} MB")
        
        del selected_spatial, relative_pe
        
        # =========================================================================
        # 4. Prepare global context
        # =========================================================================
        if L_global > 0:
            global_latents = latents[:, L_spatial:, :]
            global_pe_zeros = torch.zeros(B, L_global, D_pe, device=device)
            global_context = torch.cat([global_latents, global_pe_zeros], dim=-1)
            if diagnose:
                print_memory("4. After global_context")
                print(f"   global_context: {global_context.shape} | {global_context.numel() * 4 / 1024**2:.2f} MB")
        else:
            global_context = None
        
        # =========================================================================
        # 5. Process queries
        # =========================================================================
        processed_queries, _, _ = self.transform.process_data(
            query_tokens, query_mask, query=True
        )
        if diagnose:
            print_memory("5. After process_data (queries)")
            print(f"   processed_queries: {processed_queries.shape} | {processed_queries.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 6. Cross-attention - THIS IS WHERE OOM HAPPENS
        # =========================================================================
        if diagnose:
            print_memory("6a. BEFORE decoder_cross_attn")
            print(f"   Query: {processed_queries.shape}")
            print(f"   Spatial context: {spatial_context.shape}")
            print(f"   Global context: {global_context.shape if global_context is not None else None}")
            
            # Estimate attention sizes
            heads = self.decoder_cross_attn.heads if hasattr(self.decoder_cross_attn, 'heads') else 8
            
            # Spatial attention: [B, N, heads, k_spatial]
            spatial_attn_mb = B * N * heads * k_spatial * 4 / 1024**2
            print(f"   [ESTIMATE] spatial attention: {B}×{N}×{heads}×{k_spatial} = {spatial_attn_mb:.2f} MB")
            
            # Global attention: [B, N, heads, L_global]
            global_attn_mb = B * N * heads * L_global * 4 / 1024**2
            print(f"   [ESTIMATE] global attention: {B}×{N}×{heads}×{L_global} = {global_attn_mb:.2f} MB")
            
            # K, V projections for spatial: [B, N, k_spatial, heads, dim_head]
            dim_head = 64  # typical
            kv_spatial_mb = 2 * B * N * k_spatial * heads * dim_head * 4 / 1024**2
            print(f"   [ESTIMATE] spatial K+V: 2×{B}×{N}×{k_spatial}×{heads}×{dim_head} = {kv_spatial_mb:.2f} MB")
        
        output = self.decoder_cross_attn(
            processed_queries,
            spatial_context=spatial_context,
            global_context=global_context
        )
        
        if diagnose:
            print_memory("6b. AFTER decoder_cross_attn")
            print(f"   output: {output.shape} | {output.numel() * 4 / 1024**2:.2f} MB")
        
        # =========================================================================
        # 7. Output head
        # =========================================================================
        result = self.output_head(output)
        if diagnose:
            print_memory("7. After output_head")
            print(f"   result: {result.shape}")
            print("=" * 80)
        
        return result
    
    def classify(self, latents):
        """Classify from latent representations"""
        return self.to_logits(latents)
    
    def forward(self, data, mask, mae_tokens=None, mae_tokens_mask=None, 
            training=True, task="reconstruction"):
        """Forward pass of the Atomizer"""
        #print_memory("before encoder ")
        latents = self.encode(data, mask, training=training)
        #print_memory("after encode ")
        if task == "encoder":
            return latents
        
        elif task == "reconstruction" or task == "vizualisation":
            # Chunked reconstruction (already implemented!)
            chunk_size = 10000
            N = mae_tokens.shape[1]
            
            if N > chunk_size:
                preds_list = []
                for i in range(0, N, chunk_size):
                    chunk_tokens = mae_tokens[:, i:i + chunk_size]
                    chunk_mask = mae_tokens_mask[:, i:i + chunk_size]
                    
                    p = self.reconstruct(latents, chunk_tokens, chunk_mask)
                    preds_list.append(p)
                return torch.cat(preds_list, dim=1)
            else:
                return self.reconstruct(latents, mae_tokens, mae_tokens_mask)
        
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