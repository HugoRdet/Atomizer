import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from einops import repeat
from .nn_comp import PreNorm, CrossAttention, SelfAttention, FeedForward, LatentAttentionPooling


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


class Atomiser_FLAIR(pl.LightningModule):
    """
    Clean implementation of the Atomizer model for satellite image processing.
    
    The model processes tokens representing individual pixel-band measurements
    with metadata (position, wavelength, etc.) through cross-attention and 
    self-attention layers to learn representations for classification and reconstruction.
    """
    
    def __init__(self, *, config, transform):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        
        # Store config and transform
        self.config = config
        self.transform = transform
        
        # Extract model architecture parameters
        self.depth = config["Atomiser"]["depth"]
        self.num_latents = config["Atomiser"]["num_latents"]
        self.latent_dim = config["Atomiser"]["latent_dim"]
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
        
        
        #self.VV = nn.Parameter(torch.empty(dw))
        #self.VH = nn.Parameter(torch.empty(dw))
        #nn.init.trunc_normal_(self.VH, std=0.02, a=-2., b=2.)
        
        # Token limits for different phases
        self.max_tokens_forward = config["trainer"]["max_tokens_forward"]
        self.max_tokens_val = config["trainer"]["max_tokens_val"]
        
        # Compute input dimensions based on encoding configuration
        self.input_dim = self._compute_input_dim()
        self.query_dim_recon = self._compute_query_dim_recon()
        
        # Initialize model components
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
    
    def _compute_input_dim(self):
        """Compute total input dimension from all encodings"""
        pos_dim = self._get_encoding_dim("pos")
        wavelength_dim = self._get_encoding_dim("wavelength") 
        bandvalue_dim = self._get_encoding_dim("bandvalue")
        return 2 * pos_dim + wavelength_dim + bandvalue_dim  # 2x pos for x,y
    
    def _compute_query_dim_recon(self):
        """Compute query dimension for reconstruction (no band values)"""
        pos_dim = self._get_encoding_dim("pos")
        wavelength_dim = self._get_encoding_dim("wavelength")
        return 2 * pos_dim + wavelength_dim  # position + wavelength only
    
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
        """Initialize learnable latent vectors"""
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing"""
        # Create cached layer factories for weight sharing
        get_cross_attn = cache_fn(lambda: 
            CrossAttention(
                query_dim=self.latent_dim,
                context_dim=self.input_dim,
                heads=self.cross_heads,
                dim_head=self.cross_dim_head,
                dropout=self.attn_dropout,
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
            # Enable caching for weight sharing (except first layer)
            cache_args = {'_cache': (i > 0 and self.weight_tie_layers)}
            
            # Cross-attention and feedforward
            cross_attn = get_cross_attn(**cache_args)
            cross_ff = get_cross_ff(**cache_args)
            
            # Self-attention blocks
            self_attns = nn.ModuleList()
            for j in range(self.self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=j),
                    get_latent_ff(**cache_args, key=j)
                ]))
            
            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))
    
    def _init_decoder(self):
        """Initialize decoder for reconstruction"""
        self.recon_cross = CrossAttention(
                query_dim=self.query_dim_recon,
                context_dim=self.latent_dim,
                heads=self.latent_heads,
                dim_head=self.latent_dim_head,
                dropout=0.0)
        
        # Simple output head (no redundant LayerNorm)
        self.recon_tologits = nn.Linear(self.query_dim_recon, self.num_classes)  # Reconstruct reflectance only
        
        self.recon_head = nn.Sequential(
            nn.LayerNorm(self.query_dim_recon),
            nn.Linear(self.query_dim_recon,self.query_dim_recon*2),
            nn.GELU(),                # ReLU works too; GELU is smoother
            nn.LayerNorm(self.query_dim_recon*2),
            nn.Linear(self.query_dim_recon*2,self.query_dim_recon),
            nn.GELU(),
            nn.LayerNorm(self.query_dim_recon),
            nn.Linear(self.query_dim_recon,self.query_dim_recon)  # linear output for regression
        )
        #self.decoder_ff = PreNorm(self.query_dim_recon, FeedForward(queries_dim))
    
    def _init_classifier(self):
        """Initialize classification head"""
        if self.final_classifier_head:
            self.classifier = nn.Sequential(
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
            self.classifier = nn.Identity()
    
    def _subsample_tokens(self, tokens, mask, max_tokens, training=True):
        """Randomly subsample tokens to fit memory constraints"""
        B, N, D = tokens.shape
        
        if N <= max_tokens:
            return tokens, mask
        
        # Random permutation for subsampling
        device = tokens.device
        perm = torch.randperm(N, device=device)[:max_tokens]
        
        return tokens[:, perm], mask[:, perm]
    
    def encode(self, tokens, mask, training=True):
        """
        Encode tokens through the transformer layers
        
        Args:
            tokens: [B, N, D] input tokens
            mask: [B, N] attention mask (True = masked/invalid)
            training: whether in training mode
            
        Returns:
            latents: [B, num_latents, latent_dim] encoded representations
        """
        B = tokens.shape[0]
        
        # Initialize latents
        latents = repeat(self.latents, 'n d -> b n d', b=B)
        
        # Process through encoder layers
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            # Subsample tokens if needed
            max_tokens = self.max_tokens_forward if training else self.max_tokens_val
            current_tokens, current_mask = self._subsample_tokens(tokens, mask, max_tokens, training)
            
            # Process tokens through transform
            
            processed_tokens, processed_mask = self.transform.process_data(current_tokens, current_mask,query=False)
            processed_mask = processed_mask.bool()
            
            
            # Mask invalid tokens
            processed_tokens = processed_tokens.masked_fill_(processed_mask.unsqueeze(-1), 0.0)
            
            # Cross-attention: latents attend to tokens
            latents = cross_attn(latents, context=processed_tokens, mask=~processed_mask) + latents
            
            # Cross feedforward
            latents = cross_ff(latents) + latents
            
            # Self-attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents
        
        return latents
    
    def reconstruct(self, latents, query_tokens, query_mask):
        """
        Reconstruct token values from latent representations
        
        Args:
            latents: [B, num_latents, latent_dim] encoded representations
            query_tokens: [B, N, D] tokens to reconstruct (without band values)
            query_mask: [B, N] mask for query tokens
            
        Returns:
            predictions: [B, N, 1] reconstructed values
            output_mask: [B, N] mask aligned with predictions
        """
        # Process query tokens (remove band values, keep position + wavelength)
        processed_query, processed_mask = self.transform.process_data(
            query_tokens, query_mask, query=True
        )
        
        # Cross-attention: query tokens attend to latents
        attended = self.recon_cross(processed_query, context=latents, mask=None)
        skip_co=self.recon_head(attended)
        # Project to output
        
        predictions = attended+skip_co
        
        predictions = self.recon_tologits(predictions)
        
        return predictions, processed_mask
    
    def classify(self, latents):
        """
        Classify from latent representations
        
        Args:
            latents: [B, num_latents, latent_dim] encoded representations
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        return self.classifier(latents)
    
    def forward(self, data, mask, mae_tokens=None, mae_tokens_mask=None, 
                training=True, task="reconstruction"):
        """
        Forward pass of the Atomizer
        
        Args:
            data: [B, N, D] input tokens
            mask: [B, N] attention mask
            mae_tokens: [B, M, D] tokens for reconstruction (optional)
            mae_tokens_mask: [B, M] mask for reconstruction tokens (optional)
            training: whether in training mode
            task: "classification", "reconstruction", or "encoder"
            
        Returns:
            For classification: [B, num_classes] logits
            For reconstruction: ([B, M, 1] predictions, [B, M] mask)
            For encoder: [B, num_latents, latent_dim] latent representations
        """
        # Encode input tokens to latent representations
        latents = self.encode(data, mask, training=training)
        
        if task == "encoder":
            return latents
        elif task == "reconstruction":
            return self.reconstruct(latents, mae_tokens, mae_tokens_mask)
        else:  # task == "classification"
            return self.classify(latents)
    
    # Utility methods for freezing/unfreezing components
    def _set_requires_grad(self, module, flag):
        """Set requires_grad for all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = flag
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        self._set_requires_grad(self.encoder_layers, False)
        self.latents.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        self._set_requires_grad(self.encoder_layers, True)
        self.latents.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        self._set_requires_grad(self.recon_cross, False)
        self._set_requires_grad(self.recon_head, False)
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters"""
        self._set_requires_grad(self.recon_cross, True)
        self._set_requires_grad(self.recon_head, True)
    
    def freeze_classifier(self):
        """Freeze classifier parameters"""
        self._set_requires_grad(self.classifier, False)
    
    def unfreeze_classifier(self):
        """Unfreeze classifier parameters"""
        self._set_requires_grad(self.classifier, True)