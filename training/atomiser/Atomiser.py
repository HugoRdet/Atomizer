import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from einops import repeat
from .nn_comp import PreNorm, CrossAttention, SelfAttention, FeedForward, LatentAttentionPooling
import einops as einops

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


class GaussianUpdateModule(nn.Module):
    def __init__(self, embedding_size, initial_lr=0.1):
        super().__init__()
        # MLP to predict deltas for each Gaussian
        self.delta_predictor = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.GELU(),
            nn.Linear(embedding_size // 2, 3)  # Δμ_x, Δμ_y, Δσ
        )
        # Learnable learning rate (how big the updates should be)
        self.update_scale = nn.Parameter(torch.tensor(initial_lr))
        
    def forward(self, latent_tokens, current_gaussians):
        """
        latent_tokens: [batch_size, 400, embedding_size]
        current_gaussians: [batch_size, 400, 2, 2] where:
            - dim 2: axis (0=x, 1=y)
            - dim 3: parameters (0=mu, 1=sigma)
        
        Returns:
            updated_gaussians: [batch_size, 400, 2, 2]
        """
        batch_size = latent_tokens.shape[0]
        
        # Predict deltas for each Gaussian
        deltas = self.delta_predictor(latent_tokens)  # [batch_size, 400, 3]
        
        # Scale deltas
        deltas = deltas * self.update_scale
        
        # Extract delta components
        delta_mu_x = deltas[:, :, 0]  # [batch_size, 400]
        delta_mu_y = deltas[:, :, 1]  # [batch_size, 400]
        delta_sigma = deltas[:, :, 2]  # [batch_size, 400]
        
        # Clone current Gaussians to avoid in-place modification
        updated_gaussians = current_gaussians.clone()
        
        # Update mu_x and mu_y
        updated_gaussians[:, :, 0, 0] = current_gaussians[:, :, 0, 0] + delta_mu_x
        updated_gaussians[:, :, 1, 0] = current_gaussians[:, :, 1, 0] + delta_mu_y
        
        # Update sigma (same for both x and y, or you could make them independent)
        updated_gaussians[:, :, 0, 1] = current_gaussians[:, :, 0, 1] + delta_sigma
        updated_gaussians[:, :, 1, 1] = current_gaussians[:, :, 1, 1] + delta_sigma
        
        # Ensure sigma stays positive
        updated_gaussians[:, :, :, 1] = F.softplus(updated_gaussians[:, :, :, 1])
        # Or: updated_gaussians[:, :, :, 1] = torch.clamp(updated_gaussians[:, :, :, 1], min=1e-3)
        
        return updated_gaussians


class Atomiser(pl.LightningModule):
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
        
        # Compute dimensions first
        self.input_dim = self._compute_input_dim()
        self.query_dim_recon = self._compute_query_dim_recon()
        
        # Extract model architecture parameters
        self.depth = config["Atomiser"]["depth"]
        self.num_latents = config["Atomiser"]["num_latents"]
        # Use config latent_dim if available, otherwise fall back to input_dim
        self.latent_dim = self._get_encoding_dim("pos")*2 #config["Atomiser"].get("latent_dim", self.input_dim)
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
        
        # Token limits for different phases
        self.max_tokens_forward = config["trainer"]["max_tokens_forward"]
        self.max_tokens_val = config["trainer"]["max_tokens_val"]
        
        # Initialize model components
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
        self.Gaussian_update=GaussianUpdateModule(self.latent_dim, initial_lr=0.1)
    
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
        reflectance_dim = self._get_encoding_dim("bandvalue")  # No band values in reconstruction queries
        return 2 * pos_dim + wavelength_dim #+ reflectance_dim  # position + wavelength only
    
    def _get_encoding_dim(self, attribute):
        """Get encoding dimension for a specific attribute"""
        encoding_type = self.config["Atomiser"][f"{attribute}_encoding"]

        #if attribute=="pos":
        #    return 396
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
        """Initialize learnable latent vectors and input projection"""
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
        
        
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing"""
        # Create cached layer factories for weight sharing
        
        get_cross_attn = cache_fn(lambda: 
            PreNorm(
                dim=self.latent_dim,
                fn=CrossAttention(
                    query_dim=self.latent_dim,
                    context_dim=self.input_dim,
                    heads=self.cross_heads,
                    dim_head=self.cross_dim_head,
                    dropout=self.attn_dropout,
                    id=1
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

            cache_key = f"cross_entropy_bias_std_layer_{i}"
            tmp_stds = nn.Parameter(torch.rand(400))
            setattr(self, cache_key, tmp_stds)
            
            # Enable caching for weight sharing (except first layer)
            cache_args = {'_cache': (i > 0 and self.weight_tie_layers)}

            
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
        """Initialize decoder for reconstruction following Perceiver IO formula"""
        # Cross-attention with PreNorm (applies LayerNorm before attention)
        #533 512
        #print(self.query_dim_recon,self.latent_dim,"fhkdfhskjlhfdsjkql"*100)
      
        self.recon_cross = PreNorm(
            dim=self.query_dim_recon,
            fn=CrossAttention(
                query_dim=self.query_dim_recon,
                context_dim=self.latent_dim,
                heads=self.latent_heads,
                dim_head=self.latent_dim_head,
                dropout=0.0,
                id=0
            ),
            context_dim= self.latent_dim # For normalizing context (latents)
        )
        
        # MLP with PreNorm (applies LayerNorm before MLP)
        self.recon_ff = PreNorm(
            dim=self.query_dim_recon,
            fn=FeedForward(
                dim=self.query_dim_recon, 
                mult=4, 
                dropout=0.0
            )
        )
        
        # Final output projection
        self.recon_to_logits = nn.Linear(self.query_dim_recon, self.num_classes)  # Single reflectance value
    
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
    
    def _subsample_tokens(self, tokens, mask, max_tokens, training=True):
        """Randomly subsample tokens to fit memory constraints"""
        B, N, D = tokens.shape
        
        #if N <= max_tokens:
        #    return tokens, mask
        
        # Random permutation for subsampling
        device = tokens.device
        perm = torch.randperm(N, device=device)[:max_tokens]
        
        return tokens[:, perm], mask[:, perm],perm
    


    def bias(self, data,idx):
        """
        Compute the integral of 2D Gaussians over rectangular domains.
        
        Args:
            data: tuple of (tokens_positions, latent_gaussians)
                tokens_positions: [batch_size, nb_tokens, 2, 2]
                    - dim 2: axis (0=x, 1=y)
                    - dim 3: bounds (0=min, 1=max)
                latent_gaussians: [batch_size, 400, 2, 2]
                    - dim 2: axis (0=x, 1=y)
                    - dim 3: parameters (0=center/mu, 1=std/sigma)
        
        Returns:
            torch.Tensor: [batch_size, 400, nb_tokens] - integral values
        """
        


        tokens_positions, latent_gaussians = data

        cache_key = f"cross_entropy_bias_std_layer_{idx}"
        learned_sigmas = getattr(self, cache_key)
        learned_sigmas = torch.unsqueeze(learned_sigmas, 0)  # [1, 400]
        learned_sigmas = torch.unsqueeze(learned_sigmas, -1)
        learned_sigmas = torch.exp(learned_sigmas)  # Ensure positivity
        learned_sigmas = einops.repeat(learned_sigmas, 'b n c -> (b b1) n c', b1=tokens_positions.shape[0])
        
        batch_size = tokens_positions.shape[0]
        nb_tokens = tokens_positions.shape[1]
        nb_gaussians = latent_gaussians.shape[1]
        
        # Extract parameters
        # tokens_positions: [batch_size, nb_tokens, 2, 2]
        x_min = tokens_positions[:, :, 0, 0]  # [batch_size, nb_tokens]
        x_max = tokens_positions[:, :, 0, 1]  # [batch_size, nb_tokens]
        y_min = tokens_positions[:, :, 1, 0]  # [batch_size, nb_tokens]
        y_max = tokens_positions[:, :, 1, 1]  # [batch_size, nb_tokens]
        
        # latent_gaussians: [batch_size, 400, 2, 2]
        mu_x = latent_gaussians[:, :, 0, 0]  # [batch_size, 400]
        sigma_x = latent_gaussians[:, :, 0, 1]  # [batch_size, 400]
        mu_y = latent_gaussians[:, :, 1, 0]  # [batch_size, 400]
        sigma_y = latent_gaussians[:, :, 1, 1]  # [batch_size, 400]
        
        # Reshape for broadcasting: we want to compute all pairs (gaussian, token)
        # Shape transformations for broadcasting
        mu_x = mu_x[:, :, None]  # [batch_size, 400, 1]
        sigma_x = sigma_x[:, :, None]*learned_sigmas  # [batch_size, 400, 1]
        mu_y = mu_y[:, :, None]  # [batch_size, 400, 1]
        sigma_y = sigma_y[:, :, None]*learned_sigmas  # [batch_size, 400, 1]
        
        
        x_min = x_min[:, None, :]  # [batch_size, 1, nb_tokens]
        x_max = x_max[:, None, :]  # [batch_size, 1, nb_tokens]
        y_min = y_min[:, None, :]  # [batch_size, 1, nb_tokens]
        y_max = y_max[:, None, :]  # [batch_size, 1, nb_tokens]
        
        # For a 1D Gaussian N(mu, sigma^2), the integral from a to b is:
        # (1/2) * [erf((b - mu)/(sigma * sqrt(2))) - erf((a - mu)/(sigma * sqrt(2)))]
        # This is equivalent to: Phi((b - mu)/sigma) - Phi((a - mu)/sigma)
        # where Phi is the standard normal CDF
        
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=latent_gaussians.device))
        
        # Compute CDF values for x-axis
        # Using the identity: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        z_x_min = (x_min - mu_x) / (sigma_x * sqrt_2)
        z_x_max = (x_max - mu_x) / (sigma_x * sqrt_2)
        
        # Integral in x direction
        integral_x = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
        
        # Compute CDF values for y-axis
        z_y_min = (y_min - mu_y) / (sigma_y * sqrt_2)
        z_y_max = (y_max - mu_y) / (sigma_y * sqrt_2)
        
        # Integral in y direction
        integral_y = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
        
        # For a 2D Gaussian with independent components, the integral is the product
        result =integral_x * integral_y 
        result = result/result.max(dim=-1, keepdim=True)[0]
        result/=0.2**2
        #result = torch.softmax(result, dim=1)

        # For visualization: normalize to [0, 1] per token
        #result_min = result.min(dim=1, keepdim=True)[0]
        #result_max = result.max(dim=1, keepdim=True)[0]
        #result = (result - result_min) / (result_max - result_min + 1e-8)


        result =torch.log( result + 1e-8 )  # [batch_size, 400, nb_tokens]
        
        
        return result




    
    def encode(self, tokens, mask, training=True,viz=False):
        """
        Encode tokens through the transformer layers
        
        Args:
            tokens: [B, N, D] input tokens
            mask: [B, N] attention mask (True = masked/invalid)
            training: whether in training mode
            
        Returns:
            latents: [B, num_latents, latent_dim] encoded representations
        """
        #B = tokens.shape[0]

        # FIXED: Initialize latents from learnable parameters (not from input tokens!)
        #latents = repeat(self.latents, 'n d -> b n d', b=B)

        latents_s=None
        matrix_biais=None
        
        
        latents = repeat(self.latents, 'n d -> b n d', b=tokens.shape[0])
        

        
        
        # Process through encoder layers
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):

            # Subsample tokens if needed
            max_tokens = self.max_tokens_forward if training else self.max_tokens_val
            
            current_tokens, current_mask,_ = self._subsample_tokens(tokens, mask, max_tokens, training)
            
            # Process tokens through transform
            processed_tokens, processed_mask, tmp_latents,tmp_bias = self.transform.process_data(current_tokens, current_mask, query=False)

            processed_mask = processed_mask.bool()
            processed_tokens = processed_tokens.masked_fill_(processed_mask.unsqueeze(-1), 0.0)

            
            latents_s=latents[:,:400,:]
            latents_l=latents[:,400:,:]
            
        

            if viz and layer_idx==1:
                max_tokens_b=75000
                matrix_biais=[]
                matrix_attention=[]
                
                
                tokens_viz, mask_viz,perm_viz = self._subsample_tokens(tokens, mask, tokens.shape[1], training)

                for tmp_idx in range(0,tokens.shape[1],max_tokens_b):
                    tmp_tokens_viz=tokens_viz[:,tmp_idx:tmp_idx+max_tokens_b]
                    tmp_mask_viz=mask_viz[:,tmp_idx:tmp_idx+max_tokens_b]

                    
                    

                    processed_tokens_tmp, processed_mask_tmp, tmp_latents,tmp_bias_viz = self.transform.process_data(tmp_tokens_viz, tmp_mask_viz, query=False)
                    
                
                    processed_mask_tmp = processed_mask_tmp.bool()
                    processed_tokens_tmp = processed_tokens_tmp.masked_fill_(processed_mask_tmp.unsqueeze(-1), 0.0)
        
                    tmp_b=self.bias(tmp_bias_viz,layer_idx)
                    
                    cross_attn.fn.viz=True
                    
                    aya=cross_attn(latents_s, context=processed_tokens_tmp, mask=~processed_mask_tmp, bias=tmp_b, id=layer_idx)
                    
                    cross_attn.fn.viz=False

                    matrix_biais.append(tmp_b[0])
                    matrix_attention.append(aya[0].mean(dim=0))
                
                matrix_biais=torch.cat(matrix_biais,dim=-1)
                inv_perm = torch.argsort(perm_viz)
                matrix_biais = matrix_biais[:, inv_perm]
                
                #matrix_biais+=torch.abs(matrix_biais.min(dim=-1, keepdim=True)[0])
                #matrix_biais = matrix_biais/matrix_biais.max(dim=-1, keepdim=True)[0]
                matrix_biais=einops.rearrange(matrix_biais,"l (c h w) -> l c h w",c=5,h=512,w=512)
                matrix_biais=matrix_biais.mean(dim=1)

                matrix_attention=torch.cat(matrix_attention,dim=-1)
                matrix_attention=matrix_attention[:, inv_perm]
                #matrix_attention=F.softmax(matrix_attention, dim=-1)
                matrix_attention=einops.rearrange(matrix_attention,"l (c h w) -> l c h w",c=5,h=512,w=512)
                matrix_attention=matrix_attention.mean(dim=1)
                print("scores test",matrix_attention[0,30:,30:].sum()," ayou: ",matrix_attention[0,:30,:30].sum())
                
                #matrix of size [400,512,512[]]
                #here mat has a shape [400,nb_tokens]
                #it's possible to do something like [400,5,512,512]
                #then do the average to get someting like [400,512,512]

                    

            
            b=self.bias(tmp_bias,layer_idx)
            
            term_1=cross_attn(latents_s, context=processed_tokens, mask=~processed_mask,bias=b,id=layer_idx)
           
            latents_s =  term_1 + latents_s
            latents_s = cross_ff(latents_s) + latents_s
            
            latents = torch.cat([latents_s, latents_l], dim=1)
            
            # Cross feedforward with residual connection
            
            
            # Self-attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) +latents
            
  
            
        
        if viz:
            return latents,(matrix_biais,matrix_attention)
        
        return latents
    
    
    
    def reconstruct(self, latents, query_tokens, query_mask):
        """
        Reconstruct token values from latent representations following Perceiver IO
        
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
        
        # CORRECT Perceiver IO decoder formula:
        # Step 1: Cross-attention - queries attend to latents
        # X_QKV = Attn(layerNorm(X_Q), layerNorm(X_KV)) + X_Q
       
        x = self.recon_cross(processed_query, context=latents, mask=None) #+ processed_query
        
        # Step 2: MLP with residual connection  
        # X_QKV = X_QKV + MLP(layerNorm(X_QKV))
        x = x + self.recon_ff(x)
        
        # Step 3: Final projection to output space
        predictions = self.recon_to_logits(x)  # [B, N, 1]

        

        #predictions = einops.rearrange( predictions, "b (q u) c -> b q u c", 
        #                                b=predictions.shape[0], 
        #                                c=predictions.shape[-1], 
        #                                q=5)
    
        #predictions = predictions.mean(dim=1)
        
        return predictions
    
    def classify(self, latents):
        """
        Classify from latent representations
        
        Args:
            latents: [B, num_latents, latent_dim] encoded representations
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        return self.to_logits(latents)
    
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
        latents=None
        matrix_biais=None
        if task == "vizualisation":
            latents,matrix_biais = self.encode(data, mask, training=training,viz=True)
        else:
            latents = self.encode(data, mask, training=training)
        
        if task == "encoder":
            return latents
        elif task == "reconstruction" or task == "vizualisation":
            # Handle large token sequences by chunking
            nb_tokens_processed = 100000
            if mae_tokens.shape[1] > nb_tokens_processed:
                preds_list = []
                
               
                for i in range(0, mae_tokens.shape[1], nb_tokens_processed):
                    chunk_tokens = mae_tokens[:, i:i+nb_tokens_processed]
                    chunk_mask = mae_tokens_mask[:, i:i+nb_tokens_processed]
                    p = self.reconstruct(latents, chunk_tokens, chunk_mask)
                    preds_list.append(p)
                    
                
                preds = torch.cat(preds_list, dim=1)


                if task=="vizualisation":
                    return preds,matrix_biais
                
                
                
                return preds
            else:
                return self.reconstruct(latents, mae_tokens, mae_tokens_mask)
            

            
        #forward_viz(self, x, context, mask=None, bias=None, id=0)
            
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
        self._set_requires_grad(self.input_projection, False)
        self.latents.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        self._set_requires_grad(self.encoder_layers, True)
        self._set_requires_grad(self.input_projection, True)
        self.latents.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        self._set_requires_grad(self.recon_cross, False)
        self._set_requires_grad(self.recon_ff, False)
        self._set_requires_grad(self.recon_to_logits, False)
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters"""
        self._set_requires_grad(self.recon_cross, True)
        self._set_requires_grad(self.recon_ff, True)
        self._set_requires_grad(self.recon_to_logits, True)
    
    def freeze_classifier(self):
        """Freeze classifier parameters"""
        self._set_requires_grad(self.to_logits, False)
    
    def unfreeze_classifier(self):
        """Unfreeze classifier parameters"""
        self._set_requires_grad(self.to_logits, True)