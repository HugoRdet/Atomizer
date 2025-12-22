"""
Script to inspect sigma values in Gaussian Bias Self-Attention.

Usage:
    python inspect_sigma.py path/to/checkpoint.ckpt
"""

import torch
import sys
import math
import re

def inspect_sigma(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Get state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Find all log_sigma keys
    sigma_keys = sorted([k for k in state_dict.keys() if 'log_sigma' in k.lower()])
    
    if not sigma_keys:
        print("No log_sigma parameters found!")
        return
    
    print("\n" + "="*80)
    print("LEARNED SIGMA VALUES (Gaussian Bias Self-Attention)")
    print("="*80)
    print(f"Initial σ = 3.0m (log_sigma = {math.log(3.0):.4f})")
    print("="*80)
    
    all_sigmas = []
    
    for key in sigma_keys:
        # Parse layer structure from key
        # Pattern: encoder.encoder_layers.X.2.Y.0.fn.log_sigma
        # X = encoder block, Y = self-attention index within block
        
        match = re.search(r'encoder_layers\.(\d+)\.2\.(\d+)', key)
        if match:
            block_idx = int(match.group(1))
            self_attn_idx = int(match.group(2))
            layer_name = f"Encoder Block {block_idx} - Self Attention {self_attn_idx}"
        else:
            layer_name = key
        
        log_sigma = state_dict[key]
        sigma = torch.exp(log_sigma)
        
        all_sigmas.append(sigma)
        
        print(f"\n┌─ {layer_name}")
        print(f"│  Key: {key}")
        print(f"│")
        print(f"│  Per-head σ values (meters):")
        print(f"│  ┌" + "─"*60)
        
        for head_idx, (ls, s) in enumerate(zip(log_sigma.tolist(), sigma.tolist())):
            # Visual bar
            bar_length = int(s * 5)  # Scale for visualization
            bar = "█" * min(bar_length, 40)
            change = s - 3.0  # Change from initial
            change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
            print(f"│  │ Head {head_idx}: σ = {s:6.3f}m ({change_str}m) {bar}")
        
        print(f"│  └" + "─"*60)
        print(f"│")
        print(f"│  Summary:")
        print(f"│    Mean σ = {sigma.mean().item():.3f}m")
        print(f"│    Min  σ = {sigma.min().item():.3f}m (Head {sigma.argmin().item()})")
        print(f"│    Max  σ = {sigma.max().item():.3f}m (Head {sigma.argmax().item()})")
        print(f"│    Std  σ = {sigma.std().item():.3f}m")
        print(f"│")
        
        # Interpretation
        mean_sigma = sigma.mean().item()
        effective_k = math.pi * mean_sigma**2 / 9  # Assuming 3m spacing
        print(f"│  Interpretation (spacing=3m):")
        print(f"│    Effective neighbors: ~{effective_k:.1f}")
        print(f"│    99% attention within: {3*mean_sigma:.1f}m radius")
        print(f"└" + "─"*70)
    
    # Global summary
    print("\n" + "="*80)
    print("GLOBAL SUMMARY ACROSS ALL LAYERS")
    print("="*80)
    
    all_sigmas_flat = torch.cat(all_sigmas)
    
    print(f"\nAll σ values:")
    print(f"  Global Mean: {all_sigmas_flat.mean().item():.3f}m")
    print(f"  Global Min:  {all_sigmas_flat.min().item():.3f}m")
    print(f"  Global Max:  {all_sigmas_flat.max().item():.3f}m")
    print(f"  Global Std:  {all_sigmas_flat.std().item():.3f}m")
    
    # Check if sigma changed significantly from init
    init_sigma = 3.0
    mean_change = all_sigmas_flat.mean().item() - init_sigma
    
    print(f"\nChange from initial σ=3.0m:")
    if abs(mean_change) < 0.5:
        print(f"  → σ stayed close to init ({mean_change:+.2f}m): Model liked this locality")
    elif mean_change > 0.5:
        print(f"  → σ increased ({mean_change:+.2f}m): Model wanted MORE context (wider attention)")
    else:
        print(f"  → σ decreased ({mean_change:+.2f}m): Model wanted SHARPER attention (more local)")
    
    # Distribution visualization
    print("\n" + "="*80)
    print("DISTRIBUTION OF LEARNED σ VALUES")
    print("="*80)
    
    # Simple histogram
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, float('inf')]
    bin_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-10', '10-15', '15-20', '20+']
    
    counts = []
    for i in range(len(bins)-1):
        count = ((all_sigmas_flat >= bins[i]) & (all_sigmas_flat < bins[i+1])).sum().item()
        counts.append(count)
    
    max_count = max(counts) if max(counts) > 0 else 1
    
    print("\nHistogram of σ values (meters):")
    for label, count in zip(bin_labels, counts):
        bar = "█" * int(count / max_count * 30)
        print(f"  {label:>6}m: {bar} ({count})")
    
    # Global bias values
    print("\n" + "="*80)
    print("GLOBAL BIAS VALUES (for global latents)")
    print("="*80)
    
    global_bias_keys = sorted([k for k in state_dict.keys() if 'log_sigma' in k.lower()])
    
    if global_bias_keys:
        for key in global_bias_keys:
            match = re.search(r'encoder_layers\.(\d+)\.2\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                self_attn_idx = int(match.group(2))
                layer_name = f"Encoder Block {block_idx} - Self Attention {self_attn_idx}"
            else:
                layer_name = key
            
            value = state_dict[key].item()
            print(f"  {layer_name}: global_bias = {value:.4f}")
    else:
        print("  No global_bias parameters found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_sigma.py <checkpoint_path>")
        print("\nExample: python inspect_sigma.py ./checkpoints/model.ckpt")
    else:
        inspect_sigma(sys.argv[1])