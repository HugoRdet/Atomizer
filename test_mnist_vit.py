"""
ViT baseline for the MNIST dark-pixel-drop vs random-pixel-drop ablation,
matched as closely as possible to Atomizer's actual setup
(see MNISTSparseCanvas dataset: per-pixel tokenization, subsample_mode
'uniform' vs 'background_only', pixel_threshold=0.5).

Design choices, matched to Atomizer:
- Per-PIXEL tokenization (patch size 1): 784 tokens for a 28x28 image,
  matching MNISTSparseCanvas's NUM_BANDS=1 -> 784 tokens/image. Each token's
  feature is just the (normalized) pixel intensity; positions are injected
  via a learned absolute positional embedding table (see note on RoPE below).
- ViT-Small-ish sizing (dim=480, depth=12, heads=8, mlp_dim=1920 = 4x) to
  land close to Atomizer's ~34M parameters. Exact count is printed at
  runtime -- nudge `DEPTH` by +-1 if you need to close the gap further.
- Learned ABSOLUTE positional embeddings, NO RoPE (deliberate -- this is
  precisely the architectural property under test: Atomizer uses relative/
  RoPE-style encoding and is architecturally exempt from fixed-slot brittleness;
  standard ViT is not).
- CLS token readout (not attention pooling), as the fairer "standard ViT" baseline.
- Training: full 784-token sequences only, canonical order, no masking or
  dropout of tokens at train time -- mirrors Atomizer's training regime,
  which is presumably also trained at subsample_keep_rate=1.0.
- Train/val/test split: validation carved out of the training set (test set
  untouched during training/model selection). Best-validation-accuracy
  checkpoint is saved and reloaded before final evaluation.
- Inference-time ablation, TWO drop strategies matching MNISTSparseCanvas exactly:
    * "uniform"          : keep a uniformly random subset of ALL 784 tokens
                            at the given subsample_keep_rate. (= "random drop")
    * "background_only"  : ALWAYS keep every foreground pixel (raw intensity
                            >= pixel_threshold=0.5); only the background
                            pixels are subsampled, at subsample_keep_rate.
                            (= "dark-pixel drop" / "drk" in the paper table)
  IMPORTANT: at a given subsample_keep_rate, "background_only" generally
  retains MORE total tokens than "uniform", by construction (foreground is
  never dropped). This matches Atomizer's actual dataset code -- the x-axis
  in the results table is the *rate parameter*, not a literal matched token
  count between conditions.
- Swept rates: {1.00, 0.75, 0.50, 0.25, 0.10}, matching
  tab:input_size_generalization for direct comparison against Atomizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import math

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG = 28
NUM_TOKENS = IMG * IMG  # 784, one token per pixel (patch size 1)

# Downscaled to match Atomizer's actual trained size (~7.4M params).
DIM = 224
DEPTH = 12
HEADS = 4
MLP_DIM = 4 * DIM  # 896

BATCH = 64
EPOCHS = 50            # matches Atomizer config: trainer.epochs = 50
VAL_SIZE = 5000
CKPT_PATH = "./vit_small_mnist_best.pt"

PIXEL_THRESHOLD = 0.5  # raw [0,1] intensity threshold, matches Atomizer's default

# Matched to Atomizer's config_test-MNIST.yaml
LR = 1e-4              # trainer.lr
WEIGHT_DECAY = 1e-3    # trainer.weight_decay
ATTN_DROPOUT = 0.05    # Atomiser.attn_dropout
FF_DROPOUT = 0.10      # Atomiser.ff_dropout
WARMUP_FRAC = 0.05     # matches Model_MNIST's default: max(1, 0.05 * total_steps)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
transform = transforms.ToTensor()
full_train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Stratified train/val split (matches Atomizer's MNISTSparseCanvas dataset:
# sklearn train_test_split, stratify=digit label, same seed=42), so class
# balance in the held-out val slice is controlled the same way on both sides
# of the comparison rather than left to a plain random split.
from sklearn.model_selection import train_test_split
import numpy as np

all_targets = full_train_ds.targets.numpy()
all_indices = np.arange(len(full_train_ds))
train_idx, val_idx = train_test_split(
    all_indices, test_size=VAL_SIZE, random_state=42, stratify=all_targets,
)
train_ds = Subset(full_train_ds, train_idx.tolist())
val_ds = Subset(full_train_ds, val_idx.tolist())
print(f"Train/val split: {len(train_ds)} train, {len(val_ds)} val (stratified, seed=42)")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)


def to_tokens(images):
    """
    images: [B, 1, 28, 28] raw [0,1] pixel values.
    Returns:
        tokens:    [B, 784, 1]  raw pixel intensity per token (patch size 1)
        intensity: [B, 784]     same values, used for foreground/background masking
    """
    B = images.shape[0]
    flat = images.view(B, NUM_TOKENS)      # [B, 784], row-major (y*28+x), matches
                                            # a standard raster scan of the image
    tokens = flat.unsqueeze(-1)            # [B, 784, 1]
    return tokens, flat


# ----------------------------------------------------------------------
# Model: ViT-Small-ish, patch size 1 (per-pixel tokens), CLS token, learned
# absolute positional embeddings, NO RoPE.
# ----------------------------------------------------------------------
class ViT(nn.Module):
    """
    NOTE on dropout: nn.TransformerEncoderLayer applies a single `dropout`
    rate to both the attention-output projection and the feedforward block.
    Atomizer uses two distinct rates (attn_dropout=0.05, ff_dropout=0.10).
    To match that split with the stock PyTorch layer, we patch each
    sub-layer's dropout module after construction rather than passing a
    single blended rate.
    """
    def __init__(self, patch_dim=1, dim=DIM, depth=DEPTH, heads=HEADS,
                 mlp_dim=MLP_DIM, num_tokens=NUM_TOKENS, num_classes=10,
                 attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, dim) * 0.02)
        self.dropout = nn.Dropout(attn_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=ff_dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Patch attention-side dropout modules to attn_dropout, leaving the
        # two FF-block dropouts (self.dropout1 after self_attn residual add
        # is actually the attn-output dropout; self.dropout / self.dropout2
        # inside the layer are FF-path) at ff_dropout as set above.
        for layer in self.encoder.layers:
            layer.dropout1 = nn.Dropout(attn_dropout)  # applied right after self-attn output

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens, token_indices=None, key_padding_mask=None):
        """
        tokens: [B, N, patch_dim]
        token_indices: [B, N] long tensor giving the ORIGINAL pixel index
                       (0..783) of each token, so the correct absolute
                       positional embedding is looked up even when tokens
                       were dropped. If None, assumes canonical 0..N-1.
        key_padding_mask: [B, N] bool, True where PADDED (to be ignored).
        """
        B, N, _ = tokens.shape
        x = self.patch_embed(tokens)  # [B, N, dim]

        if token_indices is None:
            pos = self.pos_embed[:, 1:1 + N, :].expand(B, -1, -1)
        else:
            pos_table = self.pos_embed[0, 1:, :]  # [NUM_TOKENS, dim]
            pos = pos_table[token_indices]  # [B, N, dim]

        x = x + pos
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, 0:1, :]
        x = torch.cat([cls, x], dim=1)  # [B, N+1, dim]

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        else:
            full_mask = None

        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=full_mask)
        x = self.norm(x)
        return self.head(x[:, 0])


model = ViT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameter count: {n_params:,}  (downscaled config for fast iteration -- "
      f"match this ratio when downscaling Atomizer)")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Step-based linear-warmup + cosine-decay schedule, matching Atomizer's
# transformers.get_cosine_schedule_with_warmup (num_cycles=0.5 default),
# stepped once per optimizer step (not once per epoch).
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(WARMUP_FRAC * total_steps))


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
print(f"LR schedule: total_steps={total_steps}, warmup_steps={warmup_steps}, peak_lr={LR}")


# ----------------------------------------------------------------------
# Training: ALWAYS full 784-token sequences, canonical order, no masking.
# ----------------------------------------------------------------------
def train_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        tokens, _ = to_tokens(imgs)
        logits = model(tokens, token_indices=None, key_padding_mask=None)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()  # step-based schedule, matches Atomizer's "interval": "step"
        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_loader(loader):
    model.eval()
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        tokens, _ = to_tokens(imgs)
        logits = model(tokens, token_indices=None, key_padding_mask=None)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


print("Training ViT (per-pixel tokens, CLS token, learned absolute pos-embed, no RoPE) on MNIST...")
best_val_acc = -1.0
for ep in range(EPOCHS):
    tr_loss, tr_acc = train_epoch()
    val_acc = eval_loader(val_loader)
    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {"model_state": model.state_dict(), "epoch": ep, "val_acc": val_acc},
            CKPT_PATH,
        )
        improved = "  <- best so far, checkpoint saved"
    print(f"epoch {ep+1}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}{improved}")

# Reload best checkpoint (by validation accuracy) before any test-set evaluation.
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
print(f"\nLoaded best checkpoint from epoch {ckpt['epoch']+1} (val_acc={ckpt['val_acc']:.4f})")

full_acc = eval_loader(test_loader)
print(f"Full-token (784/784) test accuracy (best checkpoint): {full_acc:.4f}")


# ----------------------------------------------------------------------
# Inference-time token removal, using the best checkpoint.
# Two strategies matching MNISTSparseCanvas's subsample_mode exactly.
# Variable-length sequences within a batch are handled via padding + a
# key_padding_mask, so padded slots never contribute to attention.
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_with_drop(rate, strategy):
    """
    rate: subsample_keep_rate in (0, 1].
    strategy: "uniform" (random drop over all 784 tokens) or
              "background_only" (always keep foreground >= PIXEL_THRESHOLD,
              subsample only background at `rate`).
    """
    assert strategy in ("uniform", "background_only")
    model.eval()
    total, correct = 0, 0
    total_kept_tokens, total_possible_tokens = 0, 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        tokens, intensity = to_tokens(imgs)  # [B,784,1], [B,784]
        B = imgs.size(0)

        # Build a per-sample boolean keep-mask [B, 784].
        if strategy == "uniform":
            keep_mask = torch.rand(B, NUM_TOKENS, device=device) < rate
            # guarantee at least one token survives per sample
            no_keep = keep_mask.sum(dim=1) == 0
            if no_keep.any():
                fallback_idx = torch.randint(0, NUM_TOKENS, (int(no_keep.sum()),), device=device)
                keep_mask[no_keep, fallback_idx] = True
        else:  # background_only
            is_fg = intensity >= PIXEL_THRESHOLD          # [B, 784], always kept
            bg_random = torch.rand(B, NUM_TOKENS, device=device) < rate
            keep_mask = is_fg | (~is_fg & bg_random)
            no_keep = keep_mask.sum(dim=1) == 0
            if no_keep.any():
                fallback_idx = torch.randint(0, NUM_TOKENS, (int(no_keep.sum()),), device=device)
                keep_mask[no_keep, fallback_idx] = True

        n_keep_per_sample = keep_mask.sum(dim=1)  # [B]
        max_keep = int(n_keep_per_sample.max().item())

        # Gather kept token indices, right-pad to max_keep within the batch.
        batch_indices = torch.full((B, max_keep), -1, dtype=torch.long, device=device)
        for b in range(B):
            idx = torch.nonzero(keep_mask[b], as_tuple=True)[0]
            batch_indices[b, : idx.numel()] = idx

        pad_mask = batch_indices == -1  # [B, max_keep], True = padding
        safe_indices = batch_indices.clamp(min=0)  # avoid -1 indexing; padded slots are masked anyway

        kept_tokens = torch.gather(
            tokens, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 1)
        )  # [B, max_keep, 1]
        kept_tokens = kept_tokens.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        logits = model(kept_tokens, token_indices=safe_indices, key_padding_mask=pad_mask)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += imgs.size(0)

        total_kept_tokens += n_keep_per_sample.sum().item()
        total_possible_tokens += B * NUM_TOKENS

    avg_keep_frac = total_kept_tokens / total_possible_tokens
    return correct / total, avg_keep_frac


rates = [1.00, 0.75, 0.50, 0.25, 0.10]
results = {"n_params": n_params, "full_acc": full_acc, "uniform": {}, "background_only": {}}

print("\nEvaluating token-removal ablation (matched to Atomizer's MNISTSparseCanvas protocol)...")
for rate in rates:
    if rate == 1.00:
        acc_u, frac_u = full_acc, 1.0
        acc_b, frac_b = full_acc, 1.0
    else:
        acc_u, frac_u = eval_with_drop(rate, "uniform")
        acc_b, frac_b = eval_with_drop(rate, "background_only")
    results["uniform"][rate] = {"acc": acc_u, "avg_keep_frac": frac_u}
    results["background_only"][rate] = {"acc": acc_b, "avg_keep_frac": frac_b}
    print(f"rate={rate:.2f}  uniform_acc={acc_u:.4f} (avg kept {frac_u:.3f})  "
          f"bg_only_acc={acc_b:.4f} (avg kept {frac_b:.3f})")

with open("./vit_mnist_drop_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to ./vit_mnist_drop_results.json")

# --- Primary result (goes in the main paper table): dark-pixel drop only ---
# This is the condition that isolates the mechanism under test (content-aware
# removal preserving foreground). Uniform/random-drop numbers are still
# computed and saved above (results["uniform"]) for an appendix table, per
# the scoping note agreed for the main text.
print("\n--- Main table: MNIST dark-pixel drop (ViT), accuracy % ---")
print(f"{'Tokens kept':<14}{'ViT (drk)':<12}")
for rate in rates:
    b = results["background_only"][rate]
    print(f"{rate:<14.2f}{b['acc']*100:>6.2f}")

print("\n--- Appendix table: MNIST random-drop vs dark-drop (ViT), accuracy % "
      "(avg fraction of 784 tokens actually kept) ---")
print(f"{'rate':<8}{'MNIST (rnd)':<16}{'MNIST (drk)':<16}")
for rate in rates:
    u = results["uniform"][rate]
    b = results["background_only"][rate]
    print(f"{rate:<8.2f}{u['acc']*100:>6.2f}% ({u['avg_keep_frac']:.2f})   "
          f"{b['acc']*100:>6.2f}% ({b['avg_keep_frac']:.2f})")
