"""
Eval-only entry point for the ViT MNIST drop experiment.

Use this when a training run already produced ./vit_small_mnist_best.pt
(saved on every val-accuracy improvement in vit_mnist_drop_experiment.py)
and you just want the final test accuracy + drop-ablation sweep, without
re-running the training loop.

Usage:
    python vit_mnist_eval_only.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH = None  # unused, kept for clarity: tokenization is per-pixel (patch=1)
NUM_TOKENS = 28 * 28  # 784

DIM = 224
DEPTH = 12
HEADS = 4
MLP_DIM = 4 * DIM  # 896
ATTN_DROPOUT = 0.05
FF_DROPOUT = 0.10

BATCH = 64
CKPT_PATH = "./vit_small_mnist_best.pt"
PIXEL_THRESHOLD = 0.5


# ----------------------------------------------------------------------
# Data — only test set is needed here.
# ----------------------------------------------------------------------
transform = transforms.ToTensor()
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)


def to_tokens(images):
    B = images.shape[0]
    flat = images.view(B, NUM_TOKENS)
    tokens = flat.unsqueeze(-1)
    return tokens, flat


# ----------------------------------------------------------------------
# Model — identical definition to the training script.
# ----------------------------------------------------------------------
class ViT(nn.Module):
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
        for layer in self.encoder.layers:
            layer.dropout1 = nn.Dropout(attn_dropout)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens, token_indices=None, key_padding_mask=None):
        B, N, _ = tokens.shape
        x = self.patch_embed(tokens)

        if token_indices is None:
            pos = self.pos_embed[:, 1:1 + N, :].expand(B, -1, -1)
        else:
            pos_table = self.pos_embed[0, 1:, :]
            pos = pos_table[token_indices]

        x = x + pos
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, 0:1, :]
        x = torch.cat([cls, x], dim=1)

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
print(f"Model parameter count: {n_params:,}")

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']+1} (val_acc={ckpt['val_acc']:.4f})")


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


full_acc = eval_loader(test_loader)
print(f"Full-token (784/784) test accuracy: {full_acc:.4f}")


@torch.no_grad()
def eval_with_drop(rate, strategy):
    assert strategy in ("uniform", "background_only")
    model.eval()
    total, correct = 0, 0
    total_kept_tokens, total_possible_tokens = 0, 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        tokens, intensity = to_tokens(imgs)
        B = imgs.size(0)

        if strategy == "uniform":
            keep_mask = torch.rand(B, NUM_TOKENS, device=device) < rate
            no_keep = keep_mask.sum(dim=1) == 0
            if no_keep.any():
                fallback_idx = torch.randint(0, NUM_TOKENS, (int(no_keep.sum()),), device=device)
                keep_mask[no_keep, fallback_idx] = True
        else:
            is_fg = intensity >= PIXEL_THRESHOLD
            bg_random = torch.rand(B, NUM_TOKENS, device=device) < rate
            keep_mask = is_fg | (~is_fg & bg_random)
            no_keep = keep_mask.sum(dim=1) == 0
            if no_keep.any():
                fallback_idx = torch.randint(0, NUM_TOKENS, (int(no_keep.sum()),), device=device)
                keep_mask[no_keep, fallback_idx] = True

        n_keep_per_sample = keep_mask.sum(dim=1)
        max_keep = int(n_keep_per_sample.max().item())

        batch_indices = torch.full((B, max_keep), -1, dtype=torch.long, device=device)
        for b in range(B):
            idx = torch.nonzero(keep_mask[b], as_tuple=True)[0]
            batch_indices[b, : idx.numel()] = idx

        pad_mask = batch_indices == -1
        safe_indices = batch_indices.clamp(min=0)

        kept_tokens = torch.gather(
            tokens, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 1)
        )
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

print("\nEvaluating token-removal ablation...")
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
