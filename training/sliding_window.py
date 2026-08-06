"""
Sliding-Window Inference
==========================

Tiled inference for models that can't afford a full high-resolution image
in one forward pass. Crops the input into overlapping windows, runs the
model on each, and stitches logits back together with overlap-averaging.

Built for RAMEN on Sen1Floods11: RAMENBackbone tokenizes at the pixel
level (no patch embedding), so a 512x512 image produces effective_size**2
tokens per modality through full O(N^2) self-attention — intractable.
Training uses small random crops (see script_train_senflood_baseline.py's
--ramen_window_size); this function handles inference over the full
512x512 image at eval/test time using that same small-window model.

Works for both:
  - dict input models (RAMEN):        image = {"optical": [B,C,H,W], "sar": [B,C,H,W]}
  - single-tensor input models:       image = [B,C,H,W]
"""

import torch


def _spatial_size(image):
    """Return (H, W) regardless of whether image is a Tensor or a dict of Tensors."""
    if isinstance(image, dict):
        ref = next(iter(image.values()))
        return ref.shape[-2], ref.shape[-1]
    return image.shape[-2], image.shape[-1]


def _batch_size_and_device(image):
    if isinstance(image, dict):
        ref = next(iter(image.values()))
    else:
        ref = image
    return ref.shape[0], ref.device, ref.dtype


def _crop(image, y0, y1, x0, x1):
    """Crop a Tensor or a dict of Tensors to [..., y0:y1, x0:x1]."""
    if isinstance(image, dict):
        return {k: v[:, :, y0:y1, x0:x1] for k, v in image.items()}
    return image[:, :, y0:y1, x0:x1]


def _window_starts(size: int, window: int, stride: int) -> list:
    """
    Compute top-left starting offsets covering [0, size) with the given
    window/stride, guaranteeing the last window is flush with the edge
    (so no border pixels are left uncovered when size isn't evenly
    divisible by stride).
    """
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


@torch.no_grad()
def sliding_window_inference(
    model,
    image,
    window_size: int,
    stride: int,
    num_classes: int,
) -> torch.Tensor:
    """
    Run `model` over `image` in overlapping windows and stitch the
    resulting logits back into a full-resolution map.

    Args:
        model:       callable, model(window_crop) -> [B, num_classes, window_size, window_size]
                     (window_crop has the same type — Tensor or dict — as `image`,
                     cropped to window_size x window_size)
        image:       [B, C, H, W] Tensor, or dict[modality] -> [B, C, H, W] Tensor
        window_size: spatial size the model was built/trained for
        stride:      step between windows; stride < window_size gives overlap,
                     which is averaged in the output. stride == window_size
                     gives non-overlapping tiling (fastest, no averaging benefit).
        num_classes: number of output channels for the logits map

    Returns:
        logits: [B, num_classes, H, W]
    """
    H, W = _spatial_size(image)
    B, device, dtype = _batch_size_and_device(image)

    if H <= window_size and W <= window_size:
        # Image already fits in one window — no tiling needed.
        return model(image)

    logits_sum = torch.zeros(B, num_classes, H, W, device=device, dtype=torch.float32)
    count_map = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    y_starts = _window_starts(H, window_size, stride)
    x_starts = _window_starts(W, window_size, stride)

    for y0 in y_starts:
        y1 = y0 + window_size
        for x0 in x_starts:
            x1 = x0 + window_size
            crop = _crop(image, y0, y1, x0, x1)
            out = model(crop)  # [B, num_classes, window_size, window_size]
            logits_sum[:, :, y0:y1, x0:x1] += out.to(torch.float32)
            count_map[:, :, y0:y1, x0:x1] += 1.0

    logits = logits_sum / count_map.clamp(min=1.0)
    return logits.to(dtype)
