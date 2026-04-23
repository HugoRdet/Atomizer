"""
Sliding Window Utilities for Atomizer
======================================

Two functions:
  1. compute_crop_positions — grid of (y0, x0) for overlapping crops
  2. stitch_predictions — average overlapping logits into full-tile map
"""

import torch


def compute_crop_positions(full_h, full_w, crop_h, crop_w, stride_h, stride_w):
    """
    Compute top-left (y0, x0) positions for sliding window crops.
    Ensures full coverage: last crop clamped to image boundary.

    Returns:
        list of (y0, x0) tuples
    """
    ys = list(range(0, full_h - crop_h + 1, stride_h))
    if len(ys) == 0 or ys[-1] + crop_h < full_h:
        ys.append(max(0, full_h - crop_h))
    ys = sorted(set(ys))

    xs = list(range(0, full_w - crop_w + 1, stride_w))
    if len(xs) == 0 or xs[-1] + crop_w < full_w:
        xs.append(max(0, full_w - crop_w))
    xs = sorted(set(xs))

    return [(y, x) for y in ys for x in xs]


def stitch_predictions(crop_logits_list, crop_positions, crop_h, crop_w,full_h, full_w, num_classes):
    """
    Average overlapping crop logits into a full-tile prediction map.

    Args:
        crop_logits_list: list of [crop_h*crop_w, num_classes] tensors
        crop_positions:   list of (y0, x0) at 10m
        crop_h, crop_w:   crop size at 10m
        full_h, full_w:   full tile size at 10m
        num_classes:       number of output classes

    Returns:
        prediction: [full_h, full_w] argmax class indices
        logits_avg: [num_classes, full_h, full_w] averaged logits
    """
    device = crop_logits_list[0].device
    logits_sum = torch.zeros(num_classes, full_h, full_w, device=device)
    counts = torch.zeros(1, full_h, full_w, device=device)

    for logits, (y0, x0) in zip(crop_logits_list, crop_positions):
        logits_2d = logits.view(crop_h, crop_w, num_classes).permute(2, 0, 1)
        h = min(crop_h, full_h - y0)
        w = min(crop_w, full_w - x0)
        logits_sum[:, y0:y0 + h, x0:x0 + w] += logits_2d[:, :h, :w]
        counts[:, y0:y0 + h, x0:x0 + w] += 1.0

    counts = counts.clamp(min=1.0)
    logits_avg = logits_sum / counts
    prediction = logits_avg.argmax(dim=0)
    return prediction, logits_avg