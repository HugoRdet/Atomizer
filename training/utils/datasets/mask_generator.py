import torch
import random
import math
import matplotlib.pyplot as plt

class IJEPAStyleMaskGenerator:
    def __init__(self, 
                 input_size,                    # (H, W)
                 mask_ratio_range=(0.4, 0.8),   # % of image area to mask
                 patch_ratio_range=(0.1, 0.4),  # % of image height for patch size
                 aspect_ratio_range=(0.75, 1.33),
                 num_attempts=100):
        assert input_size[0] == input_size[1], "This version assumes square images"
        self.H, self.W = input_size
        self.mask_ratio_range = mask_ratio_range
        self.patch_ratio_range = patch_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.num_attempts = num_attempts

    def generate_mask(self):
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        # Sample mask ratio
        mask_ratio = random.uniform(*self.mask_ratio_range)
        total_area = self.H * self.W
        target_area = int(total_area * mask_ratio)
        masked_area = 0

        for _ in range(self.num_attempts):
            if masked_area >= target_area:
                break

            # Sample patch side ratio, and convert to pixel area
            patch_ratio = random.uniform(*self.patch_ratio_range)
            patch_size = int(patch_ratio * self.H)
            area = patch_size ** 2

            # Random aspect ratio
            aspect = random.uniform(*self.aspect_ratio_range)
            h = int(round(math.sqrt(area / aspect)))
            w = int(round(h * aspect))

            if h <= 0 or w <= 0 or h > self.H or w > self.W:
                continue

            top = random.randint(0, self.H - h)
            left = random.randint(0, self.W - w)

            current_mask = mask[top:top+h, left:left+w]
            new_area = (~current_mask).sum().item()
            if new_area == 0:
                continue

            mask[top:top+h, left:left+w] = True
            masked_area += new_area

        return mask

    def generate_batch(self, batch_size):
        return torch.stack([self.generate_mask() for _ in range(batch_size)])
    
    
def visualize_mask(mask, title="Mask", figsize=(5, 5), save_path=None):
    """
    Visualize a binary mask using matplotlib.

    Args:
        mask (torch.Tensor or np.ndarray): Binary mask of shape [H, W] or [B, H, W]
        title (str): Title of the plot
        figsize (tuple): Size of the figure
        save_path (str): Optional path to save the figure
    """
    if mask.ndim == 3:
        # If batch, visualize the first one
        mask = mask[0]
    
    plt.figure(figsize=figsize)
    plt.imshow(mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()