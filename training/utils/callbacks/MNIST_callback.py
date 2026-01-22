import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange
import wandb
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Dict, Any

class MNISTVisualizationCallback(pl.Callback):
    """Base Callback for MNIST Visualization."""
    
    TRAJECTORY_COLORS = ['#10002b', '#240046', '#3c096c', '#5a189a', '#7b2cbf', '#9d4edd', '#c77dff', '#e0aaff']
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.log_every_n_epochs = config["debug"].get("viz_every_n_epochs", 5)
        self.sample_indices = config["debug"].get("idxs_to_viz", [0])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
        
        pl_module.eval()
        try:
            dataset_val = trainer.datamodule.val_dataset
            for i, idx in enumerate(self.sample_indices):
                self._process_single_sample(idx, i, dataset_val, pl_module, trainer.current_epoch, "val")
        except Exception as e:
            print(f"Callback Error: {e}")
            import traceback
            traceback.print_exc()
        pl_module.train()

    def _process_single_sample(self, dataset_idx, viz_idx, dataset, pl_module, epoch, split):
        """Stub to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement _process_single_sample")

    def _create_and_upload_visualization(self, **kwargs):
        sample_idx = kwargs.get('sample_idx')
        epoch = kwargs.get('epoch')
        split = kwargs.get('split')
        
        recon = kwargs.get('reconstruction')
        target = kwargs.get('target')
        
        # Ensure we are dealing with numpy for metric calculation
        target_np = target.cpu().numpy() if torch.is_tensor(target) else target
        recon_np = recon.cpu().numpy() if torch.is_tensor(recon) else recon
        
        mse = np.mean((target_np - recon_np)**2) if target_np.shape == recon_np.shape else 0.0

        fig = self._create_reconstruction_figure(mse=mse, **kwargs)
        
        wandb_data = {
            f"reconstruction/sample_{sample_idx}_{split}": wandb.Image(fig, caption=f"Epoch {epoch} MSE: {mse:.5f}"),
            f"metrics/sample_{sample_idx}_mse": mse
        }
        
        traj = kwargs.get('trajectory')
        if traj and len(traj) > 1:
            traj_fig = self._create_trajectory_figure(**kwargs)
            wandb_data[f"trajectory/sample_{sample_idx}_{split}"] = wandb.Image(traj_fig)

        wandb.log(wandb_data)
        plt.close(fig)

class MNISTAttentionCallback(MNISTVisualizationCallback):
    def __init__(self, config):
        super().__init__(config)
        self.attn_weights = None

    def _get_attention_hook(self):
        def hook(module, input, output):
            if hasattr(module, 'last_attn_weights'):
                self.attn_weights = module.last_attn_weights
        return hook

    def _process_single_sample(self, dataset_idx, viz_idx, dataset, pl_module, epoch, split="val"):
        # 1. Attach Hook to the Cross-Attention layer
        target_layer = pl_module.encoder.encoder_layers[0][0].fn
        handle = target_layer.register_forward_hook(self._get_attention_hook())

        # 2. Prepare Data
        result = dataset.get_samples_to_viz(dataset_idx)
        image, image_tokens, attn_mask, queries, queries_mask, label, latent_pos, bbox = (
            result if len(result) == 8 else (*result[:7], {'x':0,'y':0,'w':64,'h':64})
        )
        
        device = pl_module.device
        with torch.no_grad():
            # 3. Forward Pass
            out = pl_module.encoder(
                image_tokens.unsqueeze(0).to(device),
                attn_mask.unsqueeze(0).to(device) if attn_mask is not None else None,
                queries.unsqueeze(0).to(device),
                torch.ones(1, queries.shape[0]).to(device),
                latent_pos.unsqueeze(0).to(device),
                training=False, task="visualization", return_trajectory=True
            )

            # 4. Handle logits for visualization
            preds = out['predictions'].squeeze(0) 
            if preds.shape[-1] > 1:
                # Classification mode: Use max probability as pixel intensity
                recon_viz = torch.softmax(preds, dim=-1).max(dim=-1)[0]
            else:
                recon_viz = preds.squeeze(-1)

            logits = pl_module.encoder.classify(out['latents'])
            pred_label = logits.argmax(dim=-1).item()

            # 5. Upload
            self._create_and_upload_visualization(
                sample_idx=dataset_idx, viz_idx=viz_idx, epoch=epoch, split=split,
                original_image=image, target=queries[:, 0], reconstruction=recon_viz,
                trajectory=out.get('trajectory'), bbox=bbox,
                gt_label=int(label), pred_label=pred_label,
                attn_map=self._project_attention_to_grid(self.attn_weights, queries) if self.attn_weights is not None else None
            )

        handle.remove()

    def _project_attention_to_grid(self, weights, queries):
        grid = np.zeros((64, 64))
        if weights is None: return grid
        w = weights[0].mean(dim=(0, 1)).cpu().numpy()
        x = queries[:, 1].long().cpu().numpy()
        y = queries[:, 2].long().cpu().numpy()
        for i in range(len(w)):
            if 0 <= x[i] < 64 and 0 <= y[i] < 64:
                grid[y[i], x[i]] += w[i]
        return gaussian_filter(grid, sigma=1.0)

    def _create_reconstruction_figure(self, **kwargs) -> plt.Figure:
        attn_map = kwargs.get('attn_map')
        has_attn = attn_map is not None
        cols = 4 if has_attn else 3
        
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
        
        target = kwargs['target'].cpu().numpy() if torch.is_tensor(kwargs['target']) else kwargs['target']
        recon = kwargs['reconstruction'].cpu().numpy() if torch.is_tensor(kwargs['reconstruction']) else kwargs['reconstruction']
        
        side = int(np.sqrt(len(target)))
        gt_img = target.reshape(side, side)
        rb_img = recon.reshape(side, side)

        axes[0].imshow(gt_img, cmap='gray')
        axes[0].set_title(f"GT: {kwargs.get('gt_label')}")
        
        axes[1].imshow(rb_img, cmap='magma')
        axes[1].set_title(f"Pred: {kwargs.get('pred_label')}")
        
        error = np.abs(gt_img - rb_img)
        im2 = axes[2].imshow(error, cmap='hot')
        axes[2].set_title("Diff Map")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        if has_attn:
            im3 = axes[3].imshow(attn_map, cmap='viridis')
            axes[3].set_title("Attention")
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        return fig

    def _create_trajectory_figure(self, **kwargs) -> plt.Figure:
        trajectory = kwargs['trajectory']
        fig, ax = plt.subplots(figsize=(6, 6))
        img = kwargs['original_image']
        img_np = img.cpu().numpy() if torch.is_tensor(img) else np.array(img)
        
        ax.imshow(img_np[:,:,0] if img_np.ndim==3 else img_np, cmap='gray', alpha=0.5)
        
        traj_pts = [t[0].detach().cpu().numpy() for t in trajectory]
        for l in range(len(traj_pts)-1):
            for i in range(traj_pts[0].shape[0]):
                ax.plot([traj_pts[l][i,0], traj_pts[l+1][i,0]], 
                        [traj_pts[l][i,1], traj_pts[l+1][i,1]], 
                        color=self.TRAJECTORY_COLORS[min(l, 7)], alpha=0.3)
        ax.set_title("Latent Movement")
        ax.axis('off')
        return fig