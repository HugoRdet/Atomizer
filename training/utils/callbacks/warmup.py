from pytorch_lightning.callbacks import Callback
import torch

class DisplacementGradientWarmupCallback(Callback):
    """
    Gradually scale displacement gradients from 0 to 1.
    
    Args:
        start_epoch: Epoch to start warming up (before this, scale=0)
        warmup_epochs: Number of epochs to warm up from 0 to 1
    """
    
    def __init__(self, start_epoch: int = 10, warmup_epochs: int = 10):
        super().__init__()
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.displacement_params = []
        
    def _get_atomiser(self, pl_module):
        if hasattr(pl_module, 'atomiser'):
            return pl_module.atomiser
        elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'position_updater'):
            return pl_module.model
        return None
    
    def on_train_start(self, trainer, pl_module):
        """Cache displacement parameters."""
        atomiser = self._get_atomiser(pl_module)
        if atomiser and hasattr(atomiser, 'position_updater'):
            self.displacement_params = list(atomiser.position_updater.parameters())
            print(f"[GradientWarmup] Tracking {len(self.displacement_params)} displacement params")
            print(f"[GradientWarmup] Warmup: epoch {self.start_epoch} → {self.start_epoch + self.warmup_epochs}")
    
    def _get_scale(self, current_epoch: int) -> float:
        """Compute gradient scale factor."""
        if current_epoch < self.start_epoch:
            return 0.0
        elif current_epoch >= self.start_epoch + self.warmup_epochs:
            return 1.0
        else:
            # Linear warmup
            progress = (current_epoch - self.start_epoch) / self.warmup_epochs
            return progress
    
    def on_after_backward(self, trainer, pl_module):
        """Scale displacement gradients after backward pass."""
        scale = self._get_scale(trainer.current_epoch)
        
        if scale < 1.0:
            for param in self.displacement_params:
                if param.grad is not None:
                    param.grad.mul_(scale)
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Log current scale."""
        scale = self._get_scale(trainer.current_epoch)
        print(f"[GradientWarmup] Epoch {trainer.current_epoch}: displacement gradient scale = {scale:.2f}")
        
        if trainer.logger:
            trainer.logger.log_metrics(
                {"displacement_gradient_scale": scale}, 
                step=trainer.global_step
            )