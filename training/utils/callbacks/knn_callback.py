import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import wandb
import torchmetrics
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm

class KNN_multiclass_EvaluationCallback(pl.Callback):
    """
    Callback to evaluate learned representations using KNN classification.
    Uses a datamodule for KNN evaluation.
    """
    
    def __init__(
        self,
        config,
        knn_datamodule,
    ):
        """
        Args:
            config: Configuration dictionary with knn settings
            knn_datamodule: DataModule for KNN evaluation (should have train_dataloader() and val_dataloader())
        """
        super().__init__()
        self.config = config
        self.knn_datamodule = knn_datamodule
        self.k = config["knn"]["k"]
        self.eval_every_n_epochs = config["knn"]["knn_every_n_epochs"]
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run KNN evaluation every n epochs."""
        # Only run every n epochs
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return
            
        # Only run on rank 0 to avoid duplicated evaluation
        if trainer.global_rank != 0:
            return
        
        # Get dataloaders from the KNN datamodule
        knn_train_dataloader = self.knn_datamodule.train_dataloader()
        knn_val_dataloader = self.knn_datamodule.val_dataloader()
        
        # Run KNN evaluation for each class (binary classification)
        class_accuracies = []
        
        for class_idx in range(19):  # 19 classes in BigEarthNet
            metrics = self._evaluate_knn_binary_online(
                pl_module, 
                knn_train_dataloader, 
                knn_val_dataloader,
                class_idx=class_idx,
                k=self.k
            )
            
            class_accuracies.append(metrics['accuracy'])
                
        # Calculate overall metrics (always 19 classes)
        overall_metrics = {
            'mean_accuracy': np.mean(class_accuracies)
        }
            
        # Log results
        self._log_results(overall_metrics, trainer.current_epoch)
        
    def _evaluate_knn_binary_online(
        self,
        pl_module: pl.LightningModule,
        train_dataloader,
        val_dataloader,
        class_idx: int,
        k: int
    ) -> Dict[str, float]:
        """Evaluate KNN for binary classification without storing features."""
        
        pl_module.eval()
        
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        
        # Collect training data online
        train_features_list = []
        train_labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(train_dataloader,desc="knn - training phase"):
                # Unpack batch
                image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch
                
                # Move to device
                image = image.to(pl_module.device)
                attention_mask = attention_mask.to(pl_module.device)
                mae_tokens = mae_tokens.to(pl_module.device)
                mae_tokens_mask = mae_tokens_mask.to(pl_module.device)
                
                # Extract features
                latents = pl_module.forward(
                    image,
                    attention_mask,
                    mae_tokens,
                    mae_tokens_mask,
                    training=False,
                    task="encoder"
                )
                
                # Pool features: [batch, latents, hidden_dim] -> [batch, hidden_dim]
                features = latents.mean(dim=1)  # Mean pool over latents dimension
                
                # Get binary labels for this class
                binary_labels = labels[:, class_idx].numpy().astype(int)
                
                train_features_list.append(features.cpu().numpy())
                train_labels_list.append(binary_labels)
        
        # Concatenate training data
        train_features = np.concatenate(train_features_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        
        # Check if we have both positive and negative samples
        if len(np.unique(train_labels)) < 2:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_positive_train': int(train_labels.sum()),
                'num_positive_val': 0,
                'skipped': True
            }
        
        # Normalize training features and fit KNN
        scaler = StandardScaler()
        train_features_norm = scaler.fit_transform(train_features)
        knn.fit(train_features_norm, train_labels)
        
        # Now evaluate on validation data online
        all_predictions = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader,desc="knn - validation phase"):
                # Unpack batch
                image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch
                
                # Move to device
                image = image.to(pl_module.device)
                attention_mask = attention_mask.to(pl_module.device)
                mae_tokens = mae_tokens.to(pl_module.device)
                mae_tokens_mask = mae_tokens_mask.to(pl_module.device)
                
                # Extract features
                latents = pl_module.forward(
                    image,
                    attention_mask,
                    mae_tokens,
                    mae_tokens_mask,
                    training=False,
                    task="encoder"
                )
                
                # Pool features: [batch, latents, hidden_dim] -> [batch, hidden_dim]
                features = latents.mean(dim=1)  # Mean pool over latents dimension
                
                # Get binary labels for this class
                binary_labels = labels[:, class_idx].numpy().astype(int)
                
                # Normalize features and predict
                features_norm = scaler.transform(features.cpu().numpy())
                predictions = knn.predict(features_norm)
                
                all_predictions.append(predictions)
                all_val_labels.append(binary_labels)
        
        # Concatenate validation results
        val_predictions = np.concatenate(all_predictions, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)
        
        # Check if validation has both classes
        if len(np.unique(val_labels)) < 2:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_positive_train': int(train_labels.sum()),
                'num_positive_val': int(val_labels.sum()),
                'skipped': True
            }
        
        # Convert to torch tensors for torchmetrics
        val_predictions_torch = torch.tensor(val_predictions, dtype=torch.int)
        val_labels_torch = torch.tensor(val_labels, dtype=torch.int)
        
        # Calculate metrics using torchmetrics
        accuracy_metric = torchmetrics.Accuracy(task='binary')
        precision_metric = torchmetrics.Precision(task='binary', zero_division=0)
        recall_metric = torchmetrics.Recall(task='binary', zero_division=0)
        f1_metric = torchmetrics.F1Score(task='binary', zero_division=0)
        
        accuracy = accuracy_metric(val_predictions_torch, val_labels_torch)
        precision = precision_metric(val_predictions_torch, val_labels_torch)
        recall = recall_metric(val_predictions_torch, val_labels_torch)
        f1 = f1_metric(val_predictions_torch, val_labels_torch)
        
        return {
            'accuracy': accuracy.item() * 100,
            'precision': precision.item() * 100,
            'recall': recall.item() * 100,
            'f1': f1.item() * 100,
            'num_positive_train': int(train_labels.sum()),
            'num_positive_val': int(val_labels.sum()),
            'skipped': False
        }
    
    def _log_results(self, overall_metrics: Dict, epoch: int):
        """Log KNN evaluation results."""
        
        wandb.log({
            "knn/mean_accuracy": overall_metrics.get('mean_accuracy', 0.0)
        }, step=epoch)