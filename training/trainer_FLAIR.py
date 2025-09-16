from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.perceiverIO import *
from training.ScaleMae import*
from training.ResNet import *
from collections import defaultdict
from training import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import matplotlib.pyplot as plt
from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import torchmetrics
import warnings
import wandb
from transformers import get_cosine_schedule_with_warmup
import seaborn as sns
from pytorch_optimizer import Lamb

import torch_optimizer as optim
class Model_FLAIR(pl.LightningModule):
    def __init__(self, config, wand, name, transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        self.labels_idx = load_json_to_dict("./data/Encoded-BigEarthNet/labels.json")
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table = False
        self.comment_log = ""
        
        # Metrics
        self.metric_IoU_train = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro")
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")
        self.metric_IoU_test = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average=None)
        
        # Model
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config, transform=self.transform)
        else:
            self.encoder =PerceiverIO(
            depth=8,
            dim=5,
            queries_dim=256,
            logits_dim = None,
            num_latents = 512,
            latent_dim = 512,
            cross_heads = 1,
            latent_heads = 8,
            cross_dim_head = 64,
            latent_dim_head = 64,
            weight_tie_layers = False,
            decoder_ff = False,
            seq_dropout_prob = 0.)
    

        self.loss = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, image, attention_mask, mae_tokens, mae_tokens_mask, training=False, task="reconstruction"):
        return self.encoder(image, attention_mask, mae_tokens, mae_tokens_mask, training=training, task=task)



    def analyze_logits_overfitting(self, logits, labels, mode="train", log_metrics=True):
        """
        Analyze logits to detect overfitting patterns and query collapse in Perceiver-like models
        
        Args:
            logits: torch.Tensor of shape [batch_size, num_samples, num_classes] - raw model outputs
            labels: torch.Tensor of shape [batch_size, num_samples] - ground truth class indices
            mode: str - prefix for logging (e.g., "train")
            log_metrics: bool - whether to log metrics to wandb
        
        Returns:
            dict: Dictionary containing overfitting metrics
        """
        with torch.no_grad():
            batch_size, num_samples, num_classes = logits.shape
            
            # Reshape for easier processing: [batch_size * num_samples, num_classes]
            logits_flat = logits.view(-1, num_classes)  # [batch_size * num_samples, num_classes]
            labels_flat = labels.view(-1).long()  # [batch_size * num_samples]
            
            # Convert to probabilities
            probs = torch.softmax(logits_flat, dim=-1)  # [batch_size * num_samples, num_classes]
            
            # === QUERY COLLAPSE DETECTION ===
            
            # 1. Prediction diversity: How many different classes are being predicted?
            predicted_classes = torch.argmax(probs, dim=-1)  # [batch_size * num_samples]
            unique_predictions = torch.unique(predicted_classes)
            num_unique_predictions = len(unique_predictions)
            prediction_diversity = num_unique_predictions / num_classes  # Ratio of classes being predicted
            
            # 2. Query similarity: How similar are the outputs across different queries?
            # Reshape back to [batch_size, num_samples, num_classes] for query analysis
            probs_queries = probs.view(batch_size, num_samples, num_classes)
            
            # Calculate pairwise cosine similarity between queries within each batch
            query_similarities = []
            for b in range(batch_size):
                queries_b = probs_queries[b]  # [num_samples, num_classes]
                # Normalize for cosine similarity
                queries_norm = F.normalize(queries_b, p=2, dim=-1)
                # Compute pairwise cosine similarity
                sim_matrix = torch.mm(queries_norm, queries_norm.t())  # [num_samples, num_samples]
                # Get upper triangular part (excluding diagonal)
                mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
                if mask.sum() > 0:
                    avg_similarity = sim_matrix[mask].mean().item()
                    query_similarities.append(avg_similarity)
            
            avg_query_similarity = np.mean(query_similarities) if query_similarities else 0.0
            
            # 3. Dominant class analysis: Is one class being predicted way more than others?
            class_counts = torch.bincount(predicted_classes, minlength=num_classes)
            class_distribution = class_counts.float() / len(predicted_classes)
            
            # Entropy of class distribution (low = dominated by few classes)
            class_entropy = -torch.sum(class_distribution * torch.log(class_distribution + 1e-8)).item()
            max_class_entropy = np.log(num_classes)  # Maximum possible entropy
            normalized_class_entropy = class_entropy / max_class_entropy
            
            # Most frequent class percentage
            dominant_class_ratio = class_distribution.max().item()
            
            # 4. Within-sample consistency: Do queries within same sample agree?
            predicted_classes_queries = predicted_classes.view(batch_size, num_samples)
            within_sample_agreement = []
            for b in range(batch_size):
                sample_preds = predicted_classes_queries[b]
                # Calculate how often queries agree within this sample
                mode_class = torch.mode(sample_preds)[0]
                agreement_ratio = (sample_preds == mode_class).float().mean().item()
                within_sample_agreement.append(agreement_ratio)
            
            avg_within_sample_agreement = np.mean(within_sample_agreement)
            
            # === ORIGINAL OVERFITTING METRICS ===
            
            # 1. Confidence Gap
            sorted_logits, _ = torch.sort(logits_flat, dim=-1, descending=True)
            confidence_gaps = sorted_logits[:, 0] - sorted_logits[:, 1]
            avg_confidence_gap = confidence_gaps.mean().item()
            std_confidence_gap = confidence_gaps.std().item()
            
            # 2. Probability statistics
            max_probs, _ = torch.max(probs, dim=-1)
            min_probs, _ = torch.min(probs, dim=-1)
            mean_probs = torch.mean(probs, dim=-1)
            
            avg_max_prob = max_probs.mean().item()
            avg_min_prob = min_probs.mean().item()
            avg_mean_prob = mean_probs.mean().item()
            
            # 3. Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_entropy = entropy.mean().item()
            
            # 4. Logit statistics
            avg_logit = logits_flat.mean().item()
            std_logit = logits_flat.std().item()
            max_logit = logits_flat.max().item()
            min_logit = logits_flat.min().item()
            
            # 5. Sharpness
            sharpness = (max_probs ** 2).mean().item()
            
            # 6. Calibration analysis
            correct_predictions = (predicted_classes == labels_flat)
            
            if correct_predictions.sum() > 0:
                confidence_on_correct = max_probs[correct_predictions].mean().item()
            else:
                confidence_on_correct = 0.0
                
            if (~correct_predictions).sum() > 0:
                confidence_on_incorrect = max_probs[~correct_predictions].mean().item()
            else:
                confidence_on_incorrect = 0.0
            
            calibration_gap = confidence_on_correct - confidence_on_incorrect
            
            # 7. True class analysis
            true_class_probs = probs[torch.arange(len(labels_flat)), labels_flat]
            avg_true_class_prob = true_class_probs.mean().item()
            
            true_class_logits = logits_flat[torch.arange(len(labels_flat)), labels_flat]
            avg_true_class_logit = true_class_logits.mean().item()
            
            # 8. Accuracy
            accuracy = correct_predictions.float().mean().item()
            
            # 9. Overconfidence
            max_logits = logits_flat.max(dim=-1)[0]
            logit_overconfidence = (max_logits - true_class_logits).mean().item()
            
            results = {
                # Query collapse metrics
                'prediction_diversity': prediction_diversity,
                'num_unique_predictions': num_unique_predictions,
                'avg_query_similarity': avg_query_similarity,
                'class_entropy_normalized': normalized_class_entropy,
                'dominant_class_ratio': dominant_class_ratio,
                'within_sample_agreement': avg_within_sample_agreement,
                
                # Original overfitting metrics
                'confidence_gap_mean': avg_confidence_gap,
                'confidence_gap_std': std_confidence_gap,
                'max_prob_mean': avg_max_prob,
                'min_prob_mean': avg_min_prob,  # NEW
                'mean_prob_mean': avg_mean_prob,  # NEW
                'entropy_mean': avg_entropy,
                'logit_mean': avg_logit,
                'logit_std': std_logit,
                'logit_max': max_logit,
                'logit_min': min_logit,
                'sharpness': sharpness,
                'confidence_on_correct': confidence_on_correct,
                'confidence_on_incorrect': confidence_on_incorrect,
                'calibration_gap': calibration_gap,
                'true_class_prob_mean': avg_true_class_prob,
                'true_class_logit_mean': avg_true_class_logit,
                'accuracy': accuracy,
                'logit_overconfidence': logit_overconfidence,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'num_classes': num_classes,
                'total_predictions': batch_size * num_samples
            }
            
            # Log metrics to wandb (only for training)
            if log_metrics and mode == "train":
                # Query collapse metrics
                self.log(f'{mode}_prediction_diversity', prediction_diversity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                self.log(f'{mode}_num_unique_predictions', num_unique_predictions, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_query_similarity', avg_query_similarity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                self.log(f'{mode}_class_entropy', normalized_class_entropy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_dominant_class_ratio', dominant_class_ratio, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_within_sample_agreement', avg_within_sample_agreement, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                
                # Original overfitting metrics
                self.log(f'{mode}_confidence_gap', avg_confidence_gap, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_max_prob', avg_max_prob, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_min_prob', avg_min_prob, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)  # NEW
                self.log(f'{mode}_mean_prob', avg_mean_prob, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)  # NEW
                self.log(f'{mode}_entropy', avg_entropy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_logit_std', std_logit, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_sharpness', sharpness, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_calibration_gap', calibration_gap, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_true_class_prob', avg_true_class_prob, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_logit_overconfidence', logit_overconfidence, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f'{mode}_prediction_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            
            return results


    def analyze_label_distribution(self, labels, logits, mode="train"):
        """Analyze the distribution of ground truth labels"""
        with torch.no_grad():
            batch_size, num_samples = labels.shape
            labels_flat = labels.view(-1).long()
            
            # Class distribution in this batch
            class_counts = torch.bincount(labels_flat, minlength=self.num_classes)
            class_distribution = class_counts.float() / len(labels_flat)
            
            # Most common class percentage
            dominant_class_pct = class_distribution.max().item()
            
            # How many classes appear in this batch
            classes_present = (class_counts > 0).sum().item()
            
            # Within-sample label consistency
            within_sample_consistency = []
            for b in range(batch_size):
                sample_labels = labels[b]  # [num_samples]
                unique_labels = torch.unique(sample_labels)
                # What fraction of queries in this sample have the same label?
                mode_label = torch.mode(sample_labels)[0]
                consistency = (sample_labels == mode_label).float().mean().item()
                within_sample_consistency.append(consistency)
            
            avg_within_sample_consistency = np.mean(within_sample_consistency)
            
            if mode == "train":
                self.log(f'{mode}_dominant_class_pct', dominant_class_pct, on_epoch=True, logger=True)
                self.log(f'{mode}_classes_present', classes_present, on_epoch=True, logger=True)
                self.log(f'{mode}_label_consistency', avg_within_sample_consistency, on_epoch=True, logger=True)
            
            return {
                'dominant_class_pct': dominant_class_pct,
                'classes_present': classes_present,
                'within_sample_consistency': avg_within_sample_consistency,
                'class_distribution': class_distribution
            }

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch

        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        #labels = mae_tokens[:,::5,4]
        labels = mae_tokens[:,:,4]

        
        
        labels_loss=rearrange(labels,"b p -> (b p)")
        y_hat_loss =rearrange(y_hat.clone() ,"b t c -> (b t) c")
        
        
        
        loss = self.loss(y_hat_loss, labels_loss.long())
        
        preds = torch.argmax(y_hat.clone(), dim=-1)
        self.metric_IoU_train.update(preds, labels)
        
        # Log the loss directly here instead of manually tracking
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss
    
    def on_fit_start(self):
        # Model setup
        return
        self.encoder.unfreeze_encoder()
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
    
    def on_train_epoch_start(self):
        return
        self.encoder.unfreeze_encoder()    
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
        
    def on_train_epoch_end(self):
        # Compute and log IoU
        train_iou = self.metric_IoU_train.compute()
        self.log("train_IoU", train_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_train.reset()
    
    def on_validation_epoch_start(self):
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch
        
        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        #labels = mae_tokens[:,::5,4]
        labels = mae_tokens[:,:,4]
        
        
        
        labels_loss=rearrange(labels,"b p -> (b p)")
        y_hat_loss =rearrange(y_hat.clone() ,"b t c -> (b t) c")
        
        
        
        loss = self.loss(y_hat_loss, labels_loss.long())
        
        preds = torch.argmax(y_hat.clone(), dim=-1)
        self.metric_IoU_val.update(preds, labels)
        
        # Log the loss directly here instead of manually tracking
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        return loss

    def on_validation_epoch_end(self):
      
        # Compute and log IoU
        val_iou = self.metric_IoU_val.compute()
        self.log("val_IoU", val_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_val.reset()
        
        # Reset dataset mode
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")

    def test_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch

        # Forward pass
        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        # Get labels and predictions - keep original format
        labels = mae_tokens[:, 4]
        preds = torch.argmax(y_hat.clone(), dim=-1)
        y_hat = y_hat.squeeze(-1)
        
        # CRITICAL: Clamp labels to valid range [0, num_classes-1]
        labels = torch.clamp(labels, 0, self.num_classes - 1).long()
        
        # Calculate loss
        loss = self.loss(y_hat, labels)
        
        # Update metrics
        self.metric_IoU_test.update(preds, labels)
        
        # Log loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_test_epoch_end(self):
        # Compute per-class IoU
        test_iou_per_class = self.metric_IoU_test.compute()
        
        # Log mean IoU
        mean_iou = test_iou_per_class.mean()
        self.log("test_IoU_mean", mean_iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Log per-class IoU if needed
        for i, iou in enumerate(test_iou_per_class):
            self.log(f"test_IoU_class_{i}", iou, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.metric_IoU_test.reset()
        
    def save_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))

    def debug_data_ranges(self, dataloader):
        """
        Debug method to check label ranges in your dataset.
        Call this before training to identify issues.
        """
        print("=== DEBUGGING DATA RANGES ===")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Check first 5 batches
                break
                
            image, attention_mask, mae_tokens, mae_tokens_mask, _ = batch
            labels = mae_tokens[:, :, 4]
            
            print(f"Batch {batch_idx}:")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Labels min: {labels.min().item()}")
            print(f"  Labels max: {labels.max().item()}")
            print(f"  Unique labels: {torch.unique(labels).cpu().tolist()}")
            print(f"  Expected range: [0, {self.num_classes-1}]")
            
            # Check for problematic values
            invalid_labels = (labels < 0) | (labels >= self.num_classes)
            if invalid_labels.any():
                print(f"  ⚠️  FOUND {invalid_labels.sum().item()} INVALID LABELS!")
                invalid_values = labels[invalid_labels].unique()
                print(f"  Invalid values: {invalid_values.cpu().tolist()}")
            else:
                print("  ✅ All labels are in valid range")
            print()
        
        print("=== END DEBUG ===")

    def configure_optimizers(self):
        base_lr = self.lr
        wd = self.weight_decay

        if self.config["optimizer"]["name"] == "ADAM":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=wd)
        else:
            import torch_optimizer as optim
            optimizer = optim.Lamb(self.parameters(), lr=base_lr, weight_decay=wd,
                                betas=(0.9, 0.999), eps=1e-6)

        # total optimizer steps for the entire fit (already accounts for grad accumulation & epochs)
        total_steps = int(self.trainer.estimated_stepping_batches)
        
        # pick a % warmup or your fixed value, but keep it <= total_steps
        warmup_steps = min(self.config["optimizer"]["warmup_steps"], max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # per-step schedule
                # no 'monitor' here
            },
        }
