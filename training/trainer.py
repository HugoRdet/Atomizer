from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
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
import pytorch_lightning as pl
#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model(pl.LightningModule):
    def __init__(self, config, wand, name,transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform=transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        self.labels_idx = load_json_to_dict("./data/Encoded-BigEarthNet/labels.json")
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table=False
        self.comment_log=""
        

        
        self.metric_train_AP_per_class = torchmetrics.classification.MultilabelAveragePrecision(self.num_classes, average=None, thresholds=None)
        self.metric_train_accuracy_per_class = torchmetrics.classification.MultilabelAccuracy(self.num_classes, threshold=0.5, average=None)

        self.metric_val_mod_val_AP_per_class = torchmetrics.classification.MultilabelAveragePrecision(self.num_classes, average=None, thresholds=None)
        self.metric_val_mod_val_accuracy_per_class = torchmetrics.classification.MultilabelAccuracy(self.num_classes, threshold=0.5, average=None)
        self.metric_val_mod_train_AP_per_class = torchmetrics.classification.MultilabelAveragePrecision(self.num_classes, average=None, thresholds=None)
        self.metric_val_mod_train_accuracy_per_class = torchmetrics.classification.MultilabelAccuracy(self.num_classes, threshold=0.5, average=None)

        self.metric_test_AP_per_class = torchmetrics.classification.MultilabelAveragePrecision(self.num_classes, average=None, thresholds=None)
        self.metric_test_accuracy_per_class = torchmetrics.classification.MultilabelAccuracy(self.num_classes, threshold=0.5, average=None)

        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        if config["encoder"] == "ViT":
            ViT_conf = config["ViT"]["config"]
            self.encoder = SimpleViT(
                image_size=config["ViT"]["image_size"],
                patch_size=config["ViT"]["patch_size"],
                num_classes=self.num_classes,
                dim=config["ViT"][ViT_conf]["dim"],
                depth=config["ViT"][ViT_conf]["depth"],
                heads=config["ViT"][ViT_conf]["heads"],
                mlp_dim=config["ViT"][ViT_conf]["mlp_dim"],
                channels=12,
                dim_head=config["ViT"][ViT_conf]["dim_head"]
            )
        if config["encoder"] == "ResNet50":
            self.encoder = ResNet50(config["trainer"]["num_classes"], channels=12)
        if config["encoder"] == "ResNet101":
            self.encoder = ResNet101(config["trainer"]["num_classes"], channels=12)
        if config["encoder"] == "ResNet152":
            self.encoder = ResNet152(config["trainer"]["num_classes"], channels=12)
        if config["encoder"] == "ResNetSmall":
            self.encoder = ResNetSmall(config["trainer"]["num_classes"], channels=12)
        if config["encoder"] == "ResNetSuperSmall":
            self.encoder = ResNetSuperSmall(config["trainer"]["num_classes"], channels=12)
        if config["encoder"] == "Perceiver":
            self.encoder = Perceiver(
                num_freq_bands=config["Perceiver"]["num_freq_bands"],
                depth=config["Perceiver"]["depth"],
                max_freq=config["Perceiver"]["max_freq"],
                input_channels=12,
                input_axis=2,
                num_latents=config["Perceiver"]["num_latents"],
                latent_dim=config["Perceiver"]["latent_dim"],
                cross_heads=config["Perceiver"]["cross_heads"],
                latent_heads=config["Perceiver"]["latent_heads"],
                cross_dim_head=config["Perceiver"]["cross_dim_head"],
                latent_dim_head=config["Perceiver"]["latent_dim_head"],
                num_classes=config["trainer"]["num_classes"],
                attn_dropout=config["Perceiver"]["attn_dropout"],
                ff_dropout=config["Perceiver"]["ff_dropout"],
                weight_tie_layers=config["Perceiver"]["weight_tie_layers"],
                fourier_encode_data=config["Perceiver"]["fourier_encode_data"],
                self_per_cross_attn=config["Perceiver"]["self_per_cross_attn"],
                final_classifier_head=config["Perceiver"]["final_classifier_head"]
            )

        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config,transform=self.transform)
            
        if config["encoder"] == "Atomiser_tradi":
            self.atos_masking=config["Atomiser"]["masking"]

            self.resolutions=torch.from_numpy(np.array([60,10,10,10,20,20,20,10,20,60,60,20]))
            
            

            self.encoder = Atomiser_tradi(
                config=self.config,
                transform=self.transform,
                depth=config["Atomiser"]["depth"],
                num_latents=config["Atomiser"]["num_latents"],
                latent_dim=config["Atomiser"]["latent_dim"],
                cross_heads=config["Atomiser"]["cross_heads"],
                latent_heads=config["Atomiser"]["latent_heads"],
                cross_dim_head=config["Atomiser"]["cross_dim_head"],
                latent_dim_head=config["Atomiser"]["latent_dim_head"],
                num_classes=config["trainer"]["num_classes"],
                attn_dropout=config["Atomiser"]["attn_dropout"],
                ff_dropout=config["Atomiser"]["ff_dropout"],
                weight_tie_layers=config["Atomiser"]["weight_tie_layers"],
                self_per_cross_attn=config["Atomiser"]["self_per_cross_attn"],
                final_classifier_head=config["Atomiser"]["final_classifier_head"],
                masking=config["Atomiser"]["masking"]
            )

        
        if config["encoder"] == "ScaleMAE":
            self.encoder=CustomScaleMAE()
            
        


        self.loss = nn.BCEWithLogitsLoss()
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, x,mask,resolution,size,training=True):
        if "Atomiser_tradi" in self.config["encoder"]:
            return self.encoder(x,mask,resolution,size,training=training)
        if "Atomiser" in self.config["encoder"]:
            return self.encoder(x,mask,resolution,size,training=training,task="classification")
        else:
            if "Perceiver" in self.config["encoder"]:
                tmp_resolutions=10/resolution#self.resolutions/resolution
                return self.encoder(x,res=tmp_resolutions,mask=mask)
            
            elif "ScaleMAE" in self.config["encoder"]:
                tmp_resolutions=10/resolution#self.resolutions/resolution
                return self.encoder(x,res=tmp_resolutions)
            
      
                
            
            return self.encoder(x)
        
    def on_fit_start(self):
        # if starting with MAE
        #self.encoder.unfreeze_encoder()
        #self.encoder.freeze_decoder()
        #self.encoder.unfreeze_classifier()
        pass
                
            
    def training_step(self, batch, batch_idx):
        img, mask, _,_, labels= batch
        resolution=1.0
        size=1.0

        #print(f"[DEBUG] inital img diff (sample 0 vs 1): {(img[0,:,0] - img[1,:,0]).abs().mean().item()}, img0 mean: {img[0,:,0].mean().item()}, img0 max: {img[0,:,0].max().item()}, img0 min: {img[0,:,0].min().item()}")
        #print(f"[DEBUG] inital img diff (sample 0 vs 2): {(img[0,:,0] - img[2,:,0]).abs().mean().item()}, img1 mean: {img[1,:,0].mean().item()}, img1 max: {img[1,:,0].max().item()}, img1 min: {img[1,:,0].min().item()}")
        #print(f"[DEBUG] inital img diff (sample 0 vs 3): {(img[0,:,0] - img[3,:,0]).abs().mean().item()}, img2 mean: {img[2,:,0].mean().item()}, img2 max: {img[2,:,0].max().item()}, img2 min: {img[2,:,0].min().item()}")

        y_hat = self.forward(img, mask, resolution,size, training=True)
        loss = self.loss(y_hat, labels.float())

        
        # Update metrics
        #with record_function("metrics computations"):
        y_hat = torch.sigmoid(y_hat)
        
        
        
    
        self.metric_train_accuracy_per_class.update(y_hat, labels.to(torch.int))
        self.metric_train_AP_per_class.update(y_hat, labels.to(torch.int))
        
        # Log losses
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, logger=False, sync_dist=False)
        #self.log("train_total_loss", total_loss.item(), on_step=False, on_epoch=True, logger=False, sync_dist=False)
        
        return loss
    
    
        
    def on_train_epoch_end(self):
        self.compute_metrics(mode="train", all_classes=False,table=self.table)
        
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("train_loss", float("inf"))
        train_ap = metrics.get("train_ap", float("-inf"))
        
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, logger=True)

        #self.log("log train_loss", torch.log(train_loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_ap", train_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        
        return {"train_loss": train_loss, "train_ap": train_ap}

    
    def on_validation_epoch_start(self):
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")
        
    #def on_after_backward(self):
    # Print gradient norm for key layers
    #    for name, param in self.named_parameters():
    #        if param.requires_grad:
    #            grad_norm = None if param.grad is None else param.grad.norm().item()
    #            print(f"{name}: grad_norm={grad_norm}")


        
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        img, mask, _,_, labels= batch
        resolution=1.0
        size=1.0
        
        

   
        y_hat = self.forward(img,mask,resolution,size,training=False)
    
      

        loss = self.loss(y_hat, labels.float())
        y_hat = torch.sigmoid(y_hat)

        

        if dataloader_idx==0:
            self.metric_val_mod_val_accuracy_per_class.update(y_hat, labels.to(torch.int))
            self.metric_val_mod_val_AP_per_class.update(y_hat, labels.to(torch.int))

            if not self.table:
                self.log("val_mod_val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=False)
        else:
            self.metric_val_mod_train_accuracy_per_class.update(y_hat, labels.to(torch.int))
            self.metric_val_mod_train_AP_per_class.update(y_hat, labels.to(torch.int))

            if not self.table:
                self.log("val_mod_train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=False)


        return loss        

    def on_validation_epoch_end(self):

        self.compute_metrics(mode="val_mod_val", all_classes=False,table=self.table)
        self.compute_metrics(mode="val_mod_train", all_classes=False,table=self.table)
        

        self.trainer.datamodule.val_dataset.reset_modality_mode()
        
        if self.table:
            return None
        
        
        
    


    
        
    def test_step(self, batch, batch_idx):
        img, mask, resolution,size, labels, _ = batch
        y_hat = self.forward(img, mask, resolution,size, training=False)
        
        # Update metrics
        self.metric_test_accuracy_per_class.update(y_hat, labels.to(torch.int))
        self.metric_test_AP_per_class.update(y_hat, labels.to(torch.int))
        
        # Visualize attention for the first 10 batches only
        #max_batch=256
        #if batch_idx < max_batch:
            # Pass the current global step and batch_idx for better tracking
            #current_step = self.trainer.global_step if hasattr(self.trainer, 'global_step') else batch_idx
            #self.encoder.visualize_attention(img, mask, resolution, step=current_step, batch_id=batch_idx,MAX_BB=max_batch)
        
        return y_hat

    def on_test_epoch_end(self):
        modality = self.trainer.datamodule.test_dataset.modality_mode
      
     
        self.compute_metrics(mode="test", table=True, all_classes=False, modality=modality)
        
        
    def compute_metrics(self, mode, table=False, all_classes=True, modality=None):
        
        
        if mode=="train":
            metric_accuracy=self.metric_train_accuracy_per_class
            metric_AP=self.metric_train_AP_per_class

        if mode=="val_mod_val":
            metric_accuracy=self.metric_val_mod_val_accuracy_per_class
            metric_AP=self.metric_val_mod_val_AP_per_class

        if mode=="val_mod_train":
            metric_accuracy=self.metric_val_mod_train_accuracy_per_class
            metric_AP=self.metric_val_mod_train_AP_per_class

        if mode=="test":
            metric_accuracy=self.metric_test_accuracy_per_class
            metric_AP=self.metric_test_AP_per_class
            
        per_class_acc = metric_accuracy.compute()*100
        overall_accuracy = per_class_acc.mean().item()
        ap = metric_AP.compute()*100
        mean_ap = ap.mean().item()

        metric_accuracy.reset()
        metric_AP.reset()

        if mode=="val_mod_val":
            self.log(self.comment_log+"val_mod_val_ap", mean_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.comment_log+"val_mod_val_accuracy", overall_accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if mode=="val_mod_train":
            self.log(self.comment_log+"val_mod_train_ap", mean_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.comment_log+"val_mod_train_accuracy", overall_accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if mode=="train":
            self.log(self.comment_log+"train_ap", mean_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.comment_log+"train_accuracy", overall_accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if mode=="test":
            self.log(self.comment_log+"test_ap", mean_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.comment_log+"test_accuracy", overall_accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
    
        metric_accuracy.reset()
        metric_AP.reset()
        
        table_data = []
        
        
        
        if self.wand and table:


            for idx in range(self.num_classes):
                
                class_name = self.labels_idx[str(int(idx))]
                table_data.append([class_name, per_class_acc[idx].item(), ap[idx].item()])
            table_data.append(["Average", overall_accuracy, mean_ap])
        
            wandb_table = wandb.Table(columns=["Class Name", "Accuracy (%)", "mAP (%)"], data=table_data)
            if modality!=None:
                wandb.log({f"{self.comment_log} Metrics Table ({mode}), modality: {modality} ": wandb_table})
            else:
                wandb.log({f"{self.comment_log} Metrics Table ({mode})": wandb_table})
                
        if modality !=None:
            self.log(f"{self.comment_log} {mode} ,modality: {modality} average accuracy", overall_accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{self.comment_log} {mode} ,modality: {modality} average AP", mean_ap, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
        if all_classes:
            for idx in range(self.num_classes):
                class_name = self.labels_idx[str(int(idx))]
                self.log(f"{mode}_{class_name}_AP", ap[idx].item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)


    
    
        
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
        

    def configure_optimizers(self):
        # Import LAMB optimizer
        from pytorch_optimizer import Lamb
        
        # LAMB optimizer optimized for MAE training
        # Consider increasing batch size when using LAMB
        optimizer = Lamb(
            self.parameters(), 
            lr=self.lr * 2.0,    # LAMB often works with higher learning rates
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),  
            eps=1e-6,            
            # For MAE, you might want to experiment with:
            # - Higher learning rates (2x-10x of Adam)
            # - Larger batch sizes (if memory allows)
        )

        accumulate_grad_batches = 64
        batches_per_epoch = self.trainer.estimated_stepping_batches/self.config["trainer"]["epochs"]
        steps_per_epoch = batches_per_epoch // accumulate_grad_batches

        total_steps = self.config["trainer"]["epochs"] * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'val_reconstruction_loss'
            }
        }
        
    