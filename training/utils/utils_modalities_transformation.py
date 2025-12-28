import torch
from training.utils.datasets.utils_dataset import read_yaml,save_yaml
from training.utils.datasets.image_utils import *
from training.utils.datasets.files_utils import*
from math import pi
import einops 
import numpy as np
import random


def apply_spatial_transforms(img: torch.Tensor) -> torch.Tensor:
    """
    Apply a random 90-degree rotation and random flips to an image.

    Args:
        img (torch.Tensor): Input image tensor of shape [C, H, W]

    Returns:
        torch.Tensor: Transformed image tensor of shape [C, H, W]
    """
    if img.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor [C, H, W], got {img.ndim}-D.")

    img_out = img.clone()

    # --- Random 90° rotation ---
    k = random.randint(0, 3)  # 0, 1, 2, or 3 quarter-turns
    if k > 0:
        # dims (1,2) correspond to H and W
        img_out = torch.rot90(img_out, k=k, dims=(1, 2))

    # --- Random horizontal flip ---
    if random.random() > 0.5:
        img_out = img_out.flip(dims=(2,))  # flip width axis

    # --- Random vertical flip ---
    if random.random() > 0.5:
        img_out = img_out.flip(dims=(1,))  # flip height axis

    return img_out




class modalities_transformations_config:

    def __init__(self,configs_dataset,model,path_imgs_config="./data/Tiny_BigEarthNet/",bands_infos="./data/bands_info/bands.yaml",name_config="",force_modality=None):

        self.force_modality=force_modality
        self.model=model
        self.name_config=name_config
        self.configs_dataset=read_yaml(configs_dataset)
        self.groups=self.configs_dataset["groups"]
        self.bands_infos=read_yaml(bands_infos)["bands_sen2_info"]


        self.path=path_imgs_config+"transformations"
        ensure_folder_exists(self.path)
        self.encoded_fourier_cc=dict()
        self.encoded_fourier_wavength=dict()

        self.gaussian_means=[]
        self.gaussian_stds=[]


        self.dico_group_channels=dict()
        if self.path[-1]=="\\":
            self.path=path_imgs_config[:-1]

    def get_band_identifier(self,channel_idx):
        for band_key in self.bands_infos:
            band=self.bands_infos[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None
    




            

    def get_band_infos(self,band_identifier):
        return self.bands_infos[band_identifier]

    
    def get_group(self,channel_idx):
        channel_name=self.get_band_identifier(channel_idx)
        for group in self.groups:
            if channel_name in self.groups[group]:
                return group

    def get_opposite_bands(self,tmp_group):
        group_bands=self.configs_dataset["groups"][tmp_group]
        res_bands=[]

        for band in list(self.bands_infos.keys()):
            if not band in group_bands:
                res_bands.append(band)
        return res_bands
            
            
    def get_channels_from_froup(self,group):
        if group in self.dico_group_channels:
            return self.dico_group_channels[group]
        group_bands=self.configs_dataset["groups"][group]
        res_idxs=[]
        for band in group_bands:
            res_idxs.append(self.get_band_infos(band)["idx"])
        
        self.dico_group_channels[group]=res_idxs
        return res_idxs

    def get_opposite_channels_from_froup(self,group):
        groupe_name=f"opposite_{group}"
        if groupe_name in self.dico_group_channels:
            return self.dico_group_channels[groupe_name]
        
        group_bands=self.get_opposite_bands(group)
        res_idxs=[]
        for band in group_bands:
            res_idxs.append(self.get_band_infos(band)["idx"])
        
        self.dico_group_channels[group]=res_idxs
        return res_idxs
            
    def get_random_attribute_id(self,L_attributes):
        if type(L_attributes)==list:
            rand_idx=int(torch.rand(())*len(L_attributes))
            return rand_idx
        if type(L_attributes)==dict:
            rand_idx=int(torch.rand(())*len(L_attributes.keys()))
            return list(L_attributes.keys())[rand_idx]
            
    def create_transform_image_dico(self,img_idx,mode,modality_folder=None):
        """create a transform yaml for each image
        """
        target_dico=dict()
        modalities=self.configs_dataset[mode]

        if modality_folder==None:
            modality_folder=mode
        
        modality_index=self.get_random_attribute_id(modalities)
        selected_modality=self.configs_dataset[mode][modality_index]
   
        for transfo_key in selected_modality.keys():
            if selected_modality[transfo_key]==None:
                continue
            
            transfo_val=selected_modality[transfo_key]

            if (type(transfo_val)==float and transfo_val==1.0) or transfo_val=="None":
                continue
            
            

            if transfo_key=="size":

                orig_img_size=self.configs_dataset["metadata"]["img_size"]
                new_size=None
                if type(transfo_val)==dict:
                    #infinite modalities
                    min_value=transfo_val["min"]
                    max_value=transfo_val["max"]
                    step=transfo_val["step"]

                    new_size=random_value_from_range(min_value, max_value, step)
                    new_size=int(orig_img_size*float(new_size))
                else:
                    new_size=int(orig_img_size*float(transfo_val))
                transfo_val=change_size_get_only_coordinates(orig_img_size,new_size)
            
            if transfo_key=="resolution":
                if type(transfo_val)==dict:
                    #infinite modalities
                    min_value=transfo_val["min"]
                    max_value=transfo_val["max"]
                    step=transfo_val["step"]

                    transfo_val=random_value_from_range(min_value, max_value, step)

            if transfo_key=="remove" or transfo_key=="keep":
                if transfo_val=="random":
                    transfo_val=self.get_random_attribute_id(self.groups)
                
            
            target_dico[transfo_key]=transfo_val

        folder_path=save_path=f"{self.path}/{modality_folder}/"

        if self.name_config!="":
            folder_path=save_path=f"{self.path}/{self.name_config}"
            ensure_folder_exists(folder_path)
            folder_path=save_path=f"{self.path}/{self.name_config}/{modality_folder}/"
            ensure_folder_exists(folder_path)
        else:
            ensure_folder_exists(folder_path)



        save_path=f"{self.path}/{modality_folder}/{img_idx}_transfos_{mode}.yaml"
        
        if self.name_config!="":
            save_path=f"{self.path}/{self.name_config}/{modality_folder}/{img_idx}_transfos_{mode}.yaml"

        save_yaml(save_path,target_dico)

    def apply_transformations(self,img,mask,idx,mode="train",modality_mode=None,f_s=None,f_r=None):
        """
        apply transformations specified in {self.path}/{idx}_transfos.yaml file.
        This is the function you should call in the get_item
        """
        return img,mask,1.0
     
        
      

        if modality_mode==None:
            modality_mode=mode

        img=apply_spatial_transforms(img)


        transfos=self.force_modality
        
        
        
        if self.force_modality==None:
            file_path=f"{self.path}/{mode}/{idx}_transfos_{modality_mode}.yaml"
            if self.name_config!="":
                file_path=f"{self.path}/{self.name_config}/{mode}/{idx}_transfos_{modality_mode}.yaml"

            transfos=read_yaml(file_path)
        else:
          
            file_path=f"./data/Tiny_BigEarthNet/custom_modalities/configs_dataset_"+self.force_modality+".yaml"
            transfos=read_yaml(file_path)
     
            

        resolution_change=1.0
     
        
        if "resolution" in transfos:
            resolution_change=float(transfos["resolution"])
            new_resolution=int(img.shape[1]*float(transfos["resolution"]))
            img,mask=change_resolution(img=img,mask=mask,target_size=new_resolution)
            
        

        

      
        if "size" in transfos:     
            if type(transfos["size"])==float:
                if transfos["size"]<1:
                    transfo_val=change_size_get_only_coordinates(120,int(120*transfos["size"]),center=True)
                    img,mask=change_size(img,mask,transfo_val)
            else:
                img,mask=change_size(img,mask,transfos["size"])


        if "remove"in transfos:
            img,mask=remove_bands(img,mask,self.get_channels_from_froup(transfos["remove"]))

        if "keep" in transfos:
            img,mask=remove_bands(img,mask,self.get_opposite_channels_from_froup(transfos["keep"]))

    
        

        
        return img,mask,resolution_change
    

    




        
     
      



        

    

    






    


    


        
        
    
    
