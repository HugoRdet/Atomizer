import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from .FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
from .lookup_positional import*

 


def del_file(path):
    if os.path.exists(path):
        os.remove(path)



def compute_channel_mean_std_dico(dico_idxs,ds):
    """
    Computes the mean and std for each of the 12 channels in a large list of tensors.
    Each tensor in the list has shape (12, 120, 120).

    Args:
        tensor_list (list): A list of PyTorch tensors, each with shape (12, 120, 120).

    Returns:
        A tensor of shape (12, 2), where [:, 0] contains the means and [:, 1] the std.
    """

    # We assume each tensor is (12, 120, 120)
    # so each channel has 120 * 120 elements per tensor.
    # We'll accumulate sums and sums of squares in double precision to reduce numerical issues.
    
    num_channels = 14
    sums = torch.zeros(num_channels, dtype=torch.float64)
    sums_of_squares = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0  # total number of pixels across all tensors for each channel

    for key in dico_idxs:
        
        L_idxs=dico_idxs[key]
        
        
        for idx in L_idxs:
            
            tensor,_=ds[idx]
            # Ensure the tensor is on CPU and in float64 for stable accumulations
            t = tensor.to(dtype=torch.float64)
    
            # Sum over spatial dimensions (dim=1,2) => shape [12]
            channel_sums = t.sum(dim=(1, 2))
            # Sum of squares over spatial dimensions
            channel_sums_of_squares = (t * t).sum(dim=(1, 2))
    
            sums += channel_sums
            sums_of_squares += channel_sums_of_squares
            # Each tensor contributes 120 * 120 pixels per channel
            total_pixels += t.shape[1] * t.shape[2]

    # Compute mean per channel
    mean = sums / total_pixels

    # Compute variance = E[X^2] - (E[X])^2
    # E[X^2] = sums_of_squares / total_pixels
    var = (sums_of_squares / total_pixels) - mean**2

    # To be safe against negative rounding errors, clamp var to 0
    var = var.clamp_min(0.0)
    std = var.sqrt()

    # Stack into (12, 2): first column is mean, second is std
    stats = torch.stack((mean, std), dim=-1).float()

    


    return stats






def compute_channel_mean_std(dico_idxs,ds):
    if dico_idxs!=None:
        return compute_channel_mean_std_dico(dico_idxs,ds)
    
    num_channels = 14
    sums = torch.zeros(num_channels, dtype=torch.float64)
    sums_of_squares = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0  # total number of pixels across all tensors for each channel

    for idx in range(len(ds)):
        
        tensor,_=ds[idx]
        # Ensure the tensor is on CPU and in float64 for stable accumulations
        t = tensor.to(dtype=torch.float64)

        # Sum over spatial dimensions (dim=1,2) => shape [12]
        channel_sums = t.sum(dim=(1, 2))
        # Sum of squares over spatial dimensions
        channel_sums_of_squares = (t * t).sum(dim=(1, 2))

        sums += channel_sums
        sums_of_squares += channel_sums_of_squares
        # Each tensor contributes 120 * 120 pixels per channel
        total_pixels += t.shape[1] * t.shape[2]

    # Compute mean per channel
    mean = sums / total_pixels

    # Compute variance = E[X^2] - (E[X])^2
    # E[X^2] = sums_of_squares / total_pixels
    var = (sums_of_squares / total_pixels) - mean**2

    # To be safe against negative rounding errors, clamp var to 0
    var = var.clamp_min(0.0)
    std = var.sqrt()

    # Stack into (12, 2): first column is mean, second is std
    stats = torch.stack((mean, std), dim=-1).float()
    torch.save(stats, "./data/stats.pt")
    return stats


def create_dataset(dico_idxs, ds,df, name="tiny", mode="train",trans_config=None,trans_tokens=None, stats=None,max_len=-1):
    if dico_idxs!=None:
        return create_dataset_dico(dico_idxs, ds, name, mode,trans_config,trans_tokens=trans_tokens, stats=stats)
    
    # 1) Clean up any existing file
    

    h5_path = f'./data/Tiny_BigEarthNet/{name}_{mode}.h5'
    del_file(h5_path)
    db = h5py.File(h5_path, 'w')
    
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
        if os.path.exists("data/normalisation/stats.pt"):
            stats=torch.load("data/normalisation/stats.pt",weights_only=True)
            print("ok!")
        else:
            stats = compute_channel_mean_std( dico_idxs,ds)

    ids=set()
    dico_stats=dict()


        

    # Make sure stats is a torch float tensor
    stats = stats.float()
    # Separate mean/std for easy broadcasting
    means = stats[:, 0].view(-1, 1, 1)  # shape: [12, 1, 1]
    stds = stats[:, 1].view(-1, 1, 1)   # shape: [12, 1, 1]

    # 3) Create a new HDF5 file
    cpt_train = 0
    

    if max_len!=-1:
        max_len=int(len(ds)*max_len)    



    for elem_id in tqdm(range(len(ds))):
        
        if cpt_train>max_len and max_len!=-1:
            print("oui Milgram",cpt_train,"   ",max_len)
            print(cpt_train)
            break

        

        split=get_split(df,elem_id)

        if split!=mode:
            continue
        
        img, label = ds[elem_id]  # shape (12, 120, 120)


        tmp_id=get_one_hot_indices(label)
        for tmp_id_elem in tmp_id:
            if not tmp_id_elem in dico_stats:
                dico_stats[tmp_id_elem]=1
            else:
                dico_stats[tmp_id_elem]+=1
        

        if trans_config!=None:
            trans_config.create_transform_image_dico(int(elem_id),mode="train",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="test",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="validation",modality_folder=mode)

      

    

        # Convert to float (if needed) before normalization
        img = img.float()

        # Apply per-channel normalization
        # normalized_value = (value - mean[channel]) / std[channel]
        img = (img - means) / stds



        # Convert back to numpy to store in HDF5
        db.create_dataset(f'image_{cpt_train}', data=img.numpy().astype(np.float16))
        db.create_dataset(f'label_{cpt_train}', data=label.numpy().astype(np.float16))
        db.create_dataset(f'id_{cpt_train}', data=int(elem_id))


        cpt_train += 1


    db.close()

    return stats


def get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode,modality_mode):
    tmp_img=img[2:,:,:]


    tmp_mask=torch.ones(tmp_img.shape)
    tmp_img,tmp_mask=trans_config.apply_transformations(tmp_img,tmp_mask,int(elem_id),mode=mode,modality_mode=modality_mode)
    tmp_img,tmp_mask=trans_tokens.process_data(tmp_img.unsqueeze(0),tmp_mask.unsqueeze(0))
    tmp_img=tmp_img.squeeze(0)
    tmp_mask=tmp_mask.squeeze(0)

    cond=tmp_mask==1
    image=tmp_img[cond]
    return int(image.shape[0])

def create_dataset_dico(dico_idxs, ds, name="tiny", mode="train",trans_config=None,trans_tokens=None, stats=None):
    """
    Creates an HDF5 dataset using the given sample indices (dico_idxs) from ds.
    If stats (per-channel mean/std) is None, computes it on-the-fly in a streaming fashion.
    Then applies normalization to each image: (image - mean) / std

    Args:
        dico_idxs (dict): Mapping from some key to a list of sample indices
        ds: A dataset that supports ds[idx] -> (image, label)
            where image is shape (12, 120, 120).
        name (str): HDF5 file prefix
        mode (str): e.g. "train" or "test"
        stats (torch.Tensor or None): shape (12,2) with [:,0] as mean, [:,1] as std

    Returns:
        stats (torch.Tensor): The per-channel mean/std used for normalization
    """

    # 1) Clean up any existing file
    h5_path = f'./data/Tiny_BigEarthNet/{name}_{mode}.h5'
    del_file(h5_path)
    ids=set()
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
   
        stats = compute_channel_mean_std( dico_idxs,ds)
        

    # Make sure stats is a torch float tensor
    stats = stats.float()
    # Separate mean/std for easy broadcasting
    means = stats[:, 0].view(-1, 1, 1)  # shape: [12, 1, 1]
    stds = stats[:, 1].view(-1, 1, 1)   # shape: [12, 1, 1]

    # 3) Create a new HDF5 file
    db = h5py.File(h5_path, 'w')

    cpt_train = 0

    # 4) Iterate through your dictionary of IDs, fetch images, and store them
    for idx in tqdm(dico_idxs.keys()):
        l_samples = dico_idxs[idx]

        for elem_id in l_samples:
            img, label = ds[elem_id]  # shape (12, 120, 120)
        

            trans_config.create_transform_image_dico(int(elem_id),mode="train",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="test",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="validation",modality_folder=mode)

            # Convert to float (if needed) before normalization
            img = img.float()

            # Apply per-channel normalization
            # normalized_value = (value - mean[channel]) / std[channel]
            img = (img - means) / stds

            #shape_train=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="train")
            #shape_validation=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="validation")
            #shape_test=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="test")

            
            

            # Convert back to numpy to store in HDF5
            db.create_dataset(f'image_{cpt_train}', data=img.numpy().astype(np.float16))
            db.create_dataset(f'label_{cpt_train}', data=label.numpy().astype(int))
            db.create_dataset(f'id_{cpt_train}', data=int(elem_id))
            db.create_dataset(f'shape_train_{cpt_train}', data=int(0))
            db.create_dataset(f'shape_test_{cpt_train}', data=int(0))
            db.create_dataset(f'shape_validation_{cpt_train}', data=int(0))

            

            cpt_train += 1

    db.close()
    return stats