from torchvision.transforms import v2
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import torch

def normalize(img: torch.tensor):
    tmp_img=img.clone()
    # Normalize image 

    tmp_img = tmp_img - tmp_img.min() 
    tmp_img = tmp_img / tmp_img.max() 
    tmp_img = (tmp_img * 255).to(torch.uint8) 
    return tmp_img

def to_format_hwc(img):
    #chw -> hwc
    return rearrange(img,"a b c -> b c a")

def to_format_chw(img):
    #hwc -> chw
    return rearrange(img,"b c a -> a b c")

def random_between(a, b):
    """
    Return a random floating-point number x in the range [a, b),
    generated using PyTorch (as a Python float).
    """
    return ((b - a) * torch.rand(()) + a).item()

def normalize_image(img, means, stds):
    return (img - means) / stds


def random_rotate_flip(image: torch.Tensor):
    """
    Apply random rotation (0°, 90°, 180°, 270°) and random horizontal/vertical flip
    to a satellite image of shape [C=12, H=120, W=120].

    Args:
        image (torch.Tensor): Tensor of shape [12, H, W]

    Returns:
        torch.Tensor: Transformed image with same shape
    """
    assert image.ndim == 3 and image.shape[0] == 12, "Expected shape [12, H, W]"

    # Random rotation: 0, 90, 180, 270 degrees (implemented as number of 90° rotations)
    k = torch.randint(0, 4, (1,)).item()
    image = torch.rot90(image, k=k, dims=[1, 2])

    # Random horizontal flip
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[2])  # Flip width (left-right)

    # Random vertical flip
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[1])  # Flip height (top-bottom)

    return image

def rotate_image_torch(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return image.rot90(1, [-2, -1])
    elif angle == 180:
        return image.rot90(2, [-2, -1])
    elif angle == 270:
        return image.rot90(3, [-2, -1])
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270 degrees.")
    
def flip_image_torch(image, horizontal=False, vertical=False):
    if horizontal:
        image = image.flip(-1)
    if vertical:
        image = image.flip(-2)
    return image

def change_size_get_only_coordinates(orig_size,new_size,center=False):
    

    if center:
        new_center_x=60
        new_center_y=60
    else:
            
        new_center_x=int(random_between(int(new_size/2.0),int(orig_size-new_size/2.0)+1))
        new_center_y=int(random_between(int(new_size/2.0),int(orig_size-new_size/2.0)+1))
    
    
    x_min=int(new_center_x-new_size/2.0)
    x_max=int(new_center_x+new_size/2.0)
    y_min=int(new_center_y-new_size/2.0)
    y_max=int(new_center_y+new_size/2.0)
    return (x_min,x_max,y_min,y_max)

def change_size(img,mask,coordinates):
    orig_size=img.shape[1]

    x_min,x_max,y_min,y_max=coordinates
    
    new_img=img[:,x_min:x_max,y_min:y_max]
    padded_img=torch.zeros(img.shape)
    padding_size=int((orig_size-(x_max-x_min))/2.0)
    
    padded_img[:,padding_size:-padding_size,padding_size:-padding_size]=new_img
    mask[:,:,:]=1.0
    mask[:,padding_size:-padding_size,padding_size:-padding_size]=0.0
    
    return padded_img,mask

def random_value_from_range(min_value, max_value, step):
        """
        Returns a random value chosen uniformly from the set:
        {min_value, min_value+step, ..., max_value}
        Assumes that (max_value - min_value) is exactly divisible by step.
        
        Parameters:
            min_value (int or float): The minimum value.
            max_value (int or float): The maximum value.
            step (int or float): The step increment.
        
        Returns:
            A randomly chosen value from the defined range.
        """
        # Calculate the number of steps between min_value and max_value.
        # For instance, if min_value=0, max_value=10, and step=5, then n_steps = 2.
        n_steps = (max_value - min_value) // step
        
        # There are n_steps + 1 possible values.
        n_values = n_steps + 1
        
        # Choose a random index in the range [0, n_values-1] uniformly.
        rand_index = torch.randint(low=0, high=int(n_values), size=(1,)).item()
        
        # Compute the random value using the index.
        random_value = min_value + rand_index * step
        
        return random_value


def change_resolution(img,mask,target_size,keep_res=True):   
    orig_size=img.shape[1]
    img=v2.Resize(size=target_size)(img)
    img=v2.Resize(size=orig_size)(img)
    return img,mask

def remove_bands(img, mask, bands_ids, remove=False):

    img[bands_ids] = 0.0
    mask[bands_ids] = 1.0
    

    return img, mask



def apply_dico_transformations(img,dico,keep_shape=True):

    img=to_format_hwc(img)

    for key in dico:
        if key=="size":
            img=change_size(img,dico[key]["target_size"])
        if key=="resolution":
            img=change_resolution(img,dico[key]["target_resolution"])
        if key=="bands":
            img=remove_bands(img,list(dico[key]["bands_to_remove"]),replace=keep_shape)

    img=to_format_chw(img)
    
    return img
        
def create_dico_transfo(path,img_id,sizes=[],resolutions=[],channels=[]):
    """create a JSON 
    """
    pass




import torch
import numpy as np
import random
from typing import Tuple, Union

def augment_image_and_label(
    image: Union[torch.Tensor, np.ndarray], 
    label: Union[torch.Tensor, np.ndarray],
    rotation_angle: int = None,
    flip_horizontal: bool = None,
    flip_vertical: bool = None
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Apply the same rotation and flip transformations to both image and label.
    
    Args:
        image: Image tensor of shape [5, 512, 512] or numpy array
        label: Label tensor of shape [1, 512, 512] or numpy array  
        rotation_angle: Rotation angle in degrees [0, 90, 180, 270]. If None, randomly chosen.
        flip_horizontal: Whether to flip horizontally. If None, randomly chosen.
        flip_vertical: Whether to flip vertically. If None, randomly chosen.
        
    Returns:
        Tuple of (transformed_image, transformed_label)
    """
    
    # Convert to torch tensors if numpy arrays
    is_numpy_image = isinstance(image, np.ndarray)
    is_numpy_label = isinstance(label, np.ndarray)
    
    if is_numpy_image:
        image = torch.from_numpy(image)
    if is_numpy_label:
        label = torch.from_numpy(label)
    
    # Randomly choose transformations if not specified
    if rotation_angle is None:
        rotation_angle = random.choice([0, 90, 180, 270])
    if flip_horizontal is None:
        flip_horizontal = random.choice([True, False])
    if flip_vertical is None:
        flip_vertical = random.choice([True, False])
    
    # Apply rotation
    if rotation_angle == 90:
        # Rotate 90 degrees clockwise: transpose then flip horizontally
        image = torch.transpose(image, -2, -1).flip(-1)
        label = torch.transpose(label, -2, -1).flip(-1)
    elif rotation_angle == 180:
        # Rotate 180 degrees: flip both dimensions
        image = image.flip(-2).flip(-1)
        label = label.flip(-2).flip(-1)
    elif rotation_angle == 270:
        # Rotate 270 degrees clockwise: transpose then flip vertically
        image = torch.transpose(image, -2, -1).flip(-2)
        label = torch.transpose(label, -2, -1).flip(-2)
    # rotation_angle == 0: no rotation needed
    
    # Apply horizontal flip
    if flip_horizontal:
        image = image.flip(-1)  # flip along width dimension
        label = label.flip(-1)
    
    # Apply vertical flip
    if flip_vertical:
        image = image.flip(-2)  # flip along height dimension
        label = label.flip(-2)
    
    # Convert back to numpy if input was numpy
    if is_numpy_image:
        image = image.numpy()
    if is_numpy_label:
        label = label.numpy()
    
    image = torch.from_numpy(image) if is_numpy_image else image
    label = torch.from_numpy(label) if is_numpy_label else label
    return image, label


def random_augment_image_and_label(
    image: Union[torch.Tensor, np.ndarray], 
    label: Union[torch.Tensor, np.ndarray]
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Apply random rotation and flip transformations to both image and label.
    
    Args:
        image: Image tensor of shape [5, 512, 512] or numpy array
        label: Label tensor of shape [1, 512, 512] or numpy array
        
    Returns:
        Tuple of (transformed_image, transformed_label)
    """
    return augment_image_and_label(image, label)

