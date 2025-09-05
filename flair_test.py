import torch
from math import pi
import einops as einops
from training.utils.FLAIR_2 import*
import matplotlib.pyplot as plt
from training.perceiver import*
from training.utils import*
from training.losses import*
from training.VIT import*
from training.ResNet import*
from collections import defaultdict
from training import*


from pytorch_lightning import Trainer,seed_everything
seed_everything(42, workers=True)

#config_path = "./data/flair_2_dataset/flair-2-config.yml" # Change to yours
#config_path = "./data/flair_2_toy_dataset/flair-2-config.yml" # Change to yours
config_path = "./data/flair/flair-2-config.yml" # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
# Creation of the train, val and test dictionnaries with the data file paths
d_train, d_val, d_test = load_data(config)

images = d_train["PATH_IMG"]
labels = d_train["PATH_LABELS"]
sentinel_images = d_train["PATH_SP_DATA"]
sentinel_masks = d_train["PATH_SP_MASKS"] # Cloud masks
sentinel_products = d_train["PATH_SP_DATES"] # Needed to get the dates of the sentinel images
centroids = d_train["SP_COORDS"] # Position of the aerial image in the sentinel super area
aerial_mtds=d_train["MTD_AERIAL"]
print("images train: ",len(images))


stats=create_dataset_flair(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtds, name="o_regular", mode="train", stats=None,max_samples=50)


images = d_val["PATH_IMG"]
labels = d_val["PATH_LABELS"]
sentinel_images = d_val["PATH_SP_DATA"]
sentinel_masks = d_val["PATH_SP_MASKS"] # Cloud masks
sentinel_products = d_val["PATH_SP_DATES"] # Needed to get the dates of the sentinel images
centroids = d_val["SP_COORDS"] # Position of the aerial image in the sentinel super area
aerial_mtds=d_val["MTD_AERIAL"]

create_dataset_flair(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtds, name="o_regular", mode="val", stats=stats,max_samples=50)



images = d_test["PATH_IMG"]
labels = d_test["PATH_LABELS"]
sentinel_images = d_test["PATH_SP_DATA"]
sentinel_masks = d_test["PATH_SP_MASKS"] # Cloud masks
sentinel_products = d_test["PATH_SP_DATES"] # Needed to get the dates of the sentinel images
centroids = d_test["SP_COORDS"] # Position of the aerial image in the sentinel super area
aerial_mtds=d_test["MTD_AERIAL"]

print("images test: ",len(labels))
create_dataset_flair(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtds, name="o_regular", mode="test", stats=stats,max_samples=50)

