from training.utils import *
from training.utils.datasets import *
import argparse

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Test dataset loading")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset_path = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), 
    read_yaml(bands_yaml), 
    config_model
)

# =============================================================================
# DATASET CONFIG
# =============================================================================
dataset_config = read_yaml(configs_dataset_path)
dataset_config["bands_info"] = read_yaml(bands_yaml)

# =============================================================================
# INSTANTIATE DATASET
# =============================================================================
h5_path = f"./data/Tiny_BigEarthNet/{args.dataset_name}_train.h5"

dataset = Tiny_BigEarthNet(
    root_path=h5_path,
    transform=None,
    model=config_model["encoder"],
    modality_mode="train",
    mode="train",
    dataset_config=dataset_config,
    config_model=config_model,
    look_up=lookup_table,
)

# =============================================================================
# TEST: print shapes from element 0
# =============================================================================
image, attention_mask, label, id_img = dataset[0]

print(f"\n{'='*60}")
print(f"Dataset element 0:")
print(f"  image shape:       {image.shape}")
print(f"  image dtype:       {image.dtype}")
print(f"  mask shape:        {attention_mask.shape}")
print(f"  mask dtype:        {attention_mask.dtype}")
print(f"  valid pixels:      {int((attention_mask == 0).sum())} / {int(attention_mask.numel())}")
print(f"  valid bands:       {int((attention_mask == 0).any(dim=-1).any(dim=-1).sum())} / {image.shape[0]}")
print(f"  label shape:       {label.shape}")
print(f"  label values:      {label}")
print(f"  id_img:            {id_img}")
print(f"{'='*60}")