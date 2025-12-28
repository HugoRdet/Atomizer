import os
import yaml
import json
import pandas as pd

def load_json_to_dict(filepath):
    with open(filepath, 'r') as json_file:
        return json.load(json_file)
    
def save_dict_to_json(dictionary, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

def read_yaml(path):
    # Open the YAML file
    with open(path, 'r') as file:
        # Load YAML content into a Python dictionary
        data = yaml.safe_load(file)
    return data

def save_yaml(path, data):
    """
    Save a dictionary as a YAML file at the specified path.
    
    :param path: The file path where the YAML file will be saved.
    :param data: The dictionary to save as YAML.
    """
    with open(path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)



def open_parquet(path):
    return pd.read_parquet(path, engine='fastparquet')


def ensure_folder_exists(folder_path: str) -> None:
    """
    Checks if a folder exists at the given path; creates it if it does not exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
