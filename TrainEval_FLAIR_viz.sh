#!/bin/bash

# Check if all three required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <xp_name> <config_model> <config_dataset>"
    exit 1
fi

XP_NAME="$1"
CONFIG_MODEL="$2"
CONFIG_DATASET="$3"

#echo "Starting training..."
python -u viz_callback.py --xp_name "$XP_NAME" --config_model "$CONFIG_MODEL" --dataset_name "$CONFIG_DATASET"

# Read the WandB run ID from the wandb_runs directory using the xp_name variable for the filename
RUN_FILE="training/wandb_runs/${XP_NAME}.txt"
if [ ! -f "$RUN_FILE" ]; then
    echo "Error: $RUN_FILE not found."
    exit 1
fi



#python script_evaluation_spec.py --run_id "$RUN_ID" --xp_name "$XP_NAME" --config_model "$CONFIG_MODEL" --dataset_name "$CONFIG_DATASET" 

#python script_evaluation.py --run_id AYAYdsdsqA --xp_name "$XP_NAME" --config_model "$CONFIG_MODEL" --dataset_name "$CONFIG_DATASET" 
