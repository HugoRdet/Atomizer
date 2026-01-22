#!/bin/bash

# Check if all three required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <xp_name> <config_model> <config_dataset>"
    exit 1
fi

XP_NAME="$1"
CONFIG_MODEL="$2"
CONFIG_DATASET="$3"

echo "=== Starting Distributed Training on $(hostname) ==="

# Use torchrun to launch the training. 
# The variables $NODE_RANK, $MASTER_ADDR, and $MASTER_PORT 
# must be exported by your main launch script.
# Inside TrainEval_FLAIR_multinodes.sh
cd $(dirname "$0") # Move to the directory where this script is located
echo "Current directory on $(hostname): $(pwd)"

torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    script_train_flair.py \
    --xp_name "$XP_NAME" \
    --config_model "$CONFIG_MODEL" \
    --dataset_name "$CONFIG_DATASET"

# --- POST-TRAINING EVALUATION ---
# We only want to run evaluation ONCE (on the master node, rank 0)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Training complete on Master Node. Starting evaluation..."

    # Read the WandB run ID
    RUN_FILE="training/wandb_runs/${XP_NAME}.txt"
    
    # Wait a few seconds to ensure the training script finished writing the file
    sleep 5 

    if [ ! -f "$RUN_FILE" ]; then
        echo "Error: $RUN_FILE not found."
        exit 1
    fi

    RUN_ID=$(cat "$RUN_FILE" | tr -d '[:space:]')

    if [ -z "$RUN_ID" ]; then
        echo "Error: WandB run ID not found in file."
        exit 1
    fi

    echo "Captured WandB Run ID: $RUN_ID"
    
    # Run evaluation normally (usually evaluation doesn't need DDP/distributed)
    python script_evaluation.py \
        --run_id "$RUN_ID" \
        --xp_name "$XP_NAME" \
        --config_model "$CONFIG_MODEL" \
        --dataset_name "$CONFIG_DATASET"

else
    echo "Training complete on Worker Node ($NODE_RANK). Exiting to let Master handle evaluation."
fi