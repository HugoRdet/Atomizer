#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda
conda activate venv

# === Config ===
SWEEP_CONFIG="sweep_config.yaml"
NUM_AGENTS=2  # Number of parallel agents (1 per GPU). Set to 1 if single GPU.

# === Create the sweep (once) ===
echo "Creating W&B sweep..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output (format: entity/project/sweep_id)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | grep -oP '[^\s]+$')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create sweep. Check output above."
    exit 1
fi

echo ""
echo "========================================="
echo "Sweep created: $SWEEP_ID"
echo "Launching $NUM_AGENTS agent(s)..."
echo "========================================="
echo ""

# === Launch agents (one per GPU) ===
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    echo "Starting agent on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i wandb agent "$SWEEP_ID" &
    sleep 2  # Small delay to avoid race conditions
done

echo ""
echo "All agents launched."
echo "To stop: kill %1 %2 ... or Ctrl+C"

# Wait for all background agents to finish
wait