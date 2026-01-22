#!/bin/bash
#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda

# 1. Get Cluster Info
NODES=($(uniq $OAR_NODEFILE))
MASTER_ADDR=${NODES[0]}
NUM_NODES=${#NODES[@]}
MASTER_PORT=12345

# Generate a random experiment name if none is provided
if [ -z "$1" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1)
  EXPERIMENT_NAME="xp_${TIMESTAMP}_${RANDOM_SUFFIX}"
  echo "No experiment name provided. Using generated name: $EXPERIMENT_NAME"
else
  EXPERIMENT_NAME=$1
fi

## === Then load the module and activate your env ===
conda activate venv

# Call training script with experiment name used in the arguments
#sh TrainEval.sh "$EXPERIMENT_NAME" config_test-Atomiser_Atos.yaml regular
MODEL_NAME=config_test-Atomiser_Atos_One.yaml
#python3 flair_test.py
#sh TrainEval_MAE.sh "$EXPERIMENT_NAME" "$MODEL_NAME" regular
# 2. Launch on ALL nodes
# 2. Launch on ALL nodes
for i in "${!NODES[@]}"; do
  NODE=${NODES[$i]}
  echo "Launching on $NODE as Rank $i"
  
  # ADD "cd $(pwd);" at the start of the oarsh command string
  oarsh $NODE "cd $(pwd); \
    source /etc/profile.d/lmod.sh; \
    module load conda; \
    conda activate venv; \
    export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export NODE_RANK=$i; \
    export WORLD_SIZE=$((NUM_NODES * 2)); \
    bash TrainEval_FLAIR_multinodes.sh '$EXPERIMENT_NAME' '$MODEL_NAME' u_regular" &
done

wait

