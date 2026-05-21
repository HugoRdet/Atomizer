#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda

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
#python3 flair_test.py
#sh TrainEval_MAE.sh "$EXPERIMENT_NAME" "$MODEL_NAME" regular

#python3 script_train_baselines_flairhub.py --xp_name vit_pm_v1_spot_train --model vit_pm  --subset_indices ./data/FLAIR-HUB/subset_indices.json --use_vhr false --use_spot true --spot_norm_as_vhr false --use_dem true --use_s2 true --use_s1 true --multi_temporal 6 --epochs 30
python3 script_train_baselines_flairhub.py --xp_name resnet_concat_v1_spot_train --model resnet --resnet_variant resnet50 --subset_indices ./data/FLAIR-HUB/subset_indices.json --use_vhr false --use_spot true  --spot_norm_as_vhr false  --use_dem true --use_s2 true --use_s1 true --multi_temporal 6 --epochs 30