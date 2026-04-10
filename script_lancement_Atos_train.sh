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

python3 script_train_senflood.py --xp_name senflood_baseline --config_model config_test-Atomiser_Atos_One.yaml --dataset_name  u_regular  --clipping
python3 script_train_senflood.py --xp_name senflood_baseline --config_model config_test-Atomiser_Atos_One.yaml --dataset_name  u_regular 

python3 script_train_senflood.py --xp_name senflood_baseline --config_model config_test-Atomiser_Atos_two.yaml --dataset_name  u_regular 
python3 script_train_senflood.py --xp_name senflood_baseline --config_model config_test-Atomiser_Atos_two.yaml --dataset_name  u_regular  --clipping


 
# Call training script with experiment name used in the arguments
#sh TrainEval.sh "$EXPERIMENT_NAME" config_test-Atomiser_Atos.yaml regular
#MODEL_NAME=config_test-Atomiser_Atos_One.yaml
#python3 flair_test.py
#sh TrainEval_MAE.sh "$EXPERIMENT_NAME" "$MODEL_NAME" regular
#python script_train_pretraining.py --xp_name esa_test --config_model config_test-Atomiser_Atos_One.yaml --dataset_name mmearth --task all

#sh TrainEval_SENFLOOD.sh "$EXPERIMENT_NAME" "$MODEL_NAME" u_regular

