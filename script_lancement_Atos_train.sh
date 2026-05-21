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

#python script_train_PASTIS.py --xp_name s2s1_768_last12 --use_s1  --use_spot --multi_temporal 6 --ckpt_path ./checkpoints/pastis/pastis_pastis_e2e_ltae-epoch=21-val_mIoU=0.4640.ckpt --wandb_run_id lz6bfex8
#python script_train_flair.py --xp_name ayaaa --num_workers 32 --subset_indices ./data/FLAIR-HUB/subset_indices.json --ckpt_path ./checkpoints/flairhub/atomiser_flairhub_ayaaa-epoch=19-val_mIoU=0.5100.ckpt --wandb_run_id x0syh1sx
#python3 script_ablation_senflood.py --fraction 1.0 --ckpt_path ./checkpoints/ATOMIZERsenflood_baseline-val_loss-epoch=139-val_loss=0.0668.ckpt --xp_name dqdsq
#python3 script_ablation_senflood.py --fraction 0.75 --ckpt_path ./checkpoints/ATOMIZERsenflood_baseline-val_loss-epoch=139-val_loss=0.0668.ckpt --xp_name dQ
#python3 script_ablation_senflood.py --fraction 0.50 --ckpt_path ./checkpoints/ATOMIZERsenflood_baseline-val_loss-epoch=139-val_loss=0.0668.ckpt --xp_name DSQ
#python3 script_ablation_senflood.py --fraction 0.25 --ckpt_path ./checkpoints/ATOMIZERsenflood_baseline-val_loss-epoch=139-val_loss=0.0668.ckpt --xp_name DSQqsd
#python3 script_ablation_senflood.py --fraction 0.1 --ckpt_path ./checkpoints/ATOMIZERsenflood_baseline-val_loss-epoch=139-val_loss=0.0668.ckpt --xp_name bouhouhou
 python3 script_train_flair.py --xp_name flair_v1_spot_train --use_vhr false --use_spot true --spot_as_vhr true --spot_norm_as_vhr false --use_dem true --use_s2 true --use_s1 true --subset_indices ./data/FLAIR-HUB/subset_indices.json --multi_temporal 6 
#python3 script_train_senflood.py --xp_name senflood_baseline --clipping   
#python3 script_train_xview.py --xp_name fdhslkjqfd
#python3 script_train_baselines_flairhub.py --xp_name resnet_pm_v1  --model resnet_pm --resnet_variant resnet50  --subset_indices ./data/FLAIR-HUB/subset_indices.json  --epochs 30 --batch_size 2 --grad_accum 4  --num_workers 4 --lr 1e-4 
#python3 script_train_baselines_flairhub.py --xp_name vit_pm_v1 --model vit_pm --subset_indices ./data/FLAIR-HUB/subset_indices.json --epochs 30  --batch_size 2 --grad_accum 1 --num_workers 8 --img_size 512 --lr 1e-4 
#python3 script_train_baselines_flairhub.py --xp_name vit_concat_v1  --model vit  --subset_indices ./data/FLAIR-HUB/subset_indices.json  --epochs 30   --batch_size 1   --grad_accum 3   --num_workers 4   --img_size 512 
#python script_train_baselines_flairhub.py --xp_name resnet50_v1 --model resnet --resnet_variant resnet50 --subset_indices ./data/FLAIR-HUB/subset_indices.json --epochs 30  --batch_size 2 --num_workers 32
# Call training script with experiment name used in the arguments
#sh TrainEval.sh "$EXPERIMENT_NAME" config_test-Atomiser_Atos.yaml regular
#MODEL_NAME=config_test-Atomiser_Atos_One.yaml
#python3 flair_test.py
#sh TrainEval_MAE.sh "$EXPERIMENT_NAME" "$MODEL_NAME" regular
#python script_train_pretraining.py --xp_name esa_test --config_model config_test-Atomiser_Atos_One.yaml --dataset_name mmearth --task all

#sh TrainEval_SENFLOOD.sh "$EXPERIMENT_NAME" "$MODEL_NAME" u_regular

