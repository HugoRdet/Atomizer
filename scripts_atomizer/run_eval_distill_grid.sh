#!/usr/bin/env bash
# Full grid for one distilled checkpoint + the random-init control.
# Usage: bash scripts_atomizer/run_eval_distill_grid.sh <encoder_*.pth> <distill_config.yaml> [bn_mode]
set -euo pipefail
CKPT=$1; CFG=$2; BN=${3:-batch}
COMMON="--distill_config $CFG --bands_yaml ./data/bands_info/bands.yaml --bn_mode $BN --seeds 0 1 2"

for CK in "$CKPT" none; do
  for DS in sen1floods11 burnscars pastis; do
    EB=2; [ "$DS" = pastis ] && EB=4
    for P in linear decoder; do
      python scripts_atomizer/eval_distill_ptv3.py --dataset $DS --protocol $P --ckpt $CK --enc_bs $EB \
        --label_fracs 0.1 1.0 $COMMON
    done
  done
  python scripts_atomizer/eval_distill_ptv3.py --dataset eurosat --protocol knn    --ckpt $CK --enc_bs 32 --label_fracs 0.1 1.0 $COMMON
  python scripts_atomizer/eval_distill_ptv3.py --dataset eurosat --protocol linear --ckpt $CK --enc_bs 32 --label_fracs 0.1 1.0 $COMMON
done
# Fine-tuning (expensive: one encoder pass per epoch): distilled vs from scratch, full labels
for CK in "$CKPT" none; do
  python scripts_atomizer/eval_distill_ptv3.py --dataset sen1floods11 --protocol finetune --ckpt $CK --epochs 30 \
    --lr 1e-4 --enc_bs 2 --label_fracs 1.0 $COMMON
done
