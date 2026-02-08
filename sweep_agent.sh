source /etc/profile.d/lmod.sh
module load conda
conda activate venv

CUDA_VISIBLE_DEVICES=0 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=1 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=2 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=3 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=4 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=5 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=6 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
CUDA_VISIBLE_DEVICES=7 wandb agent hugordet-inria/Atomiser_BigEarthNet/5oyvlv3b &
wait