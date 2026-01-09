#!/bin/bash
# Usage: ./script_setup.sh <port> <destination_host>

# Check if both required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <port> <destination_host> <dataset_name>"
    exit 1
fi

PORT="$1"
HOST="$2"
DATA="$3"

# Execute commands on the remote host via SSH using a here-document
ssh -p "$PORT" root@"$HOST" <<'EOF'
# Change to /home directory
cd /home/

# Clone the repository
git clone https://github.com/HugoRdet/Atomizer
git config --global user.email "hugordet@gmail.com"
# (If needed, you can perform any additional authentication steps here.)
# Install zip non-interactively
sudo apt install -y zip


EOF

# After the remote commands, execute the scp command to copy your file
scp -P "$PORT" -r ./data/custom_flair/"$DATA"_*.h5 root@"$HOST":/home/Atomizer/data/custom_flair/
scp -P "$PORT" -r ./data/custom_flair/precomputed_* root@"$HOST":/home/Atomizer/data/custom_flair/
scp -P "$PORT" -r ./data/bands_info/* root@"$HOST":/home/Atomizer/data/bands_info/
scp -P "$PORT" -r ./training/configs/* root@"$HOST":/home/Atomizer/training/configs/
scp -P "$PORT" -r ./requirements.txt root@"$HOST":/home/Atomizer/




ssh -p "$PORT" root@"$HOST" <<'EOF'
# Change to /home directory
cd /home/Atomizer/

pip install -r requirements.txt

# Uninstall and install specific version of wandb
pip uninstall wandb -y
pip install wandb==0.15.12

# Log in to wandb (ensure this token is secured)
wandb login 243a5bece03ed91d556f0d6c5280bdc533666345



chmod +x script_lancement_Atos_train.sh
chmod +x script_lancement_Atos_viz.sh


EOF



