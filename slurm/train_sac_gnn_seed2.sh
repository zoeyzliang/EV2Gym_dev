#!/bin/bash
#SBATCH --job-name=sac_gnn_seed2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_seed2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_seed2.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

python train_sac_gnn.py \
    --agent sac_gnn \
    --episodes 1500 \
    --seed 2 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_real_seed2
