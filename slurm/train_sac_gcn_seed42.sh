#!/bin/bash
#SBATCH --job-name=sac_gcn_seed42
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gcn_seed42.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gcn_seed42.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

python train_sac_gnn.py \
    --agent sac_gcn \
    --episodes 1500 \
    --seed 42 \
    --results_dir results/sac_gcn_real_seed42
