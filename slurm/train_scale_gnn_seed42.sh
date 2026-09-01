#!/bin/bash
#SBATCH --job-name=scale_gnn42
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_scale_gnn42.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_scale_gnn42.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

# 32-hub scaling experiment (RQ4: does SAC-GNN's advantage over SAC-GCN
# grow, shrink, or stay flat at larger hub count?). All hyperparameters
# identical to the 21-hub inner_melbourne runs — only --zone differs.
python train_sac_gnn.py \
    --agent sac_gnn \
    --episodes 1500 \
    --seed 42 \
    --zone greater_melbourne \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_scaling32_seed42
