#!/bin/bash
#SBATCH --job-name=gnn_32h_s1_fresh
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s1_fresh.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s1_fresh.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh (non-resumed) 2500-episode training, companion experiment to
# train_sac_gnn_32hub_seed1_extended.sh (which RESUMES seed1's existing
# 1500-episode checkpoint and continues to 2500). This script instead
# trains from scratch for a full 2500 episodes, deliberately NOT
# reusing the existing checkpoint -- new random initialisation, new
# replay buffer, new optimiser state.
#
# Purpose: separates two different questions. The resumed run answers
# "was THIS SPECIFIC seed1 trajectory cut off before converging?" This
# fresh run answers "does a full 2500-episode budget produce a
# different/better/more stable outcome than 1500, in general, for
# 32-hub SAC-GNN?" -- independent of whatever seed1's particular first
# 1500 episodes happened to do. Results directory is distinct
# (_fresh2500) from both the original (_20260904) and resumed
# (_extended2500) runs, so none of the three can collide.
python train_sac_gnn.py \
    --agent sac_gnn \
    --zone greater_melbourne \
    --seed 1 \
    --episodes 2500 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904_fresh2500
