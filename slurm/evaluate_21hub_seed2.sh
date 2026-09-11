#!/bin/bash
#SBATCH --job-name=eval_21h_s2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_21h_s2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_21h_s2.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Final Q1-consistency evaluation. All checkpoints from the 20260904
# full-retrain batch (all fixes: DOE direction, DOE float32 tolerance,
# EV participation degradation, CUDA allocator). n_runs=100 matches
# Orfanoudakis et al. 2025's precedent for the flagship result.
python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_21hub_seed2_20260904/checkpoints/best.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_21hub_seed2_20260904/checkpoints/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_flat_21hub_seed2_20260904/checkpoints/best.pt \
    --n_runs 100 \
    --seed 2 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_21hub_seed2_20260904
