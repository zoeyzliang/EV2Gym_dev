#!/bin/bash
#SBATCH --job-name=gcn_32h_s1_ext
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gcn_32h_s1_ext.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gcn_32h_s1_ext.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SAC-GCN 32-hub, seed 1: warm-start extension from 1500 to 2500 episodes, matching
# the protocol applied to SAC-GNN (train_sac_gnn_32hub_seed1_extended_v2.sh), so the
# GNN-vs-GCN comparison at 32 hubs is not confounded by unequal training budget.
# Checkpoints store network weights, log_alpha and step counters only; the replay
# buffer and optimiser states are re-initialised at the restart (same as the GNN runs).
# --start_episode is omitted deliberately: inferred from the checkpoint.
# Expected duration ~1.5 days (1000 episodes at ~2.2 min/episode), i.e. well inside
# the process lengths that have completed on this cluster (a 2500-episode single
# process was terminated by a CUDA allocator assertion at ~episode 2480).
python train_sac_gnn.py \
    --agent sac_gcn \
    --zone greater_melbourne \
    --seed 1 \
    --episodes 2500 \
    --resume /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_32hub_seed1_20260904/checkpoints/final.pt \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_32hub_seed1_20260904_extended2500
