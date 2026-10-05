#!/bin/bash
#SBATCH --job-name=extended_seed1_gcnvgnn_final
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_extended_seed1_gcnvgnn_final.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_extended_seed1_gcnvgnn_final.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Matched-budget evaluation: both SAC-GNN and SAC-GCN warm-started from
# their 1500-episode checkpoints to 2500 episodes (seed 1, post-recovery
# final.pt for both). SAC-GCN's collapse (episodes ~2100-2250, DOE
# violation up to 371,077 kW) is longer than SAC-GNN's (episodes
# 2150-2200) but both fully recover by episode 2300-2500.
# SAC-Flat intentionally points at a nonexistent path (never trained at
# 32-hub scale) -- skipped gracefully with a warning.
python evaluate.py \
    --energy_model legacy \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904_extended2500_v2/checkpoints/final.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_32hub_seed1_20260904_extended2500/checkpoints/final.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --n_runs 100 \
    --seed 1 \
    --graph_path data/graphs/greater_melbourne.pkl \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_32hub_seed1_matched2500_final_20260904
