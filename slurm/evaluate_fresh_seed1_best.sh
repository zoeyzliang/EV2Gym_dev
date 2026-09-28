#!/bin/bash
#SBATCH --job-name=eval_fresh_s1_best
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_fresh_s1_best.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_fresh_s1_best.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh 2500-episode SAC-GNN (32-hub, seed 1) evaluation, checkpoint best.pt.
# This run was trained from scratch for a 2500-episode target but terminated by a
# PyTorch CUDA caching-allocator assertion (Aborted/core dumped) at ~episode 2480, so
# no final.pt exists; best.pt is the checkpoint evaluated. SAC-GCN is the ORIGINAL
# 1500-episode seed 1 checkpoint (matched comparison point). SAC-Flat points at a
# nonexistent path (never trained at 32-hub scale) and is skipped with a warning.
python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904_fresh2500/checkpoints/best.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_32hub_seed1_20260904/checkpoints/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --n_runs 100 \
    --seed 1 \
    --graph_path data/graphs/greater_melbourne.pkl \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_32hub_seed1_fresh2500_best_20260904
