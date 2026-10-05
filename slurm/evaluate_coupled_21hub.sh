#!/bin/bash
#SBATCH --job-name=eval_cpl_21h
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Evaluate one seed of the coupled-energy-model batch (train_coupled_21hub.sh):
# SAC-GNN, SAC-GCN, SAC-Flat, both edgeless controls, and the three
# non-learning baselines, paired on identical environment realisations.
# Missing checkpoints are skipped with a warning by evaluate.py.
#
# Usage:  sbatch --job-name=eval_cpl_s42 slurm/evaluate_coupled_21hub.sh <seed> <lambda_conf>

set -euo pipefail

SEED=${1:?seed required}
LAMBDA=${2:?lambda_conf required (must match the training batch)}
R=/scratch2/fr57/zlia0072/ev2gym_training/results
SUFFIX=21hub_seed${SEED}_coupled_lc${LAMBDA}_20261004

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main
echo "Code revision: $(git rev-parse HEAD)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate.py \
    --energy_model coupled \
    --sac_gnn_checkpoint        $R/sac_gnn_${SUFFIX}/checkpoints/best.pt \
    --sac_gcn_checkpoint        $R/sac_gcn_${SUFFIX}/checkpoints/best.pt \
    --sac_flat_checkpoint       $R/sac_flat_${SUFFIX}/checkpoints/best.pt \
    --sac_gnn_noedge_checkpoint $R/sac_gnn_noedge_${SUFFIX}/checkpoints/best.pt \
    --sac_gcn_noedge_checkpoint $R/sac_gcn_noedge_${SUFFIX}/checkpoints/best.pt \
    --n_runs 100 \
    --seed "$SEED" \
    --results_dir $R/evaluation_${SUFFIX}
