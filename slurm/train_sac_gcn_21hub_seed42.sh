#!/bin/bash
#SBATCH --job-name=gcn_21hub_s42
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gcn_21hub_s42.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gcn_21hub_s42.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full-consistency training run for Q1 submission: current code includes
# DOE direction-specific violation fix, DOE float32-precision tolerance
# fix, and EV participation degradation-cost fix (see repo history for
# details). Results directory is date-stamped (20260904) so this run's
# provenance (exact code state at submission time) is unambiguous
# relative to any earlier or later run of the same agent/seed/scale.
python train_sac_gnn.py \
    --agent sac_gcn \
    --episodes 1500 \
    --seed 42 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_21hub_seed42_20260904
