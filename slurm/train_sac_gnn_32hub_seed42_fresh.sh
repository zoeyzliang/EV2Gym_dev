#!/bin/bash
#SBATCH --job-name=gnn_32h_s42_fresh
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s42_fresh.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s42_fresh.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh (non-resumed) 2500-episode training, companion experiment to
# train_sac_gnn_32hub_seed42_extended.sh (which RESUMES seed42's
# existing 1500-episode checkpoint and continues to 2500). This script
# instead trains from scratch for a full 2500 episodes, deliberately
# NOT reusing the existing checkpoint (which had plateaued at a poor,
# noisy, oscillating -$7,500 to -$14,300 level with no recovery trend
# across ep800-1500).
#
# Purpose: separates two different questions. The resumed run tests
# whether seed42's specific stuck trajectory can EVER escape its
# current poor local optimum given more gradient steps from where it
# already is. This fresh run tests whether seed 42's random
# initialisation, given a full 2500-episode budget from the start
# (rather than 1500), avoids landing in that same poor optimum at all.
# Results directory is distinct (_fresh2500) from both the original
# (_20260904) and resumed (_extended2500) runs.
python train_sac_gnn.py \
    --agent sac_gnn \
    --zone greater_melbourne \
    --seed 42 \
    --episodes 2500 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed42_20260904_fresh2500
