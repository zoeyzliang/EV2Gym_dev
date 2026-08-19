#!/bin/bash
#SBATCH --job-name=eval_seed42
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_seed42.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_seed42.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

# Robust conda activation for non-interactive SLURM batch shells.
# "source activate ev2gym" (legacy syntax) intermittently fails on compute
# nodes with "activate: No such file or directory" -- depends on an old
# standalone activate script being reachable, which isn't guaranteed on
# every compute node. Sourcing conda.sh directly avoids this.
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_real_seed42/checkpoints/best.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_real_seed42/checkpoints/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_flat_real_seed42/checkpoints/best.pt \
    --n_runs 30 \
    --seed 42 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_seed42
