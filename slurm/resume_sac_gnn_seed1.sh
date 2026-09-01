#!/bin/bash
#SBATCH --job-name=sac_gnn_resume_seed1
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_resume_seed1.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_resume_seed1.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

# Edit to the checkpoint you want to resume from (best.pt or episode_N.pt).
# --start_episode is inferred from the checkpoint filename if not given.
CHECKPOINT=/scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_real_seed1/checkpoints/best.pt

python train_sac_gnn.py \
    --agent sac_gnn \
    --seed 1 \
    --resume "$CHECKPOINT" \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_real_seed1
