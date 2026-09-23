#!/bin/bash
#SBATCH --job-name=extended_seed1_final
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_extended_seed1_final.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_extended_seed1_final.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Extended-training follow-up evaluation (post-recovery final checkpoint, ep2500, includes the ep2150 collapse+recovery).
# SAC-GNN checkpoint is the final.pt from the resumed
# 2500-episode extended-training experiment (train_sac_gnn_32hub_seed1_extended_v2.sh),
# testing whether the original 1500-episode 32-hub SAC-GNN
# seed1 result (Table~\ref{tab:scaling32}) was a training-budget
# artefact. SAC-GCN checkpoint is the ORIGINAL (non-extended, 1500-episode)
# seed1 checkpoint, providing a direct, matched comparison point.
# SAC-Flat intentionally points at a nonexistent path (never trained at
# 32-hub scale, per the established scoping decision) -- skipped
# gracefully with a warning.
python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904_extended2500_v2/checkpoints/final.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_32hub_seed1_20260904/checkpoints/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --n_runs 100 \
    --seed 1 \
    --zone greater_melbourne \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_32hub_seed1_extended2500_final_20260904
