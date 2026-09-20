#!/bin/bash
#SBATCH --job-name=gnn_32h_s1_ext
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s1_ext.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s1_ext.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Extended-training experiment: does SAC-GNN at 32-hub scale need more
# than 1500 episodes to converge? eval_log.csv for this seed shows
# mean_net_profit_normal still clearly rising at the ep1500 cutoff
# (-$165 @ ep1300 -> -$199 @ ep1350 -> +$2,447 @ ep1400 -> +$1,779 @
# ep1500, no sign of plateauing), unlike every other 21-hub and 32-hub
# run, which had genuinely plateaued by ep1500. Resuming from this
# seed's own final.pt (not restarting from scratch) and extending to
# 2500 total episodes to see whether profit continues improving.
# --start_episode omitted deliberately: inferred automatically from the
# checkpoint, avoiding a hardcoded episode number that could be wrong.
python train_sac_gnn.py \
    --agent sac_gnn \
    --zone greater_melbourne \
    --seed 1 \
    --episodes 2500 \
    --resume /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904/checkpoints/final.pt \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed1_20260904_extended2500
