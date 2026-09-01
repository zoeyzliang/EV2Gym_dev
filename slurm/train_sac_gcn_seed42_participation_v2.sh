#!/bin/bash
#SBATCH --job-name=sac_gcn_s42_partv2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gcn_seed42_partv2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gcn_seed42_partv2.err

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

# Baseline comparison for the participation-model robustness check —
# same degradation-cost + domain-randomized participation model as
# sac_gnn_real_seed42_participation_v2, isolating whether SAC-GNN's
# advantage over SAC-GCN (learned per-edge attention vs fixed averaging)
# survives under a more complete/uncertain participation model, or was
# specific to the earlier fixed, simpler participation curve.
python train_sac_gnn.py \
    --agent sac_gcn \
    --episodes 1500 \
    --seed 42 \
    --randomize_participation \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gcn_real_seed42_participation_v2
