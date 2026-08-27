#!/bin/bash
#SBATCH --job-name=sac_gnn_s42_partv2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_seed42_partv2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sac_gnn_seed42_partv2.err

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

# Focused robustness retrain (see thesis discussion): SAC-GNN under the
# battery-degradation-cost participation model, with domain randomization
# of beta_1/beta_3/participation_gamma each training episode so the policy
# is not overfit to one fixed, uncalibrated participation curve.
# Fixed evaluation days are NOT randomized (see make_env() split guard in
# train_sac_gnn.py) — checkpoint selection and reported eval numbers stay
# against a known, fixed curve.
# results_dir is intentionally distinct from sac_gnn_real_seed42 — this is
# a secondary robustness/sensitivity result, NOT a replacement for the
# existing headline checkpoints, which are left untouched.
python train_sac_gnn.py \
    --agent sac_gnn \
    --episodes 1500 \
    --seed 42 \
    --randomize_participation \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_real_seed42_participation_v2
