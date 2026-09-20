#!/bin/bash
#SBATCH --job-name=eval_lc400_v2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_lc400_v2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_lc400_v2.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sensitivity analysis evaluation (lambda_conf=400), Q1-consistency batch.
# _v2 note: the original train_sensitivity_lc400.sh job (59746349, 7-day
# wall time, no GPU constraint) spent 4+ days pending due to cluster-wide
# maintenance reservations (confirmed by M3 support), so a parallel
# resubmission (train_sensitivity_lc400_v2.sh: --constraint=L40S,
# --time=3-00:00:00) was submitted alongside it as a fallback. The _v2
# job completed first (COMPLETED, 2d05h43m, m3g103); the original was
# cancelled once _v2 finished. This script therefore points at the _v2
# checkpoint path, not the original (non-_v2) one.
#
# SAC-GCN/Flat checkpoints intentionally point at a nonexistent path --
# evaluate.py's load_agents() skips them gracefully with a warning. This
# evaluates SAC-GNN (at this lambda_conf) against Greedy/RulePrice/OracleMPC
# only, matching the original sensitivity-analysis scope (SAC-GNN, seed42
# only, 21-hub). lambda_conf=200 is NOT re-evaluated separately here -- it
# reuses the main comparison's own SAC-GNN seed42 result (identical
# checkpoint and conditions), evaluated in evaluate_21hub_seed42.sh.
python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_21hub_seed42_lc400_20260904_v2/checkpoints/best.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --n_runs 100 \
    --seed 42 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_sensitivity_lc400_20260904_v2
