#!/bin/bash
#SBATCH --job-name=sens_lc400_v2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sens_lc400_v2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sens_lc400_v2.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sensitivity analysis (lambda_conf sweep), full-consistency Q1 rerun.
# NOTE: lambda_conf=200 is the EnvConfig default and is intentionally
# NOT re-run here -- it is identical in every respect (agent, seed,
# zone, lambda_conf) to the main sac_gnn_21hub_seed42_20260904 run in
# this same batch, so that run's result IS the lc200 sensitivity data
# point. Only lc100 and lc400 (genuinely different configurations) are
# run as separate jobs.
#
# _v2 / redundant-submission note:
# This is a parallel resubmission of the original sens_lc400 job
# (59746349), which has been pending 4+ days due to cluster-wide
# maintenance reservations (confirmed by M3 support) rather than any
# issue with this job itself. This version adds --constraint=L40S
# (avoiding the T4/slow-hardware issue identified earlier this
# project) and a realistic --time=3-00:00:00 (down from 7-00:00:00,
# to make it easier for the scheduler to fit around the maintenance
# reservation windows, per M3 support's explanation). The original
# job is being kept in the queue as a fallback rather than cancelled;
# --results_dir below is deliberately distinct (suffixed _v2) so the
# two jobs cannot collide/overwrite each other's checkpoints or logs
# if both happen to start. Whichever job starts running first should
# be kept; cancel the other at that point.
python train_sac_gnn.py \
    --agent sac_gnn \
    --episodes 1500 \
    --seed 42 \
    --lambda_conf 400.0 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_21hub_seed42_lc400_20260904_v2
