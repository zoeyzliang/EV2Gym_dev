#!/bin/bash
#SBATCH --job-name=eval_pilot
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Evaluate the lambda pilot (train_feeder_pack.sh, batch tag "pilot"): the
# best.pt of each lambda run side by side with the NoV2G / GreedyTOU /
# RulePrice baselines, on the 36 stratified held-out days, 2 paired reps.
# No LP bound (not needed to choose lambda). About 10-20 min.
#
# Choose lambda by profit before penalties (arbitrage_profit) and limit
# violations (limit_viol_kwh, compliance) — both independent of lambda —
# taking the SMALLEST lambda whose violations are no worse than NoV2G's.
#
# Usage:  sbatch slurm/evaluate_pilot.sh <pilot_batch_dir>
#   e.g.  sbatch slurm/evaluate_pilot.sh feeder_pilot_20261006

set -euo pipefail
BATCH=${1:?pilot batch dir required, e.g. feeder_pilot_20261006}

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv

R=/scratch2/fr57/zlia0072/ev2gym_training/results/$BATCH
AGENTS=()
for run in "$R"/sac_gnn_lc*_seed42; do
    lc=$(basename "$run" | sed -E 's/sac_gnn_lc(.*)_seed42/\1/')
    ckpt=$run/checkpoints/best.pt
    if [ -f "$ckpt" ]; then AGENTS+=(--agent "SAC-GNN-lc$lc=sac_gnn:$ckpt"); else echo "skip $run (no best.pt)"; fi
done

python evaluate_feeder.py "${AGENTS[@]}" --doe_mode per_hub --n_reps 2 \
    --results_dir "$R/evaluation_pilot"
