#!/bin/bash
#SBATCH --job-name=eval_lambda
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Evaluate one seed of the full-length lambda study (train_feeder_pack.sh,
# runs sac_gnn_lc<lambda>_seed<N>): every lambda's best.pt side by side with
# NoV2G / GreedyTOU / RulePrice on the pre-registered days (36 representative
# + stress set), 3 paired repetitions, no LP bound. About 10-20 min.
# Then apply the pre-registered criterion across seeds with decide_lambda.py.
#
# Usage:  sbatch --job-name=eval_lam_s42 slurm/evaluate_lambda.sh <batch_dir> <seed> [extra eval args]
#   e.g.  sbatch slurm/evaluate_lambda.sh feeder_lambda_s42_20261007 42
#         sbatch slurm/evaluate_lambda.sh feeder_lambdav2_s42_<date> 42 --action_scale feasible

set -euo pipefail
BATCH=${1:?batch dir required}
SEED=${2:?seed required}
shift 2

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv

R=/scratch2/fr57/zlia0072/ev2gym_training/results/$BATCH
AGENTS=()
for run in "$R"/sac_gnn_lc*_seed${SEED}; do
    lc=$(basename "$run" | sed -E "s/sac_gnn_lc(.*)_seed${SEED}/\1/")
    ckpt=$run/checkpoints/best.pt
    if [ -f "$ckpt" ]; then AGENTS+=(--agent "SAC-GNN-lc$lc=sac_gnn:$ckpt"); else echo "MISSING $run (no best.pt)"; exit 1; fi
done
[ ${#AGENTS[@]} -gt 0 ] || { echo "no runs found in $R for seed $SEED"; exit 1; }

python evaluate_feeder.py "${AGENTS[@]}" --doe_mode per_hub --n_reps 3 "$@" \
    --results_dir "$R/evaluation_seed${SEED}"
