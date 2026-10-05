#!/bin/bash
#SBATCH --job-name=eval_fdr
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Evaluate one seed of a feeder batch (train_feeder_pack.sh) on the 36
# stratified held-out 2024 days, 3 paired repetitions each, with the
# perfect-foresight LP bound. About 30-45 min.
#
# Usage:
#   sbatch --job-name=eval_fdr_s42 slurm/evaluate_feeder.sh <batch_dir> <seed> [extra eval args...]
#     <batch_dir>  e.g. feeder_perhub_lc1_20261010 (under results/)
#     extra args   must match training, e.g. --doe_mode per_hub
#
# Missing checkpoints (e.g. no noedge run) are skipped.

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

R=/scratch2/fr57/zlia0072/ev2gym_training/results/$BATCH
AGENTS=()
for pair in "SAC-GNN=sac_gnn:sac_gnn" "SAC-GCN=sac_gcn:sac_gcn" "SAC-Flat=sac_flat:sac_flat" \
            "SAC-GNN-NoEdge=sac_gnn:sac_gnn_noedge" "SAC-GCN-NoEdge=sac_gcn:sac_gcn_noedge"; do
    NAME=${pair%%=*}; REST=${pair#*=}; KIND=${REST%%:*}; DIR=${REST#*:}
    CKPT=$R/${DIR}_seed${SEED}/checkpoints/best.pt
    if [ -f "$CKPT" ]; then AGENTS+=(--agent "$NAME=$KIND:$CKPT"); else echo "skip $NAME (no $CKPT)"; fi
done

python evaluate_feeder.py "${AGENTS[@]}" "$@" --n_reps 3 --lp \
    --results_dir "$R/evaluation_seed${SEED}"
