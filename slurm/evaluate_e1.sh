#!/bin/bash
#SBATCH --job-name=eval_e1
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

# Evaluate one seed of E1 (spec §6): SAC-GNN, SAC-GCN, SAC-Flat and
# SAC-GNN-NoEdge at λ = 0.5 with NoV2G / GreedyTOU / RulePrice on the
# pre-registered days, 3 paired repetitions. No LP (evaluate_lp_bound.sh
# computes it once). The λ-study runs are reused for SAC-GNN at seeds
# 42, 1, 2 (sac_gnn_lc0.5_seed<N>, same code and settings as E1).
# The λ-study evaluation took ~5 min with 6 agents.
#
# Usage:  sbatch --job-name=eval_e1_s42 slurm/evaluate_e1.sh <seed>
# Results: results/e1_evaluation/seed<N>/

set -euo pipefail
SEED=${1:?seed required}

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv

R=/scratch2/fr57/zlia0072/ev2gym_training/results
one() {   # exactly one directory must match
    local m=( $1 ); [ ${#m[@]} -eq 1 ] && [ -d "${m[0]}" ] || { echo "expected one match for $1, got: ${m[*]}" >&2; exit 1; }
    echo "${m[0]}"
}
E1=$(one "$R/feeder_e1_s${SEED}_*")
case "$SEED" in
    42|1|2) GNN=$(one "$R/feeder_lambda_s${SEED}_*")/sac_gnn_lc0.5_seed${SEED} ;;
    *)      GNN=$E1/sac_gnn_seed${SEED} ;;
esac
AGENTS=()
for spec in "SAC-GNN=sac_gnn:$GNN" "SAC-GCN=sac_gcn:$E1/sac_gcn_seed${SEED}" \
            "SAC-Flat=sac_flat:$E1/sac_flat_seed${SEED}" "SAC-GNN-NoEdge=sac_gnn:$E1/sac_gnn_noedge_seed${SEED}"; do
    ckpt=${spec#*:}/checkpoints/best.pt
    [ -f "$ckpt" ] || { echo "MISSING $ckpt" >&2; exit 1; }
    AGENTS+=(--agent "${spec%%:*}:$ckpt")
    echo "agent ${spec%%=*}: $ckpt"
done

python evaluate_feeder.py "${AGENTS[@]}" --doe_mode per_hub --n_reps 3 \
    --results_dir "$R/e1_evaluation/seed${SEED}"
