#!/bin/bash
#SBATCH --job-name=cpl_21hub
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Coupled-energy-model rerun (referee M1/M4 fix), 21-hub, 1500 episodes.
#
# Usage:  sbatch --job-name=<name> slurm/train_coupled_21hub.sh <agent> <seed> <lambda_conf> [noedge]
#   agent        sac_gnn | sac_gcn | sac_flat
#   seed         e.g. 42, 1, 2
#   lambda_conf  DOE penalty weight; required, choose it from
#                pilot_coupled_lambda.sh first (200 is likely far too large
#                for the coupled reward scale)
#   noedge       optional; edgeless control (sac_gnn/sac_gcn only, referee M3(ii))
#
# Examples:
#   sbatch --job-name=cpl_gnn_s42        slurm/train_coupled_21hub.sh sac_gnn 42 10
#   sbatch --job-name=cpl_gnn_noedge_s42 slurm/train_coupled_21hub.sh sac_gnn 42 10 noedge
#
# Results: results/{agent}[_noedge]_21hub_seed{N}_coupled_lc{lambda}_20261004

set -euo pipefail

AGENT=${1:?agent required (sac_gnn|sac_gcn|sac_flat)}
SEED=${2:?seed required}
LAMBDA=${3:?lambda_conf required (run pilot_coupled_lambda.sh first)}
VARIANT=${4:-}

EXTRA_ARGS=()
TAG=${AGENT}
if [[ "$VARIANT" == "noedge" ]]; then
    EXTRA_ARGS+=(--no_edges)
    TAG=${AGENT}_noedge
elif [[ -n "$VARIANT" ]]; then
    echo "Unknown variant '$VARIANT' (expected 'noedge' or nothing)" >&2
    exit 1
fi

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main
echo "Code revision: $(git rev-parse HEAD)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sac_gnn.py \
    --energy_model coupled \
    --agent "$AGENT" \
    --episodes 1500 \
    --seed "$SEED" \
    --lambda_conf "$LAMBDA" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/${TAG}_21hub_seed${SEED}_coupled_lc${LAMBDA}_20261004
