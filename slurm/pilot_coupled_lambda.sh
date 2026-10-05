#!/bin/bash
#SBATCH --job-name=pilot_cpl
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Reward-scale pilot for the coupled energy model, run BEFORE the full batch.
#
# Under "coupled", Oracle earns only ~$0.7-1.0 per step (~$200-300/day) on
# normal days, so lambda_conf=200 $/kW-step makes a single 1 kW over-request
# cost more than a day's profit. Because over-requesting beyond the DOE now
# yields no extra energy, any lambda > 0 already makes it strictly
# dominated; the pilot checks which lambda trains a policy that both trades
# and stays compliant. Compare eval_log.csv profit and doe_violation across
# lambda values, then fix lambda for the full batch.
#
# Usage:  sbatch --job-name=pilot_lc1 slurm/pilot_coupled_lambda.sh <lambda_conf> [episodes]
#   e.g.  sbatch --job-name=pilot_lc200 slurm/pilot_coupled_lambda.sh 200
#         sbatch --job-name=pilot_lc10  slurm/pilot_coupled_lambda.sh 10
#         sbatch --job-name=pilot_lc1   slurm/pilot_coupled_lambda.sh 1

set -euo pipefail

LAMBDA=${1:?lambda_conf required}
EPISODES=${2:-300}

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
    --agent sac_gnn \
    --episodes "$EPISODES" \
    --seed 42 \
    --lambda_conf "$LAMBDA" \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/pilot_sac_gnn_21hub_seed42_coupled_lc${LAMBDA}_ep${EPISODES}_20261005
