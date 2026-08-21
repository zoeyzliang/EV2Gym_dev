#!/bin/bash
#SBATCH --job-name=sens_lc200
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sens_lc200.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_sens_lc200.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

# lambda_conf=200 is the same as the default EnvConfig.lambda_conformance —
# included explicitly (rather than relying on the main sac_gnn_seed42 run)
# so the sensitivity comparison uses 3 runs from an identical training
# script invocation and results_dir naming convention, not one run drawn
# from a differently-purposed job.
python train_sac_gnn.py \
    --agent sac_gnn \
    --episodes 1500 \
    --seed 42 \
    --lambda_conf 200.0 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sensitivity_lambda_conf_200
