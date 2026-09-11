#!/bin/bash
#SBATCH --job-name=eval_lc100
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_lc100.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_eval_lc100.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

# SAC-GCN/Flat checkpoint paths intentionally point at nonexistent
# files — evaluate.py's load_agents() skips any agent whose checkpoint
# doesn't exist (with a warning), rather than crashing. This correctly
# evaluates SAC-GNN in isolation against Greedy/RulePrice/OracleMPC for
# this specific lambda_conf value, without needing GCN/Flat checkpoints
# at this lambda_conf (which were never trained — sensitivity analysis
# scope is SAC-GNN only, seed42 only, per the earlier scoping decision).
python evaluate.py \
    --sac_gnn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/sensitivity_lambda_conf_100/checkpoints/best.pt \
    --sac_gcn_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --sac_flat_checkpoint /scratch2/fr57/zlia0072/ev2gym_training/results/_no_such_checkpoint/best.pt \
    --n_runs 30 \
    --seed 42 \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/evaluation_sensitivity_lc100
