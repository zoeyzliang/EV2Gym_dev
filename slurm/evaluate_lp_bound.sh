#!/bin/bash
#SBATCH --job-name=eval_lp
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Perfect-foresight LP bound (and the rule baselines) on the pre-registered
# days, every paired repetition. No trained agents: the bound does not depend
# on them, so it runs once per environment setting, not once per seed.
# Rows pair with the agents' per_run.csv on (date, rep).
#
# Time: ~50 s per episode (HiGHS, measured locally) x 41 days x 3 reps
# = ~1.7 h; 3 h allows for a slower CPU. per_run.csv is rewritten after each
# day, so a timeout keeps the finished days. CPU-only work: the GPU type does
# not affect the result, so the fit pool may be used to avoid the normal-QOS
# GPU limit:  sbatch --partition=fit --qos=fitq --constraint=A100-80G --gres=gpu:A100:1 ...
#
# Usage:  sbatch slurm/evaluate_lp_bound.sh <out_name> [env args matching training...]
#   e.g.  sbatch slurm/evaluate_lp_bound.sh lp_bound_perhub --doe_mode per_hub
# Results: results/<out_name>/

set -euo pipefail
NAME=${1:?output name required}
shift 1

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"

python evaluate_feeder.py --n_reps 3 --lp --lp_reps 3 "$@" \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/$NAME
