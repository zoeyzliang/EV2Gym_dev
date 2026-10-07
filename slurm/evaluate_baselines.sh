#!/bin/bash
#SBATCH --job-name=eval_base
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Non-learned benchmark on the pre-registered days, 3 paired repetitions:
# NoV2G / GreedyTOU / RulePrice, plus whatever the extra args add
# (--lp --lp_reps 3: perfect-foresight bound; --mpc_incentive c [--mpc_perfect]:
# forecast MPC). Rows pair with agents' per_run.csv on (date, rep).
# Time: ~40 s per episode with the LP (123 episodes ~1.4 h, measured locally);
# without the LP ~2 s per episode (shorten with --time=00:30:00).
#
# Usage:  sbatch --job-name=<name> slurm/evaluate_baselines.sh <out_name> [eval args...]
#   e.g.  sbatch slurm/evaluate_baselines.sh benchmark_perhub_billing --participant_billing --lp --lp_reps 3 --mpc_incentive 0.2 --mpc_perfect
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

python evaluate_feeder.py --n_reps 3 "$@" \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/$NAME
