#!/bin/bash
#SBATCH --job-name=coord
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Value of coordination (spec §6, C2): perfect-foresight LP with per-hub DOEs
# vs joint feeder constraints, all pre-registered days x 3 reps, one PV level.
# CPU only (HiGHS). Time: ~80 s per (day, rep) measured locally (12 LPs:
# 6 incentives x 2 modes) -> 123 x 80 s = 2.7 h; 4 h = +~45% for a slower CPU.
# per_run.csv is rewritten after each day.
#
# Usage:  sbatch --job-name=coord_pv0.9 slurm/coordination_study.sh <pv> [extra args]
#   e.g.  sbatch slurm/coordination_study.sh 0.9 --participant_billing
# Results: results/coordination_pv<pv>/

set -euo pipefail
PV=${1:?pv penetration required}
shift 1

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"

python coordination_study.py --pv_penetration "$PV" "$@" \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/coordination_pv${PV}
