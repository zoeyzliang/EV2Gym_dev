#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Robustness checks S1–S3 (spec §6), CPU only (HiGHS LPs + MPC), packed: each
# group runs its studies as parallel single-threaded processes. Each study is
# 123 (day, rep) episodes; measured locally ~80 s per episode for the LP
# studies (~2.7 h), M3 ran the same studies ~2–3x faster. 4 h limit.
#
# Usage:  sbatch --job-name=robust_A slurm/robustness_pack.sh A
#         sbatch --job-name=robust_B slurm/robustness_pack.sh B
# Results: results/robust_<name>/

set -euo pipefail
GROUP=${1:?group A or B required}
WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=/scratch2/fr57/zlia0072/ev2gym_training/results
LOGS=/scratch2/fr57/zlia0072/ev2gym_training/logs
B="--participant_billing"
EV="--lp --lp_reps 3 --mpc_incentive 0.2 --n_reps 3 $B"

declare -a CMDS
case "$GROUP" in
  A) CMDS=(
       "shared0.3|python coordination_study.py --mode shared --shared_frac 0.3 $B --results_dir $R/robust_shared0.3"
       "shared0.5|python coordination_study.py --mode shared --shared_frac 0.5 $B --results_dir $R/robust_shared0.5"
       "thermal0.8|python coordination_study.py --mode network --thermal_margin 0.8 $B --results_dir $R/robust_thermal0.8"
       "beta1x0.5|python evaluate_feeder.py $EV --beta1_scale 0.5 --results_dir $R/robust_beta1x0.5"
       "beta1x1.5|python evaluate_feeder.py $EV --beta1_scale 1.5 --results_dir $R/robust_beta1x1.5"
     ) ;;
  B) CMDS=(
       "deg0.05|python evaluate_feeder.py $EV --deg_cost 0.05 --results_dir $R/robust_deg0.05"
       "deg0.10|python evaluate_feeder.py $EV --deg_cost 0.10 --results_dir $R/robust_deg0.10"
       "tariff50|python evaluate_feeder.py $EV --import_tariff 50 --results_dir $R/robust_tariff50"
       "tariff100|python evaluate_feeder.py $EV --import_tariff 100 --results_dir $R/robust_tariff100"
     ) ;;
  *) echo "unknown group $GROUP" >&2; exit 1 ;;
esac

PIDS=()
for c in "${CMDS[@]}"; do
    name=${c%%|*}; cmd=${c#*|}
    $cmd > "$LOGS/slurm_${SLURM_JOB_ID:-local}_robust_${name}.log" 2>&1 &
    PIDS+=($!); echo "started $name (pid $!)"
done
FAIL=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "FAILED: ${CMDS[$i]%%|*}"; FAIL=1; }; done
exit $FAIL
