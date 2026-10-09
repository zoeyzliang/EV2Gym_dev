#!/bin/bash
#SBATCH --job-name=cre21r
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Resume of cre21_pack.sh after job 60852177 hit its 3 h limit (9 Oct 2026).
# Benchmarks finished there and are not rerun. The 4 coordination studies
# resume from their per-day per_run.csv (29-30 of 41 days done; ~6 min per day
# on M3, so ~12 days ~75 min). The 2 DOE-statistics runs failed on an import
# path (fixed) and run in full: ~100 s per date locally, and M3 ran ~1.3x slower
# than local here, so ~90 min. 2.5 h = ~90 min + ~30% + margin.
#
# Usage:  sbatch slurm/cre21_resume.sh

set -euo pipefail
WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=/scratch2/fr57/zlia0072/ev2gym_training/results
LOGS=/scratch2/fr57/zlia0072/ev2gym_training/logs
N="--feeder_network node_cre21 --feeder_v_base 22 --kappa_load 1.0"
mkdir -p $R/cre21

declare -a CMDS=(
  "doe_base|python tools/doe_stats.py $N --siting base --out $R/cre21/doe_stats_base.csv"
  "doe_constrained|python tools/doe_stats.py $N --siting constrained --out $R/cre21/doe_stats_constrained.csv"
  "coord_base_pv0.6|python coordination_study.py $N --siting base --pv_penetration 0.6 --participant_billing --results_dir $R/cre21/coord_base_pv0.6"
  "coord_base_pv0.9|python coordination_study.py $N --siting base --pv_penetration 0.9 --participant_billing --results_dir $R/cre21/coord_base_pv0.9"
  "coord_constrained_pv0.6|python coordination_study.py $N --siting constrained --pv_penetration 0.6 --participant_billing --results_dir $R/cre21/coord_constrained_pv0.6"
  "coord_constrained_pv0.9|python coordination_study.py $N --siting constrained --pv_penetration 0.9 --participant_billing --results_dir $R/cre21/coord_constrained_pv0.9"
)
PIDS=()
for c in "${CMDS[@]}"; do
    name=${c%%|*}; cmd=${c#*|}
    ( eval "$cmd" ) > "$LOGS/slurm_${SLURM_JOB_ID:-local}_cre21_${name}.log" 2>&1 &
    PIDS+=($!); echo "started $name (pid $!)"
done
FAIL=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "FAILED: ${CMDS[$i]%%|*}"; FAIL=1; }; done
exit $FAIL
