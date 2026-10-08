#!/bin/bash
#SBATCH --job-name=cre21
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# CRE21 network study (spec §6), CPU only, packed: DOE statistics, coordination
# LP (both sitings, PV 0.6 / 0.9) and benchmark (MPC incentive re-selected on
# validation days, then baselines + MPC + LP bound) for both sitings.
# Time: coordination ~280 s per (day, rep) locally; M3 ran the node_34 studies
# ~10x faster -> ~1 h; DOEs ~100 s per date locally. 3 h covers a 3x
# slower-than-expected run. Results: results/cre21_*/
#
# Usage:  sbatch slurm/cre21_pack.sh

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

bench() {   # $1 = siting: select the MPC incentive on validation days, then evaluate
    python select_mpc_incentive.py $N --siting $1 --participant_billing --out $R/cre21/mpc_incentive_$1.json
    c=$(python -c "import json;print(json.load(open('$R/cre21/mpc_incentive_$1.json'))['chosen'])")
    python evaluate_feeder.py $N --siting $1 --participant_billing --n_reps 3 --lp --lp_reps 3 \
        --mpc_incentive $c --mpc_perfect --results_dir $R/cre21/benchmark_$1
}

declare -a CMDS=(
  "doe_base|python tools/doe_stats.py $N --siting base --out $R/cre21/doe_stats_base.csv"
  "doe_constrained|python tools/doe_stats.py $N --siting constrained --out $R/cre21/doe_stats_constrained.csv"
  "coord_base_pv0.6|python coordination_study.py $N --siting base --pv_penetration 0.6 --participant_billing --results_dir $R/cre21/coord_base_pv0.6"
  "coord_base_pv0.9|python coordination_study.py $N --siting base --pv_penetration 0.9 --participant_billing --results_dir $R/cre21/coord_base_pv0.9"
  "coord_constrained_pv0.6|python coordination_study.py $N --siting constrained --pv_penetration 0.6 --participant_billing --results_dir $R/cre21/coord_constrained_pv0.6"
  "coord_constrained_pv0.9|python coordination_study.py $N --siting constrained --pv_penetration 0.9 --participant_billing --results_dir $R/cre21/coord_constrained_pv0.9"
  "bench_base|bench base"
  "bench_constrained|bench constrained"
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
