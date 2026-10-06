#!/bin/bash
#SBATCH --job-name=fdr_pack
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Packed training on the feeder environment (NEMFeederEnv): several training
# runs share one GPU job. A single run uses ~1 CPU core and ~3 GB RAM and
# leaves the GPU mostly idle (sacct of the 20260904 batch), so packing runs
# multiplies throughput under the 4-concurrent-GPU-job limit.
#
# Time limit: benchmark_train_speed.sh (job 60746661, L40S) measured 20 ms/step
# for one SAC-GNN run (2.4 h per 1500 episodes) and 24-29 ms/step with four
# runs packed (~3.5 h). Adding the feeder env's per-date DOE computation
# (~25 min per run) gives ~4.5 h; 6 h = measured + ~30%. (The feeder env
# places hubs on 29 loaded buses, so there are no 32-hub feeder runs.)
#
# Usage:
#   sbatch --job-name=<name> slurm/train_feeder_pack.sh "<runs>" <batch_tag> [extra train args...]
#     <runs>       space-separated agent:seed[:noedge][:lc<lambda>], e.g.
#                  "sac_gnn:42 sac_gcn:42 sac_flat:42 sac_gnn:42:noedge"
#                  per-run lc overrides --lambda_conf (used by the lambda pilot)
#     <batch_tag>  results subfolder, e.g. perhub_pv0.6_sig0.05_lc1
#     extra args   passed to every run, e.g. --doe_mode per_hub --lambda_conf 1 --episodes 1500
#
# Example (one seed of the main comparison, 4 runs in one GPU job):
#   sbatch --job-name=fdr_s42 slurm/train_feeder_pack.sh \
#       "sac_gnn:42 sac_gcn:42 sac_flat:42 sac_gnn:42:noedge" perhub_lc1 \
#       --doe_mode per_hub --lambda_conf 1 --episodes 1500
#
# Lambda pilot (3 runs, 300 episodes, one GPU job; ~1 h, so shorten the limit):
#   sbatch --job-name=fdr_pilot --time=02:00:00 slurm/train_feeder_pack.sh \
#       "sac_gnn:42:lc0.5 sac_gnn:42:lc2 sac_gnn:42:lc10" pilot --doe_mode per_hub --episodes 300
#
# Results: results/feeder_<batch_tag>_<YYYYMMDD>/<agent>[_noedge][_lc<l>]_seed<N>/

set -euo pipefail

RUNS=${1:?runs required, e.g. "sac_gnn:42 sac_gcn:42"}
TAG=${2:?batch tag required}
shift 2
EXTRA=("$@")

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read -r -a RUN_LIST <<< "$RUNS"
N=${#RUN_LIST[@]}
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / N )); [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
OUT=/scratch2/fr57/zlia0072/ev2gym_training/results/feeder_${TAG}_$(date +%Y%m%d)
LOGS=/scratch2/fr57/zlia0072/ev2gym_training/logs
mkdir -p "$OUT"
echo "Packing $N runs, $THREADS threads each -> $OUT"

PIDS=()
for spec in "${RUN_LIST[@]}"; do
    IFS=: read -r -a F <<< "$spec"
    AGENT=${F[0]}; SEED=${F[1]}; SUFFIX=""; RUN_FLAGS=()
    for opt in "${F[@]:2}"; do
        case "$opt" in
            noedge) RUN_FLAGS+=(--no_edges); SUFFIX+="_noedge" ;;
            lc*)    RUN_FLAGS+=(--lambda_conf "${opt#lc}"); SUFFIX+="_$opt" ;;
            *)      echo "unknown run option '$opt' in '$spec'" >&2; exit 1 ;;
        esac
    done
    NAME=${AGENT}${SUFFIX}_seed${SEED}
    # per-run flags come last so a per-run lc overrides a common --lambda_conf
    FLAGS=(--env feeder --agent "$AGENT" --seed "$SEED" --results_dir "$OUT/$NAME")
    python train_sac_gnn.py "${FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
        > "$LOGS/slurm_${SLURM_JOB_ID:-local}_${NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "started $NAME (pid $!)"
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${RUN_LIST[$i]}"; FAIL=1
    fi
done
exit $FAIL
