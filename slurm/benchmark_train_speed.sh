#!/bin/bash
#SBATCH --job-name=bench_speed
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_%x.err

# Training-throughput benchmark (about 30 min). Answers: how fast is one
# training step on an L40S after the vectorised update, and does running
# several training processes on one GPU increase total throughput?
#
# Usage:  sbatch slurm/benchmark_train_speed.sh
# Read:   grep RESULT /scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_<jobid>_bench_speed.out

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

git pull origin main
echo "Code revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 1 process per agent, 21-hub ==="
for agent in sac_gnn sac_gcn sac_flat; do
    python benchmark_train_speed.py --agent $agent --steps 300 --threads 8
done
echo "=== 1 process, 32-hub GNN ==="
python benchmark_train_speed.py --agent sac_gnn --zone greater_melbourne --steps 300 --threads 8

for n in 2 4; do
    echo "=== $n concurrent sac_gnn processes on one GPU (each line = one process) ==="
    for i in $(seq 1 $n); do
        python benchmark_train_speed.py --agent sac_gnn --steps 300 --threads $((8 / n)) &
    done
    wait
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
