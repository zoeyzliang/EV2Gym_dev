#!/bin/bash
#SBATCH --job-name=gnn_32h_s42_ext_v2
#SBATCH --account=fr57
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s42_ext_v2.out
#SBATCH --error=/scratch2/fr57/zlia0072/ev2gym_training/logs/slurm_%j_gnn_32h_s42_ext_v2.err

set -euo pipefail

WORKDIR=/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev
cd "$WORKDIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ev2gym

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

git pull origin main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Extended-training experiment (companion to seed1's version): this
# seed's eval_log.csv shows mean_net_profit_normal oscillating between
# roughly -$7,500 and -$14,300 for the entire ep800-ep1500 window, with
# no visible upward trend -- unlike seed1, which was still clearly
# rising at the same cutoff. This looks like a stuck poor local optimum
# (DOE compliance stayed at 100% throughout, so this is a policy-quality
# failure, not a constraint-violation failure) rather than an
# insufficient-training-budget problem. Resuming from this seed's own
# final.pt and extending to 2500 total episodes to test directly
# whether it EVER escapes this plateau given more gradient steps, or
# stays stuck -- either outcome is informative for the paper's
# training-stability discussion (Section VII).
# --start_episode omitted deliberately: inferred automatically from the
# checkpoint, avoiding a hardcoded episode number that could be wrong.
python train_sac_gnn.py \
    --agent sac_gnn \
    --zone greater_melbourne \
    --seed 42 \
    --episodes 2500 \
    --resume /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed42_20260904/checkpoints/final.pt \
    --results_dir /scratch2/fr57/zlia0072/ev2gym_training/results/sac_gnn_32hub_seed42_20260904_extended2500_v2
