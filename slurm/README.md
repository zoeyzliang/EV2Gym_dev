# slurm/

Job scripts for training the SAC-GNN / SAC-GCN / SAC-Flat baselines on
Monash M3, driven through `train_sac_gnn.py --agent {sac_gnn,sac_gcn,sac_flat}`.

## Fixes applied (this revision)

**Path fix:** `--results_dir` (and `CHECKPOINT=` in resume scripts) used a
relative `results/...` path, which after `cd $WORKDIR` resolves to
`EV2Gym_dev/results/...` — NOT where any actual checkpoint has ever been
written. Every real training run to date wrote to
`/scratch2/fr57/zlia0072/ev2gym_training/results/...` (one level above the
repo). All `--results_dir` and `CHECKPOINT=` values are now absolute paths
pointing there.

### Original fixes

The first draft of these scripts was reconstructed from `logs/*.err` and
`train_sac_gnn.py` without a working script to copy from, and missed a
few things that every previously *successful* submission had. Fixed here:

1. **`#SBATCH --partition=gpu` was missing from all 9 scripts.** Every job
   that actually ran on GPU used this. Without it, scheduling behaviour is
   unverified and could silently reintroduce the 0%-GPU-utilisation issue
   this repo already hit once (see `baselines/gnn_rl/agent.py` device fix).
2. **Conda activation used `source ~/.bashrc; conda activate ev2gym`**,
   which risks silently no-op'ing if `.bashrc` returns early for
   non-interactive shells (a common default). This exact failure mode is
   already documented below for job 58510001. Reverted to the
   proven-working `source activate ev2gym`.
3. **`--output`/`--error` used relative `logs/...` paths**, fragile to
   whatever directory `sbatch` is invoked from. Changed to absolute paths
   under `/scratch2/fr57/zlia0072/ev2gym_training/logs/`.
4. **Missing the CUDA sanity check** (`python -c "import torch; ..."`)
   that every prior working script had as an immediate first-line-of-log
   diagnostic. Added back after `source activate ev2gym`.
5. **Missing explicit `--episodes 1500`** on the `train_*` scripts (not
   `resume_*`, which correctly infer remaining episodes from the
   checkpoint). Not a functional bug — `DEFAULT_CONFIG` already defaults
   to 1500 — but added back for self-documentation.

## Before first use

These scripts were reconstructed from `logs/*.err` and `train_sac_gnn.py`,
not copied from an existing submission script (none was checked into the
repo). Confirmed against `sacctmgr -p show assoc user=zlia0072` /
`sacctmgr -p show qos`:

- `#SBATCH --account=fr57` - confirmed; the only account on this login.
- `#SBATCH --qos=normal` - no dedicated GPU partition exists on this
  allocation (M3 schedules by QOS here); `normal` allows `gres/gpu=4`
  and has the longest wall-time budget (`MaxWall=7-00:00:00`) of the
  QOSs available to this account (`desktopq, fitcq, fitq, irq, m3h,
  normal, rtq, shortq`).
- `#SBATCH --time=7-00:00:00` - set to `normal`'s max wall time so long
  training runs (1500 episodes logged at ~2-4 min/ep in the past) don't
  get killed early. Lower it if you want jobs to queue faster.

All scripts `cd` to `/fs04/scratch2/fr57/zlia0072/ev2gym_training/EV2Gym_dev`
and `git pull origin main` before running, matching what past job logs show.

## Scripts → results directory

| Script | `--results_dir` |
|---|---|
| `train_sac_gnn_seed42.sh` | `results/sac_gnn_real_seed42` |
| `train_sac_gnn_seed1.sh` | `results/sac_gnn_real_seed1` |
| `train_sac_gnn_seed2.sh` | `results/sac_gnn_real_seed2` |
| `train_sac_gcn_seed42.sh` | `results/sac_gcn_real_seed42` |
| `train_sac_gcn_seed1.sh` | `results/sac_gcn_real_seed1` |
| `train_sac_gcn_seed2.sh` | `results/sac_gcn_real_seed2` |
| `train_sac_flat_seed42.sh` | `results/sac_flat_real_seed42` |
| `resume_sac_gnn_seed1.sh` | `results/sac_gnn_real_seed1` (from a checkpoint) |
| `resume_sac_gcn_seed1.sh` | `results/sac_gcn_real_seed1` (from a checkpoint) |

The resume scripts hard-code `CHECKPOINT=.../checkpoints/best.pt` — edit
that line to point at whichever `episode_N.pt` / `best.pt` you want to
continue from before submitting.

## Which script produced which historical results (from `logs/*.err`)

`logs/` is gitignored, so these `.err` files aren't tracked, but they're
still on disk locally and record what actually ran:

| Job ID | Log file | Agent / seed | Notes |
|---|---|---|---|
| 58747283 | `slurm_58747283_sac_gnn_seed1.err` | sac_gnn, seed 1 | matches `train_sac_gnn_seed1.sh` |
| 58747284 | `slurm_58747284_sac_gnn_seed2.err` | sac_gnn, seed 2 | matches `train_sac_gnn_seed2.sh` |
| 58747285 | `slurm_58747285_sac_gcn_seed1.err` | sac_gcn, seed 1 | matches `train_sac_gcn_seed1.sh` |
| 58747286 | `slurm_58747286_sac_gcn_seed2.err` | sac_gcn, seed 2 | matches `train_sac_gcn_seed2.sh` |
| 58537944 | `slurm_58537944_sac_gcn_seed42.err` | sac_gcn, seed 42 | matches `train_sac_gcn_seed42.sh` |
| 58541330 | `slurm_58541330_sac_flat_seed42.err` | sac_flat, seed 42 | matches `train_sac_flat_seed42.sh` |
| 58510001 | `slurm_58510001_real_seed42.err` | sac_gnn, seed 42 | matches `train_sac_gnn_seed42.sh`; one attempt failed on `conda activate` before succeeding — this is exactly the `.bashrc` risk fixed in this revision (see "Fixes applied" above); current script uses `source activate ev2gym` directly, matching the run that succeeded |
| 58612598 | `slurm_58612598_sac_gnn_resume.err` | sac_gnn, seed 42 (resumed) | a resume of the seed-42 run, not seed 1 — if you need to resume seed 42, copy `resume_sac_gnn_seed1.sh` and change `--seed`/`--results_dir` accordingly |
| 57942364 | `slurm_57942364.err` | sac_gnn | wrote to `results/sac_gnn_doe_m3/`, an earlier/legacy results layout that predates the `*_real_seed*` convention — no current script reproduces this exactly |
| 58308859 | `slurm_58308859.err` | sac_gnn | same legacy `results/sac_gnn_doe_m3/` run, continued |

## Submitting

```bash
sbatch slurm/train_sac_gnn_seed42.sh
squeue -u $USER
```

Logs land in `logs/slurm_<jobid>_<label>.out` / `.err`. `logs/` is
gitignored, so copy anything you want to keep in version control
elsewhere (or remove `logs/` from `.gitignore` if you want it tracked).
