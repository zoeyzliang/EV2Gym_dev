# slurm/

Job scripts for training SAC-GNN / SAC-GCN / SAC-Flat on Monash M3, driven
through `train_sac_gnn.py --agent {sac_gnn,sac_gcn,sac_flat}`, at both the
21-hub (`inner_melbourne`) and 32-hub (`greater_melbourne`) scales, plus a
`lambda_conf` sensitivity sweep.

## Time limits (`#SBATCH --time`)

Keep requests tight: a shorter request lets the scheduler backfill the job
into gaps, so it starts sooner. Evaluations finish in about 1 h, so every
`eval_*`/`evaluate_*` script asks for 2 h. Training limits are set from
measured speed (`benchmark_train_speed.sh`) with about 30% headroom; until
that benchmark has run on the vectorised update, `train_coupled_21hub.sh`
asks for 3 days and `pilot_coupled_lambda.sh` for 12 h. Lower the limit of a
job that is already pending with
`scontrol update JobId=<id> TimeLimit=HH:MM:SS` (it can only be lowered).

## GPU pools (three separate 4-GPU allowances)

`sacctmgr show assoc user=zlia0072` lists the QOS available to this account.
Each GPU pool below has its own per-user limit of 4 GPUs, so jobs can be
spread across them (up to 12 GPUs at once, if no account-level cap applies).
`sbatch` options on the command line override the script's `#SBATCH` lines,
including `--constraint=L40S`.

| Pool | sbatch options | GPUs | Max wall time |
|---|---|---|---|
| gpu (default) | *(none)* | L40S (also A100/A40/T4 nodes) | 7 days |
| gpu, any type | `--constraint= --exclude=m3t100` | L40S/A100/A40, **no T4** | 7 days |
| fit | `--partition=fit --qos=fitq --constraint=` | A100, H200 | 1 day |
| m3h | `--partition=m3h --qos=m3h --constraint=` | H100 | 2 days |

`--constraint=` (empty) clears the L40S requirement; M3's node feature names
differ from the GRES names, so `--constraint=A100` is rejected (check with
`sinfo -p fit,m3h -o "%22N %G %f"` if a specific GPU type is ever needed).
Without the L40S constraint the `gpu` partition can also start a job on
`m3t100`, the only T4 node (much slower: the 2026-10-06 pilot landed there),
so add `--exclude=m3t100` whenever the constraint is cleared on `gpu`.
Results do not depend on GPU type; only wall-clock times do, and each pack
job logs its GPU (`nvidia-smi`). Quote timings from the L40S benchmark.

Example: three seeds of E1 started at once on three pools:

```
sbatch --job-name=fdr_s42 slurm/train_feeder_pack.sh "sac_gnn:42 sac_gcn:42 sac_flat:42 sac_gnn:42:noedge" perhub_lc<λ> --doe_mode per_hub --lambda_conf <λ>
sbatch --partition=fit --qos=fitq --constraint= --job-name=fdr_s1 slurm/train_feeder_pack.sh "sac_gnn:1 sac_gcn:1 sac_flat:1 sac_gnn:1:noedge" perhub_lc<λ> --doe_mode per_hub --lambda_conf <λ>
sbatch --partition=m3h --qos=m3h --constraint= --job-name=fdr_s2 slurm/train_feeder_pack.sh "sac_gnn:2 sac_gcn:2 sac_flat:2 sac_gnn:2:noedge" perhub_lc<λ> --doe_mode per_hub --lambda_conf <λ>
```

## Feeder-environment batch (redesign, from 2026-10-06)

Environment: `NEMFeederEnv` (`--env feeder`), spec in `docs/env_redesign_spec.md`.
Several training runs share one GPU job (`train_feeder_pack.sh`): a single
run uses ~1 CPU core and ~3 GB and leaves the GPU mostly idle.

**Step 1: λ pilot** (one GPU job, three penalty levels, 300 episodes):

```
sbatch --job-name=fdr_pilot --time=02:00:00 slurm/train_feeder_pack.sh \
    "sac_gnn:42:lc0.5 sac_gnn:42:lc2 sac_gnn:42:lc10" pilot --doe_mode per_hub --episodes 300
```

Evaluate each pilot run with `evaluate_feeder.py` and choose the **smallest λ
whose limit violations are no worse than the NoV2G baseline's**, then report
arbitrage profit at that λ. The reward itself is not comparable across λ.

**Step 2: main comparison (E1), one job per seed**, 4 runs packed:

```
sbatch --job-name=fdr_s42 slurm/train_feeder_pack.sh \
    "sac_gnn:42 sac_gcn:42 sac_flat:42 sac_gnn:42:noedge" perhub_lc<λ> \
    --doe_mode per_hub --lambda_conf <λ> --episodes 1500
```

E2 (network-aware limits), E3 (forecast error) and the PV sensitivity use the
same script with `--doe_mode network`, `--spatial permuted`,
`--forecast_sigma`, `--pv_penetration`.

**Step 3: evaluation**, per seed (36 stratified held-out days × 3 paired
repetitions + the perfect-foresight LP bound, about 30–45 min):

```
sbatch --job-name=eval_fdr_s42 slurm/evaluate_feeder.sh feeder_perhub_lc<λ>_<date> 42 --doe_mode per_hub
```

Time limits come from the L40S benchmark (job 60746661): one SAC-GNN run is
20 ms/step (2.4 h per 1,500 episodes), four packed runs 24–29 ms/step (~3.5 h),
plus ~25 min per run for the per-date DOE computation. Packed 1,500-episode
jobs request 6 h (measured + ~30%); the 300-episode pilot needs ~1 h, so pass `--time=02:00:00`
on the `sbatch` line (it overrides the script's default).

## Coupled-energy-model batch (2026-10-04) — referee M1/M3/M4 rerun

`EnvConfig.energy_model="coupled"`: delivered energy is
`sign(a)·min(|a_clipped|, n_respond·p_ev)·Δt`, where `p_ev` is the charger
rating capped by SoC headroom, and responders are drawn from plugged-in EVs,
`min(n_connected, n_chargers)`. The draft-v1 model (`sign(a) × n_respond ×
8 kWh`, responders from the enrolled pool) is `"legacy"`.

**The default is still `legacy`** in `EnvConfig`, `train_sac_gnn.py` and
`evaluate.py`, because `sbatch` snapshots a script at submission and the job
then `git pull`s new code — a queued legacy job must not silently switch
model. New runs pass `--energy_model coupled` explicitly. Every older script
is also pinned to `--energy_model legacy`. `evaluate.py` checks each
checkpoint's `config.json` (missing key = legacy) and aborts on an
energy-model, no-edge-slot or graph mismatch.

**Step 1 — pilot λ first.** Under the coupled model Oracle earns only
~$0.7–1.0 per step on normal days, so λ=200 $/kW-step is likely far too large.

```
sbatch --job-name=pilot_lc200 slurm/pilot_coupled_lambda.sh 200
sbatch --job-name=pilot_lc10  slurm/pilot_coupled_lambda.sh 10
sbatch --job-name=pilot_lc1   slurm/pilot_coupled_lambda.sh 1
```

**Step 2 — batch, with the chosen λ** (e.g. 10):

```
sbatch --job-name=cpl_gnn_s42        slurm/train_coupled_21hub.sh sac_gnn  42 10
sbatch --job-name=cpl_gcn_s42        slurm/train_coupled_21hub.sh sac_gcn  42 10
sbatch --job-name=cpl_flat_s42       slurm/train_coupled_21hub.sh sac_flat 42 10
sbatch --job-name=cpl_gnn_noedge_s42 slurm/train_coupled_21hub.sh sac_gnn  42 10 noedge
sbatch --job-name=cpl_gcn_noedge_s42 slurm/train_coupled_21hub.sh sac_gcn  42 10 noedge
# ... repeat for seeds 1 and 2, then per seed:
sbatch --job-name=eval_cpl_s42       slurm/evaluate_coupled_21hub.sh 42 10
```

`--results_dir`: `{agent}[_noedge]_21hub_seed{N}_coupled_lc{λ}_20261004`.
`noedge` = edgeless control (GAT/GCN on a self-loop-only graph): no
hub-to-hub message passing in the encoder or dispatch head; the price head
and critics still mean-pool across hubs.

## Previous batch (2026-09-04) — Q1 full-consistency rerun (legacy energy model)

Every checkpoint referenced anywhere in the thesis/paper should come from
**this** batch, not from `archive_preQ1consistency_20260904/`. This batch
is the first (and, unless another bug is found, only) point at which every
job ran under identical code, specifically:

- DOE violation metric: direction-specific limit (export for discharge,
  import for charge), not the earlier symmetric `min(import, export)` bug.
- DOE violation float32-precision tolerance (`1e-3` kW) — without this,
  any agent dispatching exactly at the DOE boundary (notably OracleMPC)
  registers spurious near-zero-kW "violations" from the observation/action
  space's `float32` round-trip, producing an apparent 0% compliance rate
  that has nothing to do with actual constraint satisfaction. Confirmed
  fixed via `evaluation_DIAGNOSTIC_oracle_check3` (OracleMPC: 0.0% → 100.0%
  DOE compliance, all other metrics ~unchanged from `check2`).
- EV participation model: battery degradation cost term (`-γ·g_{i,t}`,
  Hematiboroujeni et al. 2026) wired through, including in `oracle_mpc.py`
  (previously silently omitted there, meaning Oracle was not actually
  using "the true" participation model it claims privileged access to).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — resolves a CUDA
  allocator crash (`CUDACachingAllocator.cpp` internal assert) that hit
  32-hub SAC-GCN specifically.

**Why a full rerun rather than patching individual checkpoints:** three of
the fixes above landed at different times across earlier submissions,
producing a batch where e.g. one seed of a given agent/scale had the
participation-degradation fix and another didn't (traced via each run's
`git pull`-at-job-start timestamp vs. each fix's commit timestamp). For a
Q1 submission this needed to be eliminated outright, not documented as a
caveat — hence discarding and rerunning everything together.

### Naming convention

Script filenames carry **no date** (they're reusable tools, kept in sync
with `main` via `git pull` at job start). Each script's `--results_dir`
**does** carry a date, since that's a run *output* and its provenance
(which exact commit produced it) matters:

```
{agent}_{scale}_seed{N}_{YYYYMMDD}
```

| Script | `--results_dir` |
|---|---|
| `train_sac_gnn_21hub_seed42.sh` | `sac_gnn_21hub_seed42_20260904` |
| `train_sac_gnn_21hub_seed1.sh` | `sac_gnn_21hub_seed1_20260904` |
| `train_sac_gnn_21hub_seed2.sh` | `sac_gnn_21hub_seed2_20260904` |
| `train_sac_gcn_21hub_seed42.sh` | `sac_gcn_21hub_seed42_20260904` |
| `train_sac_gcn_21hub_seed1.sh` | `sac_gcn_21hub_seed1_20260904` |
| `train_sac_gcn_21hub_seed2.sh` | `sac_gcn_21hub_seed2_20260904` |
| `train_sac_flat_21hub_seed42.sh` | `sac_flat_21hub_seed42_20260904` |
| `train_sac_flat_21hub_seed1.sh` | `sac_flat_21hub_seed1_20260904` |
| `train_sac_flat_21hub_seed2.sh` | `sac_flat_21hub_seed2_20260904` |
| `train_sac_gnn_32hub_seed42.sh` | `sac_gnn_32hub_seed42_20260904` |
| `train_sac_gnn_32hub_seed1.sh` | `sac_gnn_32hub_seed1_20260904` |
| `train_sac_gcn_32hub_seed42.sh` | `sac_gcn_32hub_seed42_20260904` |
| `train_sac_gcn_32hub_seed1.sh` | `sac_gcn_32hub_seed1_20260904` |
| `train_sensitivity_lc100.sh` | `sac_gnn_21hub_seed42_lc100_20260904` |
| `train_sensitivity_lc400.sh` | `sac_gnn_21hub_seed42_lc400_20260904` |

**`lambda_conf=200` is intentionally not in this table.** It's the
`EnvConfig` default, so it's identical in every respect (agent, seed,
zone, `lambda_conf`) to `train_sac_gnn_21hub_seed42.sh` in this same
batch — that run's result **is** the `lc200` sensitivity data point.
Running it a third time would just be training the same configuration
twice.

The 32-hub scripts pass `--zone greater_melbourne`; the 21-hub scripts
omit `--zone` (defaults to `inner_melbourne`).

### Submitting

```bash
for f in slurm/train_sac_gnn_21hub_seed42.sh slurm/train_sac_gnn_21hub_seed1.sh slurm/train_sac_gnn_21hub_seed2.sh \
         slurm/train_sac_gcn_21hub_seed42.sh slurm/train_sac_gcn_21hub_seed1.sh slurm/train_sac_gcn_21hub_seed2.sh \
         slurm/train_sac_flat_21hub_seed42.sh slurm/train_sac_flat_21hub_seed1.sh slurm/train_sac_flat_21hub_seed2.sh \
         slurm/train_sac_gnn_32hub_seed42.sh slurm/train_sac_gnn_32hub_seed1.sh \
         slurm/train_sac_gcn_32hub_seed42.sh slurm/train_sac_gcn_32hub_seed1.sh \
         slurm/train_sensitivity_lc100.sh slurm/train_sensitivity_lc400.sh; do
  sbatch "$f"
done
squeue -u $USER
```

Before submitting, confirm every script has the CUDA allocator fix
(should print nothing — empty output means all have it):

```bash
grep -L "PYTORCH_CUDA_ALLOC_CONF" slurm/train_sac_*.sh slurm/train_sensitivity_*.sh
```

15 jobs against a 4-GPU-per-user concurrency limit (`QOSMaxGRESPerUser`)
→ expect ~4 scheduling waves.

## `archive_preQ1consistency_20260904/`

Every script that trained a checkpoint used anywhere in earlier drafts,
under code predating one or more of the fixes above. Kept (not deleted)
for provenance — if asked "what changed between an earlier and the final
run," the actual script is more useful than a description of one. **Do
not `sbatch` anything in this folder**; results from it are archived
under matching `_OLD_...` suffixes in `results/` and are superseded.

Includes the old `_participation_v2`-suffixed scripts (an earlier,
narrower attempt at exactly this consistency problem — only reran
`seed42` for GNN/GCN, not the full batch) and the old `resume_*.sh`
scripts (superseded; also see the general caution below).

## Evaluation scripts

`evaluate_seed{1,2,42}.sh`, `evaluate_seed{1,2,42}_doefixed.sh`,
`evaluate_sensitivity_lc{100,200,400}.sh` currently point at pre-Q1-batch
checkpoint paths and **need rebuilding** once the current 15-job batch
completes, pointing at the new `*_20260904` results directories. Not yet
done as of this revision — do this before running the real (`n_runs=100`)
final evaluation.

## General fixes (apply to any script in this repo, current or archived)

1. **`#SBATCH --partition=gpu` is required.** Its absence has previously
   caused silent 0%-GPU-utilisation runs.
2. **Conda activation:** use
   `source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate ev2gym`,
   not `source ~/.bashrc; conda activate ev2gym` — the latter risks a
   silent no-op if `.bashrc` returns early for non-interactive shells
   (this exact failure hit job 58510001's first attempt).
3. **Absolute paths only** for `--output`/`--error`/`--results_dir`/
   `CHECKPOINT=` — a relative `results/...` resolves to `EV2Gym_dev/
   results/...` after `cd $WORKDIR`, not
   `/scratch2/fr57/zlia0072/ev2gym_training/results/...`, which is where
   every real checkpoint has ever actually been written.
4. **Include the CUDA sanity check** (`python -c "import torch; ..."`)
   immediately after activating conda, as a first-line-of-log diagnostic.
5. **`--episodes 1500`** — not functionally required (`DEFAULT_CONFIG`
   already defaults to 1500) but kept explicit for self-documentation.
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — see "Current
   batch" above; costs nothing for architectures that don't need it.

## `.gitignore` notes

`data/graphs/*.pkl`, `data/graphs/*_config.json`, `data/nem_cache/`,
`cache/`, and `training_report_*.html` are all gitignored — these are
regenerable build/run artifacts (hub graphs, price caches, visualisation
reports), not source. `logs/` is also gitignored; copy anything from it
you want version-controlled elsewhere.

If you ever add a pattern via `cat >> .gitignore`, check the file already
ends in a newline first — appending without one silently merges onto the
previous line (e.g. `foo/` + `bar.html` → `foo/bar.html`, matching
nothing). Always `git status` after editing `.gitignore` to confirm the
intended files actually disappear from "Untracked files."

## Historical run log (pre-2026-09-04, superseded)

Job IDs and which now-archived script produced them — kept for provenance
only; none of these checkpoints should be cited as final results.

| Job ID | Agent / seed / scale | Notes |
|---|---|---|
| 58510001 | sac_gnn, seed 42, 21-hub | first attempt failed on `conda activate` (`.bashrc` issue, see fix #2 above) |
| 58537944 | sac_gcn, seed 42, 21-hub | |
| 58541330 | sac_flat, seed 42, 21-hub | |
| 58612598 | sac_gnn, seed 42, 21-hub (resumed) | |
| 58747283–58747286 | sac_gnn/gcn, seeds 1–2, 21-hub | |
| 59155196–59155203 | sac_gnn/gcn/flat, seeds 42/1/2, 21-hub | pre-DOE-direction-fix |
| 59313281–59313283 | sac_flat, seeds 42/1/2, 21-hub | post-alpha-clamp-fix retrain |
| 59349278–59349286 | sac_gnn/gcn/flat, seeds 42/1/2, 21-hub | post-DOE-direction-fix, pre-participation-fix |
| 59349287–59349289 | sensitivity lc100/200/400, 21-hub | pre-participation-fix |
| 59351341–59351344 | sac_gnn/gcn, seeds 42/1, 32-hub | first scaling attempt; both GCN jobs crashed (CUDA allocator), and split across pre/post participation-fix code depending on resubmission timing |
| 59524777–59524778 | sac_gnn/gcn, seed 42, 21-hub | `_participation_v2`; first (partial) consistency-fix attempt |
| 59630580–59630583 | sac_gnn/gcn, seeds 42/1, 32-hub | second scaling attempt, CUDA-allocator-fixed but pre-today's-DOE-tolerance-fix |
| 59638436–59638442 | sac_gnn/gcn/flat, seeds 1/2/42, 21-hub | cancelled before completion, superseded by current batch |
