# Feeder-grounded DOE dispatch environment (`NEMFeederEnv`): as built

Status: implemented on `main` (commits 182d9e7, 6a2546b, 25854c6), 6 Oct 2026.
This document records the design **as implemented and calibrated**, for the
thesis methodology chapter. The legacy `NEMDOEEnv` is unchanged, so draft-v1
results remain reproducible (`--energy_model legacy`, bit-identical).

## 1. Why: gaps in the legacy environment

| Gap in `NEMDOEEnv` | Consequence | Review ref. |
|---|---|---|
| Traded energy ignored the dispatch magnitude (sign × responders × 8 kWh) | Profits not physical; the DOE never limited energy | M4 |
| The DOE clipped the request, so compliance was guaranteed and free | Trained agents dispatched ~10 kW against a 35 kW mean DOE: a free safety margin, not constraint satisfaction | M1 |
| DOEs i.i.d. per hub; road-distance graph; `edge_attr` unused | Nothing for message passing to exploit (measured: 90–95% of dispatch variation came from global price/time) | M5, M3 |
| Mean SoC reverted to its average (half-life ≈ 33 min); occupancy ignored actions | Near-contextual-bandit; a one-step oracle was near-optimal | M7 |
| Participation re-decided every 5 min; incentive paid on charging too | Unrealistic participation and pricing | M9 |
| No grid physics | DOE compliance not tied to voltages or thermal limits | VIII-A |

## 2. Claims and supporting experiments

| Claim | Supported by |
|---|---|
| **C1.** A realistic, reproducible formulation of public V2G hub dispatch under power-flow-derived, location-specific DOEs on net site flow, with EV sessions, session-level opt-in and real NEM data | §4 model; §7 validation tests; benchmark ordering (§5) |
| **C2.** When graph structure helps: graph vs edgeless vs flat, with spatial structure and inter-hub coupling on vs off | E2 + policy-sensitivity diagnostics |
| **C3.** Constraint satisfaction under genuine uncertainty: profit vs violations as forecast error grows | E3, E5 |

## 3. Scope

- **Implemented:** feeder + power flow; DOEs from voltage and thermal limits; network-aware mode; net-site compliance; EV sessions with opt-in and departure targets; charger throttling; electrical graph; per-hub capacity scaling for all agents; NoV2G, GreedyTOU, RulePrice and perfect-foresight LP baselines; stratified evaluation; packed training.
- **Not implemented (journal version):** forecast-based MPC; a learned safety-layer comparison; the 32-hub study on a larger feeder (the 34-node feeder has 29 loaded buses, fewer than 32 hubs).

## 4. Model

### 4.1 Network, power flow, hub placement (`nem_env/feeder.py`, `nem_env/powerflow.py`)
- **Feeder.** The 34-node radial 11 kV feeder shipped with EV2Gym (from RL-ADN): 33 non-slack buses, 7.8 MW nominal load.
- **Power flow.** Batched fixed-point ("tensor") power flow for constant-power loads, re-implemented in NumPy. It uses the same algorithm and network model as EV2Gym's `GridTensor` and matches it to 5×10⁻⁸ pu on 2,000 scenarios. It is about 0.004 ms per scenario and has no numba/pandapower dependency.
- **Substation set point.** V_s = 1.03 pu, applied exactly: for constant-power loads, the solution with the slack at V_s equals V_s × the solution for powers / V_s².
- **Hub placement (stylised, disclosed).** The 21 hubs are ranked by radial distance from the zone centre. The loaded buses are ranked by electrical distance from the substation, 21 are chosen at evenly spaced quantiles, and hubs are assigned in rank order.
- **Electrical graph.** Hubs are connected if their buses are within 3 hops in the feeder tree (74 directed edges, mean degree 3.5; the road graph has 140). The edge attribute is the normalised path impedance. The encoders currently ignore `edge_attr`.

### 4.2 Background load and PV (`nem_env/grid_profiles.py`)
- **Data.** VIC1 `DISPATCHREGIONSUM.TOTALDEMAND` (5-min) and `ROOFTOP_PV_ACTUAL` (30-min, interpolated to 5-min) for 2022–2024, on the same dates as the RRP.
- **Normalisation.** Demand is divided by its 99th percentile, so κ·PD_b is the bus's peak load. PV is divided by **each calendar year's** 99.5th percentile, because rooftop capacity grew strongly over 2022–2024 and normalising by the overall peak would give the training years systematically less PV than the test year.
- **Background per bus:** `P_bg = κ·PD_b·D_t − π·κ·PD_b·S_t`, `Q_bg = κ·QD_b·D_t`.
  - κ = 0.7 (calibrated below).
  - π = PV peak as a fraction of the bus peak load: 0.6 by default; sensitivity at 0.3 and 0.9.
- **Calibration (no EVs, 3 years at 15-min).** With κ = 0.7 and π = 0.6:
  - minimum bus voltage at the evening peak: 0.958 pu (1st percentile);
  - maximum bus voltage at midday: 1.045 pu (99th percentile);
  - the background alone never breaches [0.95, 1.05].
  - At π = 0.9, the background exceeds 1.05 pu on 1.5% of time and reverse flow occurs on 11%.
- **Forecast vs realised.** Realised = forecast × (1 + ε). ε follows an AR(1) process (ρ = 0.9), half common to all buses and half bus-specific, with σ = 5% by default (sensitivity 0 and 10%).

### 4.3 DOEs (`Feeder.doe_day`, `Feeder.hosting_day`)
- **Limits.** Bus voltage in [0.95, 1.05] pu. Section thermal rating = 1.0 × the section's downstream design peak load (design rule; the feeder data has no ratings). Section flows are computed as radial sums of downstream bus powers.
- **Individual hosting capacity h_i.** The largest import (or export) at hub i, with all other hubs at baseline, that keeps every bus voltage and section flow within limits over the 30-min window of forecast background. Found by a two-stage grid search on batched power flow. A limit already breached by the background alone is not attributed to the hub.
- **Per-hub DOE (default mode).** `DOE_i = min(cap_i, κ*·h_i)`, where κ* ≤ 1 is the largest common factor under which all hubs at their DOE are jointly feasible under the forecast. Published every 30 min and held for 6 steps.
- **Network-aware mode (E2).** Limits are the individual hosting capacities h_i, without the joint scaling. Their sum can exceed what the feeder takes jointly, so hubs must coordinate. Violations are measured physically: voltage excursions and section overloads from the realised power flow.
- **What the DOEs look like** (π = 0.6):
  - Sunny summer day: mean export DOE 58% of capacity at noon, 86% at 9 am, 96% at night. Import DOE 93% at the evening peak.
  - Import is limited on about 16% of hub-steps, mostly at electrically weak buses.
  - Joint coupling (κ* < 1) occurs in 10% of evening-peak windows and 14% of midday windows, with κ* ≈ 0.7 when coupled. At π = 0.9 this rises to 28–37% of midday windows (κ* ≈ 0.44), and export DOEs reach zero at noon.
- **Caching.** DOEs depend only on the date and the forecast, so they are cached per date (about 1.5 s per date).
- **Spatial ablation (E2, `--spatial permuted`).** Hubs' limit fractions are permuted independently in each 30-min block. This keeps each step's distribution of limits and destroys the spatial structure.

### 4.4 EV sessions (`nem_env/sessions.py`)
Session sampling mirrors EV2Gym's public scenario (ElaadNL distributions shipped with EV2Gym):
- **Arrivals.** A free port receives an EV with a per-step probability shaped by the public arrival curve (weekday/weekend, 15-min). It is scaled so that 2.5 arrivals are *offered* per port per day; realised: about 1.4 sessions per port per day and 26% mean occupancy, since ports are often busy.
- **Energy demand** ~ N(mean(arrival half-hour), 0.5·mean), with a minimum of 5 kWh. **Dwell time** ~ N(mean(arrival half-hour), 0.2·mean) h.
- **EV models** are sampled by registrations from `ev_specs_v2g_enabled2024.json` (battery 46–77 kWh). Charge power = min(DC limit, charger kW). **Discharge ≈ 10–11 kW**, the real V2G limit for these models.
- **Battery.** Efficiency 0.95 each way. SoC floor 0.2 for V2G. Target energy = arrival energy + demand (capped at capacity).

### 4.5 Participation (one decision per session)
- At arrival, the owner is offered the current incentive c_t ($/kWh, paid on **discharged** energy only, contracted for the session). They accept with the logistic ρ(c, d_h, SoC_arr, g), using the unchanged β and γ: β₀ −2.2, β₁ 0.008 per $/MWh, β₂ −0.2, β₃ 1.5, γ 0.14.
- **Measured opt-in rates:** 11% at $0/kWh, 39% at $0.20, 78% at $0.50.
- **Participants** are dispatchable. **Non-participants** charge only, as fast as the site allows.

### 4.6 Action, allocation, safety projection (`NEMFeederEnv.step`)
- **Action.** For each hub, a normalised setpoint in [−1, 1] for its participants (+ = discharge), scaled by the hub's **own** capacity; plus one network-wide incentive in [0, 0.5] $/kWh.
- **Per-EV bounds each step.** These keep SoC within [floor, capacity] and keep the departure target reachable. **Forced charging** applies when an EV has no slack left. Participants never leave under-charged.
- **Safety projection (the deployed clip).**
  1. The participant setpoint is limited to [−export limit, import limit − expected in-step arrivals].
  2. It is then limited to the fleet's feasible bounds; forced charging overrides the limit, which is then a genuine violation.
  3. Non-participants share the remaining import headroom, least-laxity-first. This is how DOE-compliant (CSIP-AUS) chargers would throttle them.
- **Uncertainty.** EVs arriving *during* the step (with non-participants charging at once) and the background forecast error mean the realised outcome can differ from the projected one.

### 4.7 Reward ($ per step)
`r = −RRP·Σ flex·Δt/1000 − Σ rate·E_dis − λ_u·E_unmet − P_limit (− P_terminal at the end of day)`
- **Wholesale term.** Participants' net energy at RRP: charging is a cost, discharging a revenue. Non-participants' energy is a pass-through and is excluded.
- **Unmet energy.** λ_u = $1/kWh short of target at departure, for **all** customers (non-participants can be short when throttled).
- **P_limit.** Per-hub mode: λ_doe × kWh of realised net-site-flow violation. Network mode: λ_doe × kWh of section overload + λ_v × pu voltage violation. **λ_doe is chosen by the pilot (§10).**
- **Terminal term.** Energy participants still need at the end of the day is charged at the day's mean RRP, so draining batteries late in the day is not free.

### 4.8 Observation (17 features per hub; global values broadcast)
1. Import limit / capacity
2. Export limit / capacity
3. Uncontrolled load / capacity
4. Participants per port
5. Fleet max charge / capacity
6. Fleet max discharge (negative = forced charging) / capacity
7. Energy still needed (hours at full power)
8. Mean hours to departure / 24
9. Port occupancy
10. Capacity / 200
11. Last hub bus voltage deviation × 10
12. RRP / 1000
13. Mean RRP over the last 30 min / 1000
14. sin(hour)
15. cos(hour)
16. Weekend flag
17. Time to the next DOE update

## 5. Baselines (`baselines/feeder_baselines.py`)

| Baseline | Rule |
|---|---|
| **NoV2G** | Participants charge as soon as possible; no incentive. Reference for V2G value and for the λ selection rule. |
| **GreedyTOU** | Discharge at full setpoint when RRP ≥ $264/MWh, charge when ≤ $9/MWh (90th/25th percentiles of the 2022–23 *training* prices), otherwise idle; incentive $0.15/kWh. |
| **RulePrice** | Port of the draft's baseline: incentive = 50% of RRP (capped), discharge if RRP > incentive, else charge. |
| **Perfect-foresight LP** | Upper bound over constant-incentive policies. Knowing the day's prices, arrivals, opt-ins and limits, it solves the optimal per-EV schedule (HiGHS via SciPy). It is solved for c ∈ {0, 0.1, …, 0.5} and the best is kept; about 5–6 s per LP. Caveat: the RL agent may vary the incentive over time, which can change who opts in, so this bounds constant-incentive policies. |

Example (one day, seed 1):
- 13 Feb 2024 (spike to $16,600/MWh): GreedyTOU −$1,356, LP bound +$3,188. Greedy discharges early and must recharge during the same multi-hour spike, which is evidence of intertemporal structure (M7).
- 20 Dec 2023 (sunny): NoV2G −$13.5, GreedyTOU −$7.7, LP +$49.1.

## 6. Experiments (all via `train_feeder_pack.sh` / `evaluate_feeder.sh`)

| Exp. | Question | Setting |
|---|---|---|
| **E1** | Main comparison | `--doe_mode per_hub`; SAC-GNN, SAC-GCN, SAC-Flat, SAC-GNN-NoEdge; 5 seeds |
| **E2** | When does the graph help? | `--doe_mode network` (coupling on) and `--spatial permuted` (structure off), graph vs edgeless vs flat; π = 0.6 and 0.9 |
| **E3** | Uncertainty | `--forecast_sigma 0 / 0.05 / 0.10` |
| **PV** | Sensitivity | `--pv_penetration 0.3 / 0.6 / 0.9` |
| **E4** | Participation misspecification (evaluation only) | `evaluate_feeder.py --beta1_scale / --beta3_scale / --gamma_scale` |
| **E5** | Physical outcome | realised voltage and overload metrics, reported for every run |

- **Evaluation.** Held-out 2024 days, never including the 5 validation days used for checkpoint selection. 3 paired repetitions per day. Report seed-level statistics (IQM, bootstrap CIs over seeds) and pre-penalty economics separately from penalties.
  **Pre-registered day-selection rule** (7 Oct 2026, written before the completed Aug–Dec 2024 prices were inspected; the pilot evaluation showed that per-month percentiles alone select no volatile or extreme day):
  - **Representative set (main results):** for each of the 12 months, the days at the 20th, 50th and 90th percentile of that month's daily RRP standard deviation. That is 36 days, labelled by volatility tier (calm < $100/MWh ≤ normal < $500 ≤ volatile < $2,000 ≤ extreme, the curriculum thresholds) and weekday/weekend.
  - **Stress set (reported separately):** every remaining 2024 day with daily RRP standard deviation ≥ $500/MWh (the volatile and extreme tiers). If there are more than 8, the 8 with the highest standard deviation. As in the draft, stress results are never pooled with representative results.
  - **λ selection (§10)** uses the representative set only.
- **Diagnostics (to port).** Neighbour-sensitivity test and attention entropy, as run on the legacy checkpoints.

## 7. Validation tests (`tests/`, local; `tests/` is gitignored)

`test_feeder_env.py` (8 checks, all passing):
1. Power flow converges and the background stays within limits on sample days.
2. DOEs are jointly feasible under the forecast (all hubs at their DOE stay within voltage and thermal limits).
3. Hosting capacity is never tighter than the DOE, and coupling exists on a high-PV day.
4. The permuted mode preserves each step's distribution.
5. The projection formula is exact; SoC stays in range; participants are never short.
6. The terminal term is non-negative and charged once.
7. The LP bound dominates every baseline.
8. The NumPy power flow matches EV2Gym's `GridTensor`.

`test_energy_model.py`: 16 checks (coupled/legacy energy model, Oracle Monte Carlo, edgeless isolation, vectorised SAC update equivalence, config safeguard).

## 8. Data and assumptions (all disclosed)

| Item | Source | Note |
|---|---|---|
| RRP | AEMO DISPATCHPRICE (NEMOSIS) | June 2022 market suspension (12–24 Jun) excluded from training |
| Demand, PV shapes | AEMO DISPATCHREGIONSUM, ROOFTOP_PV_ACTUAL, VIC1 | same dates as RRP; PV normalised per year |
| Feeder | EV2Gym/RL-ADN 34-node | stylised placement of real hubs; thermal ratings by design rule |
| Sessions | ElaadNL public (via EV2Gym) | Dutch; no public Australian equivalent identified |
| EV models | EV2Gym V2G-enabled 2024 specs | real V2G discharge limits |
| Hubs | OpenChargeMap | V2G capability assumed |
| Participation | logistic (β, γ as in §4.5) | uncalibrated, so E4 sensitivity is run |

## 9. Code
- `nem_env/grid_profiles.py`, `feeder.py`, `powerflow.py`, `sessions.py`, `feeder_env.py`
- `baselines/feeder_baselines.py`
- `train_sac_gnn.py --env feeder` (+ `--doe_mode --spatial --forecast_sigma --pv_penetration --graph`)
- `evaluate_feeder.py`
- `slurm/train_feeder_pack.sh`, `slurm/evaluate_feeder.sh`, `slurm/benchmark_train_speed.sh`

## 10. Compute and timeline
- **Measured on the L40S** (benchmark job 60746661):
  - SAC-GNN 20.4 ms/step (2.4 h per 1,500 episodes); SAC-GCN 16.1 ms; SAC-Flat 6.7 ms.
  - Four runs packed on one GPU: 24–29 ms/step each (≈3× throughput).
  - The feeder env adds about 25 min per run for per-date DOE computation.
  - Packed jobs request 8 h.
- **Plan:**
  1. λ pilot: λ_doe ∈ {0.5, 2, 10} $/kWh, 300 episodes (job 60757441).
  2. Choose the **smallest λ whose limit violations are no worse than NoV2G's**.
     **Pre-registered criterion** (recorded 7 Oct 2026, before the per-run
     pilot data were analysed):
     - **Unit of comparison.** One paired (day, repetition) episode. Every agent faces the identical environment realisation (same seed).
     - **Measure.** d = limit_viol_kwh(λ-agent) − limit_viol_kwh(NoV2G) per paired episode; mean d with a 95% bootstrap CI (10,000 resamples over the paired episodes).
       *Multi-seed refinement* (added 7 Oct 2026, before any full-length λ-study data existed): with several training seeds, the CI is a **hierarchical bootstrap**. Each of the 10,000 resamples draws seeds with replacement, then paired episodes with replacement within each drawn seed. Mean d is the mean over seeds of each seed's mean d. So the uncertainty includes seed-to-seed variation, not only day-to-day. Implemented in `decide_lambda.py`, written and tested before the λ-study runs.
     - **"No worse than NoV2G"** means the CI's upper bound is ≤ 0.5 kWh/day: a margin of about 3% of NoV2G's ~15 kWh/day, set in advance as practically negligible.
     - **Choice.** Among the λ values meeting this, take the smallest. If none meets it, take the λ with the smallest mean d and report that compliance is not matched.
     - **Reporting.** Pre-penalty profit (arbitrage_profit) relative to NoV2G is reported for every λ with the same paired CI, but does not decide λ.
     - **Data.** Applied to the pilot evaluation on the full stratified evaluation set (36 days), after the 2024 price data are completed (§8).
  3. E1 (5 seeds), then E2, E3, PV, E4.
  4. Diagnostics.
  5. Writing.
- **Fallback.** If the feeder results are not ready by **20 Oct**, the thesis reports the corrected legacy/coupled study and moves the redesign to future work.

## 11. Decisions taken
1. Stylised placement of real hubs on the 34-node test feeder: accepted, and disclosed.
2. PV penetration 0.6 (sensitivity 0.3 / 0.9) and forecast error 5% (sensitivity 0 / 10%).
3. 2.5 offered sessions per port per day.
4. Unmet-energy cost λ_u = $1/kWh.
5. 5 seeds per configuration (affordable after the speed-up).
