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
- **Participant billing** (`--participant_billing`; added 8 Oct 2026, after E1 v1 and before any v2 run). In v1 the aggregator bought all of a participant's charging and nobody paid it back, while non-participants' energy was a pass-through. Participants therefore charged for free, every extra participant added its whole energy bill, and the best incentive was pushed towards $0 (the LP chose $0 on 65 of 108 representative episodes, and the MPC chose $0 on validation). The opt-in model, which responds only to the discharge incentive, is inconsistent with free charging.
  - **Rule:** each participant pays for the energy it asked for, up to what it received, i.e. (min(E_dep, target) − E_arr)⁺ / η. The price is the day's mean RRP, the terminal-cost convention. It is billed at departure, or for the whole request if still connected at the end of the day.
  - **Negative-mean days:** the price is max(0, day mean), as for the terminal cost. On a day whose mean RRP is ≤ 0 participants are billed nothing; a retailer would not pay customers to charge.
  - **Effect:** for a given set of participants the bill is a constant (targets are met), so it changes no dispatch incentive. It changes the reward level and the economics of the incentive decision. `r_billing` is reported and included in `arbitrage_profit` and `net_economic`.
  - **Verified:** default off reproduces v1 bit-for-bit. With billing on, the LP bound, the MPC and the env agree: MPC with perfect prices stays below the LP bound in every check, and the bill is identical across policies at a given incentive.

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

### 4.9 Agent–environment interface fix (8 Oct 2026, after E1 v1)
**Finding (E1 v1 evaluation).** At λ = 0.5, SAC-GNN, SAC-GCN and SAC-GNN-NoEdge close −4% to −1% of the gap between NoV2G and the perfect-foresight LP bound on representative days, and about 5% on stress days. Comparison is on the LP objective's terms: economics minus unmet energy, with no limit penalty. The LP's best incentive is $0 on 65 of 108 representative episodes, so the value is reachable with the participants the agents already get.

**Diagnosis (training days only).**
- **(a) Action range.** Setpoints were scaled by hub capacity, but the participants' feasible range averages 2.4% of capacity (median 0). The projection clips almost the whole action range, so most actions have identical effects.
- **(b) Reward scale.** Per-step rewards are about $0.06 (std $0.34). The legacy running-std normaliser has a floor of 1.0, so it never rescales them, and the SAC entropy weight sits at its 0.05 floor (α = 0.0500 in every log). The entropy bonus dominates the reward. Both floors were set for the legacy env, whose rewards were thousands of times larger.
- **(c) Flat agent confound.** SAC-Flat stored raw rewards, while SAC-GNN/GCN stored normalised and clipped rewards. So E1 v1's architecture comparison also differed in reward processing.

**Fix (options; defaults reproduce v1 exactly).**
- **A. `--action_scale feasible`:** a ∈ [0, 1] → 0 … the participants' maximum discharge; a ∈ [−1, 0] → 0 … their maximum charge; 0 = idle. The safety projection is unchanged. The ±1/0 baselines are bit-identical under both scales (verified on 4 days × 2 reps against E1 v1), so baseline and LP results stand.
- **B. `--reward_scale 1 --alpha_min 0.001`:** every agent (GNN, GCN, Flat) stores clip(r, ±10), so all agents get identical reward processing. Under random actions on 30 training days, the 99.9th percentile of |r| is 9.2, so the clip binds on about 0.1% of steps. The entropy floor is lowered from 0.05 to 0.001, so automatic tuning can work.
- **Evaluation** checks `action_scale` against each checkpoint's config (`ENV_KEYS`).

**Pilot (pre-registered 8 Oct 2026, before any pilot data).**
- One packed L40S job, λ = 0.5, 500 episodes: A+B at seeds 42 and 1, A only at seed 42, B only at seed 42.
- **Measure:** best validation normal-day reward (`mean_net_profit_normal`, the training script's 4 normal validation days, seeded) within 500 episodes. **No test days are used.**
- **References:**
  - NoV2G on the same validation days: −10.76 $/day;
  - the v1 λ-study runs `sac_gnn_lc0.5_seed42/seed1`: best validation value over all 1,500 episodes, **−11.8 for both** (read from the M3 logs before the pilot was submitted). v1 never beat NoV2G on validation, so NoV2G's −10.76 is the binding threshold.
- **Decision:** adopt A+B if **both** A+B runs exceed their v1 counterpart's best **and** NoV2G. If only A or only B meets this, adopt that one alone. If neither does, keep v1 and report the failure-to-learn finding.
- **If adopted:** rerun the λ study (9 runs) and E1 (17 new + 3 λ-study runs) under the same pre-registered λ rule and evaluation days. v1 results are kept and reported as the "capacity-scaled interface" ablation.

**Pilot outcome (job 60810857, 8 Oct 2026; rule applied as written): not adopted.** Best validation normal-day reward within 500 episodes, against the thresholds NoV2G −10.76 and v1 best −11.8:

| Run | Best | Passes | Validation curve (10 checks) |
|---|---|---|---|
| A+B seed 1 | −7.5 | yes | −24.3 … −7.5 zigzag, best at the last check; no trend (best-of-10 noise) |
| A+B seed 42 | −11.8 | no | −131.8 … −11.8; unstable (critic loss 871 at ep 500) |
| A only seed 42 | −12.8 | no | flat −15.0 … −12.8 (stable, passive) |
| B only seed 42 | −12.4 | no | declines to −72.1 (unstable) |

Both A+B seeds were required, so the fix is not adopted and v1 remains the reference RL result. Fix A is harmless but does not produce learning. Fix B destabilises training. No further interface iteration is planned: the curves give no evidence that a reward-scale change would succeed, and further tuning would be open-ended.

### 4.10 Final RL stage: SAC-NoEdge-F (recorded 8 Oct 2026, before any code or run)
**Why this stage.** The coordination LP and the E1 ablation show that message passing adds nothing here, so the graph is not the issue. But "SAC fails to capture the value" is not yet a firm conclusion. Two plausible causes are untested:
- **Information:** RL sees no price forecasts or next-window DOEs, while MPC does.
- **Signal:** the controllable reward is small beside uncontrollable terms.

This is the last RL iteration; its outcome is reported whichever way it goes.

**Agent.** SAC with the shared per-hub network (GAT with self-loops only, i.e. NoEdge), plus:
- **Information parity** (`--forecast_features`). Seven more node features (17 → 24), from the same sources and timing rules as MPC-Predispatch: the latest predispatch run published at least 5 min before the interval starts. Features, broadcast to every hub, RRP/1000 clipped like feature 11:
  - forecast mean RRP over the next 1 h and next 3 h;
  - forecast max and min over the next 6 h;
  - forecast mean to the end of the day.

  Plus the hub's next-window import and export limits / capacity. Predispatch history 2022–24 (`nem_env/predispatch.py`).
- **Baseline-subtracted learning signal** (`--baseline_reward`). Each training step stores r_t − b_t, where b_t is the reward of NoV2G (all hubs −1, incentive 0) at step t on a shadow copy of the same episode. The copy is made at reset, so arrivals, EV draws, opt-in random draws and forecast errors are identical.
  - b_t never depends on the agent's actions, so Σ(r − b) differs from Σr by a policy-independent amount, and the optimal policy is unchanged.
  - Logging, validation and checkpoint selection use the true reward r.
- **Environment:** participant billing on, `--action_scale feasible`, λ_doe = 0.5 (v1 λ; not re-selected).
- **Reward processing:**
  - fixed scale k, set before the pilot by the earlier rule: the 99.9th percentile of |k(r − b)| under random actions on 30 training days equals the clip, 10;
  - entropy floor α_min ∈ {0.001, 0.01} compared in the pilot.

**Values fixed before the pilot** (8 Oct 2026):
- Reward scale **k = 1.34**: |r − b| under random actions on 30 training days has 99.9th percentile 7.47, and 10/7.47 = 1.34.
- NoV2G validation threshold **−5.15 $/day** (normal days; billing on, λ 0.5, forecast env).
- Predispatch coverage: the AEMO archive has no 2022-10 predispatch price file. **33 training days** where some decision lacks a run published within 2 h are excluded from sampling. 2024 (validation and test) is fully covered.

**Pilot (validation days only).**
- One packed job, 500 episodes: α_min 0.001 and 0.01, each at seeds 42 and 1.
- **Measure:** best validation normal-day reward.
- **Threshold:** NoV2G's validation reward in the same environment (billing on, λ 0.5), computed before the pilot.
- **Adopt** the α_min whose two seeds both beat NoV2G; if both qualify, the higher mean. If neither qualifies, stop: the conclusion is that RL fails even with information parity and a cleaner signal.

**If adopted.** 5 seeds × 1,500 episodes. Evaluated on the pre-registered days against MPC-Predispatch (incentive 0.2, billing on, the same information) and the LP bound. Main comparison: net economics minus unmet energy vs NoV2G, with the seed-level hierarchical bootstrap.

## 5. Baselines (`baselines/feeder_baselines.py`)

| Baseline | Rule |
|---|---|
| **NoV2G** | Participants charge as soon as possible; no incentive. Reference for V2G value and for the λ selection rule. |
| **GreedyTOU** | Discharge at full setpoint when RRP ≥ $264/MWh, charge when ≤ $9/MWh (90th/25th percentiles of the 2022–23 *training* prices), otherwise idle; incentive $0.15/kWh. |
| **RulePrice** | Port of the draft's baseline: incentive = 50% of RRP (capped), discharge if RRP > incentive, else charge. |
| **Perfect-foresight LP** | Upper bound over constant-incentive policies. Knowing the day's prices, arrivals, opt-ins and limits, it solves the optimal per-EV schedule (HiGHS via SciPy). It is solved for c ∈ {0, 0.1, …, 0.5} and the best is kept; about 5–6 s per LP. Caveat: the RL agent may vary the incentive over time, which can change who opts in, so this bounds constant-incentive policies. |

| **Forecast MPC** (`baselines/forecast_mpc.py`, added 8 Oct 2026) | What a real aggregator could run. Every 30 min it solves a shrinking-horizon LP to the end of the day for the connected participants and follows the plan until the next re-plan. **Information at each decision:** the current interval's RRP; for later intervals, the latest AEMO **predispatch** run published at least 5 min before the interval starts (`nem_env/predispatch.py`, full 2024 run history from NEMweb MMSDM, 48 runs/day); connected EVs' energy, departure and target (declared at plug-in); the day's DOE schedule (published day-ahead). It does not know future arrivals or opt-ins. **Incentive:** constant, chosen on the validation days by `select_mpc_incentive.py` (grid 0–0.5, mean reward on the normal validation days). Chosen: $0.0 with v1 accounting, **$0.2 with participant billing**. |
| **MPC-PerfectPrice** (diagnostic) | Same controller with realised prices. With the LP bound and MPC-Predispatch it separates the value of price foresight from that of arrival foresight. |

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
- **Economic reporting rule** (added 7 Oct 2026, after the λ-study evaluation and before any E1 run was submitted; it changes only what is reported, not the environment, reward, training or λ choice). The λ study showed that pre-penalty profit (`arbitrage_profit`) leaves out `p_terminal`: the cost of energy still owed to participants at the end of the day, bought at the day's mean price. That is a real energy purchase, not a penalty, and agents that discharge late shift cost into it. For λ = 0.5 on representative days, relative to NoV2G, pre-penalty profit is +2.29 $/day but net of terminal cost it is −0.57 [−1.40, +0.23] $/day.
  - **Main economic metric:** `net_economic` = `arbitrage_profit` − `p_terminal`, reported for every agent.
  - `arbitrage_profit` and `p_terminal` are reported alongside, so the split stays visible.
  - Penalties (`p_unmet`, `p_limit`) are still reported separately and never mixed into the economic metric.
  - Implemented in `evaluate_feeder.py`: a per-episode `net_economic` column, and `net_economic` and `p_terminal` in `summary.csv`. For the λ-study files it is computed from the same columns.
- **Benchmark and value of coordination** (recorded 8 Oct 2026, after the interface pilot was not adopted and before these full runs; checked on 2 days × 2 PV levels only).
  - **Benchmark** (`slurm/evaluate_baselines.sh`): NoV2G, GreedyTOU, RulePrice, MPC-Predispatch, MPC-PerfectPrice and the LP bound on the pre-registered days × 3 reps. Run twice: **with participant billing** (MPC incentive 0.2, the main benchmark) and **with v1 accounting** (MPC incentive 0.0, directly comparable to the v1 RL agents, which were trained without billing).
  - **Value of coordination, C2 answered with optimisation** (`coordination_study.py`, `baselines/coordination.py`):
    - **Comparison:** the perfect-foresight LP with per-hub DOEs (κ*·h_i) against the same LP with the feeder's joint constraints. These are hub capacity, bus voltages in [0.95, 1.05] pu (linearised at the forecast background), and exact radial section flows, each at the 30-min window's worst-case background, as in the DOE rule.
    - **Measure:** value of coordination = LP_network − LP_perhub (best constant incentive each), per (day, rep), at π = 0.6 and 0.9, with billing.
    - **Checks:** the network plan is re-run through the full power flow on the forecast background (linearisation error) and on the realised background (forecast error). The per-hub path reproduces the existing LP bound exactly (26.639 on 2024-01-04, rep 0).
    - **Two-day check:** value of coordination +0.005 to +0.07 $/day, with zero voltage or thermal violations of the network plan on the forecast background. The full runs decide whether this holds.
  - **v1 RL agents** stay the reference RL result (§4.9 pilot not adopted) and are reported under v1 accounting.
- **Robustness checks of the benchmark conclusions** (recorded 8 Oct 2026, after the benchmark and coordination results, before any of these runs). All use the LP bound and/or MPC-Predispatch, with billing, on the pre-registered days and reps. CPU only, no RL training. They bound the main stylised assumptions; they are not new claims.
  - **S1, omitted costs** (each alone; values are a sensitivity grid, not estimates):
    - battery degradation cost on discharged energy: {0, 0.05, 0.10} $/kWh;
    - network-tariff adder on participants' grid imports: {0, 50, 100} $/MWh.
    - Reported: LP bound and MPC value vs NoV2G at each value.
  - **S2, network:**
    - **(a) Shared-limit contrast** (the parking-lot structure in our setting). Hubs are grouped behind one shared connection limit. The coordinated LP keeps Σ_group flows within the group limit; the uncoordinated LP splits that limit across the group's hubs in proportion to capacity. Value of coordination = the difference.
      - Groups: hubs within the same 3-hop electrical neighbourhood (graph components).
      - Group limit: {0.3, 0.5} × the group's total hub capacity.
    - **(b) Tighter thermal ratings:** thermal_margin 0.8 in the existing per-hub vs network coordination LP.
  - **S3, participation:** opt-in price sensitivity β₁ × {0.5, 1.5} (E4 flags), on the LP bound and MPC.
  - **S4, FCAS availability bound.** For the participants plugged in under MPC-Predispatch (incentive 0.2), per 5-min step:
    - raise headroom = current participant set-point − max(feasible max discharge, −export limit);
    - lower headroom = min(feasible max charge, import limit − expected arrivals) − set-point.
    - Valued at the 2024 VIC1 FCAS enablement prices (AEMO DISPATCHPRICE: contingency raise and lower 1 s, 6 s, 60 s, 5 min, and regulation).
    - Reported as a range. **Lower bound:** headroom × the single highest-priced service each interval. **Upper bound:** headroom × the sum of all raise (or lower) services, as if co-enabled.
    - Ignores the 1 MW minimum aggregation, telemetry costs and FCAS trapezium limits. It is an indicative upper range against which to compare arbitrage value, not a dispatch result.
  - **Order:** S4, S2a, S1, S3, S2b.
  - **S1b, corrected network tariff** (recorded 8 Oct 2026 after the S1 results, before this run). As pre-registered, S1 charged the tariff on all participant imports, while participants were billed only the energy price. That is a worst case in which the aggregator absorbs every network charge, including on drivers' own requested energy, which a retail bill would pass on. S1b (`--tariff_passthrough`):
    - participants are billed for their requested energy at (day-mean RRP + tariff), and end-of-day purchases also pay the tariff;
    - the aggregator therefore bears the tariff only on extra energy cycled for arbitrage;
    - tariff {50, 100} $/MWh; the LP bound re-optimises its incentive as before;
    - **the MPC incentive is re-selected per tariff level** on the validation days (`select_mpc_incentive.py`, same rule).
    - S1 results are kept and reported as the worst case.
- **CRE21 network study** (design recorded 8 Oct 2026, before any import code or result). It addresses the IEEE-34 realism objection and tests the coordination mechanism on a real Australian urban network (also used by HetGPS, arXiv 2608.00679).
  - **Source.** Team-Nando/MV-LV-Networks (BSD-3; AusNet Services data; cite Ochoa et al.). CRE21 is urban, 22 kV / 0.4 kV:
    - 643 MV buses in one radial tree (649 line rows, 7 of them parallel duplicates);
    - 30.5 km of MV line, all three-phase;
    - 79 distribution transformers (70 residential, 9 C&I; 10–1,500 kVA), 3,383 customers.
  - **Model** (balanced positive-sequence, as in HetGPS):
    - slack at the 22 kV side of the zone substation (bus 111), set point as in the base study;
    - MV lines from linecode r1, x1 × length; parallel duplicates combined; line charging ignored;
    - **section ratings from real ampacities** (√3 × 22 kV × A) in place of the design rule;
    - each distribution transformer modelled explicitly (MV bus → LV busbar, impedance from xhl and load loss on its kVA base), with its **kVA rating as a thermal section**;
    - LV networks below the transformers are not modelled, so LV voltages are out of scope (disclosed).
  - **Background load.** Residential transformer peak = customers × per-customer diversified peak computed once from the CRE21 profile pool (value recorded before use); C&I transformer peak = 0.5 × kVA (assumption, disclosed). Time shape and PV as in the base study (VIC1 demand and PV shapes, π rule). **No impedance or load retuning:** the network is used as published, and the resulting voltage range is reported.
  - **Hub siting.** The 21 hubs go to the LV busbars of distinct transformers whose kVA ≥ hub capacity, chosen at evenly spaced electrical-distance quantiles as in the base placement rule (*base siting*). Sensitivity, *constrained siting*: each hub, in decreasing capacity order, at the smallest feasible free transformer. Both are stylised (anonymised network; no real geography match).
  - **Validation before use.** Our power flow against OpenDSS (dss_python) on the same balanced MV model at peak and midday snapshots; acceptance: max |ΔV| ≤ 0.005 pu.
  - **Studies.** CPU only, pre-registered days × 3 reps, billing on. RL agents are not retrained:
    - (a) DOE statistics: how often import and export limits bind, and minimum and maximum voltages;
    - (b) coordination LP, per-site DOEs vs joint constraints (now including transformer ratings), π = 0.6 and 0.9, both sitings;
    - (c) benchmark: NoV2G, GreedyTOU, RulePrice, MPC-Predispatch (incentive re-selected on validation days), LP bound.
    - Optional Tier 2: added uncontrolled residential EV charging per customer, evening-weighted, to approach the fully electrified regime of HetGPS.
  - **Build and validation** (8 Oct 2026, before any full study run; `tools/build_cre21.py`, network files in `ev2gym/data/network_data/node_cre21/`):
    - 722 buses / 721 branches (642 MV lines, 7 parallel-combined, plus 79 transformers);
    - residential diversified peak 1.32 kW per customer (seed 2026); total peak 5.95 MW;
    - nominal-peak voltages 0.98–1.03 pu, so no retuning was applied.
    - **OpenDSS validation passed:** max |ΔV| ≤ 6.3e-6 pu at peak, midday-PV and peak-with-full-hub snapshots (tolerance 0.005 pu).
    - Smoke runs, which are not results: DOE import limits bind on ~10–34% of hub-steps depending on siting and day.
- **Value of the extended model** (recorded 8 Oct 2026, before any code or result; responds to the supervisor's framing that a "new problem" must extend a literature model and be shown to matter).
  - **Old problem (literature baseline):** the EV2Gym V2G profit-maximisation problem (Orfanoudakis et al., IEEE T-ITS 2025), also used by EV-GNN (2025). Its assumptions, relative to ours:
    - **O1** no DOEs: only hub/transformer capacity limits flow;
    - **O2** every EV is controllable: no opt-in, no incentive;
    - **O3** discharge paid 1.2× the charging price;
    - **O4** soft departure: unmet energy is a mild penalty, here λ_unmet = $0.1/kWh in place of $1 and no forced charging in the plan.
  - **Study VEM-1, cost of planning with the old model.**
    - **Controllers:** MPC-Predispatch (incentive 0.2, billing on), planned with the new model (*MPC-new*) or with one old assumption switched on (*MPC-O1 … MPC-O4*), plus all four together (*MPC-old*).
    - **Same information, same controller, same executed environment** (the realistic one); only the planner's internal model differs.
    - O2 planning treats connected non-participants as dispatchable; their requested discharge is then clipped by the environment, as it would be in reality.
    - **Metrics,** paired against MPC-new on the pre-registered days × 3 reps: net economics minus unmet energy, realised limit violations (kWh), unmet driver energy (kWh), overload, voltage range. Representative and stress sets reported separately, with 95% bootstrap CIs.
    - **Interpretation:** a significant loss or extra violations from an assumption shows that this extension matters for decisions.
  - **Study VEM-2, how the old model values V2G.** The perfect-foresight LP and MPC are evaluated in an environment with the old assumptions (O1–O3 as env options; O4 can't be relaxed in the env, because forced charging is part of the session model). Reported: the factor by which the old model overstates achievable V2G value, and the coordination value under it. This is the "EV-GNN conditions" check.
  - **Network:** IEEE-34 (main). CRE21 if time allows, with the same rules.
  - **Not done:** RL is not retrained under the old model.
- **Run plan for E2, E3 and PV, in priority tiers** (recorded 7 Oct 2026, while the λ-study runs were training, before any λ-study or E1 results existed). All runs use the λ chosen by §10, `--episodes 1500`, 3 seeds (42, 1, 2), and are evaluated like E1. Lower tiers are dropped first if time runs short (fallback date 20 Oct). Dropping a tier is decided by the calendar, never by results.
  - **Tier 1 (required, 27 runs).** All at π = 0.9, where inter-hub coupling is strongest (κ* < 1 in 37% of midday windows, vs 14% at π = 0.6). SAC-GNN vs SAC-GNN-NoEdge vs SAC-Flat in each:

    | Condition | Flags | Role |
    |---|---|---|
    | Coupling on | `--doe_mode network --pv_penetration 0.9` | Does the graph help when hubs are electrically coupled? |
    | Control: coupling off | `--doe_mode per_hub --pv_penetration 0.9` | Same PV level without coupling, so a network-vs-per-hub difference is due to coupling, not PV level. Also the π = 0.9 point of the PV sensitivity |
    | Structure off | `--doe_mode per_hub --spatial permuted --pv_penetration 0.9` | Limits shuffled across hubs, which breaks the link between a hub's limits and its place in the graph. Does any graph benefit vanish? |

    The permutation is applied in per-hub mode only. In network mode the penalty is the realised overload and voltage from the true power flow, so permuted limits would also make the projection's limits physically wrong. Structure removal would then be confounded with infeasible limits.
  - **Tier 2 (if time allows).**
    - Network mode at π = 0.6, the same three agents (9 runs). This gives a coupling dose–response: per-hub (E1), then 14%, then 37%.
    - E3 at `--forecast_sigma 0` and `0.10`, SAC-GNN and SAC-Flat only (12 runs). σ = 0.05 comes from E1.
  - **Tier 3 (dropped first).** PV sensitivity at π = 0.3, per-hub mode, the same three agents (9 runs). π = 0.6 comes from E1 and π = 0.9 from Tier 1.
  - **Scheduling.** E1 (17 new runs) plus Tier 1 is about 11 packed L40S jobs, about 3 rounds of about 4.5 h under the 4-job limit, about 14 h of compute plus queueing.
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
  - Packed jobs request 6 h (≈4.5 h expected + ~30%). The λ study measured 8.4 s/episode with three runs packed, about 3.6 h per run.
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
     - **Disclosure** (added 7 Oct 2026, while the λ-study runs were training, before any results): λ is selected on the same representative days that E1 reports on. The thesis states: "λ was selected by a pre-registered, compliance-only criterion on the evaluation days; profit did not enter the selection; full results for every λ are reported." The λ table (violations and pre-penalty profit vs NoV2G, all λ) is reported as the λ sensitivity analysis. Its scope is SAC-GNN in the main setting only (per-hub DOEs, σ = 0.05, π = 0.6), which is stated as a limitation.
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
