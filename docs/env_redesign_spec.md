# Environment redesign spec: feeder-grounded DOE dispatch (`NEMFeederEnv`)

Status: draft for sign-off (Zoey, Terrence). Target: thesis results by about 16 Oct 2026.
The legacy `NEMDOEEnv` stays unchanged so that draft-v1 results remain reproducible.

## 1. Why: what the current environment lacks

| Gap in `NEMDOEEnv` (legacy/coupled) | Consequence | Review ref. |
|---|---|---|
| The DOE clips the agent's request, so compliance is guaranteed and free | "Compliance" measures intent; agents keep a large free margin (~10 kW dispatched vs a 35 kW mean DOE) | M1 |
| DOEs are i.i.d. per hub, the graph uses road distance, `edge_attr` is unused | No spatial structure for a graph encoder to exploit | M5 |
| Mean SoC reverts to its average (half-life ≈ 33 min); occupancy ignores actions | Near-contextual-bandit; a myopic oracle is near-optimal, so the case for RL is weak | M7 |
| Re-decided participation every 5 min; incentive paid on charging | Unrealistic participation and pricing | M9 |
| No grid physics | DOE compliance is not tied to voltages or thermal limits | VIII-A |

## 2. Claims and the experiment that supports each

| Claim | Supported by |
|---|---|
| **C1. A realistic, reproducible formulation** of public V2G hub dispatch under power-flow-derived, location-specific DOEs applied to *net site flow*, with EV sessions, opt-in participation and real NEM data | §4 model; §7 validation tests; benchmarks that behave as expected (§5) |
| **C2. When graph structure helps:** graph vs edgeless vs flat, with spatial DOE structure on vs off | Experiment E2 + policy-sensitivity diagnostics |
| **C3. Compliance under genuine uncertainty:** profit vs violation trade-off as forecast error grows | Experiment E3 (+ voltage outcomes, E5) |

## 3. Scope

- **Must (thesis):** feeder + power-flow DOEs, net-site compliance, EV sessions, opt-in, electrical graph, per-hub capacity scaling, perfect-foresight LP benchmark, experiments E1–E5.
- **Stretch:** a learned safety-margin layer; the 32-hub feeder variant; Australian EV session data in place of ElaadNL.

## 4. Model

### 4.1 Network and hub mapping
- **Feeder.** The 34-node radial feeder shipped with EV2Gym (from RL-ADN; 11 kV, 33 load buses, 7.8 MW nominal load). Power flow uses `GridTensor` (tensor/Laurent solver): 0.06 ms per solve, 3.9 ms for a 200-scenario batch.
  - The installed pandapower (2.13) does not import under numpy 2. We set `np.Inf = np.inf` before import; no environment change is needed.
- **Hub → bus mapping (stylised, disclosed).** Sort the 21 hubs by road distance from the zone centroid and the 33 buses by electrical distance from the substation, then assign in order. Hubs far from the centre therefore sit on electrically weak buses.
- **Graph (electrical).** Hubs are adjacent if their buses are within *k* hops in the feeder tree, after contracting buses that carry no hub. Edge attribute: the impedance distance between the two buses. The road-distance graph is kept as an ablation.

### 4.2 Background (non-EV) load and PV, from AEMO data on the same dates as RRP
- **Per bus *b*, at step *t*:**
  `P^bg_{b,t} = PD_b · κ_load · D_t − PV_b · S_t`
  - `D_t` is VIC1 `TOTALDEMAND` (DISPATCHREGIONSUM, 5-min), normalised to its annual mean.
  - `S_t` is VIC1 `ROOFTOP_PV_ACTUAL` (30-min, interpolated), normalised to annual peak.
  - `PV_b = π · PD_b`, where π is the PV penetration (default 0.6; sensitivity values {0.3, 0.6, 0.9}).
- **Calibration.** Choose `κ_load` and π so that, with no EVs, voltages stay inside [0.95, 1.05] pu but approach the lower limit at the evening peak and the upper limit at the midday PV peak. DOEs then bind on import in the evening and on export at midday, the Australian pattern. This also ties price and grid stress together naturally: midday PV brings low or negative RRP *and* tight export DOEs.
- **Forecast vs realised.** `P^real = P^fcst · (1 + ε)`. ε follows an AR(1) process per feeder lateral, so errors are correlated within a lateral; σ defaults to 5% (sensitivity {0, 5, 10}%).

### 4.3 DOE computation (DNSP side, every 30 min)
1. **Individual hosting capacity.** For each hub *i*, take the forecast worst case over the next 30 min with all other hubs at forecast baseline. `h_i` is the largest export (and, separately, import) that keeps every bus voltage within limits. Found by bisection on batched power flow.
2. **Joint feasibility.** Set `DOE_i = min(cap_i, κ* · h_i)`, where κ* ≤ 1 is the largest common scale under which *all* hubs at their DOE are jointly feasible (found by bisection). This is "scaled individual hosting capacity": location-specific (weak buses get less), time-varying, and jointly safe under the *forecast*.
3. DOEs are published to the agent and held for 6 steps.

### 4.4 EV sessions (per hub, per charger port)
- **Arrivals.** Poisson with the ElaadNL public time-of-day profile (weekday/weekend, 15-min, from EV2Gym), scaled to `sessions_per_port_per_day` (assumption: 2.5). An arrival is lost if every port is busy (logged).
- **Dwell time.** EV2Gym's `time_of_connection_vs_hour` matrix (conditional on arrival hour).
- **Energy demand.** ElaadNL public distribution, capped by battery headroom.
- **EV model.** Sampled by registrations from `ev_specs_v2g_enabled2024.json`.
  - Battery 46–77 kWh.
  - Charge power = min(EV DC limit, charger kW).
  - **Discharge ≈ 10–11 kW**, the real V2G limit for these models.
- **SoC.** Arrival SoC = target SoC (0.8) − demand / capacity, clipped to [0.1, 0.8].

### 4.5 Participation (opt-in once, at arrival)
- The owner is offered the current incentive `c_t` (paid per kWh **discharged** only) and accepts with the existing logistic ρ(c, d_h, SoC_arr, g). β and γ are unchanged and disclosed; *g* is the session's anticipated V2G discharge.
- **Participants** are dispatchable, subject to: SoC ≥ 0.2; reaching the target energy by departure; per-EV power limits.
- **Non-participants** charge immediately at full power until satisfied. This is uncontrolled site load the agent cannot dispatch, and its realised value is uncertain.

### 4.6 Action, allocation, safety projection
- **Action.** For each hub, a normalised dispatch for its participating fleet, scaled by the hub's own capacity (fixes the uniform 100 kW issue, B3), plus one network-wide incentive price.
- **Allocation inside a hub.**
  - Charging goes to EVs in least-laxity-first order; discharging comes from EVs in most-slack-first order.
  - Each EV's power is projected onto its feasible set this step, including "can still reach target by departure".
  - The executed flexible power is the sum.
- **Safety projection (the deployed clip).**
  - The DOE applies to **net site flow**: flexible fleet + uncontrolled charging + site base load.
  - The clip can only use the **forecast** of the uncontrolled part, so the **realised** net flow can still breach the DOE.
  - Compliance is therefore a real, uncertain outcome. This answers M1.

### 4.7 Reward ($ per step)
`r_t = RRP_t·E^flex_t/1000 − c_t·E^dis_t − λ_u·E^unmet_t − λ_doe·Δt·Σ_i max(0, |P^net,real_i| − DOE_i)`
- `E^flex`: net flexible energy (kWh). Charging is a cost at RRP and discharging a revenue.
- `E^dis`: discharged energy paid to owners.
- `E^unmet`: energy short of targets at departures (λ_u is a $/kWh dissatisfaction cost).
- The DOE term is in **energy** units ($/kWh of violation), consistent with revenue. λ_doe is set by the pilot.
- Revenue from uncontrolled charging is a pass-through and excluded.
- Realised voltages are **reported**, not rewarded, in the main runs (stretch: a voltage penalty).

### 4.8 Observation (per hub node; global values broadcast)
1. DOE import / export
2. Forecast uncontrolled site load
3. Participating EV count
4. Flexible energy needed by departures / storable
5. Fleet max charge / discharge power
6. Mean hours to departure
7. Port occupancy
8. Mean participant SoC
9. Hub capacity
10. RRP_t
11. Mean RRP over the last 30 min
12. Hour (sin/cos)
13. Weekend flag
14. Steps to next DOE update

About 16 features. `NetworkConfig.node_feature_dim` becomes a parameter (it is currently hard-coded to 9).

## 5. Baselines
- **Uncontrolled** (no V2G; everyone charges immediately): reference.
- **Greedy:** price threshold, full power.
- **RulePrice.**
- **Myopic Oracle:** knows ρ and the forecasts; one-step.
- **Perfect-foresight LP (upper bound):** realised prices, sessions, opt-ins and background load are known. Per-hub aggregate energy-reservoir model; constant incentive chosen by grid search. Solved with `scipy.optimize.milp` (HiGHS, already installed). Size ≈ 21 hubs × 288 steps.

## 6. Experiments
- **E1 (main):** SAC-GNN, SAC-GCN, SAC-Flat, SAC-GNN-NoEdge, + baselines.
  - 21 hubs; ≥3 seeds (5 if packing allows).
  - ≥30 held-out 2024 days stratified by season and price volatility, with stress days labelled honestly.
  - Seed-level IQM with bootstrap CIs; per-day paired tests reported as secondary.
- **E2 (C2):** spatial structure on (power-flow DOEs + correlated errors) vs off (i.i.d. DOEs with matched marginals). Graph vs edgeless vs flat in both.
- **E3 (C3):** forecast error σ ∈ {0, 5, 10}%, giving profit vs DOE-violation rate.
- **E4 (M9):** evaluate trained policies under perturbed β/γ (eval-only).
- **E5 (VIII-A):** realised voltage violations for each policy, from power flow.
- **Diagnostics:** neighbour-sensitivity test (does hub *i*'s dispatch respond to hub *j*'s state, by electrical distance?) and attention entropy.

## 7. Validation tests (before any training)
- Power flow converges on every step of 50 sampled days.
- DOEs are jointly feasible under the forecast: all hubs at DOE ⇒ voltages within limits.
- Session energy is conserved: Σ delivered + unmet = Σ requested.
- No EV goes below SoC 0.2 or above 1.0.
- A participant that is always feasible is fully charged at departure.
- With σ = 0 and the projection active, the DOE violation is 0.
- The LP benchmark ≥ every policy on the same realisations.
- Edgeless isolation (existing perturbation test, re-run for the new features).
- The legacy env is still bit-identical.

## 8. Data and assumptions (all disclosed)

| Item | Source | Note |
|---|---|---|
| RRP | AEMO DISPATCHPRICE (NEMOSIS) | existing pipeline |
| Demand shape | AEMO DISPATCHREGIONSUM, VIC1 | same dates as RRP |
| PV shape | AEMO ROOFTOP_PV_ACTUAL, VIC1 | 30-min, interpolated |
| Feeder | EV2Gym/RL-ADN 34-node | stylised mapping of real hubs |
| Sessions | ElaadNL public (via EV2Gym) | Dutch; no public Australian equivalent identified yet |
| EV models | EV2Gym V2G-enabled 2024 specs | real discharge limits |
| Hubs | OpenChargeMap (existing) | V2G capability assumed |
| Participation | existing logistic (β, γ) | uncalibrated, so E4 sensitivity is run |

Training-period caveat: exclude the June 2022 market suspension window from training days.

## 9. Code structure
- `nem_env/feeder.py`: network load, hub mapping, power flow, DOE computation, electrical graph.
- `nem_env/sessions.py`: arrivals, EV sampling, opt-in, allocation, feasibility projection.
- `nem_env/feeder_env.py`: `NEMFeederEnv` (gym API; flags for E2/E3 ablations).
- `nem_env/aemo_loader.py` extension: demand and PV tables alongside RRP.
- `baselines/mpc/perfect_foresight_lp.py`; Greedy/RulePrice/Myopic Oracle adapted.
- Agents: parametric `node_feature_dim`; per-hub capacity scaling.
- `train_sac_gnn.py` / `evaluate.py`: `--env feeder`, held-out day sets, the new metrics.

## 10. Timeline and decision points
| Date | Milestone |
|---|---|
| 6–7 Oct | feeder.py + AEMO data + calibration; sessions.py; tests |
| 8–9 Oct | feeder_env.py, agents, baselines, LP; full test suite |
| 9–10 Oct | throughput benchmark on the new env; λ pilot |
| 10–15 Oct | E1–E3 training (packed jobs); fallback coupled batch runs in parallel |
| 15–17 Oct | evaluation E1–E5, diagnostics |
| **20 Oct** | last point to fall back to the coupled re-run; anything unfinished goes to limitations |
| 17–24 Oct | writing |

## 11. Decisions needed
1. Approve the stylised 34-bus mapping of real hubs (vs. waiting for real DNSP network data, which is not available in time).
2. PV penetration default 0.6 and forecast error 5%: accept as defaults with sensitivity?
3. `sessions_per_port_per_day` = 2.5 (ElaadNL public-charger order of magnitude): accept?
4. Unmet-energy cost λ_u (proposal: $1/kWh, about 2–3× a public charging tariff).
5. Seeds: 3 (safe) or 5 (if packing allows).
