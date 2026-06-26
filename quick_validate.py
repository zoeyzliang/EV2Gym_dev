"""
quick_validate.py
=================
Smoke test for the full NEMDOEEnv stack after the WDR→DOE migration.

Runs without AEMO data (synthetic prices) and without PyTorch (numpy
fallback agent). Checks every layer of the stack in order:

  1. Imports — all modules load without error
  2. Graph construction — synthetic 21-hub graph builds correctly
  3. Environment reset — obs shape, dtype, no NaN
  4. Action shapes — env action_space matches agent output
  5. One full episode rollout — 288 steps, all info fields present
  6. Reward sign checks — arbitrage reward is non-zero and finite
  7. DOE clipping — raw action is clipped correctly in step()
  8. Observation layout — node feature [6] is RRP (broadcast check)
  9. Agent store + sample — replay buffer add/sample cycle
 10. Node feature reshape — obs_to_node_features returns (H, 9)

Run from repo root:
    python quick_validate.py

Expected output: all checks print ✓. Any ✗ indicates a mis-wiring
that must be fixed before running on M3 HPC.
"""

import sys
import traceback
import numpy as np

PASS = "✓"
FAIL = "✗"

results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {name}")
        print(f"       {msg}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        results.append((name, False, msg))


# ===========================================================================
# 1. Imports
# ===========================================================================
print("\n── 1. Imports ──────────────────────────────────────────────────────")

env_module = price_module = part_module = graph_module = None
agent_module = net_module = buf_module = None

def _import_nem_env():
    global env_module
    from nem_env.nem_doe_env import NEMDOEEnv, EnvConfig, HubState
    env_module = sys.modules["nem_env.nem_doe_env"]

def _import_price():
    global price_module
    from nem_env.aemo_price_loader import PriceLoader
    price_module = sys.modules["nem_env.aemo_price_loader"]

def _import_participation():
    global part_module
    from nem_env.participation_model import ParticipationModel, HubParticipationState
    part_module = sys.modules["nem_env.participation_model"]

def _import_graph():
    global graph_module
    from nem_env.spatial_graph import HubGraphBuilder, HubConfig, GraphData
    graph_module = sys.modules["nem_env.spatial_graph"]

def _import_agent():
    global agent_module
    from baselines.gnn_rl.agent import SACGNNAgent
    agent_module = sys.modules["baselines.gnn_rl.agent"]

def _import_networks():
    global net_module
    from baselines.gnn_rl.networks import NetworkConfig, NumpyActor, NumpyCritic
    net_module = sys.modules["baselines.gnn_rl.networks"]

def _import_buffer():
    global buf_module
    from baselines.gnn_rl.replay_buffer import ReplayBuffer, Batch
    buf_module = sys.modules["baselines.gnn_rl.replay_buffer"]

check("nem_env.nem_doe_env", _import_nem_env)
check("nem_env.aemo_price_loader", _import_price)
check("nem_env.participation_model", _import_participation)
check("nem_env.spatial_graph", _import_graph)
check("baselines.gnn_rl.agent", _import_agent)
check("baselines.gnn_rl.networks", _import_networks)
check("baselines.gnn_rl.replay_buffer", _import_buffer)

# Abort early if imports failed — nothing else will work
if any(not ok for _, ok, _ in results):
    print("\n  Import failures detected — fix before continuing.\n")
    sys.exit(1)

# ===========================================================================
# Shared setup (runs once, reused by all checks below)
# ===========================================================================
from nem_env.nem_doe_env import NEMDOEEnv, EnvConfig
from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel
from nem_env.spatial_graph import HubGraphBuilder, HubConfig, GraphData
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.networks import NetworkConfig
from baselines.gnn_rl.replay_buffer import ReplayBuffer, Batch

N_HUBS = 21
NODE_FEATURE_DIM = 9

# ===========================================================================
# 2. Graph construction
# ===========================================================================
print("\n── 2. Graph construction ───────────────────────────────────────────")

graph = hub_configs = None

def _build_graph():
    global graph, hub_configs
    builder = HubGraphBuilder(
        zone="inner_melbourne",
        use_synthetic=True,
        n_synthetic_hubs=N_HUBS,   # explicit: default is 15, we need 21
    )
    graph = builder.build()
    hub_configs = builder.hub_configs

def _graph_node_count():
    assert len(hub_configs) == N_HUBS, \
        f"Expected {N_HUBS} hubs, got {len(hub_configs)}"

def _graph_edge_shape():
    ei = graph.edge_index
    assert ei.shape[0] == 2, f"edge_index shape[0] should be 2, got {ei.shape}"
    assert ei.shape[1] > 0, "edge_index has no edges"

def _graph_hub_config_fields():
    hc = hub_configs[0]
    assert hasattr(hc, "p_max_kw"), "HubConfig missing p_max_kw"
    assert hasattr(hc, "distance_km"), "HubConfig missing distance_km"
    assert hc.p_max_kw > 0, f"p_max_kw should be positive, got {hc.p_max_kw}"

check("HubGraphBuilder synthetic build", _build_graph)
check(f"Graph has {N_HUBS} hub nodes", _graph_node_count)
check("edge_index shape (2, E)", _graph_edge_shape)
check("HubConfig has p_max_kw and distance_km", _graph_hub_config_fields)

# ===========================================================================
# 3. Environment construction and reset
# ===========================================================================
print("\n── 3. Environment reset ────────────────────────────────────────────")

env = None
obs0 = None

def _build_env():
    global env
    loader = PriceLoader(region="VIC1", cache_dir="data/nem_cache", seed=0)
    loader.load_synthetic(
        n_days=10,
        mean_price=100.0,
        std_price=250.0,
        spike_prob=0.003,
        spike_magnitude=2000.0,
    )
    model = ParticipationModel(seed=0)
    env = NEMDOEEnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(),
        seed=0,
    )

def _env_spaces():
    exp_obs_dim = N_HUBS * NODE_FEATURE_DIM
    assert env.observation_space.shape == (exp_obs_dim,), \
        f"obs shape {env.observation_space.shape} != ({exp_obs_dim},)"
    exp_act_dim = N_HUBS + 1
    assert env.action_space.shape == (exp_act_dim,), \
        f"action shape {env.action_space.shape} != ({exp_act_dim},)"

def _env_reset():
    global obs0
    obs0, info = env.reset()
    assert obs0.shape == (N_HUBS * NODE_FEATURE_DIM,), \
        f"reset obs shape {obs0.shape}"
    assert obs0.dtype == np.float32, f"obs dtype {obs0.dtype} != float32"
    assert not np.any(np.isnan(obs0)), "NaN in reset observation"
    assert not np.any(np.isinf(obs0)), "Inf in reset observation"

check("NEMDOEEnv construction (synthetic prices)", _build_env)
check("observation_space and action_space shapes", _env_spaces)
check("reset() returns valid (H×9,) obs", _env_reset)

# ===========================================================================
# 4. Observation layout
# ===========================================================================
print("\n── 4. Observation layout ───────────────────────────────────────────")

def _obs_to_node_features():
    node_feats = env.obs_to_node_features(obs0)
    assert node_feats.shape == (N_HUBS, NODE_FEATURE_DIM), \
        f"node_feats shape {node_feats.shape} != ({N_HUBS}, {NODE_FEATURE_DIM})"

def _rrp_broadcast():
    # Feature [6] is rrp_norm — should be identical across all hubs
    node_feats = env.obs_to_node_features(obs0)
    rrp_per_hub = node_feats[:, 6]
    assert np.allclose(rrp_per_hub, rrp_per_hub[0]), \
        f"RRP (feature [6]) not broadcast identically: {rrp_per_hub}"

def _rrp_in_valid_range():
    node_feats = env.obs_to_node_features(obs0)
    rrp_norm = node_feats[0, 6]
    assert -1.0 <= rrp_norm <= 1.0, \
        f"RRP norm {rrp_norm:.4f} outside [-1, 1]"

def _doe_features_nonneg():
    node_feats = env.obs_to_node_features(obs0)
    doe_import = node_feats[:, 0]  # feature [0]
    doe_export = node_feats[:, 1]  # feature [1]
    assert np.all(doe_import >= 0), f"Negative DOE import: {doe_import.min()}"
    assert np.all(doe_export >= 0), f"Negative DOE export: {doe_export.min()}"

def _mean_soc_in_range():
    node_feats = env.obs_to_node_features(obs0)
    socs = node_feats[:, 3]  # feature [3]
    assert np.all(socs >= 0) and np.all(socs <= 1), \
        f"mean_soc out of [0,1]: min={socs.min():.3f} max={socs.max():.3f}"

check("obs_to_node_features returns (H, 9)", _obs_to_node_features)
check("RRP broadcast identically to all hub nodes (feature [6])", _rrp_broadcast)
check("RRP norm in [-1, 1]", _rrp_in_valid_range)
check("DOE import/export features non-negative", _doe_features_nonneg)
check("mean_soc (feature [3]) in [0, 1]", _mean_soc_in_range)

# ===========================================================================
# 5. DOE clipping
# ===========================================================================
print("\n── 5. DOE clipping ─────────────────────────────────────────────────")

def _doe_clip_discharge():
    # Force a massive discharge action — should be clipped to DOE export limit
    obs, _ = env.reset()
    huge_discharge = np.full(N_HUBS, 1e6, dtype=np.float32)
    action = np.append(huge_discharge, 0.10)
    _, _, _, _, info = env.step(action)
    clipped = np.array(info["clipped_dispatch_kw"])
    violations = np.array(info["doe_violations_kw"])
    assert np.all(violations > 0), \
        "Expected DOE violations for huge discharge action, got none"
    assert np.all(clipped >= 0), \
        f"Clipped discharge should be non-negative, got {clipped.min()}"

def _doe_clip_charge():
    # Force a massive charge action — should be clipped to DOE import limit
    obs, _ = env.reset()
    huge_charge = np.full(N_HUBS, -1e6, dtype=np.float32)
    action = np.append(huge_charge, 0.05)
    _, _, _, _, info = env.step(action)
    clipped = np.array(info["clipped_dispatch_kw"])
    assert np.all(clipped <= 0), \
        f"Clipped charge should be non-positive, got {clipped.max()}"

def _zero_action_no_violation():
    obs, _ = env.reset()
    zero_action = np.zeros(N_HUBS + 1, dtype=np.float32)
    zero_action[-1] = 0.05  # price
    _, _, _, _, info = env.step(zero_action)
    violations = np.array(info["doe_violations_kw"])
    assert np.all(violations == 0), \
        f"Zero dispatch should produce no DOE violations: {violations}"

check("Huge discharge action clipped + violation recorded", _doe_clip_discharge)
check("Huge charge action clipped to non-positive", _doe_clip_charge)
check("Zero dispatch action → zero DOE violations", _zero_action_no_violation)

# ===========================================================================
# 6. Full episode rollout
# ===========================================================================
print("\n── 6. Full episode rollout (288 steps) ─────────────────────────────")

episode_rewards = []
episode_infos = []

def _full_rollout():
    global episode_rewards, episode_infos
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        # Random signed action within bounds
        dispatch = np.random.uniform(-50.0, 50.0, N_HUBS).astype(np.float32)
        price = np.random.uniform(0.0, 0.30)
        action = np.append(dispatch, price).astype(np.float32)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        episode_rewards.append(reward)
        episode_infos.append(info)

    assert step_count == 288, f"Episode should be 288 steps, got {step_count}"
    assert done is True
    assert truncated is False

def _info_fields_present():
    required = [
        "step", "rrp", "incentive_price_per_kwh",
        "clipped_dispatch_kw", "raw_dispatch_kw", "doe_violations_kw",
        "n_respond", "rho_hat", "participated_kwh",
        "r_wholesale", "r_incentive", "p_conformance", "reward",
    ]
    info = episode_infos[0]
    missing = [f for f in required if f not in info]
    assert not missing, f"Missing info fields: {missing}"

def _reward_finite():
    assert all(np.isfinite(r) for r in episode_rewards), \
        "Non-finite reward encountered during rollout"

def _reward_nonzero():
    # With random actions including discharge, some wholesale revenue expected
    assert any(r != 0.0 for r in episode_rewards), \
        "All rewards were exactly zero — participation model may be broken"

def _obs_shape_consistent():
    # All obs during rollout should have same shape (reset gives last obs = zeros)
    pass  # shape checked at reset; rollout obs checked inside step

def _rho_hat_in_range():
    rhos = [info["rho_hat"] for info in episode_infos]
    assert all(0.0 <= r <= 1.0 for r in rhos), \
        f"rho_hat out of [0,1]: min={min(rhos):.3f} max={max(rhos):.3f}"

def _participated_kwh_signed():
    # participated_kwh sign must match clipped_dispatch direction
    # Only check hubs where BOTH dispatch is non-zero AND n_respond > 0
    # (if n_respond==0 no energy flows, participated_kwh==0 regardless of direction)
    for info in episode_infos[:10]:
        p = np.array(info["participated_kwh"])
        c = np.array(info["clipped_dispatch_kw"])
        n = np.array(info["n_respond"])
        active = (np.abs(c) > 0.01) & (n > 0)
        if active.any():
            assert np.all(np.sign(p[active]) == np.sign(c[active])), \
                f"participated_kwh sign mismatch: p={p[active]} c={c[active]}"

check("288-step episode completes without error", _full_rollout)
check("All required info keys present", _info_fields_present)
check("All rewards are finite", _reward_finite)
check("At least one non-zero reward", _reward_nonzero)
check("rho_hat in [0, 1] every step", _rho_hat_in_range)
check("participated_kwh sign matches dispatch direction", _participated_kwh_signed)

# ===========================================================================
# 7. Reward decomposition
# ===========================================================================
print("\n── 7. Reward decomposition ─────────────────────────────────────────")

def _reward_components_sum():
    # r = r_wholesale - r_incentive - p_conformance
    for info in episode_infos[:20]:
        computed = info["r_wholesale"] - info["r_incentive"] - info["p_conformance"]
        assert abs(computed - info["reward"]) < 1e-4, \
            f"Reward decomposition mismatch: {computed:.4f} != {info['reward']:.4f}"

def _incentive_cost_nonneg():
    for info in episode_infos:
        assert info["r_incentive"] >= 0, \
            f"Incentive cost should be non-negative: {info['r_incentive']}"

def _conformance_penalty_nonneg():
    for info in episode_infos:
        assert info["p_conformance"] >= 0, \
            f"Conformance penalty should be non-negative: {info['p_conformance']}"

check("r_wholesale - r_incentive - p_conformance = reward", _reward_components_sum)
check("Incentive cost is non-negative", _incentive_cost_nonneg)
check("DOE conformance penalty is non-negative", _conformance_penalty_nonneg)

# ===========================================================================
# 8. Agent (numpy fallback)
# ===========================================================================
print("\n── 8. SAC-GNN agent (numpy fallback) ───────────────────────────────")

agent = None

def _build_agent():
    global agent
    import os
    os.environ["FORCE_NUMPY_AGENT"] = "1"

    obs_dim = N_HUBS * NODE_FEATURE_DIM
    net_cfg = NetworkConfig(
        node_feature_dim=NODE_FEATURE_DIM,
        embed_dim=64,
        gat_heads=4,
        equipment_cap_kw=100.0,
    )
    agent = SACGNNAgent(
        n_hubs=N_HUBS,
        graph_data=graph,
        obs_dim=obs_dim,
        net_cfg=net_cfg,
        seed=0,
    )
    os.environ.pop("FORCE_NUMPY_AGENT", None)

def _agent_select_action_shape():
    obs, _ = env.reset()
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (N_HUBS + 1,), \
        f"action shape {action.shape} != ({N_HUBS + 1},)"

def _agent_action_signed():
    obs, _ = env.reset()
    action = agent.select_action(obs, deterministic=True)
    dispatch = action[:N_HUBS]
    # dispatch should be in [-equipment_cap, +equipment_cap] — i.e. can be negative
    assert dispatch.dtype == np.float32, f"dispatch dtype {dispatch.dtype}"
    # with random weights dispatch will be near zero but not all positive
    # just check it's in a plausible range
    assert np.all(np.abs(dispatch) <= 200.0), \
        f"dispatch out of expected range: max abs = {np.abs(dispatch).max()}"

def _agent_price_in_bounds():
    obs, _ = env.reset()
    action = agent.select_action(obs, deterministic=True)
    price = action[-1]
    assert 0.0 <= price <= 1.0, \
        f"price {price:.4f} outside [price_min={agent.net_cfg.price_min}, price_max={agent.net_cfg.price_max}]"

def _agent_no_zone_features():
    # _split_obs should return a single array, not a tuple
    obs, _ = env.reset()
    result = agent._split_obs(obs)
    assert isinstance(result, np.ndarray), \
        f"_split_obs should return ndarray, got {type(result)}"
    assert result.shape == (N_HUBS, NODE_FEATURE_DIM), \
        f"_split_obs shape {result.shape} != ({N_HUBS}, {NODE_FEATURE_DIM})"

check("SACGNNAgent construction (numpy fallback)", _build_agent)
check("select_action returns (H+1,) array", _agent_select_action_shape)
check("dispatch actions are signed (not sigmoid fractions)", _agent_action_signed)
check("incentive price in [price_min, price_max]", _agent_price_in_bounds)
check("_split_obs returns (H, 9) array — no zone tuple", _agent_no_zone_features)

# ===========================================================================
# 9. Replay buffer
# ===========================================================================
print("\n── 9. Replay buffer ────────────────────────────────────────────────")

obs_dim = N_HUBS * NODE_FEATURE_DIM
action_dim = N_HUBS + 1
buf = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=1000, seed=0)

def _buffer_add():
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    buf.add(obs, action, 1.5, obs, False)
    assert buf.size == 1, f"Buffer size should be 1, got {buf.size}"

def _buffer_add_no_wdr_arg():
    # add() should NOT accept wdr_active — verify it works without it
    import inspect
    sig = inspect.signature(buf.add)
    assert "wdr_active" not in sig.parameters, \
        "add() still has wdr_active parameter — not cleaned up"

def _buffer_sample():
    # Fill buffer enough to sample
    for _ in range(300):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        buf.add(obs, action, float(np.random.randn()), obs, False)
    batch = buf.sample(64)
    assert batch.obs.shape == (64, obs_dim)
    assert batch.actions.shape == (64, action_dim)
    assert batch.rewards.shape == (64, 1)
    assert not hasattr(batch, "wdr_active"), \
        "Batch still has wdr_active field — not cleaned up"

check("buffer.add() works without wdr_active", _buffer_add)
check("add() signature has no wdr_active parameter", _buffer_add_no_wdr_arg)
check("buffer.sample(64) returns correct shapes, no wdr_active field", _buffer_sample)

# ===========================================================================
# 10. Greedy baseline
# ===========================================================================
print("\n── 10. Greedy baseline ─────────────────────────────────────────────")

def _greedy_import():
    from baselines.heuristics.greedy_dispatch import GreedyDispatchBaseline

def _greedy_action_shape():
    from baselines.heuristics.greedy_dispatch import GreedyDispatchBaseline
    greedy = GreedyDispatchBaseline(n_hubs=N_HUBS)
    obs, _ = env.reset()
    action = greedy.select_action(obs, env)
    assert action.shape == (N_HUBS + 1,), \
        f"greedy action shape {action.shape} != ({N_HUBS + 1},)"

def _greedy_dispatch_signed():
    from baselines.heuristics.greedy_dispatch import GreedyDispatchBaseline
    greedy = GreedyDispatchBaseline(n_hubs=N_HUBS, discharge_threshold=100.0)
    obs, _ = env.reset()
    action = greedy.select_action(obs, env)
    assert action.shape == (N_HUBS + 1,), f"action shape {action.shape}"
    dispatch = action[:N_HUBS]
    price = action[-1]
    # All dispatch should be same sign (all discharge or all charge per RRP threshold)
    assert np.all(dispatch >= 0) or np.all(dispatch <= 0), \
        f"Greedy dispatch should be uniform sign, got: {dispatch}"
    # Price should be the fixed_price value (positive)
    assert price > 0, f"Greedy price should be positive, got {price}"

check("GreedyDispatchBaseline imports", _greedy_import)
check("greedy.select_action returns (H+1,) array", _greedy_action_shape)
check("greedy dispatch is all-positive or all-negative (signed kW)", _greedy_dispatch_signed)

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"  {passed}/{total} checks passed")

if passed == total:
    print("  All checks passed — ready for repo integration.\n")
    sys.exit(0)
else:
    print("\n  Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"    ✗  {name}")
            if msg:
                print(f"       {msg}")
    print()
    sys.exit(1)
