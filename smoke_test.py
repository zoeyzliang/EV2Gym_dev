from nem_env.spatial_graph import HubGraphBuilder
from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel
from nem_env.nem_wdr_env import NEMWDREnv, EnvConfig
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.networks import NetworkConfig
import numpy as np

# Load graph
graph, hub_configs = HubGraphBuilder.load("data/graphs/inner_melbourne.pkl")

# Build env
loader = PriceLoader(seed=0)
loader.load_synthetic(n_days=30)
model = ParticipationModel(seed=0)
env = NEMWDREnv(hub_configs=hub_configs, price_loader=loader,
                participation_model=model, seed=0)

# Build agent
obs_dim = env.observation_space.shape[0]
agent = SACGNNAgent(n_hubs=len(hub_configs), graph_data=graph,
                    obs_dim=obs_dim, learning_starts=100)

# Run 3 episodes
for ep in range(3):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done,
                               wdr_active=info["wdr_active"])
        losses = agent.update()
        obs = next_obs
        total_reward += reward
    print(f"Episode {ep+1}: reward={total_reward:.2f}, "
          f"buffer={agent.buffer.size}")

print("Smoke test passed.")