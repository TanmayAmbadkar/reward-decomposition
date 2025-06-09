import os
import numpy as np
import torch
import io
import matplotlib.pyplot as plt
from ppo.agent import DiscreteAgent, ContinuousAgent
import mo_gymnasium as mo_gym
from morl_baselines.common.performance_indicators import hypervolume, sparsity
from tqdm import tqdm

# … your existing setup, loading, rollouts, etc. …
import envs

# env_agent = mo_gym.make("minecart-v0")
env_agent = mo_gym.make("mo-reacher-v5")
# eval_agent = ContinuousAgent(env_agent, reward_size=2).to("cpu")
eval_agent = DiscreteAgent(env_agent, reward_size=4).to("cuda")
# model_path = "66runs/mo-reacher-v5__main_ppo__2025-05-09 19:42:00.901109__1/main_ppo.rl_model"
model_path = "runs/mo-reacher-v5__main_ppo__2025-06-03 22:01:54.263704__1/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))
eval_agent.eval()

labels = ["Target 1", "Target 2", "Target 3", "Target 4"]

rewards_list = []
weights = []

for i in tqdm(range(2000)):
    
    # for j in tqdm(range(50)):
    weight = torch.distributions.dirichlet.Dirichlet(torch.ones((eval_agent.reward_size, ))).sample()
    obs, _ = env_agent.reset()
    w = weight.to("cpu")
    done = trunc = False
    rewards = []
    while not (done or trunc):
        action, value = eval_agent.predict(obs, w, deterministic=True, device="cpu")
        # action, value = env_agent.action_space.sample(), 0
        obs, rew, done, trunc, _ = env_agent.step(action[0])
        # rew[1] = rew[1] - 1
        rewards.append(rew)  # Collect unweighted rewards
        
    rewards = np.array(rewards)
    weights.append(weight.cpu().numpy())
    rewards_list.append(np.sum(rewards, axis=0))
# after you’ve collected `rewards_list` as an (N, d) array:
rewards_list = np.array(rewards_list)

def pareto_front(points: np.ndarray) -> np.ndarray:
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            # Mark as False any point that is dominated by c (i.e., all objectives less than or equal to c, and strictly less in at least one)
            is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | np.all(points[is_efficient] == c, axis=1)
    return is_efficient

mask = pareto_front(rewards_list)
front = rewards_list[mask]
dominated = rewards_list[~mask]

# --- Add this block ---
ref_point = front.min(axis=0) - 1e-6
hv = hypervolume(ref_point=ref_point, points=front)
sprs = sparsity(front)
print("Hypervolume of Pareto front:", hv)
print("Sparsity of Pareto front:", sprs)
# ----------------------

plt.figure(figsize=(7,5))
plt.scatter(dominated[:, 0], dominated[:, 1], alpha=0.4, label="Dominated", color="blue")
plt.scatter(front[:, 0], front[:, 1], alpha=0.8, label="Pareto front", color="red", marker='o', edgecolors='k', s=60)
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.title("Pareto Front vs. Dominated Points")
plt.legend()
plt.tight_layout()
plt.savefig("pareto_front_vs_dominated.png", dpi=150)
plt.show()

