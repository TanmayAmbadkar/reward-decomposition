import numpy as np
import torch
import matplotlib.pyplot as plt
from ppo.agent import ContinuousAgent, DiscreteAgent
import mo_gymnasium as mo_gym
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility
from tqdm import tqdm
import pickle
import envs
import os
from envs.building_env import BuildingEnv_9d
from envs.utils_building import ParameterGenerator
# Set up vectorized env
env_id = "building"  # or "mo-reacher-v5"
num_envs = 4
reward_size = 9
episodes_to_collect = 128
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Adjust based on the environment
ref_point = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # Reference point for hypervolume calculation

if not os.path.exists(f"results/{env_id}"):
    os.makedirs(f"results/{env_id}", exist_ok=True)

if env_id == "building":
    # Special case for BuildingEnv_9d
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        lambda: BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')) 
        for _ in range(num_envs)
    )
else:
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        [lambda: mo_gym.make(env_id, max_episode_steps = 500) for _ in range(num_envs)]
    )
vec_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(vec_envs)

# Agent
if env_id == "building":
    # Special case for BuildingEnv_9d
    env_temp = BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso'))
else:
    env_temp = mo_gym.make(env_id)
eval_agent = ContinuousAgent(env_temp, reward_size=reward_size).to("cuda")
# eval_agent = DiscreteAgent(mo_gym.make(env_id), reward_size=reward_size).to("cuda")
model_path = "runs/building__main_ppo__2025-06-08 23:30:47.097506__1/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))
eval_agent.eval()

# Buffers
rewards_list = []
weights_list = []

# Initial env reset and per-env state
obs, _ = vec_envs.reset()
curr_weights = torch.distributions.dirichlet.Dirichlet(torch.ones(reward_size)).sample((num_envs,))
env_rewards = np.zeros((num_envs, reward_size))

episodes_collected = 0
pbar = tqdm(total=episodes_to_collect)

while episodes_collected < episodes_to_collect:
    # Agent action for each env, given obs and per-env weights
    actions = []
    actions, _ = eval_agent.predict(obs, curr_weights, deterministic=True, device="cuda")
    # Step all envs
    next_obs, rews, dones, truncs, infos = vec_envs.step(actions)
    env_rewards += rews
    # Handle episode completion for each env
    
    terminations = np.logical_or(dones, truncs)
    if np.any(terminations):
        rewards_list.append(env_rewards[terminations])
        weights_list.append(curr_weights[terminations].cpu().numpy())
        episodes_collected += sum(terminations)
        pbar.update(sum(terminations))
        env_rewards[terminations] = 0  # Reset rewards for finished envs
        # Reset the finished env
        # single_obs, _ = vec_envs.reset(env_idx)
        # next_obs[env_idx] = single_obs
        # Sample a new weight for this env
        curr_weights[terminations] = torch.distributions.dirichlet.Dirichlet(torch.ones(reward_size)).sample((np.sum(terminations), ))
        # env_rewards[terminations] = []
    obs = next_obs

pbar.close()
rewards_list = np.vstack(rewards_list)
weights_list = np.vstack(weights_list)

# Pareto front calculation (robust and correct for maximization)
def pareto_front(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if all(points[j] >= points[i]) and any(points[j] > points[i]):
                is_efficient[i] = False
                break
    return is_efficient

mask = pareto_front(rewards_list)
front = rewards_list[mask]
dominated = rewards_list[~mask]
print(weights_list[mask])

# Hypervolume and sparsity
# ref_point = front.min(axis=0) - 1e-6
import itertools

n_obj = front.shape[1]
pairs = list(itertools.combinations(range(n_obj), 2))

for i, j in pairs:
    xlabel = labels[i] if i < len(labels) else f"Objective {i}"
    ylabel = labels[j] if j < len(labels) else f"Objective {j}"
    
    # 1. Pareto front vs. Dominated Points
    plt.figure(figsize=(7,5))
    plt.scatter(dominated[:, i], dominated[:, j], alpha=0.4, label="Dominated", color="blue")
    plt.scatter(front[:, i], front[:, j], alpha=0.8, label="Pareto front", color="red",
                marker='o', edgecolors='k', s=60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front vs. Dominated Points ({xlabel} vs. {ylabel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pareto_front_vs_dominated_{env_id}_({xlabel} vs. {ylabel}).png", dpi=150)
    plt.close()
    
    # 2. Pareto front only
    plt.figure(figsize=(7,5))
    plt.scatter(front[:, i], front[:, j], alpha=0.8, label="Pareto front", color="red",
                marker='o', edgecolors='k', s=60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front ({xlabel} vs. {ylabel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pareto_front_{env_id}_({xlabel} vs. {ylabel}).png", dpi=150)
    plt.close()

hv = hypervolume(ref_point=ref_point, points=front)
sprs = sparsity(front)
print("Hypervolume of Pareto front:", hv)
print("Sparsity of Pareto front:", sprs)
print("Expected utility of Pareto front:", expected_utility(front, weights_list[mask]))
pickle.dump({
    "rewards": rewards_list,
    "weights": weights_list,
    "mask": mask,
    "pareto_front": front,
    "hypervolume": hv,
    "sparsity": sprs,
    "expected_utility": expected_utility(front, weights_list[mask]),
}, open(f"results/{env_id}/eval_results_{env_id}.pkl", "wb"))