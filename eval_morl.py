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
from envs.dst import DeepSeaTreasureEnv
from envs.utils_building import ParameterGenerator

from gymnasium.wrappers.vector import NormalizeObservation
# Set up vectorized env
# env_id = "minecart-v0"  # or "mo-reacher-v5"
# env_id = "mo-hopper-2obj-v5"  # or "mo-reacher-v5"
env_id = "deep-sea-treasure-v1"  # or "mo-reacher-v5"
num_envs = 16
reward_size = 2
episodes_to_collect = 128
labels = [str(i) for i in range(reward_size)]  # Adjust based on the environment
# ref_point = np.array([-100, -100])  # Reference point for hypervolume calculation
ref_point = np.array([-1000, -1])  # Reference point for hypervolume calculation
# ref_point = np.array([-101, -1001, -101, -101])  # Reference point for hypervolume calculation
# ref_point = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # Reference point for hypervolume calculation
gamma = 1.0
n_to_select = 2048

# model_path = "runs/fruit-tree-v0__main_ppo__2025-11-22 16:10:17.332955__3/"
model_path = "runs/deep-sea-treasure-v1__main_ppo__2025-11-25 22:03:16.736830__1/"

if not os.path.exists(f"results/{env_id}"):
    os.makedirs(f"results/{env_id}", exist_ok=True)

if env_id == "building":
    # Special case for BuildingEnv_9d
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        lambda: BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')) 
        for _ in range(num_envs)
    )
elif env_id == "deep-sea-treasure-v1":
    # Special case for BuildingEnv_9d
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        lambda: DeepSeaTreasureEnv() 
        for _ in range(num_envs)
    )
else:
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        [lambda: mo_gym.make(env_id, max_episode_steps = 500) for _ in range(num_envs)]
    )


try: 
    norm_stats = pickle.load(open(model_path + "norm_stats.pkl", "rb"))
    print(norm_stats)
    mean = norm_stats.mean
    std = np.sqrt(norm_stats.var)
except:
    mean = np.zeros(vec_envs.single_observation_space.shape)
    std = np.ones(vec_envs.single_observation_space.shape)
vec_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(vec_envs)

# Agent
if env_id == "building":
    # Special case for BuildingEnv_9d
    env_temp = BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso'))
elif env_id == "deep-sea-treasure-v1":
    # Special case for BuildingEnv_9d
   env_temp = DeepSeaTreasureEnv()
else:
    env_temp = mo_gym.make(env_id)

if env_temp.action_space.__class__.__name__ == "Box":
    eval_agent = ContinuousAgent(env_temp, reward_size=reward_size).to("cpu")
else:
    eval_agent = DiscreteAgent(env_temp, reward_size=reward_size).to("cpu")
    
eval_agent.load_state_dict(torch.load(model_path + "main_ppo.rl_model"))
# eval_agent.eval()

# Buffers
rewards_list = []
weights_list = []

# Initial env reset and per-env state
obs, _ = vec_envs.reset()
curr_weights = torch.distributions.dirichlet.Dirichlet(torch.ones(reward_size)).sample((num_envs,))
# curr_weights = torch.distributions.uniform.Uniform(low = 0, high = 1).sample((num_envs,reward_size))
env_rewards = np.zeros((num_envs, reward_size))

episodes_collected = 0
pbar = tqdm(total=episodes_to_collect)

gammas = np.ones((num_envs, 1))  # Assuming no discounting for simplicity
while episodes_collected < episodes_to_collect:
    # Agent action for each env, given obs and per-env weights
    obs = (obs - mean)/(std + 1e-8)  # Normalize observations
    actions = []
    actions, _ = eval_agent.predict(obs, curr_weights, deterministic=False, device="cpu")
    # Step all envs
    next_obs, rews, dones, truncs, infos = vec_envs.step(actions)
    env_rewards += gammas * rews
    # Handle episode completion for each env
    gammas *= gamma
    terminations = np.logical_or(dones, truncs)
    if np.any(terminations):
        rewards_list.append(env_rewards[terminations])
        weights_list.append(curr_weights[terminations].cpu().numpy())
        episodes_collected += sum(terminations)
        pbar.update(sum(terminations))
        env_rewards[terminations] = 0  # Reset rewards for finished envs
        gammas[terminations] = 1.0  # Reset gammas for finished envs
        # Reset the finished env
        # single_obs, _ = vec_envs.reset(env_idx)
        # next_obs[env_idx] = single_obs
        # Sample a new weight for this env
        curr_weights[terminations] = torch.distributions.dirichlet.Dirichlet(torch.ones(reward_size)).sample((np.sum(terminations), ))
        # curr_weights[terminations] = torch.distributions.uniform.Uniform(low = 0, high = 1).sample(( np.sum(terminations), reward_size))
        # env_rewards[terminations] = []
    obs = next_obs

pbar.close()

# Additional evaluation with extreme (one-hot) weights
print("Evaluating on extreme (one-hot) preference weights...")
extreme_rewards = []
extreme_weights = []

rewards_list = np.vstack(rewards_list)
weights_list = np.vstack(weights_list)

# rewards_list = np.vstack(rewards_list)
# weights_list = np.vstack(weights_list)

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


def select_points_by_crowd_distance(pareto_points, n_to_select):
    """
    Selects a subset of points from a Pareto front using the crowd distance metric.

    This function identifies the N most spread-out points, which are crucial for
    getting a representative sample of the front, especially for computationally
    expensive tasks like hypervolume calculation.

    Args:
        pareto_points (np.ndarray): A NumPy array of shape (num_points, num_objectives)
                                    representing the points on the Pareto front.
        n_to_select (int): The number of points to select from the front.

    Returns:
        np.ndarray: A NumPy array of shape (n_to_select, num_objectives) containing
                    the selected points.
    """
    num_points, num_objectives = pareto_points.shape

    # If the number to select is greater than or equal to the number of points,
    # return all the points.
    if n_to_select >= num_points:
        return pareto_points

    # --- Step 1: Initialize Distances ---
    # Create an array to store the crowd distance for each point.
    crowding_distances = np.zeros(num_points)

    # --- Step 2: Loop Through Each Objective ---
    for i in range(num_objectives):
        # a. Sort points based on the current objective
        # We get the sorted indices to keep track of the original points
        sorted_indices = np.argsort(pareto_points[:, i])
        sorted_points = pareto_points[sorted_indices]

        # b. Assign infinite distance to boundary points
        # This ensures the extreme points of the front are always selected
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf
        
        # Get the min and max values for normalization
        min_val = sorted_points[0, i]
        max_val = sorted_points[-1, i]
        
        # Avoid division by zero if all values for an objective are the same
        if max_val == min_val:
            continue

        # c. Calculate distance for interior points
        for j in range(1, num_points - 1):
            distance = sorted_points[j + 1, i] - sorted_points[j - 1, i]
            normalized_distance = distance / (max_val - min_val)
            
            # d. Add to the total crowd distance
            crowding_distances[sorted_indices[j]] += normalized_distance

    # --- Step 3: Select the Top N Points ---
    # Sort the original indices based on the calculated crowding distances in descending order
    top_n_indices = np.argsort(crowding_distances)[::-1]

    # Select the first n_to_select indices from the sorted list
    selected_indices = top_n_indices[:n_to_select]

    # Return the corresponding points
    return pareto_points[selected_indices]

mask = pareto_front(rewards_list)
front = rewards_list[mask]
print(rewards_list)
dominated = rewards_list[~mask]
# print(weights_list[mask])
# print(front)
print("Pareto front shape:", front.shape)

# Hypervolume and sparsity
# ref_point = front.min(axis=0) - 1e-6
import itertools

n_obj = front.shape[1]
pairs = list(itertools.combinations(range(reward_size), 2))

print(pairs)

for i, j in pairs:
    xlabel = labels[i] if i < len(labels) else f"Objective {i}"
    ylabel = labels[j] if j < len(labels) else f"Objective {j}"
    
    print(f"Plotting {xlabel} vs {ylabel} for Pareto front and dominated points...")
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

filtered_front = select_points_by_crowd_distance(front, n_to_select)
hv = hypervolume(ref_point=ref_point, points=filtered_front)
sprs = sparsity(front)
print("Hypervolume of Pareto front:", hv)
print("Sparsity of Pareto front:", sprs)
print("Expected utility of Pareto front:", expected_utility(front, weights_list))

print(np.max(front, axis=0))
pickle.dump({
    "rewards": rewards_list,
    "weights": weights_list,
    "mask": mask,
    "pareto_front": front,
    "hypervolume": hv,
    "sparsity": sprs,
    "expected_utility": expected_utility(front, weights_list[mask]),
}, open(f"results/{env_id}/eval_results_{env_id}.pkl", "wb"))