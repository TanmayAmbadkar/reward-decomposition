import os
import random
import time
from datetime import datetime
from functools import partial
import itertools
import pickle
import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

# Custom imports
from func_to_script import script
from ppo.agent import ContinuousAgent, DiscreteAgent
from ppo.ppo import PPO, PPOLogger
import envs
from envs.utils import SyncVectorEnv, RecordEpisodeStatistics
import mo_gymnasium as mo_gym
from envs.building_env import BuildingEnv_9d
from envs.utils_building import ParameterGenerator
from envs.dst import DeepSeaTreasureEnv
from envs.lander import MOLunarLanderEnv
from envs.mo_gym_hopper import MOHopperEnv
from gymnasium.wrappers.vector import NormalizeObservation
from ppo.utils import RunningMeanStd
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility

# Utility Functions
from utility_functions import *

# ==========================================
# Environment & Utility Configuration
# ==========================================

ENVIRONMENT_UTILITY_MAP = {
    "deep-sea-treasure-1": [DSTDebtUtility()],
    "deep-sea-treasure-3": [
        DSTGeneralUtility(mode='linear'), 
        DSTGeneralUtility(mode='threshold'), 
        DSTGeneralUtility(mode='ratio')
    ],
    "fruit-tree-prod": [FTNProductUtility()],
    "fruit-tree-max": [FTNMaxUtility()],
    "fruit-tree-min": [FTNMinUtility()],
    "fruit-tree-2prod": [FTN2ProductUtility()],
    "fruit-tree-mixed": [FTNMixedUtility()],
    "fruit-tree-dist": [FTNDistanceUtility()],
    "fruit-tree-5": [
        FTNMaxUtility(),
        FTNMinUtility(),
        FTN2ProductUtility(),
        FTNMixedUtility(),
        FTNDistanceUtility(),
    ],
    "minecart-nsw-speed": [NSWSpeedRatioUtility()],
    # Single-utility Lunar Lander experiments (learning curves, Fig 1 equivalent)
    "lunar-lander-fuel": [LLTrajectoryQuality()],
    "lunar-lander-joint": [LLJointSuccess()],
    "lunar-lander-3": [
        LLTrajectoryQuality(),
        LLJointSuccess(),
        LLSafetyFirst(),
    ],
    "hopper-linear": [HopperLinearCalibration()],
}

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def load_and_evaluate_model(
    run_name,
    env_id,
    env_is_discrete,
    normalize_observations,
    obs_rms,
    envs,
    num_envs,
    agent_class,
    device,
    model_path,
    gamma,
    capture_video,
    eval_task_id=None
):
    # Run comprehensive evaluation matching PPO.evaluate logic
    eval_episodes = 10
    eval_envs = envs

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()
    frames_per_env = [[] for _ in range(num_envs)]  # one list of frames per env

    # Get Task Size & Utils
    task_size = eval_agent.task_size
    utility_funcs = ENVIRONMENT_UTILITY_MAP.get(env_id, None)
    if utility_funcs is None:
        utility_funcs = [lambda r: r.sum(-1)] # Default to sum
        
    num_tasks = len(utility_funcs)
    reward_size = eval_agent.reward_size

    print(f"--- Starting Comprehensive Evaluation ---")
    
    # Dictionary to store results: task_id -> list of utilities
    results = {t_id: [] for t_id in range(num_tasks)}
    
    # We process tasks in chunks based on available eval envs
    # If eval_task_id is specified, only evaluate that task
    if eval_task_id is not None:
        tasks_to_run = np.repeat([eval_task_id], eval_episodes)
    else:
        tasks_to_run = np.repeat(np.arange(num_tasks), eval_episodes)
        
    total_episodes_needed = len(tasks_to_run)
    
    obs, _ = eval_envs.reset()
    if normalize_observations:
        obs = (obs - obs_rms.mean) / ((obs_rms.var + 1e-8) ** 0.5)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    
    acc_rewards = torch.zeros((num_envs, reward_size), device=device)
    acc_gamma = torch.ones((num_envs, 1), device=device)
    
    active_tasks = torch.zeros(num_envs, dtype=torch.long, device=device)
    env_task_ptr = np.full(num_envs, -1, dtype=np.int32)
    
    # Fill initially
    params_ptr = 0
    for i in range(num_envs):
        if params_ptr < total_episodes_needed:
            active_tasks[i] = int(tasks_to_run[params_ptr])
            env_task_ptr[i] = params_ptr
            params_ptr += 1
    
    def get_one_hot_task(task_idx_tensor, batch_size):
        task_one_hot = torch.zeros((batch_size, num_tasks), device=device)
        if num_tasks > 0:
            task_idx_tensor = task_idx_tensor.long()
            task_one_hot.scatter_(1, task_idx_tensor.unsqueeze(1), 1.0)
        return task_one_hot

    while (env_task_ptr != -1).any():
        task_one_hot = get_one_hot_task(active_tasks, num_envs)
        
        with torch.no_grad():
            actions, _ = eval_agent.predict(
                obs, 
                acc_rewards,
                task_one_hot,
                deterministic=True, 
                device=device
            )
        
        next_obs, rews, dones, truncs, infos = eval_envs.step(actions)
        if normalize_observations:
            next_obs = (next_obs - obs_rms.mean) / ((obs_rms.var + 1e-8) ** 0.5)
        
        reward_tens = torch.tensor(rews, dtype=torch.float32, device=device).reshape(num_envs, reward_size)
        
        # Mask out idle environments
        idle_mask = torch.tensor(env_task_ptr == -1, device=device).unsqueeze(1)
        reward_tens[idle_mask.squeeze(1)] = 0.0

        acc_rewards += acc_gamma * reward_tens
        acc_gamma *= gamma
        
        is_done = torch.logical_or(torch.tensor(dones), torch.tensor(truncs)).to(device)
        
        if is_done.any():
            done_indices = torch.where(is_done)[0]
            for idx in done_indices:
                if env_task_ptr[idx.item()] != -1:
                    task_id = active_tasks[idx].item()
                    final_vec = acc_rewards[idx]
                    u_val = utility_funcs[task_id](final_vec).item()
                    results[task_id].append(u_val)
                    
                    print(f"Task {task_id} finished. Vec={final_vec.cpu().numpy()}, Utility={u_val:.3f}")
                    
                    acc_rewards[idx] = 0
                    acc_gamma[idx] = 1.0
                    
                    if params_ptr < total_episodes_needed:
                        active_tasks[idx] = int(tasks_to_run[params_ptr])
                        env_task_ptr[idx.item()] = params_ptr
                        params_ptr += 1
                    else:
                        env_task_ptr[idx.item()] = -1 
        
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        if capture_video:
            all_frames = eval_envs.render()
            for i in range(num_envs):
                # Only record frames if the environment is active
                if env_task_ptr[i] != -1:
                    frames_per_env[i].append(all_frames[i])

    eval_envs.close()

    print("\n--- Evaluation Summary ---")
    for t_id, vals in results.items():
        if len(vals) > 0:
            print(f"Task {t_id}: Min={np.min(vals):.3f} Mean={np.mean(vals):.3f} Max={np.max(vals):.3f} | Returns={vals}")

    # Once done, save each environment's frames to an individual GIF
    if capture_video:
        for i in range(num_envs):
            gif_name = f"gifs/{run_name}_eval_env_{i}.gif"
            if len(frames_per_env[i]) > 0:
                imageio.mimsave(gif_name, frames_per_env[i], fps=30)
                print(f"Saved GIF for env {i}: {gif_name}")
                
    return frames_per_env

@script
def run_ppo(
    env_id: str = "deep-sea-treasure-1", # Default to new ID
    env_is_discrete: bool = True, # DST is discrete
    num_envs: int = 4,
    convex: bool = True,
    scalar_reward: bool = False,
    total_timesteps: int = 5000000,
    num_rollout_steps: int = 512,
    update_epochs: int = 10,
    num_minibatches: int = 32,
    learning_rate: float = 0.0003,
    gamma: float = 1.0,
    eval_gamma: float = 1.0,
    gae_lambda: float = 1.0,
    surrogate_clip_threshold: float = 0.2,
    entropy_loss_coefficient: float = 0.02,
    policy_gradient_loss_coefficient: float = 1.0,
    value_function_loss_coefficient: float = 0.5,
    normalize_advantages: bool = True,
    normalize_observations: bool = True, # Often False for GridWorlds
    normalize_rewards: bool = False,
    clip_value_function_loss: bool = False,
    max_grad_norm: float = 10.0,
    target_kl: float = None,
    anneal_lr: bool = False,
    rpo_alpha: float = None,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = True,
    save_model: bool = True,
    eval_interval: int = 10000,
    num_eval_episodes: int = 10
):
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.
    """

    if env_is_discrete and rpo_alpha is not None:
        print(
            f"rpo_alpha is not used in discrete environments. Ignoring rpo_alpha={rpo_alpha}"
        )
    # After the env_id parameter is received, before environment construction
    # Set up run name and logging
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}__{exp_name}__{datetime.now()}__{seed}"
    set_seed(seed, torch_deterministic)

    # Set up device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ==========================
    # Environment Initialization
    # ==========================
    
    # 1. Select Utility Functions based on Env ID map
    utility_functions = ENVIRONMENT_UTILITY_MAP.get(env_id, None)
    if utility_functions is None:
        # Fallback / Warning
        print(f"Warning: No utility functions mapped for {env_id}. Defaulting to Sum.")
        
    # 2. Select Environment Constructor (User Specified Logic)
    if env_id == "building":
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')) 
            for _ in range(num_envs)
        )
    elif env_id.startswith("deep-sea-treasure"):
        # Map variants to specific env args if needed, or just use base
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: DeepSeaTreasureEnv() 
            for _ in range(num_envs)
        )
    elif env_id.startswith("fruit-tree"):
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: mo_gym.make("fruit-tree-v0", depth = 7) 
            for _ in range(num_envs)
        )
    elif env_id.startswith("minecart"):
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: mo_gym.make("minecart-v0") 
            for _ in range(num_envs)
        )
    elif env_id.startswith("lunar-lander"):
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            [lambda: MOLunarLanderEnv(continuous=not env_is_discrete)
            for _ in range(num_envs)]
        )
    elif env_id.startswith("hopper"):
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            [lambda: MOHopperEnv()
            for _ in range(num_envs)]
        )
    else:
        # Generic MO-Gym
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: gym.wrappers.RecordVideo(mo_gym.make(env_id, render_mode = "rgb_array"), f"runs/{run_name}/videos") for _ in range(num_envs)
        )
        
    if normalize_observations:
        envs = NormalizeObservation(envs)
    envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(envs, gamma=eval_gamma)

    # 3. Create Evaluation Environments (Mirroring User Logic)
    # We replicate the construction logic to ensure eval envs match training envs exactly.
    # Note: We skip RecordVideo for eval_envs to reduce I/O overhead during periodic eval.
    if env_id == "building":
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')) 
            for _ in range(num_envs)
        )
    elif env_id.startswith("deep-sea-treasure"):
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: DeepSeaTreasureEnv() 
            for _ in range(num_envs)
        )
    elif env_id.startswith("fruit-tree"):
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: mo_gym.make("fruit-tree-v0", depth = 7) 
            for _ in range(num_envs)
        )
    elif env_id.startswith("minecart"):
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: mo_gym.make("minecart-v0") 
            for _ in range(num_envs)
        )
    elif env_id.startswith("lunar-lander"):
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            [lambda: gym.make("lunar-lander-v1")
            for _ in range(num_envs)]
        )
    elif env_id.startswith("hopper"):
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            [lambda: MOHopperEnv()
            for _ in range(num_envs)]
        )
    else:
        # Generic MO-Gym (No RecordVideo for internal eval)
        eval_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: mo_gym.make(env_id, render_mode = "rgb_array") for _ in range(num_envs)
        )

    if normalize_observations:
        eval_envs = NormalizeObservation(eval_envs)
    eval_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(eval_envs, gamma=eval_gamma)

    print(f"Env: {env_id}, Reward Shape: {envs.rewards_shape}")
    print(f"Obs Space: {envs.observation_space}, Action Space: {envs.action_space}")
    print(f"Selected Utility Functions: {len(utility_functions) if utility_functions else 0}")

    # Set up agent
    
    reward_size = envs.rewards_shape[-1]
    
    # Task Size = Number of Utility Functions
    task_size = len(utility_functions) if utility_functions else 0
    
    agent_class = (
        partial(DiscreteAgent, reward_size=reward_size, task_size=task_size)
        if env_is_discrete
        else partial(ContinuousAgent, rpo_alpha=rpo_alpha, reward_size=reward_size, task_size=task_size)
    )

    if "mario" in env_id or "rgb" in env_id:
        agent_class = partial(CNNDiscreteAgent, reward_size=reward_size, task_size=task_size)

    agent = agent_class(envs).to(device)

    # Optimizer
    # Define actor parameters
    # actor_params = list(agent.actor.parameters()) + list(agent.actor_head.parameters() if env_is_discrete else agent.actor.parameters())
    
    # Define critic parameters
    # critic_params = list(agent.critic_body.parameters()) + list(agent.critic_utility_head.parameters()) + list(agent.critic_returns_head.parameters())

    optimizer = [
        torch.optim.Adam(agent.actor.parameters(), lr=learning_rate, eps=1e-5),
        torch.optim.Adam(agent.critic.parameters(), lr=learning_rate, eps=1e-5)
    ]

    if normalize_rewards:
        reward_rms = RunningMeanStd(reward_size)

    logger = PPOLogger(run_name, use_tensorboard, reward_size=reward_size)
    
    pareto_archive = ParetoArchive()
    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        envs=envs,
        eval_envs=eval_envs, # Pass the separate eval environments
        utility_functions=utility_functions, 
        env_is_discrete=env_is_discrete,
        reward_size=reward_size,
        learning_rate=learning_rate,
        num_rollout_steps=num_rollout_steps,
        num_envs=num_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        surrogate_clip_threshold=surrogate_clip_threshold,
        entropy_loss_coefficient=entropy_loss_coefficient,
        value_function_loss_coefficient=value_function_loss_coefficient,
        policy_gradient_loss_coefficient = policy_gradient_loss_coefficient,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        normalize_advantages=normalize_advantages,
        reward_rms=reward_rms if normalize_rewards else None,
        clip_value_function_loss=clip_value_function_loss,
        target_kl=target_kl,
        anneal_lr=anneal_lr,
        seed=seed,
        logger=logger,
        convex=convex,
        scalar_reward=scalar_reward,
        pareto_archive=pareto_archive,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes
    )
    
    # Train the agent
    trained_agent = ppo.learn(total_timesteps)

    if save_model:
        if not os.path.exists(f"runs/{run_name}"):
            os.mkdir(f"runs/{run_name}")
        model_path = f"runs/{run_name}/{exp_name}.rl_model"
        hparams_path = f"runs/{run_name}/hparams.json"
        
        obs_rms = None
        if normalize_observations:
            stats_path = f"runs/{run_name}/norm_stats.pkl"
            obs_rms = envs.env.obs_rms
            pickle.dump(obs_rms, open(stats_path, "wb"))
            
        
        hparams_to_json = {
            "env_id": env_id,
            "env_is_discrete": env_is_discrete,
            "num_envs": num_envs,
            "convex": convex,
            "scalar_reward": scalar_reward,
            "total_timesteps": total_timesteps,
            "num_rollout_steps": num_rollout_steps,
            "update_epochs": update_epochs,
            "num_minibatches": num_minibatches,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "surrogate_clip_threshold": surrogate_clip_threshold,
            "entropy_loss_coefficient": entropy_loss_coefficient,
            "value_function_loss_coefficient": value_function_loss_coefficient,
            "policy_gradient_loss_coefficient": policy_gradient_loss_coefficient,
            "normalize_advantages": normalize_advantages,
            "normalize_observations": normalize_observations,
            "normalize_rewards": normalize_rewards,
            "clip_value_function_loss": clip_value_function_loss,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl,
            "anneal_lr": anneal_lr,
            "rpo_alpha": rpo_alpha,
            "seed": seed,
            "eval_interval": eval_interval,
            "num_eval_episodes": num_eval_episodes
        }
        with open(hparams_path, "w") as f:
            json.dump(hparams_to_json, f, indent = 4)
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # 1. Visual Evaluation (GIFs)
        # Note: We reuse training envs for visual check at end to save re-instantiation
        frames = load_and_evaluate_model(
            run_name,
            env_id,
            env_is_discrete,
            normalize_observations,
            envs.env.obs_rms if normalize_observations else None,
            envs,
            num_envs,
            agent_class,
            device,
            model_path,
            eval_gamma,
            capture_video,
        )

        if capture_video:
            logger.write_video(frames)
        
    # Close environments
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    run_ppo()