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
from gymnasium.wrappers.vector import NormalizeObservation
from ppo.utils import RunningMeanStd
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility

# Utility Functions
from utility_functions import (
    DSTDebtUtility, 
    DSTGeneralUtility, 
    FTNProductUtility, 
    FTNBenchmarkUtility
)

# ==========================================
# Environment & Utility Configuration
# ==========================================

# Map environment IDs to their specific utility function lists
ENVIRONMENT_UTILITY_MAP = {
    "deep-sea-treasure-1": [DSTDebtUtility()],
    "deep-sea-treasure-3": [
        DSTGeneralUtility(mode='linear'), 
        DSTGeneralUtility(mode='threshold'), 
        DSTGeneralUtility(mode='ratio')
    ],
    "fruit-tree-1": [FTNProductUtility()],
    "fruit-tree-5": [
        FTNBenchmarkUtility(mode='max'), 
        FTNBenchmarkUtility(mode='min'), 
        FTNBenchmarkUtility(mode='product'), 
        FTNBenchmarkUtility(mode='mixed'), 
        FTNBenchmarkUtility(mode='distance')
    ],
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
    envs,
    num_envs,
    agent_class,
    device,
    model_path,
    gamma,
    capture_video,
):
    # Run simple evaluation to demonstrate how to load and use a trained model
    eval_episodes = 10
    eval_envs = envs

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()
    frames_per_env = [[] for _ in range(num_envs)]  # one list of frames per env

    obs, _ = eval_envs.reset()
    episodic_returns = []
    
    # Init Augmentation
    reward_size = eval_agent.reward_size
    acc_rewards = torch.zeros((num_envs, reward_size)).to(device)
    
    # Init Task (Just pick task 0 for visualization)
    task_size = eval_agent.task_size
    task_onehot = torch.zeros((num_envs, task_size)).to(device)
    if task_size > 0:
        task_onehot[:, 0] = 1.0 # Task 0

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _ = eval_agent.predict(
                torch.Tensor(obs).to(device), 
                acc_rewards,
                task_onehot,
                deterministic = True, 
                device = device
            )
        obs, rews, dones, truncs, infos = eval_envs.step(actions)
        
        # Update Acc Rewards
        acc_rewards += torch.tensor(rews).float().to(device)
        is_done = np.logical_or(dones, truncs)
        if np.any(is_done):
             acc_rewards[is_done] = 0

        if "episode" in infos:
            print(
                    f"Eval episode {len(episodic_returns)}, episodic return: {infos['episode']['r'].sum()}"
                )
            episodic_returns.append(infos["episode"]["r"].sum())

        if capture_video:
            all_frames = eval_envs.render()
            # all_frames is a list of length num_envs, each an RGB array
            for i in range(num_envs):
                frames_per_env[i].append(all_frames[i])

    eval_envs.close()

    # Once done, save each environment's frames to an individual GIF
    if capture_video:
        for i in range(num_envs):
            gif_name = f"gifs/{run_name}_env_{i}.gif"
            # Only save if we actually have frames
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
    num_rollout_steps: int = 2048,
    update_epochs: int = 10,
    num_minibatches: int = 32,
    learning_rate: float = 0.0003,
    gamma: float = 0.995,
    eval_gamma: float = 0.99,
    gae_lambda: float = 0.95,
    surrogate_clip_threshold: float = 0.2,
    entropy_loss_coefficient: float = 0.001,
    policy_gradient_loss_coefficient: float = 1.0,
    value_function_loss_coefficient: float = 0.5,
    normalize_advantages: bool = True,
    normalize_observations: bool = False, # Often False for GridWorlds
    normalize_rewards: bool = True,
    clip_value_function_loss: bool = False,
    max_grad_norm: float = 0.5,
    target_kl: float = None,
    anneal_lr: bool = False,
    rpo_alpha: float = None,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = True,
    save_model: bool = True,
):
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.
    """

    if env_is_discrete and rpo_alpha is not None:
        print(
            f"rpo_alpha is not used in discrete environments. Ignoring rpo_alpha={rpo_alpha}"
        )

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
        
    # 2. Select Environment Constructor
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
            lambda: mo_gym.make("fruit-tree-v0") 
            for _ in range(num_envs)
        )
    else:
        # Generic MO-Gym
        envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
            lambda: gym.wrappers.RecordVideo(mo_gym.make(env_id, render_mode = "rgb_array"), f"runs/{run_name}/videos") for _ in range(num_envs)
        )
        
    if normalize_observations:
        envs = NormalizeObservation(envs)
    envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(envs, gamma=eval_gamma)

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
    actor_params = list(agent.actor_body.parameters()) + list(agent.actor_head.parameters() if env_is_discrete else agent.actor.parameters())
    
    # Define critic parameters
    critic_params = list(agent.critic_body.parameters()) + list(agent.critic_utility_head.parameters()) + list(agent.critic_returns_head.parameters())

    optimizer = [
        torch.optim.Adam(actor_params, lr=learning_rate, eps=1e-5),
        torch.optim.Adam(critic_params, lr=learning_rate, eps=1e-5)
    ]

    if normalize_rewards:
        reward_rms = RunningMeanStd(reward_size)

    logger = PPOLogger(run_name, use_tensorboard, reward_size=reward_size)
    
    pareto_archive = ParetoArchive()
    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        envs=envs,
        utility_functions=utility_functions, # Pass the list of utilities
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
        pareto_archive=pareto_archive
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
        }
        with open(hparams_path, "w") as f:
            json.dump(hparams_to_json, f, indent = 4)
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # 1. Visual Evaluation (GIFs)
        frames = load_and_evaluate_model(
            run_name,
            env_id,
            env_is_discrete,
            normalize_observations,
            envs,
            num_envs,
            agent_class,
            device,
            model_path,
            gamma,
            capture_video,
        )

        if capture_video:
            logger.write_video(frames)
        
    # Close environments
    envs.close()


if __name__ == "__main__":
    run_ppo()