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
from ppo.ppo import PPO, PPOLogger   # CHANGED: import ESR_PPO
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
        DSTGeneralUtility(mode='ratio'),
    ],
    "fruit-tree-prod":  [FTNProductUtility()],
    "fruit-tree-max":   [FTNMaxUtility()],
    "fruit-tree-min":   [FTNMinUtility()],
    "fruit-tree-2prod": [FTN2ProductUtility()],
    "fruit-tree-mixed": [FTNMixedUtility()],
    "fruit-tree-dist":  [FTNDistanceUtility()],
    "fruit-tree-5": [
        FTNMaxUtility(),
        FTNMinUtility(),
        FTN2ProductUtility(),
        FTNMixedUtility(),
        FTNDistanceUtility(),
    ],
    "minecart-nsw-speed": [NSWSpeedRatioUtility()],
    # Lunar Lander
    "lunar-lander-quality": [LLTrajectoryQuality()],
    "lunar-lander-joint":   [LLJointSuccess()],
    "lunar-lander-3": [
        LLTrajectoryQuality(),
        LLJointSuccess(),
        LLSafetyFirst(),
    ],
    # Hopper
    "hopper-linear": [HopperLinearCalibration()],
}


def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def _make_envs(env_id, num_envs, run_name=None, record_video=False):
    """
    Centralised environment factory.
    Returns a MOSyncVectorEnv for any supported env_id.
    Using a single factory avoids duplicating construction logic for
    training and eval environments.
    """
    def _make_one():
        if env_id == "building":
            return BuildingEnv_9d(
                ParameterGenerator(
                    Building='OfficeLarge',
                    Weather='Warm_Marine',
                    Location='ElPaso',
                )
            )
        elif env_id.startswith("deep-sea-treasure"):
            return DeepSeaTreasureEnv()
        elif env_id.startswith("fruit-tree"):
            return mo_gym.make("fruit-tree-v0", depth=7)
        elif env_id.startswith("minecart"):
            return mo_gym.make("minecart-v0")
        elif env_id.startswith("lunar-lander"):
            # env_is_discrete is not available here — default to continuous
            # override by passing continuous= explicitly if needed
            return MOLunarLanderEnv(continuous=True)
        elif env_id.startswith("hopper"):
            return MOHopperEnv()
        else:
            if record_video and run_name is not None:
                return gym.wrappers.RecordVideo(
                    mo_gym.make(env_id, render_mode="rgb_array"),
                    f"runs/{run_name}/videos",
                )
            return mo_gym.make(env_id, render_mode="rgb_array")

    return mo_gym.wrappers.vector.MOSyncVectorEnv(
        [_make_one for _ in range(num_envs)]
    )


def load_and_evaluate_model(
    run_name,
    env_id,
    env_is_discrete,
    normalize_observations,
    obs_rms,
    eval_envs,
    num_envs,
    agent_class,
    device,
    model_path,
    capture_video,
    eval_task_id=None,
):
    """
    Post-training comprehensive evaluation.
    Accumulates raw (undiscounted) returns then applies utility functions.
    """
    eval_episodes  = 10
    utility_funcs  = ENVIRONMENT_UTILITY_MAP.get(env_id, [lambda r: r.sum(-1)])
    num_tasks      = len(utility_funcs)

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()

    reward_size    = eval_agent.reward_size
    frames_per_env = [[] for _ in range(num_envs)]
    results        = {t_id: [] for t_id in range(num_tasks)}

    if eval_task_id is not None:
        tasks_to_run = np.repeat([eval_task_id], eval_episodes)
    else:
        tasks_to_run = np.repeat(np.arange(num_tasks), eval_episodes)

    total_episodes_needed = len(tasks_to_run)

    obs, _ = eval_envs.reset()
    if normalize_observations and obs_rms is not None:
        obs = (obs - obs_rms.mean) / ((obs_rms.var + 1e-8) ** 0.5)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # FIXED: raw accumulation — no gamma discounting
    acc_rewards  = torch.zeros((num_envs, reward_size), device=device)
    active_tasks = torch.zeros(num_envs, dtype=torch.long, device=device)
    env_task_ptr = np.full(num_envs, -1, dtype=np.int32)

    params_ptr = 0
    for i in range(num_envs):
        if params_ptr < total_episodes_needed:
            active_tasks[i] = int(tasks_to_run[params_ptr])
            env_task_ptr[i] = params_ptr
            params_ptr     += 1

    def get_one_hot_task(task_idx_tensor, batch_size):
        task_one_hot = torch.zeros((batch_size, num_tasks), device=device)
        if num_tasks > 0:
            task_one_hot.scatter_(1, task_idx_tensor.long().unsqueeze(1), 1.0)
        return task_one_hot

    while (env_task_ptr != -1).any():
        task_one_hot = get_one_hot_task(active_tasks, num_envs)

        with torch.no_grad():
            actions, _ = eval_agent.predict(
                obs, acc_rewards, task_one_hot,
                deterministic=True, device=device,
            )

        next_obs, rews, dones, truncs, infos = eval_envs.step(actions)

        if normalize_observations and obs_rms is not None:
            next_obs = (next_obs - obs_rms.mean) / ((obs_rms.var + 1e-8) ** 0.5)

        reward_tens = torch.tensor(
            rews, dtype=torch.float32, device=device
        ).reshape(num_envs, reward_size)

        idle_mask = torch.tensor(env_task_ptr == -1, device=device)
        reward_tens[idle_mask] = 0.0

        # FIXED: raw accumulation, no gamma
        acc_rewards += reward_tens

        is_done = torch.logical_or(
            torch.tensor(dones), torch.tensor(truncs)
        ).to(device)

        if is_done.any():
            for idx in torch.where(is_done)[0]:
                i = idx.item()
                if env_task_ptr[i] != -1:
                    task_id   = active_tasks[idx].item()
                    final_vec = acc_rewards[idx]
                    u_val     = utility_funcs[task_id](final_vec).item()
                    results[task_id].append(u_val)

                    print(
                        f"Task {task_id} | "
                        f"Vec={final_vec.cpu().numpy().round(3)} | "
                        f"Utility={u_val:.4f}"
                    )

                    acc_rewards[idx] = 0.0

                    if params_ptr < total_episodes_needed:
                        active_tasks[idx] = int(tasks_to_run[params_ptr])
                        env_task_ptr[i]   = params_ptr
                        params_ptr       += 1
                    else:
                        env_task_ptr[i] = -1

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        if capture_video:
            all_frames = eval_envs.render()
            for i in range(num_envs):
                if env_task_ptr[i] != -1:
                    frames_per_env[i].append(all_frames[i])

    eval_envs.close()

    print("\n--- Evaluation Summary ---")
    for t_id, vals in results.items():
        if vals:
            print(
                f"Task {t_id}: "
                f"Min={np.min(vals):.4f}  "
                f"Mean={np.mean(vals):.4f}  "
                f"Max={np.max(vals):.4f} | "
                f"All={[round(v,4) for v in vals]}"
            )

    if capture_video:
        for i in range(num_envs):
            gif_name = f"gifs/{run_name}_eval_env_{i}.gif"
            if frames_per_env[i]:
                imageio.mimsave(gif_name, frames_per_env[i], fps=30)
                print(f"Saved GIF: {gif_name}")

    return frames_per_env


@script
def run_ppo(
    env_id: str                  = "deep-sea-treasure-1",
    env_is_discrete: bool        = True,
    num_envs: int                = 4,
    total_timesteps: int         = 5000000,
    # ESR-PPO episode collection params (replaces num_rollout_steps)
    episodes_per_update: int     = 64*4,
    min_episodes_per_task: int   = 64,
    max_episode_steps: int       = 1000,
    # Optimisation
    update_epochs: int           = 5,
    num_minibatches: int         = 8,
    learning_rate: float         = 3e-3,
    gamma: float                 = 1.0,       # ESR uses undiscounted returns
    discount_utility: bool            = True,      # whether to discount utility calculation
    surrogate_clip_threshold: float  = 0.2,
    entropy_loss_coefficient: float  = 0.01,
    max_grad_norm: float         = 0.5,
    target_kl: float             = None,
    anneal_lr: bool              = True,
    normalize_advantages: bool   = False,
    normalize_observations: bool = False,
    normalize_rewards: bool      = False,
    # Counterfactual IS weighting
    cf_weight_min: float         = 0.1,    # floor IS weight
    cf_weight_max: float         = 5.0,    # ceiling IS weight
   # Agent
    rpo_alpha: float             = None,
    # Misc
    seed: int                    = 1,
    torch_deterministic: bool    = True,
    capture_video: bool          = False,
    use_tensorboard: bool        = True,
    save_model: bool             = True,
    eval_interval: int           = 5000,
    num_eval_episodes: int       = 10,
    
):
    """
    Main training entry point for ESR-PPO.

    Key differences from standard PPO:
      - num_rollout_steps replaced by episodes_per_update + min_episodes_per_task
      - gae_lambda removed (no GAE — MC returns on complete episodes)
      - traj_clip_threshold added (trajectory-level PPO clip)
      - counterfactual_quantile added (adaptive IS threshold)
      - gamma fixed to 1.0 (ESR requires undiscounted returns)
    """
    if env_is_discrete and rpo_alpha is not None:
        print(f"rpo_alpha ignored for discrete environments.")

    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}__{exp_name}__{datetime.now()}__{seed}"
    set_seed(seed, torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------------
    utility_functions = ENVIRONMENT_UTILITY_MAP.get(env_id, None)
    if utility_functions is None:
        print(f"Warning: No utility functions for {env_id}. Defaulting to sum.")
        utility_functions = [lambda r: r.sum(-1)]

    # -----------------------------------------------------------------------
    # Environments
    # -----------------------------------------------------------------------
    training_envs = _make_envs(env_id, num_envs, run_name, record_video=False)
    eval_envs_raw = _make_envs(env_id, num_envs, run_name, record_video=False)

    if normalize_observations:
        training_envs = NormalizeObservation(training_envs)
        eval_envs_raw = NormalizeObservation(eval_envs_raw)

    # FIXED: always gamma=1.0 for MORecordEpisodeStatistics so dr field
    # contains raw undiscounted returns (needed for utility evaluation)
    training_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(
        training_envs, gamma=1.0
    )
    eval_envs_wrapped = mo_gym.wrappers.vector.MORecordEpisodeStatistics(
        eval_envs_raw, gamma=1.0
    )

    reward_size = training_envs.rewards_shape[-1]
    task_size   = len(utility_functions)

    print(f"Env: {env_id}")
    print(f"Reward shape: {training_envs.rewards_shape}")
    print(f"Obs space:    {training_envs.observation_space}")
    print(f"Action space: {training_envs.action_space}")
    print(f"Utility functions: {task_size}")

    # -----------------------------------------------------------------------
    # Agent
    # -----------------------------------------------------------------------
    agent_class = (
        partial(DiscreteAgent,   reward_size=reward_size, task_size=task_size)
        if env_is_discrete
        else partial(ContinuousAgent, rpo_alpha=rpo_alpha,
                     reward_size=reward_size, task_size=task_size)
    )
    agent = agent_class(training_envs).to(device)

    optimizer = [
        torch.optim.Adam(agent.actor.parameters(),  lr=learning_rate, eps=1e-5),
        torch.optim.Adam(agent.critic.parameters(), lr=learning_rate, eps=1e-5),
    ]

    reward_rms = RunningMeanStd(reward_size) if normalize_rewards else None
    logger     = PPOLogger(run_name, use_tensorboard, reward_size=reward_size)

    # -----------------------------------------------------------------------
    # ESR-PPO (CHANGED: was PPO, now ESR_PPO)
    # -----------------------------------------------------------------------
    esr_ppo = PPO(
        agent                    = agent,
        optimizer                = optimizer,
        envs                     = training_envs,
        eval_envs                = eval_envs_wrapped,
        utility_functions        = utility_functions,
        env_is_discrete          = env_is_discrete,
        reward_size              = reward_size,
        learning_rate            = learning_rate,
        # Episode collection
        episodes_per_update      = episodes_per_update,
        min_episodes_per_task    = min_episodes_per_task,
        max_episode_steps        = max_episode_steps,
        # Optimisation
        update_epochs            = update_epochs,
        num_minibatches          = num_minibatches,
        surrogate_clip_threshold = surrogate_clip_threshold,
        entropy_loss_coefficient = entropy_loss_coefficient,
        max_grad_norm            = max_grad_norm,
        normalize_advantages     = normalize_advantages,
        target_kl                = target_kl,
        anneal_lr                = anneal_lr,
        # Counterfactual
        cf_weight_min            = cf_weight_min,
        cf_weight_max            = cf_weight_max,
        # Return discounting
        discount_utility         = discount_utility,
       # Misc
        gamma                    = gamma,
        seed                     = seed,
        logger                   = logger,
        eval_interval            = eval_interval,
        num_eval_episodes        = num_eval_episodes,
        total_timesteps          = total_timesteps,
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    trained_agent = esr_ppo.learn(total_timesteps)

    # -----------------------------------------------------------------------
    # Save model + final evaluation
    # -----------------------------------------------------------------------
    if save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path   = f"runs/{run_name}/{exp_name}.rl_model"
        hparams_path = f"runs/{run_name}/hparams.json"

        obs_rms = None
        if normalize_observations:
            stats_path = f"runs/{run_name}/norm_stats.pkl"
            obs_rms    = training_envs.env.obs_rms
            pickle.dump(obs_rms, open(stats_path, "wb"))

        hparams = {
            "env_id":                    env_id,
            "env_is_discrete":           env_is_discrete,
            "num_envs":                  num_envs,
            "total_timesteps":           total_timesteps,
            "episodes_per_update":       episodes_per_update,
            "min_episodes_per_task":     min_episodes_per_task,
            "max_episode_steps":         max_episode_steps,
            "update_epochs":             update_epochs,
            "num_minibatches":           num_minibatches,
            "learning_rate":             learning_rate,
            "gamma":                     gamma,
            "discount_utility":          discount_utility,
            "surrogate_clip_threshold":  surrogate_clip_threshold,
            "entropy_loss_coefficient":  entropy_loss_coefficient,
            "max_grad_norm":             max_grad_norm,
            "normalize_advantages":      normalize_advantages,
            "normalize_observations":    normalize_observations,
            "normalize_rewards":         normalize_rewards,
            "target_kl":                 target_kl,
            "anneal_lr":                 anneal_lr,
            "cf_weight_min":             cf_weight_min,
            "cf_weight_max":             cf_weight_max,
            "discount_utility":          discount_utility,
           "rpo_alpha":                 rpo_alpha,
            "seed":                      seed,
            "eval_interval":             eval_interval,
            "num_eval_episodes":         num_eval_episodes,
        }
        with open(hparams_path, "w") as f:
            json.dump(hparams, f, indent=4)

        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved: {model_path}")

        # Final comprehensive evaluation
        final_eval_envs = _make_envs(env_id, num_envs)
        if normalize_observations:
            final_eval_envs = NormalizeObservation(final_eval_envs)
        final_eval_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(
            final_eval_envs, gamma=1.0
        )

        frames = load_and_evaluate_model(
            run_name         = run_name,
            env_id           = env_id,
            env_is_discrete  = env_is_discrete,
            normalize_observations = normalize_observations,
            obs_rms          = obs_rms,
            eval_envs        = final_eval_envs,
            num_envs         = num_envs,
            agent_class      = agent_class,
            device           = device,
            model_path       = model_path,
            capture_video    = capture_video,
        )

        if capture_video:
            logger.write_video(frames)

    training_envs.close()
    eval_envs_wrapped.close()


if __name__ == "__main__":
    run_ppo()