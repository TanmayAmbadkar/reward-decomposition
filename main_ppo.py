import os
import random
import time
from datetime import datetime
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from func_to_script import script
import imageio

from ppo.agent import ContinuousAgent, DiscreteAgent
from ppo.ppo import PPO, PPOLogger
from envs.lunar_lander import LunarLander
from envs.bipedal_walker import BipedalWalker
from envs.utils import SyncVectorEnv, RecordEpisodeStatistics

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic



def load_and_evaluate_model(
    run_name,
    env_id,
    env_is_discrete,
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

    # if not env_is_discrete:
    #     # Update normalization stats for continuous environments
    #     avg_rms_obj = (
    #         np.mean([envs.get_obs_norm_rms_obj(i) for i in range(num_envs)]) / num_envs
    #     )
    #     eval_envs.set_obs_norm_rms_obj(avg_rms_obj)

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()
    frames_per_env = [[] for _ in range(num_envs)]  # one list of frames per env

    obs, _ = eval_envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _ = eval_agent.sample_action_and_compute_log_prob(
                torch.Tensor(obs).to(device), deterministic = True
            )
        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())
        print(actions)

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
            gif_name = f"{run_name}_env_{i}.gif"
            # Only save if we actually have frames
            if len(frames_per_env[i]) > 0:
                imageio.mimsave(gif_name, frames_per_env[i][::3], fps=30)
                print(f"Saved GIF for env {i}: {gif_name}")
@script
def run_ppo(
    env_id: str = "LunarLander",
    env_is_discrete: bool = False,
    num_envs: int = 4,
    scalar_reward: bool = False,
    total_timesteps: int = 1000000,
    num_rollout_steps: int = 2048,
    update_epochs: int = 10,
    num_minibatches: int = 256,
    learning_rate: float = 0.00001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    surrogate_clip_threshold: float = 0.2,
    entropy_loss_coefficient: float = 0.0,
    value_function_loss_coefficient: float = 0.5,
    normalize_advantages: bool = True,
    clip_value_function_loss: bool = False,
    max_grad_norm: float = 0.5,
    target_kl: float = None,
    anneal_lr: bool = False,
    rpo_alpha: float = None,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = False,
    save_model: bool = True,
):
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.

    This function sets up the environment, creates the PPO agent, and runs the training process.
    It handles both discrete and continuous action spaces, and includes options for
    various PPO algorithm parameters and training configurations.

    Args:
        # Environment parameters
        env_id (str): Identifier for the Gymnasium environment to use.
        env_is_discrete (bool): Whether the environment has a discrete action space.
        num_envs (int): Number of parallel environments to run.

        # Core training parameters
        total_timesteps (int): Total number of timesteps to run the training for. This is the number of interactions with the environment
        num_rollout_steps (int): Number of steps to run in each environment per rollout.
        update_epochs (int): Number of epochs to update the policy for each rollout.
        num_minibatches (int): Number of minibatches for each update.

        # Core PPO algorithm parameters
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.

        # Loss function coefficients
        entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
        value_function_loss_coefficient (float): Coefficient for the value function loss.

        # Advanced PPO parameters
        normalize_advantages (bool): Whether to normalize advantages.
        clip_value_function_loss (bool): Whether to clip the value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.
        target_kl (float): Target KL divergence for early stopping, if not None.

        # Learning rate schedule
        anneal_lr (bool): Whether to use learning rate annealing.

        # Continuous action space specific
        rpo_alpha (float): Alpha parameter for Regularized Policy Optimization (continuous action spaces only).

        # Reproducibility and logging
        seed (int): Random seed for reproducibility.
        torch_deterministic (bool): Whether to use deterministic algorithms in PyTorch.
        capture_video (bool): Whether to capture and save videos of the agent's performance.
        use_tensorboard (bool): Whether to use TensorBoard for logging.
        save_model (bool): Whether to save the trained model to disk and validate this by running a simple evaluation.
    """

    if env_is_discrete and rpo_alpha is not None:
        print(
            f"rpo_alpha is not used in discrete environments. Ignoring rpo_alpha={rpo_alpha}"
        )

    # Set up run name and logging
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}__{exp_name}__{datetime.now()}__{seed}"
    print(exp_name, scalar_reward)

    set_seed(seed, torch_deterministic)

    # Set up device666
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Create environments
    if env_id == "LunarLander":
        envs = SyncVectorEnv(
        [
            lambda: gym.wrappers.TimeLimit(LunarLander(continuous = True, scalar_reward=scalar_reward, render_mode="rgb_array"), max_episode_steps = 500),
        ]*num_envs,
        reward_size = 1 if scalar_reward else 8
        )
    if env_id == "BipedalWalker":
        envs = SyncVectorEnv(
        [
            lambda: gym.wrappers.TimeLimit(BipedalWalker(scalar_reward=scalar_reward, render_mode="rgb_array"), max_episode_steps = 1600),
        ]*num_envs,
        reward_size = 1 if scalar_reward else 7
        )
    envs = RecordEpisodeStatistics(envs)

# Set up agent
    agent_class = (
        DiscreteAgent
        if env_is_discrete
        else partial(ContinuousAgent, rpo_alpha=rpo_alpha, reward_size = envs.env.reward_size)
    )
    agent = agent_class(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    ppo = PPO(
        agent=agent,
        reward_size = envs.env.reward_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        num_rollout_steps=num_rollout_steps,
        num_envs=num_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        surrogate_clip_threshold=surrogate_clip_threshold,
        entropy_loss_coefficient=entropy_loss_coefficient,
        value_function_loss_coefficient=value_function_loss_coefficient,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        normalize_advantages=normalize_advantages,
        clip_value_function_loss=clip_value_function_loss,
        target_kl=target_kl,
        anneal_lr=anneal_lr,
        envs=envs,
        seed=seed,
        logger=PPOLogger(run_name, use_tensorboard, envs.env.reward_size),
    )
    print(ppo.agent)
    # Train the agent
    trained_agent = ppo.learn(total_timesteps)

    if save_model:
        if not os.path.exists(f"runs/{run_name}"):
            os.mkdir(f"runs/{run_name}")
        model_path = f"runs/{run_name}/{exp_name}.rl_model"
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        load_and_evaluate_model(
            run_name,
            env_id,
            env_is_discrete,
            envs,
            num_envs,
            agent_class,
            device,
            model_path,
            gamma,
            capture_video,
        )

    # Close environments
    envs.close()


if __name__ == "__main__":
    run_ppo()