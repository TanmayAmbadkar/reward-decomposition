import os
import random
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.optim as optim
import imageio

import mo_gymnasium as mo_gym  # or gymnasium as gym if not using multi-objective
from sac.agent import VectorizedSACAgent  # Your agent class from above
from sac.sac import SACLogger          # Logger class above
from sac.sac import VectorReplayBuffer # Vectorized replay buffer
from sac.sac import SAC                       # Your full SAC class

from func_to_script import script

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def load_and_evaluate_model(
    run_name,
    env_id,
    envs,
    num_envs,
    agent_class,
    device,
    model_path,
    capture_video,
):
    eval_episodes = 10
    eval_envs = envs

    eval_agent = agent_class(
        eval_envs.single_observation_space, 
        eval_envs.single_action_space, 
        reward_size=eval_envs.rewards_shape[-1]
    ).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()
    frames_per_env = [[] for _ in range(num_envs)]

    obs, _ = eval_envs.reset()
    episodic_returns = []
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            actions, _ = eval_agent.sample_action_and_compute_log_prob(obs_tensor)
            actions = actions.cpu().numpy()
        obs, rewards, dones, truncs, infos = eval_envs.step(actions)
        episode_rewards += rewards
        for i in range(num_envs):
            if dones[i] or truncs[i]:
                episodic_returns.append(episode_rewards[i])
                episode_rewards[i] = 0

        if capture_video:
            all_frames = eval_envs.render()
            for i in range(num_envs):
                frames_per_env[i].append(all_frames[i])

    eval_envs.close()
    if capture_video:
        for i in range(num_envs):
            gif_name = f"gifs/{run_name}_env_{i}.gif"
            if len(frames_per_env[i]) > 0:
                imageio.mimsave(gif_name, frames_per_env[i], fps=30)
                print(f"Saved GIF for env {i}: {gif_name}")

@script
def run_sac(
    env_id: str = "mo-humanoid-v5",
    num_envs: int = 4,
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    learning_rate: float = 0.0003,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    automatic_entropy_tuning: bool = True,
    initial_random_steps: int = 10000,
    update_after: int = 1000,
    update_every: int = 50,
    anneal_lr: bool = False,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = True,
    save_model: bool = True,
):
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}__{exp_name}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{seed}"
    set_seed(seed, torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        [lambda: mo_gym.make(env_id) for _ in range(num_envs)]
    )
    envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(envs)

    print("Experiment:", exp_name)
    print("Reward shape:", envs.rewards_shape)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    reward_size = envs.rewards_shape[-1]

    agent_class = partial(VectorizedSACAgent, reward_size=reward_size)
    agent = agent_class(obs_space, act_space).to(device)
    optimizer_actor = optim.Adam(agent.actor_mean.parameters(), lr=learning_rate, eps=1e-5)
    optimizer_critic = optim.Adam(
        list(agent.critic1.parameters()) + list(agent.critic2.parameters()), lr=learning_rate, eps=1e-5
    )

    replay_buffer = VectorReplayBuffer(
        obs_dim=np.prod(obs_space.shape),
        act_dim=np.prod(act_space.shape),
        size=1_000_000,
        num_envs=num_envs,
        reward_size=reward_size,
    )

    logger = SACLogger(run_name, use_tensorboard, reward_size=reward_size, num_envs=num_envs)
    sac = SAC(
        agent=agent,
        env=envs,
        replay_buffer=replay_buffer,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        automatic_entropy_tuning=automatic_entropy_tuning,
        total_steps=total_steps,
        initial_random_steps=initial_random_steps,
        update_after=update_after,
        update_every=update_every,
        logger=logger,
        lr=learning_rate,
        anneal_lr=anneal_lr,
    )
    print(agent)
    # Train the agent
    trained_agent = sac.learn()

    if save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/{exp_name}.sac_model"
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        load_and_evaluate_model(
            run_name=run_name,
            env_id=env_id,
            envs=envs,
            num_envs=num_envs,
            agent_class=agent_class,
            device=device,
            model_path=model_path,
            capture_video=capture_video,
        )

    envs.close()

if __name__ == "__main__":
    run_sac()
