import os
import datetime
import itertools
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import mo_gymnasium as mo_gym
from ppo.ppo import PPOLogger

# Import your SAC agent and replay buffer
from sac.sac import SAC
from sac.replay_memory import ReplayMemory

# Import the vectorized environment utilities.

# Import the decorator for parameterized scripts
from func_to_script import script


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@script
def run_sac(
    env_id: str = "LunarLander",
    num_envs: int = 4,
    continuous: bool = True,
    scalar_reward: bool = False,
    total_steps: int = 500000,
    batch_size: int = 256,
    start_steps: int = 10000,
    updates_per_step: int = 10,
    rollout_steps: int = 40,    # Number of steps to collect per rollout batch
    hidden_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 0.0003,
    alpha: float = 0.2,
    automatic_entropy_tuning: bool = False,
    policy: str = "Gaussian",
    target_update_interval: int = 1,
    replay_size: int = 1000000,
    seed: int = 123456,
    use_cuda: bool = False,
    evaluate: bool = True,
    logger: bool = True,
):
    """
    Runs the Soft Actor-Critic (SAC) training loop using vectorized environments.

    This version uses a set of parallel environments (via a vectorized environment wrapper)
    and collects rollouts over all environments in parallel. Observations are augmented with a
    weight vector (useful for multi-objective tasks) and rewards are scaled elementwise by these weights.
    Collected transitions are stored in a replay buffer for off-policy updates.

    Args:
        env_id (str): Environment identifier ("LunarLander" or "BipedalWalker").
        num_envs (int): Number of parallel environments.
        continuous (bool): Whether to use continuous actions.
        scalar_reward (bool): Whether the reward is a single scalar.
        total_steps (int): Total number of environment steps (summed over all envs) to run training.
        batch_size (int): Batch size for training updates.
        start_steps (int): Number of initial steps to use random actions for exploration.
        updates_per_step (int): Number of gradient updates per environment step.
        rollout_steps (int): Number of rollout steps to collect in each batch.
        hidden_size (int): Hidden units in the SAC networks.
        gamma (float): Discount factor.
        tau (float): Soft target network update coefficient.
        lr (float): Learning rate.
        alpha (float): Entropy regularization coefficient.
        automatic_entropy_tuning (bool): Whether to tune α automatically.
        policy (str): Policy type ("Gaussian" or "Deterministic").
        target_update_interval (int): Update interval for the target network.
        replay_size (int): Capacity of the replay buffer.
        seed (int): Random seed.
        use_cuda (bool): If True (and available) use CUDA.
        evaluate (bool): Whether to perform periodic evaluation.
    """
    # Set seeds and determine device.
    set_seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # Create vectorized environments.
    
    envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        lambda: mo_gym.make(env_id) for _ in range(num_envs)
    )
    envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(envs)


    # Get the observation and action spaces from one environment.
    state_space = envs.single_observation_space
    action_space = envs.single_action_space

    # Instantiate the SAC agent.
    agent = SAC(state_space, action_space, envs.rewards_shape, gamma, tau, alpha, policy,
                automatic_entropy_tuning, target_update_interval, use_cuda, hidden_size, lr)

    # Set up TensorBoard logging.
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if logger:
        
        logger = PPOLogger(f'runs_sac/{run_id}_SAC_{env_id}_{policy}', True, envs.reward_size)
        writer = SummaryWriter(f'runs_sac/{run_id}_SAC_{env_id}_{policy}')

    # Create the replay memory.
    memory = ReplayMemory(replay_size, seed)

    total_numsteps = 0
    updates = 0

    ####################################
    # Vectorized Rollout Collection    #
    ####################################
    def collect_rollouts_vectorized(envs, agent, rollout_steps, reward_size, states, weights, dones, truncs):
        """
        Collects a batch of transitions from vectorized environments.

        Each environment receives its own randomly sampled weight vector (of shape [reward_size]).
        Observations are augmented (concatenated) with these weights. For each step in the rollout,
        actions are chosen (randomly or using the agent), the environment steps forward, and the reward
        is scaled elementwise by the corresponding weight. If any environments signal termination,
        they are reset and assigned new weight vectors.

        Returns:
            batch_obs: Numpy array of shape [num_steps, obs_dim + reward_size]
            batch_actions: Numpy array of shape [num_steps, action_dim]
            batch_rewards: Numpy array of shape [num_steps, reward_size]
            batch_next_obs: Numpy array of shape [num_steps, obs_dim + reward_size]
            batch_dones: Numpy array of shape [num_steps]
        """
        # Reset all environments.
        # Initialize weight vectors for each environment.
        # Containers for collected data.
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        all_dones = []

        for step in range(0, rollout_steps, num_envs):
            # Augment states by concatenating with the weight vectors.
            states_aug = np.concatenate((states, weights), axis=1)  # shape: (num_envs, obs_dim+reward_size)

            # Use list comprehension for now; vectorizing the policy call can also be done if supported.
            actions = agent.select_action(states_aug)

            # Step the vectorized environments.
            next_states, rewards, dones, truncs, infos = envs.step(actions)
            # Scale rewards elementwise by the weight vector.
            # Assume rewards returned has shape (num_envs, reward_size). If scalar, you may need to reshape.
            rewards_weighted = rewards * weights
            
            # Augment next_states.
            next_states_aug = np.concatenate((next_states, weights), axis=1)

            all_obs.append(states_aug)
            all_actions.append(np.array(actions))
            all_rewards.append(rewards_weighted)
            all_next_obs.append(next_states_aug)
            all_dones.append(dones)

            # Handle resets for finished environments.
            if len(infos) != 0:
                # print("Total Steps:", total_numsteps + rollout_steps, "Reward: ", infos['episode']['r'][infos['_episode']], "Length: ", infos['episode']['l'][infos['_episode']])
                logger.log_rollout_step(infos, total_numsteps + rollout_steps)
                
            change_weights = np.logical_or(dones, truncs)
            if reward_size != 1 and change_weights.any():
                # print(weights)  
                weights[change_weights] = np.random.uniform(-1, 1, size=weights[change_weights].shape)
                # weights[:,-1] = 1.0
                # weights = torch.ones((self.num_envs, self.reward_size)).to(self.device).type(torch.float32)

            states = next_states
            
        # Flatten the collected data over rollout_steps and num_envs.
        batch_obs = np.concatenate(all_obs, axis=0)
        batch_actions = np.concatenate(all_actions, axis=0)
        batch_rewards = np.concatenate(all_rewards, axis=0)
        batch_next_obs = np.concatenate(all_next_obs, axis=0)
        batch_dones = np.concatenate(all_dones, axis=0)

        

        is_last_observation_terminal = dones
        is_last_observation_truncated = truncs
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, states[:, :state_space.shape[0]], weights, is_last_observation_terminal, is_last_observation_truncated

    #####################################
    # Main Training Loop (Vectorized)   #
    #####################################
    states, infos = envs.reset()
    
    weights = np.random.uniform(-1, 1, size=(num_envs, envs.reward_size))
    is_last_observation_terminal = np.array([False] * num_envs)
    is_last_observation_truncated = np.array([False] * num_envs)
    while total_numsteps < total_steps:
        # Use random actions during initial exploration.


        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, states, weights, is_last_observation_terminal, is_last_observation_truncated = collect_rollouts_vectorized(
            envs, agent, rollout_steps, envs.reward_size, states, weights, is_last_observation_terminal, is_last_observation_truncated
        )
        num_transitions = batch_obs.shape[0]
        for i in range(num_transitions):
            memory.push(batch_obs[i], batch_actions[i], batch_rewards[i], batch_next_obs[i], batch_dones[i])
        total_numsteps += rollout_steps

            # Perform updates periodically.
        if len(memory) > batch_size:
            for _ in range(updates_per_step):
                qf1_loss, qf2_loss, policy_loss, ent_loss, alpha_val = agent.update_parameters(memory, batch_size, updates)
                writer.add_scalar('loss/critic_1', qf1_loss, total_numsteps)
                writer.add_scalar('loss/critic_2', qf2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temperature/alpha', alpha_val, total_numsteps)
                updates += 1

        # Log aggregated reward from the rollout batch.
        rollout_reward = np.sum(batch_rewards)
        writer.add_scalar('reward/train', rollout_reward, total_numsteps)
        # print(f"Total Steps: {total_numsteps}, Latest Rollout Reward: {rollout_reward:.2f}")

        # # Periodic evaluation.
        # if evaluate and total_numsteps % (rollout_steps * 2) == 0:
        #     avg_reward = 0.0
        #     eval_episodes = 1
        #     for _ in range(eval_episodes):
        #         states, infos = envs.reset()
        #         eval_reward = 0.0
        #         dones = np.array([False] * num_envs)
        #         # For evaluation, use a uniform weight vector.
        #         eval_weights = np.ones((num_envs, envs.reward_size))
        #         while not np.all(dones):
        #             states_aug = np.concatenate((states, eval_weights), axis=1)
        #             eval_actions = [agent.select_action(obs, evaluate=True) for obs in states_aug]
        #             states, rewards, dones, truncs, infos = envs.step(eval_actions)
        #             eval_reward += np.sum(rewards)
        #         avg_reward += eval_reward
        #     avg_reward /= eval_episodes
        #     writer.add_scalar('reward/eval', avg_reward, total_numsteps)
        #     print(f"Evaluation Reward: {avg_reward:.2f}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    run_sac()
