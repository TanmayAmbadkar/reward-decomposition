import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

#---------------------------------------------
# Hyperparameters (continuous)
#---------------------------------------------
ENV_NAME = "BipedalWalker-v3"
SEED = 42
EPISODES = 1000
STEPS_PER_ROLLOUT = 2048
PPO_EPOCHS = 10
BATCH_SIZE = 256
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 1e-3
ENTROPY_COEF = 0.0  # Typically smaller for continuous
KL_SCALE = 0.01
DEVICE = "cpu"
TOTAL_STEPS = 1000000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#---------------------------------------------
# 1. Actor-Critic (Bayesian) for continuous
#---------------------------------------------
class BayesianActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared features
        self.shared1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                       in_features=state_dim, out_features=128)
        self.shared2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                       in_features=128, out_features=128)
        
        # Actor mean head
        self.actor_mean = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                          in_features=128, out_features=action_dim)
        # We can also do a Bayesian log std, or keep a separate trainable parameter
        # For simplicity, let's do a plain parameter:
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                      in_features=128, out_features=1)
    
    def forward(self, x):
        x = torch.relu(self.shared1(x))
        x = torch.relu(self.shared2(x))
        mean = self.actor_mean(x)
        value = self.critic(x).squeeze(-1)
        return mean, self.log_std, value

#---------------------------------------------
# 2. Rollout Buffer for continuous
#---------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

#---------------------------------------------
# 3. PPO update
#---------------------------------------------
def compute_returns_and_advantages(buffer, next_value, gamma=0.99):
    returns = []
    advantages = []
    running_return = next_value
    buffer_size = len(buffer.rewards)

    for t in reversed(range(buffer_size)):
        if buffer.dones[t]:
            running_return = 0
        running_return = buffer.rewards[t] + gamma * running_return
        returns.insert(0, running_return)

    for t in range(buffer_size):
        advantages.append(returns[t] - buffer.values[t])

    return returns, advantages

def ppo_update(policy_net, optimizer, rollouts, next_value, clip_eps):
    returns, advantages = compute_returns_and_advantages(rollouts, next_value, GAMMA)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.tensor(rollouts.states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(rollouts.actions, dtype=torch.float32, device=DEVICE)
    old_log_probs = torch.tensor(rollouts.log_probs, dtype=torch.float32, device=DEVICE)

    kl_loss_fn = bnn.BKLLoss(reduction="mean", last_layer_only=False)
    dataset_size = len(states)

    for _ in range(PPO_EPOCHS):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            mean, log_std, value = policy_net(batch_states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            
            new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)  # sum over action dims
            entropy = dist.entropy().sum(axis=-1).mean()

            # Ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            value_loss = nn.MSELoss()(value, batch_returns)

            # Bayesian KL loss
            kl_bnn = kl_loss_fn(policy_net)

            # Total loss
            loss = actor_loss + 0.5 * value_loss - ENTROPY_COEF * entropy \
                   + KL_SCALE * kl_bnn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#---------------------------------------------
# 4. Training Loop (continuous)
#---------------------------------------------
def train_ppo_continuous():
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_net = BayesianActorCriticContinuous(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    rollout_buffer = RolloutBuffer()

    total_steps = 0
    timesteps = []
    episode_rewards = []
    episode = 0
    while total_steps < TOTAL_STEPS:
        state, _ = env.reset()
        episode_reward = 0

        for step in range(STEPS_PER_ROLLOUT):
            total_steps += 1
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, log_std, value = policy_net(state_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1).item()

            # Convert action to numpy, clip if necessary
            action_np = action.squeeze(0).cpu().numpy()
            # Some environments require clipping actions
            # e.g. if env.action_space is [-1, 1], do: action_np = np.clip(action_np, -1, 1)

            next_state, reward, done, truncated, info = env.step(action_np)
            episode_reward += reward

            rollout_buffer.add(
                state, action_np, log_prob, reward, done or truncated, value.item()
            )

            state = next_state
            if done or truncated:
                print(f"Episode {episode} ended with reward {episode_reward}")
                episode+=1
                state, _ = env.reset()
                episode_rewards.append(episode_reward)
                timesteps.append(total_steps)
                episode_reward = 0

        # After collecting STEPS_PER_ROLLOUT steps, do an update
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, _, next_value = policy_net(state_t)

        ppo_update(policy_net, optimizer, rollout_buffer, next_value.item(), CLIP_EPS)
        rollout_buffer.clear()

    env.close()
    print("Training finished (Continuous PPO)!")
    plt.plot(timesteps, episode_rewards)
    plt.savefig(f"ppo_{ENV_NAME}.png")

if __name__ == "__main__":
    train_ppo_continuous()
