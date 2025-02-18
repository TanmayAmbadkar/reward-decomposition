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
# Hyperparameters
#---------------------------------------------
ENV_NAME = "LunarLander-v3"  # or "CartPole-v1", "LunarLander-v2", etc.
SEED = 42
EPISODES = 1000
STEPS_PER_ROLLOUT = 2048      # How many env steps per PPO update
PPO_EPOCHS = 10               # How many optimization epochs per update
BATCH_SIZE = 256              # Mini-batch size per update epoch
GAMMA = 0.99
CLIP_EPS = 0.2                # PPO clipping epsilon
LR = 1e-3
ENTROPY_COEF = 0.01
KL_SCALE = 0.01               # Scale factor for Bayesian KL loss
DEVICE = "cpu"                # "cuda" if you have a GPU

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#---------------------------------------------
# 1. Actor-Critic Network (Bayesian)
#---------------------------------------------
class BayesianActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared feature extractor
        self.actor1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                       in_features=state_dim, out_features=128)
        self.actor2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                       in_features=128, out_features=128)
        
        self.critic1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                       in_features=state_dim, out_features=128)
        self.critic2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                       in_features=128, out_features=128)
        
        # Policy head (actor)
        self.actor_head = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                          in_features=128, out_features=action_dim)
        
        # Value head (critic)
        self.critic_head = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                           in_features=128, out_features=1)
        
    def forward(self, x):
        # Shared
        v = torch.clone(x)
        x = torch.tanh(self.actor1(x))
        x = torch.tanh(self.actor2(x))

        v = torch.tanh(self.critic1(v))
        v = torch.tanh(self.critic2(v))
        
        # Actor logits
        logits = self.actor_head(x)
        # Critic value
        value = self.critic_head(v)
        return logits, value.squeeze(-1)
    
#---------------------------------------------
# 2. PPO Storage
#---------------------------------------------
class RolloutBuffer:
    """
    We store transitions from interaction over STEPS_PER_ROLLOUT steps,
    then use them to run multiple PPO epochs.
    """
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
# 3. PPO Update Function
#---------------------------------------------
def compute_returns_and_advantages(buffer, next_value, gamma=0.99):
    """
    Simple advantage calculation (without GAE).
    We compute a set of returns for each time step t:
    R_t = r_t + gamma * r_{t+1} + ... until done.
    advantage_t = R_t - V(s_t).
    next_value is V(s_{t+1}) for the last step in the buffer (or 0 if done).
    """
    buffer_size = len(buffer.rewards)
    returns = [0] * buffer_size
    advantages = [0] * buffer_size

    running_return = next_value
    for t in reversed(range(buffer_size)):
        if buffer.dones[t]:
            running_return = 0
        running_return = buffer.rewards[t] + gamma * running_return
        returns[t] = running_return

    # advantages = returns - values
    for t in range(buffer_size):
        advantages[t] = returns[t] - buffer.values[t]

    return returns, advantages

def ppo_update(policy_net, optimizer, rollouts, next_value, clip_eps, entropy_coef):
    """
    Runs the PPO update for the data in the rollout buffer.
    """
    # 1) Compute returns & advantages
    returns, advantages = compute_returns_and_advantages(rollouts, next_value, GAMMA)
    returns = torch.tensor(np.array(returns), dtype=torch.float32, device=DEVICE)
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=DEVICE)

    # Normalize advantages (helps training)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.tensor(np.array(rollouts.states), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(np.array(rollouts.actions), dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor(np.array(rollouts.log_probs), dtype=torch.float32, device=DEVICE)

    # We'll do multiple epochs over the data
    dataset_size = len(states)
    
    kl_loss_fn = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    for _ in range(PPO_EPOCHS):
        # Shuffle indices for mini-batches
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

            # Forward pass
            logits, values = policy_net(batch_states)
            
            # Critic loss (MSE)
            value_loss = nn.MSELoss()(values, batch_returns)

            # Actor loss (clipped)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Bayesian KL loss
            kl_bnn = kl_loss_fn(policy_net)  # Summed up over all Bayesian layers

            # Total PPO loss
            total_loss = actor_loss + 0.5 * value_loss - entropy_coef * entropy \
                         + KL_SCALE * kl_bnn

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

#---------------------------------------------
# 4. Training Loop
#---------------------------------------------
def train_ppo_discrete():
    env = gym.make(ENV_NAME)
    
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = BayesianActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    rollout_buffer = RolloutBuffer()

    total_steps = 0
    episode = 0
    state, _ = env.reset(seed=SEED)
    episode_rewards = []
    timesteps = []
    while episode < EPISODES:
        episode_reward = 0

        # Collect experiences for STEPS_PER_ROLLOUT
        for step in range(STEPS_PER_ROLLOUT):
            total_steps += 1

            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = policy_net(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            # Store in rollout buffer
            rollout_buffer.add(
                state, action, log_prob, reward, done, value.item()
            )

            state = next_state
            if done or truncated:
                # Start a new episode if the environment finished
                print(f"Episode {episode} ended with reward {episode_reward}")
                state, _ = env.reset(seed=SEED)
                episode_rewards.append(episode_reward)
                timesteps.append(total_steps)
                episode_reward = 0
                episode+=1

        # After STEPS_PER_ROLLOUT, run PPO update
        # We need V(s_{t+1}) for advantage calculation (or 0 if done).
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, next_value = policy_net(state_t)

        ppo_update(policy_net, optimizer, rollout_buffer, next_value.item(), 
                   CLIP_EPS, ENTROPY_COEF)
        rollout_buffer.clear()

    env.close()
    print("Training finished (Discrete PPO)!")
    plt.plot(timesteps, episode_rewards)
    plt.savefig(f"ppo_{ENV_NAME}.png")

if __name__ == "__main__":
    train_ppo_discrete()
