import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt

from collections import deque

# -----------------------------------------------------------
# 1. Hyperparameters
# -----------------------------------------------------------
ENV_NAME = "LunarLander-v3"
SEED = 42
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000   # Minimum experiences before we start training
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 50_000        # Decay over n steps
TARGET_UPDATE_FREQ = 100
MAX_EPISODES = 500
MAX_STEPS = 10_000_000    # Just a large number; training will likely converge earlier

# -----------------------------------------------------------
# 2. Set seeds for reproducibility (optional)
# -----------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------
# 3. Replay Buffer
# -----------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------
# 4. Q-Network
# -----------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = state_dim, out_features = 128),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = 128, out_features = 128),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = 128, out_features = action_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------
# 5. Initialize environment, networks, optimizer
# -----------------------------------------------------------
env = gym.make(ENV_NAME)
env.action_space.seed(SEED)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Main network (policy network)
policy_net = QNetwork(state_dim, action_dim)
# Target network
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Epsilon schedule
eps = EPS_START
eps_decay_step = (EPS_START - EPS_END) / EPS_DECAY

# -----------------------------------------------------------
# 6. Fill the replay buffer with random transitions (warm-up)
# -----------------------------------------------------------
state, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done or truncated)
    state = next_state
    if done or truncated:
        state, _ = env.reset()

# -----------------------------------------------------------
# 7. Training Loop
# -----------------------------------------------------------
kl_loss = bnn.BKLLoss(reduction = "mean", last_layer_only  = False)
total_steps = 0
episode_rewards = []
timesteps = []
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    while True:

        # for step in range(1_000):  # limit each episode to 1000 steps
        total_steps += 1

        # Epsilon-greedy action selection
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_t)
                action = q_values.argmax(dim=1).item()

        # Step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done or truncated)
        state = next_state
        episode_reward += reward

        # Decrease epsilon
        if eps > EPS_END:
            eps -= eps_decay_step
            eps = max(eps, EPS_END)

        # Start gradient updates after MIN_REPLAY_SIZE
        if len(replay_buffer) > MIN_REPLAY_SIZE:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Convert to tensors
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            # Compute current Q values
            q_values = policy_net(states_t)
            # Gather the Q-values for the taken actions
            q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                # Double DQN variant can be used, but let's keep it simple (classic DQN)
                max_next_q_values = target_net(next_states_t).max(dim=1)[0]
                target_q_values = rewards_t + GAMMA * (1 - dones_t) * max_next_q_values

            # MSE loss (Huber loss often works better, but MSE is simpler)
            loss = nn.MSELoss()(q_values, target_q_values) + kl_loss(policy_net)*0.01

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if done or truncated:
            print(f"Episode {episode} ended with reward {episode_reward}")
            episode_rewards.append(episode_reward)
            timesteps.append(total_steps)
            break

plt.plot(timesteps, episode_rewards)
plt.savefig(f"cartpole_{ENV_NAME}.png")

print("Training finished!")
env.close()

import pickle
pickle.dump(policy_net, open(f"saved_policies/policy_{ENV_NAME}.pkl", "wb"))