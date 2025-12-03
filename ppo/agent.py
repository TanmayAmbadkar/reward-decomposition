from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module, ABC):
    @abstractmethod
    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        """
        Estimate the value (Utility and Auxiliary Returns) using the critic network.
        
        Args:
            observation: The environment state.
            accumulated_reward: The vector of rewards collected so far in the episode.
            task_id: The one-hot vector indicating the active utility function.
            
        Returns:
            Tuple (utility_value, auxiliary_returns_vector)
        """
        pass

    @abstractmethod
    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        """
        Get the action distribution.
        """
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        """
        Sample action and compute log prob.
        """
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        """
        Compute log probs and entropy for optimization.
        """
        pass
    
    def forward(self):
        pass


class DiscreteAgent(BaseAgent):
    def __init__(self, envs, reward_size=1, task_size=0):
        """
        Args:
            envs: Gym environments.
            reward_size: Number of objectives (Dimension of accumulated_reward and auxiliary head).
            task_size: Number of utility functions (Dimension of one-hot task_id). Set to 0 for single-task.
        """
        super().__init__()
        self.reward_size = reward_size
        self.task_size = task_size

        try:
            action_space = envs.single_action_space.n
            observation_space = envs.single_observation_space.shape
        except:
            action_space = envs.action_space.n
            observation_space = envs.observation_space.shape

        # Input dimension: State + Accumulated Rewards + Task ID
        self.input_dim = np.array(observation_space).prod() + self.reward_size + self.task_size

        # --- SEPARATE BODIES ---
        
        # Actor Body
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # Critic Body
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Actor Head
        self.actor_head = layer_init(nn.Linear(64, action_space), std=0.01)

        # Critic Head 1: Main Utility (Scalar)
        self.critic_utility_head = layer_init(nn.Linear(64, 1), std=1.0)

        # Critic Head 2: Auxiliary Returns (Vector = reward_size)
        self.critic_returns_head = layer_init(nn.Linear(64, reward_size), std=1.0)

    def _get_feature_embedding(self, observation, accumulated_reward, task_id=None, device="cpu"):
        """Helper to format input. Returns raw concatenated tensor."""
        # Ensure inputs are tensors and on correct device
        if not isinstance(observation, torch.Tensor):
            observation = torch.Tensor(observation).to(device)
        if not isinstance(accumulated_reward, torch.Tensor):
            accumulated_reward = torch.Tensor(accumulated_reward).to(device)
            
        # Reshape inputs if necessary
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        if len(accumulated_reward.shape) == 1:
            accumulated_reward = accumulated_reward.unsqueeze(0)

        inputs = [observation, accumulated_reward]

        # Handle Task ID
        if self.task_size > 0:
            if task_id is None:
                raise ValueError("Agent initialized with task_size > 0 but no task_id provided.")
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.Tensor(task_id).to(device)
            if len(task_id.shape) == 1:
                task_id = task_id.unsqueeze(0)
            inputs.append(task_id)

        # Concatenate: [State, History, Task]
        return torch.hstack(inputs)

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        model_input = self._get_feature_embedding(observation, accumulated_reward, task_id, device)
        
        # Pass through Critic Body
        critic_features = self.critic_body(model_input)
        
        utility_val = self.critic_utility_head(critic_features)
        returns_val = self.critic_returns_head(critic_features)
        
        return utility_val, returns_val

    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        model_input = self._get_feature_embedding(observation, accumulated_reward, task_id, device)
        
        # Pass through Actor Body
        actor_features = self.actor_body(model_input)
        
        logits = self.actor_head(actor_features)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action_dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        
        if deterministic:
            action = action_dist.logits.argmax(dim=1)
        else:
            action = action_dist.sample()
            
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        action_dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy
    
    @torch.no_grad
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action, _ = self.sample_action_and_compute_log_prob(observation, accumulated_reward, task_id, deterministic, device)
        # For prediction, we might want just the utility value, or both. Returning utility for standard compat.
        utility_val, _ = self.estimate_value_from_observation(observation, accumulated_reward, task_id, device)
        return action.cpu().numpy(), utility_val.cpu().numpy()


class ContinuousAgent(BaseAgent):
    def __init__(self, envs, reward_size=1, task_size=0, rpo_alpha=None):
        super().__init__()
        self.reward_size = reward_size
        self.task_size = task_size
        self.rpo_alpha = rpo_alpha

        try:
            action_space = envs.single_action_space.shape
            observation_space = envs.single_observation_space.shape
        except:
            action_space = envs.action_space.shape
            observation_space = envs.observation_space.shape

        # Input dimension: State + Accumulated Rewards + Task ID
        self.input_dim = np.array(observation_space).prod() + self.reward_size + self.task_size

        # --- SEPARATE BODIES ---
        
        # Actor Body
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # Critic Body
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Actor Head (Mean)
        self.actor_mean = layer_init(nn.Linear(64, np.prod(action_space)), std=0.01)
        
        # Actor LogStd
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space)))
        
        # Container for actor parameters
        self.actor = nn.Module()
        self.actor.add_module('body', self.actor_body)
        self.actor.add_module('mean', self.actor_mean)
        self.actor.register_parameter('logstd', self.actor_logstd)

        # Critic Head 1: Main Utility (Scalar)
        self.critic_utility_head = layer_init(nn.Linear(64, 1), std=1.0)

        # Critic Head 2: Auxiliary Returns (Vector = reward_size)
        self.critic_returns_head = layer_init(nn.Linear(64, reward_size), std=1.0)
        
        # Critic Container (to easily get params)
        self.critic = nn.Module()
        self.critic.add_module('body', self.critic_body)
        self.critic.add_module('utility_head', self.critic_utility_head)
        self.critic.add_module('returns_head', self.critic_returns_head)

        try:
            self.action_space_low = envs.single_action_space.low
            self.action_space_high = envs.single_action_space.high
        except:
            self.action_space_low = envs.action_space.low
            self.action_space_high = envs.action_space.high

    def _get_feature_embedding(self, observation, accumulated_reward, task_id=None, device="cpu"):
        """Helper to format input. Returns raw concatenated tensor."""
        # Ensure inputs are tensors and on correct device
        if not isinstance(observation, torch.Tensor):
            observation = torch.Tensor(observation).to(device)
        if not isinstance(accumulated_reward, torch.Tensor):
            accumulated_reward = torch.Tensor(accumulated_reward).to(device)
            
        # Reshape inputs if necessary
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        if len(accumulated_reward.shape) == 1:
            accumulated_reward = accumulated_reward.unsqueeze(0)

        inputs = [observation, accumulated_reward]

        # Handle Task ID
        if self.task_size > 0:
            if task_id is None:
                raise ValueError("Agent initialized with task_size > 0 but no task_id provided.")
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.Tensor(task_id).to(device)
            if len(task_id.shape) == 1:
                task_id = task_id.unsqueeze(0)
            inputs.append(task_id)

        # Concatenate: [State, History, Task]
        return torch.hstack(inputs)

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        model_input = self._get_feature_embedding(observation, accumulated_reward, task_id, device)
        
        # Pass through Critic Body
        critic_features = self.critic_body(model_input)
        
        utility_val = self.critic_utility_head(critic_features)
        returns_val = self.critic_returns_head(critic_features)
        
        return utility_val, returns_val

    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        model_input = self._get_feature_embedding(observation, accumulated_reward, task_id, device)
        
        # Pass through Actor Body
        actor_features = self.actor_body(model_input)
        
        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        return Normal(action_mean, action_std)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action_dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
            
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        action_dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        
        if self.rpo_alpha is not None:
            # RPO (Robust Policy Optimization) tweak
            action_mean = action_dist.mean
            z = (
                torch.FloatTensor(action_mean.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(device)
            )
            action_mean = action_mean + z
            action_dist = Normal(action_mean, action_dist.stddev)

        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy

    @torch.no_grad
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action, _ = self.sample_action_and_compute_log_prob(observation, accumulated_reward, task_id, deterministic, device)
        utility_val, _ = self.estimate_value_from_observation(observation, accumulated_reward, task_id, device)
        return action.cpu().numpy(), utility_val.cpu().numpy()
    
    def forward(self, observation, accumulated_reward, task_id=None):
        """
        Forward pass for compatibility, returning action and utility value.
        """
        return self.predict(observation, accumulated_reward, task_id)