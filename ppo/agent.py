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
    def estimate_value_from_observation(self, observation):
        """
        Estimate the value of an observation using the critic network.

        Args:
            observation: The observation to estimate.

        Returns:
            The estimated value of the observation.
        """
        pass

    @abstractmethod
    def get_action_distribution(self, observation):
        """
        Get the action distribution for a given observation.

        Args:
            observation: The observation to base the action distribution on.

        Returns:
            A probability distribution over possible actions.
        """
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations):
        """
        Sample an action from the action distribution and compute its log probability.

        Args:
            observations: The observations to base the actions on.

        Returns:
            A tuple containing:
            - The sampled action(s)
            - The log probability of the sampled action(s)
        """
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        """
        Compute the log probabilities and entropy of given actions for given observations.

        Args:
            observations: The observations corresponding to the actions.
            actions: The actions to compute probabilities and entropy for.

        Returns:
            A tuple containing:
            - The log probabilities of the actions
            - The entropy of the action distribution
        """
        pass


class DiscreteAgent(BaseAgent):
    def __init__(self, envs, reward_size = 1):
        super().__init__()
        self.reward_size = reward_size
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, reward_size), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def estimate_value_from_observation(self, observation):
        return self.critic(observation)

    def get_action_distribution(self, observation):
        logits = self.actor(observation)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy

class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, weight_dim):
        super().__init__()
        # Generate scale and bias from the weight vector
        self.film_scale = nn.Linear(weight_dim, feature_dim)
        self.film_bias = nn.Linear(weight_dim, feature_dim)
    
    def forward(self, x, weight):
        scale = self.film_scale(weight) # unsqueeze if needed to match x shape
        bias = self.film_bias(weight)
        return x * scale + bias

class ActorFiLM(nn.Module):
    def __init__(self, state_dim, weight_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.film1 = FiLMBlock(64, weight_dim)
        self.fc2 = nn.Linear(64, 64)
        self.film2 = FiLMBlock(64, weight_dim)
        self.actor_head = nn.Linear(64, action_dim)
        self.weight_dim = weight_dim
    
    def forward(self, state):
        
        weight = state[:, -self.weight_dim:]
        state = state[:, :self.weight_dim]
        x = torch.tanh(self.fc1(state))
        # x = self.film1(x, weight)
        x = torch.tanh(self.fc2(x))
        # x = self.film2(x, weight)
        mean = self.actor_head(x)
        return mean
    

class CriticFiLM(nn.Module):
    def __init__(self, state_dim, weight_dim, reward_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.film1 = FiLMBlock(64, weight_dim)
        self.fc2 = nn.Linear(64, 64)
        self.film2 = FiLMBlock(64, weight_dim)
        self.critic_head = nn.Linear(64, reward_dim)
        self.weight_dim = weight_dim
    
    def forward(self, state):

        weight = state[:, -self.weight_dim:]
        state = state[:, :self.weight_dim]
        
        x = torch.tanh(self.fc1(state))
        x = self.film1(x, weight)
        x = torch.tanh(self.fc2(x))
        x = self.film2(x, weight)
        mean = self.critic_head(x)
        return mean


class ContinuousAgent(BaseAgent):
    def __init__(self, envs, rpo_alpha=None, reward_size = 1, shield = None):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.reward_size = reward_size
        self.weight_vec_size = 0 if reward_size == 1 else reward_size
        # self.critic = CriticFiLM(np.array(envs.single_observation_space.shape).prod(), self.weight_vec_size, reward_size)
        # self.actor_mean = ActorFiLM(np.array(envs.single_observation_space.shape).prod(), self.weight_vec_size,  np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod() + self.weight_vec_size, 128)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod() + self.weight_vec_size, 128)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(
                nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )
        self.shield = shield
        
        self.action_space_low = envs.single_action_space.low
        self.action_space_high = envs.single_action_space.high

    def estimate_value_from_observation(self, observation, weights):

        assert weights.shape[0] == observation.shape[0]

        if self.weight_vec_size == 0:
            observation = observation
        elif weights is None:
            observation = torch.hstack([observation, 1/self.weight_vec_size * torch.ones((observation.shape[0], self.weight_vec_size))])
        else:
            observation =  torch.hstack([observation, weights])

        return self.critic(observation)

    def get_action_distribution(self, observation):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_dist = Normal(action_mean, action_std)

        return action_dist
    
    @torch.no_grad
    def predict(self, observation, weight = None, deterministic = False):

        observation = torch.Tensor(observation).reshape(1, -1)
        obs_critic = observation.clone()
        weight = torch.Tensor(weight).reshape(1, -1)
        if self.weight_vec_size == 0:
            observation = observation
        elif weight is None:
            observation = torch.hstack([observation, 1/self.weight_vec_size * torch.ones((observation.shape[0], self.weight_vec_size))])
        else:
            observation =  torch.hstack([observation, weight])

        action_dist = self.get_action_distribution(observation)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
        if self.shield is not None:
            action = self.shield(observation, action)
        action = torch.clamp(action, torch.Tensor(self.action_space_low).to(action.device), torch.Tensor(self.action_space_high).to(action.device))
        return action.cpu().numpy(), self.estimate_value_from_observation(obs_critic).cpu().numpy()


    def sample_action_and_compute_log_prob(self, observations, weights = None, deterministic = False):

        if weights is not None:
            assert weights.shape[0] == observations.shape[0]

        if self.weight_vec_size == 0:
            observation = observations
        elif weights is None:
            observations = torch.hstack([observations, 1/self.weight_vec_size * torch.ones((observations.shape[0], self.weight_vec_size))])
        else:
            observations =  torch.hstack([observations, weights])

        action_dist = self.get_action_distribution(observations)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
        if self.shield is not None:
            action = self.shield(observations, action)
        action = torch.clamp(action, torch.Tensor(self.action_space_low).to(action.device), torch.Tensor(self.action_space_high).to(action.device))
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, weights = None):

        if weights is not None:
            assert weights.shape[0] == observations.shape[0]

        if self.weight_vec_size == 0:
            observation = observations
        elif weights is None:
            observations = torch.hstack([observations, 1/self.weight_vec_size * torch.ones((observation.shape[0], self.weight_vec_size))])
        else:
            observations =  torch.hstack([observations, weights])

        action_dist = self.get_action_distribution(observations)
        if self.rpo_alpha is not None:
            # sample again to add stochasticity to the policy
            action_mean = action_dist.mean
            z = (
                torch.FloatTensor(action_mean.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(self.actor_logstd.device)
            )
            action_mean = action_mean + z
            action_dist = Normal(action_mean, action_dist.stddev)

        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy