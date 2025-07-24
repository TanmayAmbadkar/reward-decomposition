from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# This import is commented out as the resnet.py file was not provided,
# but it can be uncommented if needed.
# from ppo.resnet import WeightFeatureExtractorNet

import torchbnn as bnn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- NEW ACTOR CLASS ---
# This new class encapsulates the actor's logic and has the required forward method.
class ContinuousActor(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, x):
        """
        Defines the forward pass for the actor. This is what ONNX will trace.
        It returns the mean and log_std concatenated, which is what the
        client-side application will expect.
        """
        action_mean = self.actor_mean(x)
        # We don't need to expand logstd here for the ONNX model,
        # as the model will output the raw logstd parameter.
        # The expansion can be done in the main agent logic if needed.
        # For the exported model, we want a consistent output shape.
        # We will return both mean and log_std concatenated.
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return torch.cat([action_mean, action_logstd], dim=1)


class BaseAgent(nn.Module, ABC):
    @abstractmethod
    def estimate_value_from_observation(self, observation, weights=None, device="cpu"):
        pass

    @abstractmethod
    def get_action_distribution(self, observation):
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations, weights=None, deterministic=False):
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(self, observations, actions, weights=None):
        pass
    
    def forward(self, observation):
        # The base forward can be a simple pass-through for ONNX compatibility if needed,
        # but the specific agent's forward is what matters.
        pass


class DiscreteAgent(BaseAgent):
    def __init__(self, envs, reward_size=1):
        super().__init__()
        self.reward_size = reward_size
        self.weight_vec_size = 0 if reward_size == 1 else reward_size

        try:
            action_space_n = envs.single_action_space.n
            observation_space_shape = envs.single_observation_space.shape
        except AttributeError:
            action_space_n = envs.action_space.n
            observation_space_shape = envs.observation_space.shape

        input_size = np.array(observation_space_shape).prod() + self.weight_vec_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, reward_size), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_space_n), std=0.01),
        )

    def _prepare_input(self, observation, weights=None, device="cpu"):
        if self.weight_vec_size > 0:
            if weights is None:
                # Create default weights if none are provided
                weights = torch.ones((observation.shape[0], self.weight_vec_size), device=device)
            return torch.hstack([observation, weights])
        return observation

    def estimate_value_from_observation(self, observation, weights=None, device="cpu"):
        prepared_input = self._prepare_input(observation, weights, device)
        return self.critic(prepared_input)

    def get_action_distribution(self, observation, weights=None):
        prepared_input = self._prepare_input(observation, weights)
        logits = self.actor(prepared_input)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations, weights=None, deterministic=False):
        action_dist = self.get_action_distribution(observations, weights)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, weights=None):
        action_dist = self.get_action_distribution(observations, weights)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy
    
    @torch.no_grad()
    def predict(self, observation, weight=None, deterministic=False, device="cpu"):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        weight_tensor = None
        if self.weight_vec_size > 0 and weight is not None:
            weight_tensor = torch.as_tensor(weight, dtype=torch.float32, device=device).reshape(observation.shape[0], -1)

        action_dist = self.get_action_distribution(observation, weight_tensor)

        if deterministic:
            action = torch.argmax(action_dist.logits, dim=1)
        else:
            action = action_dist.sample()
        
        value = self.estimate_value_from_observation(observation, weight_tensor, device)
        return action.cpu().numpy(), value.cpu().numpy()

    def forward(self, x):
        # The actor is a sequential model, so its forward pass is implicit.
        # For ONNX export, you'd export self.actor directly.
        return self.actor(x)


class ContinuousAgent(BaseAgent):
    def __init__(self, envs, rpo_alpha=None, reward_size=1):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.reward_size = reward_size
        self.weight_vec_size = 0 if reward_size == 1 else reward_size

        try:
            action_space_shape = envs.single_action_space.shape
            observation_space_shape = envs.single_observation_space.shape
        except AttributeError:
            action_space_shape = envs.action_space.shape
            observation_space_shape = envs.observation_space.shape
            
        input_size = np.array(observation_space_shape).prod() + self.weight_vec_size
        action_size = np.prod(action_space_shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.reward_size), std=1.0),
        )
        
        # Use the new ContinuousActor class
        self.actor = ContinuousActor(input_size, action_size)

    def _prepare_input(self, observation, weights=None, device="cpu"):
        if self.weight_vec_size > 0:
            if weights is None:
                weights = torch.ones((observation.shape[0], self.weight_vec_size), device=device)
            # Ensure weights are on the same device as observation
            weights = weights.to(observation.device)
            return torch.hstack([observation, weights])
        return observation

    def estimate_value_from_observation(self, observation, weights=None, device="cpu"):
        prepared_input = self._prepare_input(observation, weights, device)
        return self.critic(prepared_input)

    def get_action_distribution(self, observation, weights=None):
        prepared_input = self._prepare_input(observation, weights)
        # The actor's forward method returns mean and log_std concatenated
        actor_output = self.actor(prepared_input)
        
        action_size = self.actor.actor_logstd.shape[1]
        action_mean = actor_output[:, :action_size]
        action_logstd = actor_output[:, action_size:]
        
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    @torch.no_grad()
    def predict(self, observation, weight=None, deterministic=False, device="cpu"):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        weight_tensor = None
        if self.weight_vec_size > 0 and weight is not None:
             weight_tensor = torch.as_tensor(weight, dtype=torch.float32, device=device).reshape(observation.shape[0], -1)

        action_dist = self.get_action_distribution(observation, weight_tensor)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
            
        value = self.estimate_value_from_observation(observation, weight_tensor, device)
        return action.cpu().numpy(), value.cpu().numpy()

    def sample_action_and_compute_log_prob(self, observations, weights=None, deterministic=False):
        action_dist = self.get_action_distribution(observations, weights)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
        
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, weights=None):
        action_dist = self.get_action_distribution(observations, weights)
        
        if self.rpo_alpha is not None:
            action_mean = action_dist.mean
            z = (torch.FloatTensor(action_mean.shape)
                 .uniform_(-self.rpo_alpha, self.rpo_alpha)
                 .to(action_mean.device))
            action_mean = action_mean + z
            action_dist = Normal(action_mean, action_dist.stddev)

        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy
    
    def forward(self, x):
        """
        Defines the forward pass for the entire agent for ONNX export.
        We only care about the actor's output for inference.
        """
        return self.actor(x)
