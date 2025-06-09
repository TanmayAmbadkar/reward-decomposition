import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class VectorizedSACAgent(nn.Module):
    """A continuous SAC actor-critic supporting vectorized rewards and envs."""
    def __init__(self, obs_space, act_space, reward_size=1, hidden_dim=256):
        super().__init__()
        obs_dim = np.prod(obs_space.shape)
        act_dim = np.prod(act_space.shape)
        self.act_dim = act_dim
        self.reward_size = reward_size
        self.weight_vec_size = 0 if reward_size == 1 else reward_size
        
        print(f"obs_dim: {obs_dim}, act_dim: {act_dim}, weight_vec_size: {self.weight_vec_size}")
        # Critic (Q1/Q2)
        self.critic1 = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim + self.weight_vec_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, reward_size))
        )
        self.critic2 = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim + self.weight_vec_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, reward_size))
        )
        # Target critics
        self.critic1_target = nn.Sequential(*[layer for layer in self.critic1])
        self.critic2_target = nn.Sequential(*[layer for layer in self.critic2])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Actor (Gaussian policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim + self.weight_vec_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, act_dim))
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        # 3) now make a *container* module and stick both parts on it
        self.actor = nn.Module()           # empty container
        # register the mean‐net as a submodule
        self.actor.add_module('mean', self.actor_mean)
        # register the logstd parameter
        self.actor.register_parameter('logstd', self.actor_logstd)

    def get_action_distribution(self, obs, weights=None):
        if self.weight_vec_size == 0:
            obs_aug = obs
        elif weights is None:
            obs_aug = torch.cat([obs, torch.ones((obs.shape[0], self.weight_vec_size), device=obs.device)], dim=-1)
        else:
            obs_aug = torch.cat([obs, weights], dim=-1)
        mean = self.actor_mean(obs_aug)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        return Normal(mean, std)

    def sample_action_and_compute_log_prob(self, obs, weights=None, deterministic=False):
        dist = self.get_action_distribution(obs, weights)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, obs, actions, weights=None):
        dist = self.get_action_distribution(obs, weights)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

    def critic(self, obs, actions, weights=None):
        if self.weight_vec_size == 0:
            obs_act = torch.cat([obs, actions], dim=-1)
        elif weights is None:
            obs_act = torch.cat([obs, actions, torch.ones((obs.shape[0], self.weight_vec_size), device=obs.device)], dim=-1)
        else:
            obs_act = torch.cat([obs, actions, weights], dim=-1)
        return self.critic1(obs_act), self.critic2(obs_act)

    def critic_target(self, obs, actions, weights=None):
        if self.weight_vec_size == 0:
            obs_act = torch.cat([obs, actions], dim=-1)
        elif weights is None:
            obs_act = torch.cat([obs, actions, torch.ones((obs.shape[0], self.weight_vec_size), device=obs.device)], dim=-1)
        else:
            obs_act = torch.cat([obs, actions, weights], dim=-1)
        return self.critic1_target(obs_act), self.critic2_target(obs_act)

    def update_targets(self, tau=0.005):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
