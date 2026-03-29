import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module):
    def __init__(self, envs, reward_size, task_size):
        super().__init__()
        self.reward_size = reward_size
        self.task_size = task_size
        self.num_tasks = max(1, task_size)

        try:
            self.obs_shape = envs.single_observation_space.shape
            self.obs_dim = np.array(self.obs_shape).prod()
        except:
            self.obs_shape = envs.observation_space.shape
            self.obs_dim = np.array(self.obs_shape).prod()

        self.actor_input_dim = self.obs_dim + self.reward_size
        self.critic_input_dim = self.obs_dim + self.reward_size + self.task_size

    def _format_inputs(self, observation, accumulated_reward, device):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=device)
        else:
            observation = observation.to(device)

        if observation.dim() > 2:
            observation = observation.reshape(observation.shape[0], -1)
        elif observation.dim() == 1:
            observation = observation.unsqueeze(0)

        if not isinstance(accumulated_reward, torch.Tensor):
            accumulated_reward = torch.tensor(accumulated_reward, dtype=torch.float32, device=device)
        else:
            accumulated_reward = accumulated_reward.to(device)

        if accumulated_reward.dim() == 1:
            accumulated_reward = accumulated_reward.unsqueeze(0)

        return observation, accumulated_reward

    def _get_task_indices(self, task_id, batch_size, device):
        if self.task_size == 0:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor(task_id, device=device)
        else:
            task_id = task_id.to(device)

        if task_id.dim() > 1 and task_id.shape[-1] > 1:
            return task_id.argmax(dim=-1)

        return task_id.flatten().long()

    def _get_critic_input(self, obs, acc_reward, task_id, device):
        if self.task_size > 0:
            if task_id is None:
                raise ValueError("Critic requires task_id")

            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.float32, device=device)
            else:
                task_id = task_id.to(device)

            if task_id.dim() == 1:
                task_id = task_id.unsqueeze(0)

            return torch.cat([obs, acc_reward, task_id], dim=1)
        else:
            return torch.cat([obs, acc_reward], dim=1)


class DiscreteAgent(BaseAgent):
    def __init__(self, envs, reward_size=1, task_size=0):
        super().__init__(envs, reward_size, task_size)

        try:
            self.n_actions = envs.single_action_space.n
        except:
            self.n_actions = envs.action_space.n

        # --- ACTOR ---
        self.actor_body = nn.ModuleList()
        for _ in range(self.num_tasks):
            self.actor_body.append(
                nn.Sequential(
                    layer_init(nn.Linear(self.actor_input_dim, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                )
            )

        self.actor_heads = nn.ModuleList()
        for _ in range(self.num_tasks):
            self.actor_heads.append(layer_init(nn.Linear(64, self.n_actions), std=0.01))

        # --- CRITIC ---
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(self.critic_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_utility_head = layer_init(nn.Linear(64, 1), std=1.0)
        self.critic_returns_head = layer_init(nn.Linear(64, reward_size), std=1.0)

        # Wrapper: all heads registered so optimizer[1] updates everything
        self.critic = nn.Module()
        self.critic.add_module('body', self.critic_body)
        self.critic.add_module('utility_head', self.critic_utility_head)
        self.critic.add_module('returns_head', self.critic_returns_head)

        self.actor = nn.Module()
        self.actor.add_module('body', self.actor_body)
        self.actor.add_module('heads', self.actor_heads)

    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        batch_size = obs.shape[0]

        actor_input = torch.cat([obs, acc], dim=1)
        task_indices = self._get_task_indices(task_id, batch_size, device)
        unique_tasks = torch.unique(task_indices)

        if self.task_size == 1:
            latent_features = self.actor_body[0](actor_input)
            logits = self.actor_heads[0](latent_features)
            return Categorical(logits=logits)

        logit_parts = []
        index_parts = []

        for t_idx in unique_tasks:
            t_int = t_idx.item()
            if t_int >= len(self.actor_heads):
                continue
            mask = (task_indices == t_idx)
            sub_latents = self.actor_body[t_int](actor_input[mask])
            sub_logits = self.actor_heads[t_int](sub_latents)
            logit_parts.append(sub_logits)
            index_parts.append(torch.nonzero(mask).flatten())

        if len(logit_parts) == 0:
            return Categorical(logits=torch.zeros(batch_size, self.n_actions, device=device, requires_grad=True))

        flat_logits = torch.cat(logit_parts, dim=0)
        flat_indices = torch.cat(index_parts, dim=0)
        argsort = torch.argsort(flat_indices)
        return Categorical(logits=flat_logits[argsort])

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        critic_input = self._get_critic_input(obs, acc, task_id, device)
        features = self.critic_body(critic_input)
        return self.critic_utility_head(features), self.critic_returns_head(features)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        action = dist.logits.argmax(dim=1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        if torch.is_floating_point(actions):
            actions = actions.long()
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        return dist.log_prob(actions), dist.entropy()

    def forward(self, observation, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observation, accumulated_reward, task_id, device)
        return dist.sample()

    @torch.no_grad()
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action, _ = self.sample_action_and_compute_log_prob(observation, accumulated_reward, task_id, deterministic, device)
        util, _ = self.estimate_value_from_observation(observation, accumulated_reward, task_id, device)
        return action.cpu().numpy(), util.cpu().numpy()


class ContinuousAgent(BaseAgent):
    def __init__(self, envs, reward_size=1, task_size=0, rpo_alpha=None):
        super().__init__(envs, reward_size, task_size)
        self.rpo_alpha = rpo_alpha

        try:
            self.action_dim = np.prod(envs.single_action_space.shape)
        except:
            self.action_dim = np.prod(envs.action_space.shape)

        # --- ACTOR ---
        self.actor_body = nn.ModuleList()
        for _ in range(self.num_tasks):
            self.actor_body.append(
                nn.Sequential(
                    layer_init(nn.Linear(self.actor_input_dim, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                )
            )

        self.actor_means = nn.ModuleList()
        self.actor_logstds = nn.ParameterList()

        for _ in range(self.num_tasks):
            self.actor_means.append(layer_init(nn.Linear(64, self.action_dim), std=1.0))
            self.actor_logstds.append(nn.Parameter(torch.zeros(1, self.action_dim)))

        # --- CRITIC ---
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(self.critic_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_utility_head = layer_init(nn.Linear(64, 1), std=1.0)
        self.critic_returns_head = layer_init(nn.Linear(64, reward_size), std=1.0)

        # PATCH 4: All three critic components registered so optimizer[1]
        # = Adam(agent.critic.parameters()) updates ALL of them.
        # Previously only 'body' was added, leaving utility_head and
        # returns_head completely frozen during training.
        self.critic = nn.Module()
        self.critic.add_module('body', self.critic_body)
        self.critic.add_module('utility_head', self.critic_utility_head)
        self.critic.add_module('returns_head', self.critic_returns_head)

        self.actor = nn.Module()
        self.actor.add_module('body', self.actor_body)
        self.actor.add_module('means', self.actor_means)
        self.actor.add_module('logstds', self.actor_logstds)

    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        batch_size = obs.shape[0]

        actor_input = torch.cat([obs, acc], dim=1)
        task_indices = self._get_task_indices(task_id, batch_size, device)
        unique_tasks = torch.unique(task_indices)

        if self.task_size <= 1:
            latent_features = self.actor_body[0](actor_input)
            mean = self.actor_means[0](latent_features)
            scale = torch.exp(self.actor_logstds[0].expand_as(mean))
            return Normal(mean, scale)

        mean_parts = []
        scale_parts = []
        index_parts = []

        for t_idx in unique_tasks:
            t_int = t_idx.item()
            if t_int >= len(self.actor_means):
                continue
            mask = (task_indices == t_idx)
            sub_latents = self.actor_body[t_int](actor_input[mask])
            sub_mean = self.actor_means[t_int](sub_latents)
            sub_scale = torch.exp(self.actor_logstds[t_int].expand_as(sub_mean))
            mean_parts.append(sub_mean)
            scale_parts.append(sub_scale)
            index_parts.append(torch.nonzero(mask).flatten())

        if len(mean_parts) == 0:
            dummy = torch.zeros(batch_size, self.action_dim, device=device, requires_grad=True)
            return Normal(dummy, torch.exp(dummy))

        flat_means = torch.cat(mean_parts, dim=0)
        flat_scales = torch.cat(scale_parts, dim=0)
        flat_indices = torch.cat(index_parts, dim=0)
        argsort = torch.argsort(flat_indices)
        return Normal(flat_means[argsort], flat_scales[argsort])

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        critic_input = self._get_critic_input(obs, acc, task_id, device)
        features = self.critic_body(critic_input)
        return self.critic_utility_head(features), self.critic_returns_head(features)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        if self.rpo_alpha is not None and self.training:
            mean = dist.mean
            z = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            dist = Normal(mean + z, dist.stddev)
        return dist.log_prob(actions).sum(1), dist.entropy().sum(1)

    def forward(self, observation, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observation, accumulated_reward, task_id, device)
        return dist.rsample()

    @torch.no_grad()
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        action, _ = self.sample_action_and_compute_log_prob(observation, accumulated_reward, task_id, deterministic, device)
        util, _ = self.estimate_value_from_observation(observation, accumulated_reward, task_id, device)
        return action.cpu().numpy(), util.cpu().numpy()