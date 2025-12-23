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

        # 1. Setup Input Dimensions
        try:
            self.obs_shape = envs.single_observation_space.shape
            self.obs_dim = np.array(self.obs_shape).prod()
        except:
            self.obs_shape = envs.observation_space.shape
            self.obs_dim = np.array(self.obs_shape).prod()

        # Actor Input: Obs + Accumulated Reward
        self.actor_input_dim = self.obs_dim + self.reward_size
        
        # Critic Input: Obs + Accumulated Reward + Task ID
        self.critic_input_dim = self.obs_dim + self.reward_size + self.task_size

    def _format_inputs(self, observation, accumulated_reward, device):
        """Standardizes inputs to (B, Dim) tensors on the correct device."""
        # Handle Observation
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=device)
        else:
            observation = observation.to(device)
        
        # Flatten if image or multidimensional
        if observation.dim() > 2:
            observation = observation.reshape(observation.shape[0], -1)
        elif observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Handle Reward
        if not isinstance(accumulated_reward, torch.Tensor):
            accumulated_reward = torch.tensor(accumulated_reward, dtype=torch.float32, device=device)
        else:
            accumulated_reward = accumulated_reward.to(device)
            
        if accumulated_reward.dim() == 1:
            accumulated_reward = accumulated_reward.unsqueeze(0)

        return observation, accumulated_reward

    def _get_task_indices(self, task_id, batch_size, device):
        """Converts diverse task_id formats (int, one-hot, list) into a flat LongTensor indices."""
        if self.task_size == 0:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor(task_id, device=device)
        else:
            task_id = task_id.to(device)

        # Handle One-Hot (B, T) -> Indices (B,)
        if task_id.dim() > 1 and task_id.shape[-1] > 1:
            return task_id.argmax(dim=-1)
        
        # Handle Raw Indices
        return task_id.flatten().long()

    def _get_critic_input(self, obs, acc_reward, task_id, device):
        """Critic explicitly needs Task ID as an input feature."""
        if self.task_size > 0:
            if task_id is None:
                raise ValueError("Critic requires task_id")
            
            # Ensure task_id is tensor
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.float32, device=device)
            else:
                task_id = task_id.to(device)

            # Ensure shapes match for concat
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
        # 1. Common Body
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

        # 2. Multi-Head Output (One Head per Task)
        self.actor_heads = nn.ModuleList()
        for _ in range(self.num_tasks):
            # std=0.01 ensures uniform probability at start
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

        # Backwards compatibility wrappers
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

        # 1. Common Body Processing
        actor_input = torch.cat([obs, acc], dim=1)

        # 2. Routing Logi
        task_indices = self._get_task_indices(task_id, batch_size, device)
        unique_tasks = torch.unique(task_indices)
        
        # We collect parts and stitch them back together
        logit_parts = []
        index_parts = []
        
        if self.task_size == 1:
            latent_features = self.actor_body[0](actor_input)
            logits = self.actor_heads[0](latent_features)
            return Categorical(logits=logits)


        for t_idx in unique_tasks:
            t_int = t_idx.item()
            if t_int >= len(self.actor_heads): 
                print(f"Task {t_int} not found in actor heads")
                continue

            # Get samples belonging to this task
            mask = (task_indices == t_idx)
            sub_latents = self.actor_body[t_int](actor_input[mask])
            
            # Pass through specific head
            sub_logits = self.actor_heads[t_int](sub_latents)
            
            logit_parts.append(sub_logits)
            index_parts.append(torch.nonzero(mask).flatten())

        # 3. Reassembly
        if len(logit_parts) == 0:
            # Fallback for graph safety
            print("No logit parts found")
            return Categorical(logits=torch.zeros(batch_size, self.n_actions, device=device, requires_grad=True))

        flat_logits = torch.cat(logit_parts, dim=0)
        flat_indices = torch.cat(index_parts, dim=0)

        # Sort indices to restore original batch order
        argsort = torch.argsort(flat_indices)
        sorted_logits = flat_logits[argsort]

        return Categorical(logits=sorted_logits)

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        
        critic_input = self._get_critic_input(obs, acc, task_id, device)
        
        features = self.critic_body(critic_input)
        return self.critic_utility_head(features), self.critic_returns_head(features)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        if deterministic:
            action = dist.logits.argmax(dim=1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        
        # Robust Discrete Handling: Cast to Long, Squeeze extra dims
        if torch.is_floating_point(actions):
            actions = actions.long()
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
            
        return dist.log_prob(actions), dist.entropy()

    def forward(self, observation, accumulated_reward, task_id=None, device="cpu"):
        """Returns sampled action. Preserves Gradients."""
        dist = self.get_action_distribution(observation, accumulated_reward, task_id, device)
        return dist.sample()

    @torch.no_grad
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        """Inference only. Returns Numpy arrays."""
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
        # 1. SEPARATE BODIES (One per task, matching DiscreteAgent style)
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

        # 2. Multi-Head Output (Mean and LogStd)
        self.actor_means = nn.ModuleList()
        self.actor_logstds = nn.ParameterList()
        
        for _ in range(self.num_tasks):
            # std=1.0 for continuous ensures we don't start with 0 variance
            self.actor_means.append(layer_init(nn.Linear(64, self.action_dim), std=1.0))
            self.actor_logstds.append(nn.Parameter(torch.zeros(1, self.action_dim)))

        # --- CRITIC ---
        # Shared Critic
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(self.critic_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_utility_head = layer_init(nn.Linear(64, 1), std=1.0)
        self.critic_returns_head = layer_init(nn.Linear(64, reward_size), std=1.0)
        
        # Backwards compatibility wrappers
        self.critic = nn.Module()
        self.critic.add_module('body', self.critic_body)

    def get_action_distribution(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        batch_size = obs.shape[0]

        # 1. Input
        actor_input = torch.cat([obs, acc], dim=1)

        # 2. Routing Logic
        task_indices = self._get_task_indices(task_id, batch_size, device)
        unique_tasks = torch.unique(task_indices)

        # Optimization for Single Task
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
            if t_int >= len(self.actor_means): continue

            mask = (task_indices == t_idx)
            
            # [CRITICAL] Route to specific Body AND Head
            sub_latents = self.actor_body[t_int](actor_input[mask])
            sub_mean = self.actor_means[t_int](sub_latents)
            sub_scale = torch.exp(self.actor_logstds[t_int].expand_as(sub_mean))
            
            mean_parts.append(sub_mean)
            scale_parts.append(sub_scale)
            index_parts.append(torch.nonzero(mask).flatten())

        # 3. Reassembly
        if len(mean_parts) == 0:
            dummy = torch.zeros(batch_size, self.action_dim, device=device, requires_grad=True)
            return Normal(dummy, torch.exp(dummy))

        flat_means = torch.cat(mean_parts, dim=0)
        flat_scales = torch.cat(scale_parts, dim=0)
        flat_indices = torch.cat(index_parts, dim=0)

        argsort = torch.argsort(flat_indices)
        sorted_means = flat_means[argsort]
        sorted_scales = flat_scales[argsort]

        return Normal(sorted_means, sorted_scales)

    def estimate_value_from_observation(self, observation, accumulated_reward, task_id=None, device="cpu"):
        obs, acc = self._format_inputs(observation, accumulated_reward, device)
        
        critic_input = self._get_critic_input(obs, acc, task_id, device)
        
        features = self.critic_body(critic_input)
        return self.critic_utility_head(features), self.critic_returns_head(features)

    def sample_action_and_compute_log_prob(self, observations, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample() # rsample allows gradients through sampling for continuous
        
        # For continuous, log_prob usually needs summation over action dimensions
        log_prob = dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observations, accumulated_reward, task_id, device)
        
        # RPO (Robust Policy Optimization) Handling
        if self.rpo_alpha is not None and self.training:
            mean = dist.mean
            z = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            # Create a new perturbed distribution
            dist = Normal(mean + z, dist.stddev)

        return dist.log_prob(actions).sum(1), dist.entropy().sum(1)

    def forward(self, observation, accumulated_reward, task_id=None, device="cpu"):
        dist = self.get_action_distribution(observation, accumulated_reward, task_id, device)
        return dist.rsample() # rsample allows gradients through sampling

    @torch.no_grad
    def predict(self, observation, accumulated_reward, task_id=None, deterministic=False, device="cpu"):
        """Inference only. Returns Numpy arrays."""
        action, _ = self.sample_action_and_compute_log_prob(observation, accumulated_reward, task_id, deterministic, device)
        util, _ = self.estimate_value_from_observation(observation, accumulated_reward, task_id, device)
        return action.cpu().numpy(), util.cpu().numpy()