import numpy as np
import torch
import torch.nn.functional as F
from sac.agent import VectorizedSACAgent
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4



class VectorReplayBuffer:
    """Replay Buffer for vectorized envs (stores experience for each env in batch)."""
    def __init__(self, obs_dim, act_dim, size, num_envs, reward_size=1):
        self.num_envs = num_envs
        self.obs_buf = np.zeros([size, num_envs, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, num_envs, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, num_envs, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, num_envs, reward_size], dtype=np.float32)
        self.done_buf = np.zeros([size, num_envs], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """All inputs shape: [num_envs, ...]"""
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        env_idxs = np.random.randint(0, self.num_envs, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs, env_idxs],
            obs2=self.next_obs_buf[idxs, env_idxs],
            acts=self.acts_buf[idxs, env_idxs],
            rews=self.rews_buf[idxs, env_idxs],
            done=self.done_buf[idxs, env_idxs],
        )
        return batch

class SACLogger:
    def __init__(self, run_name=None, use_tensorboard=True, reward_size=1, num_envs=1):
        self.use_tensorboard = use_tensorboard
        self.reward_size = reward_size
        self.num_envs = num_envs
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self._global_step = 0

    def log_rollout_step(self, infos, global_step):
        if "episode" in infos:
            non_zero_rews = infos['episode']['r'][infos['_episode']]
            non_zero_lens = infos['episode']['l'][infos['_episode']]
            non_zero_comps = []
            for i in range(self.reward_size):
                non_zero_comps.append(infos['episode'][f'r'][infos['_episode']][:,i].mean())
            print(
                f"global_step={global_step}, episodic_return={non_zero_rews.mean(axis = 0)}",
                flush=True,
            )

            if self.use_tensorboard:
                self.writer.add_scalar(
                    "charts/episodic_return", non_zero_rews.mean(), global_step
                )
                self.writer.add_scalar(
                    "charts/episodic_length", non_zero_lens.mean(), global_step
                )
                for i in range(self.reward_size):
                    self.writer.add_scalar(
                        f"charts/episodic_reward_{i}", non_zero_comps[i].mean(), global_step
                    )

    def log_training(self, update_results, step):
        """Log all the losses and diagnostics in update_results dict."""
        if self.use_tensorboard:
            for k, v in update_results.items():
                self.writer.add_scalar(k, v, step)


class SAC:
    def __init__(
        self,
        agent,
        env,                # a vectorized environment
        replay_buffer,
        optimizer_actor,
        optimizer_critic,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        target_entropy=None,
        total_steps=1_000_000,
        initial_random_steps=10_000,
        update_after=1_000,
        update_every=50,
        logger=None,
        lr=3e-4,
        anneal_lr=True,
    ):
        self.agent = agent
        self.env = env
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = (
            target_entropy or -np.prod(env.single_action_space.shape)
        )
        self.total_steps = total_steps
        self.initial_random_steps = initial_random_steps
        self.update_after = update_after
        self.update_every = update_every
        self.device = next(agent.actor.parameters()).device
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.anneal_lr = anneal_lr
        self.lr = lr

        self.lr_scheduler = None
        self.logger = logger or SACLogger()
        self._global_step = 0

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None
            self.alpha_optim = None

        self.num_envs = getattr(env, "num_envs", 1)
        self.reward_size = getattr(env, "rewards_shape", (1,))[-1]  # Last dimension of rewards shape

    def learn(self):
        obs, _ = self.env.reset(seed=None)
        episode_rewards = np.zeros((self.num_envs, self.reward_size),  dtype=np.float32)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        episode_counts = np.zeros(self.num_envs, dtype=np.int32)
        ep_returns_buffer = [[] for _ in range(self.num_envs)]
        num_updates = self.total_steps // self.update_every

        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_updates)

        for t in range(1, self.total_steps + 1):
            if t < self.initial_random_steps:
                action = np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                with torch.no_grad():
                    actions, _ = self.agent.actor.sample(obs_tensor)
                action = actions.cpu().numpy()

            next_obs, rewards, dones, truncs, infos = self.env.step(action)
            # Each of these is shape [num_envs, ...]

            # Track episodic returns
            episode_rewards += rewards
            episode_lengths += 1

            # Log completed episodes
            for i in range(self.num_envs):
                if dones[i] or truncs[i]:
                    ep_return = episode_rewards[i]
                    self.logger.log_rollout_step(infos, t)
                    ep_returns_buffer[i].append(ep_return)
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_counts[i] += 1

            # Store transition for all envs in buffer
            self.buffer.store(obs, action, rewards, next_obs, dones | truncs)

            obs = next_obs

            # Policy/critic updates after enough steps
            if t >= self.update_after and t % self.update_every == 0:
                if self.anneal_lr:
                    self.lr_scheduler.step()
                for _ in range(self.update_every):
                    self.update_parameters()

            self._global_step = t

        return self.agent

    def update_parameters(self):
        batch = self.buffer.sample_batch(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        obs2 = torch.FloatTensor(batch["obs2"]).to(self.device)
        acts = torch.FloatTensor(batch["acts"]).to(self.device)
        rews = torch.FloatTensor(batch["rews"]).to(self.device)
        done = torch.FloatTensor(batch["done"]).to(self.device)

        # Handle vector reward
        reward_size = rews.shape[-1] if rews.ndim > 1 else 1
        if reward_size == 1:
            rews = rews.unsqueeze(-1)
            done = done.unsqueeze(-1)
        else:
            # Already shape [batch, reward_size]
            done = done.unsqueeze(-1).expand(-1, reward_size)

        # === Critic update (PCGrad-style, per-objective) ===
        critic_loss = 0.0
        self.optimizer_critic.zero_grad()
        q1, q2 = self.agent.critic(obs, acts)
        with torch.no_grad():
            next_action, next_logp = self.agent.sample_action_and_compute_log_prob(obs2)
            target_q1, target_q2 = self.agent.critic_target(obs2, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logp
            q_target = rews + (1 - done) * self.gamma * target_q

        # Sum of MSE losses per reward dimension, each backwarded
        for i in range(reward_size):
            q1_loss = F.mse_loss(q1[:, i], q_target[:, i])
            q2_loss = F.mse_loss(q2[:, i], q_target[:, i])
            (q1_loss + q2_loss).backward(retain_graph=True)
            critic_loss += (q1_loss + q2_loss).item()
        self.optimizer_critic.step()

        # === Policy update (PCGrad-style, per-objective) ===
        policy_loss = 0.0
        self.optimizer_actor.zero_grad()
        pi, logp = self.agent.actor.sample(obs)
        q1_pi, q2_pi = self.agent.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # Policy loss for each reward dimension
        for i in range(reward_size):
            pol_loss = ((self.alpha * logp.squeeze(-1)) - min_q_pi[:, i]).mean()
            pol_loss.backward(retain_graph=True)
            policy_loss += pol_loss.item()
        self.optimizer_actor.step()

        # === Entropy (temperature) update ===
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0)

        # === Target network update (soft) ===
        for param, target_param in zip(self.agent.critic.parameters(), self.agent.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # === Logging ===
        info = {
            "losses/critic_loss": critic_loss,
            "losses/policy_loss": policy_loss,
            "losses/alpha_loss": alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            "alpha": self.alpha.item() if self.automatic_entropy_tuning else self.alpha,
        }
        self.logger.log_training(info, self._global_step)

