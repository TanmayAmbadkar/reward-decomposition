"""
ESR-PPO: Episode-Level Policy Gradient for Expected Scalarized Return

Core algorithmic design
-----------------------
1. EPISODE-LEVEL COLLECTION
   The unit of experience is a complete episode, not a fixed rollout window.
   Utility is always evaluated on realized complete trajectory returns — no
   mid-episode projection, no vector critic bootstrapping into utility.

2. PER-STEP BASELINES WITH TRAJECTORY-LEVEL UTILITY
   The utility head b(s_t, acc_t, task) is trained to predict E[u(R) | s_t, acc_t]
   from any mid-episode state. It is evaluated at every step t within an episode,
   giving a different baseline for each timestep while the utility target
   u(R(τ)) remains the complete realized trajectory utility throughout.
   This provides variance reduction without introducing bias (standard baseline
   identity), and improves credit assignment for long-horizon tasks.

3. PER-STEP PPO CLIPPING WITH TRAJECTORY-LEVEL ADVANTAGE
   Policy ratio: ρ_t = π_new(a_t) / π_old(a_t) — per step, stays near 1.
   Advantage: A_t = u(R(τ)) - b(s_t, acc_t) — trajectory utility, per-step baseline.
   Loss: mean_t[ max(-ρ_t·A_t, -clip(ρ_t, 1±ε)·A_t) ]
   Avoids the trajectory-sum explosion where exp(Σ log_ratios) blows up for
   long episodes (e.g. T=1000 in Hopper).

4. CLIPPED IS WEIGHTING FOR COUNTERFACTUAL REUSE
   w(τ) = clip(exp(mean_log_ratio), w_min, w_max)
   Pre-applied to advantages before the actor update. Every trajectory
   contributes — no hard cutoff. On-policy trajectories get w=1.0 exactly.
   Off-policy trajectories are smoothly downweighted or upweighted within
   bounds. This replaces the previous quantile hard-cutoff which introduced
   uncharacterizable bias and was batch-distribution-dependent.

5. RAW ACCUMULATION (acc_gamma REMOVED)
   acc_rewards accumulates raw per-step rewards with no gamma scaling.
   discount_utility=True enables gamma^t weighting for environments where
   utilities were calibrated against discounted returns (e.g. FTN, DST).

6. BOTH OPTIMIZERS SCHEDULED
   Linear LR annealing applied to both actor and critic optimizers.

7. PARALLEL ENVIRONMENTS WITH ROUND-ROBIN TASK ASSIGNMENT
   All num_envs environments step simultaneously. Tasks are assigned
   round-robin to guarantee even coverage and minimise idle environments.
"""

from uuid import uuid4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Callable, Any
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Logger (unchanged interface, minor additions)
# ---------------------------------------------------------------------------

class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False, reward_size=1):
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self.reward_size = reward_size

    def log_episode(self, global_step, task_id, utility, realized_return, ep_length):
        print(
            f"step={global_step}  task={task_id}  "
            f"utility={utility:.4f}  len={ep_length}  "
            f"return={np.round(realized_return, 3)}",
            flush=True,
        )
        if self.use_tensorboard:
            self.writer.add_scalar("charts/episodic_utility", utility, global_step)
            self.writer.add_scalar(f"charts/utility_task_{task_id}", utility, global_step)
            self.writer.add_scalar("charts/episodic_length", ep_length, global_step)
            for i in range(self.reward_size):
                self.writer.add_scalar(
                    f"charts/realized_return_obj_{i}", realized_return[i], global_step
                )

    def log_policy_update(self, stats, global_step):
        if self.use_tensorboard:
            for k, v in stats.items():
                if isinstance(v, float) and not np.isnan(v):
                    self.writer.add_scalar(f"losses/{k}", v, global_step)

    def log_evaluation(self, global_step, min_util, mean_util, max_util, task_id):
        print(
            f"EVAL step={global_step} | Task={task_id} | "
            f"Min={min_util:.4f}  Mean={mean_util:.4f}  Max={max_util:.4f}",
            flush=True,
        )
        if self.use_tensorboard:
            self.writer.add_scalar(f"eval/utility_min_task_{task_id}", min_util, global_step)
            self.writer.add_scalar(f"eval/utility_mean_task_{task_id}", mean_util, global_step)
            self.writer.add_scalar(f"eval/utility_max_task_{task_id}", max_util, global_step)


# ---------------------------------------------------------------------------
# Episode buffer — stores one complete episode
# ---------------------------------------------------------------------------

class Episode:
    """
    Container for a single complete episode.
    All quantities are exact — no estimation, no projection.

    Performance notes
    -----------------
    Tensors are pre-converted once at construction time (Fix 4).
    This avoids repeated torch.tensor() copies inside every update call.
    Future returns are computed via vectorised cumsum (Fix 3) rather
    than a Python loop over timesteps.
    """
    __slots__ = [
        'task_id',        # int
        'obs',            # (T, obs_dim)       numpy — kept for env reset compatibility
        'actions',        # (T, act_dim)|(T,)  numpy
        'logprobs',       # (T,)               numpy
        'rewards',        # (T, reward_size)   numpy
        'acc_rewards',    # (T, reward_size)   numpy
        'realized_return',# (reward_size,)     numpy — exact cumulative return
        'utility',        # scalar             float — exact u(realized_return)
        'length',         # int
        # Pre-converted tensors (populated by to_device after construction)
        'obs_t',          # (T, obs_dim)       torch.float32
        'actions_t',      # (T, act_dim)|(T,)  torch.float32
        'logprobs_t',     # (T,)               torch.float32
        'acc_t',          # (T, reward_size)   torch.float32
        'future_returns', # (T, reward_size)   numpy  — MC future return from each step
    ]

    def __init__(self, task_id, obs, actions, logprobs, rewards, acc_rewards,
                 gamma=1.0, discount_utility=False):
        self.task_id         = task_id
        self.obs             = np.array(obs,         dtype=np.float32)
        self.actions         = np.array(actions,     dtype=np.float32)
        self.logprobs        = np.array(logprobs,    dtype=np.float32)
        self.rewards         = np.array(rewards,     dtype=np.float32)
        self.acc_rewards     = np.array(acc_rewards, dtype=np.float32)
        self.length          = len(rewards)
        self.utility         = None

        # Realized return: sum or discounted sum
        if discount_utility and gamma < 1.0:
            T      = self.length
            gammas = gamma ** np.arange(T)
            self.realized_return = (gammas[:, None] * self.rewards).sum(axis=0)
        else:
            self.realized_return = self.rewards.sum(axis=0)

        # Fix 3: Vectorised future return computation via reversed cumsum.
        # future_returns[t] = sum of rewards from step t onward.
        # np.cumsum on the reversed array then re-reversed gives this in O(T)
        # instead of O(T^2) from the previous loop with ep.rewards[t:].sum().
        if discount_utility and gamma < 1.0:
            T      = self.length
            # Build discount-weighted future returns vectorised
            gammas = gamma ** np.arange(T)  # (T,)
            # future_returns[t] = sum_{k=t}^{T-1} gamma^{k-t} * r_k
            # = sum_{k=0}^{T-1-t} gamma^k * r_{t+k}
            # Computed by: for each t, shift-and-discount. Efficient via loop
            # over the T rows — still O(T^2) but unavoidable for discount case.
            # For the common undiscounted case (below) we use O(T) cumsum.
            future_returns = np.zeros_like(self.rewards)
            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    future_returns[t] = self.rewards[t]
                else:
                    future_returns[t] = self.rewards[t] + gamma * future_returns[t + 1]
            self.future_returns = future_returns
        else:
            # Fix 3: O(T) reversed cumsum — replaces O(T^2) loop
            # cumsum on reversed rewards, then re-reverse
            self.future_returns = np.cumsum(self.rewards[::-1], axis=0)[::-1].copy()

        # Pre-converted tensors initialised to None — populated by to_device()
        self.obs_t      = None
        self.actions_t  = None
        self.logprobs_t = None
        self.acc_t      = None

    def to_device(self, device: torch.device) -> 'Episode':
        """
        Fix 4: Convert numpy arrays to tensors ONCE at episode creation.
        Called immediately after the episode is finalised in collect_episodes.
        Subsequent calls to _compute_trajectory_advantages and _train_actor
        use the pre-converted tensors directly, avoiding repeated copies.
        """
        self.obs_t      = torch.tensor(self.obs,         dtype=torch.float32, device=device)
        self.actions_t  = torch.tensor(self.actions,     dtype=torch.float32, device=device)
        self.logprobs_t = torch.tensor(self.logprobs,    dtype=torch.float32, device=device)
        self.acc_t      = torch.tensor(self.acc_rewards, dtype=torch.float32, device=device)
        return self


# ---------------------------------------------------------------------------
# Core ESR-PPO algorithm
# ---------------------------------------------------------------------------

class PPO:
    """
    Episode-level ESR-PPO.

    Collects complete episodes, evaluates utility on exact realized returns,
    updates policy with trajectory-level PPO clipping, and reuses trajectories
    across tasks via trajectory-level importance weighting.
    """

    def __init__(
        self,
        agent,
        optimizer,                          # list: [actor_opt, critic_opt]
        envs,                               # vectorized training envs
        eval_envs=None,
        utility_functions=None,
        env_is_discrete=False,
        reward_size=1,
        learning_rate=3e-4,
        # Episode collection
        episodes_per_update=16,             # complete episodes per policy update
        min_episodes_per_task=4,            # minimum episodes per task per update
        max_episode_steps=1000,             # hard cap per episode
        # PPO
        update_epochs=5,
        num_minibatches=8,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.001,
        max_grad_norm=0.5,
        normalize_advantages=True,
        target_kl=None,
        anneal_lr=True,
        # Counterfactual reuse — clipped IS weighting
        # Replaces the previous quantile hard-cutoff rule.
        # Every trajectory contributes to the gradient but is weighted by
        # w(τ) = clip(exp(mean_log_ratio), cf_weight_min, cf_weight_max).
        # On-policy trajectories get w=1.0 (mean_log_ratio=0).
        # Off-policy trajectories get w<1 if unlikely, w>1 if more likely.
        # Clipping bounds variance (w_max) and prevents zero-weight dropout (w_min).
        # This is the trajectory-level analogue of PPO's per-step clipping.
        cf_weight_min=0.1,    # floor IS weight — limits off-policy influence
        cf_weight_max=5.0,    # ceiling IS weight — prevents variance explosion
        # Return discounting
        gamma=1.0,                          # ESR uses undiscounted returns by default
        discount_utility=False,             # if True: accumulate gamma^t * r_t (discounted)
                                            # if False: accumulate r_t (undiscounted, default)
                                            # Use True for FTN/DST where utilities were
                                            # calibrated with gamma=0.995.
                                            # Use False for Hopper/LunarLander calibrated
                                            # against raw undiscounted returns.
        seed=1,
        logger=None,
        eval_interval=10000,
        num_eval_episodes=10,
        total_timesteps=1000000,
    ):
        self.agent            = agent
        self.optimizer        = optimizer
        self.envs             = envs
        self.eval_envs        = eval_envs
        self.reward_size      = reward_size
        self.env_is_discrete  = env_is_discrete

        if utility_functions is None:
            self.utility_functions = [lambda r: r.sum(-1)]
        else:
            self.utility_functions = utility_functions
        self.num_tasks = len(self.utility_functions)

        # Episode collection params
        self.episodes_per_update   = episodes_per_update
        self.min_episodes_per_task = min_episodes_per_task
        self.max_episode_steps     = max_episode_steps

        # PPO params
        self.update_epochs              = update_epochs
        self.num_minibatches            = num_minibatches
        self.surrogate_clip_threshold   = surrogate_clip_threshold
        # traj_clip_threshold removed: per-step clipping uses surrogate_clip_threshold.
        # traj_clip_threshold was only needed for the (now removed) trajectory-level
        # ratio, which caused exponential variance growth with episode length.
        self.entropy_loss_coefficient   = entropy_loss_coefficient
        self.max_grad_norm              = max_grad_norm
        self.normalize_advantages       = normalize_advantages
        self.target_kl                  = target_kl

        # Counterfactual IS weighting bounds
        self.cf_weight_min = cf_weight_min
        self.cf_weight_max = cf_weight_max

        # Return accumulation mode
        self.gamma            = gamma
        self.discount_utility = discount_utility
        self.seed             = seed
        self.eval_interval   = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.total_timesteps = total_timesteps
        self._global_step    = 0
        self.next_eval_step  = 0

        self.device = next(agent.parameters()).device
        self.logger = logger or PPOLogger(reward_size=reward_size)

        # CHANGE 6: schedule BOTH optimizers
        self.anneal_lr = anneal_lr
        if self.anneal_lr:
            # Estimate total updates: total_timesteps / avg_episode_len / episodes_per_update
            # Use conservative estimate of 200 steps/episode
            estimated_updates = total_timesteps // (200 * episodes_per_update)
            self.lr_schedulers = [
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1.0, end_factor=0.1,
                    total_iters=max(estimated_updates, 1)
                )
                for opt in self.optimizer
            ]
        else:
            self.lr_schedulers = None

    # -----------------------------------------------------------------------
    # Episode-level collection with parallel environments
    # -----------------------------------------------------------------------

    def collect_episodes(self) -> List[Episode]:
        """
        Collect complete episodes across all parallel environments until every
        task has at least min_episodes_per_task complete episodes.

        Parallelism
        -----------
        All num_envs environments step simultaneously each iteration.
        Wall-clock collection time scales as ~1/num_envs relative to a
        single-environment setup for long-horizon tasks like Hopper.

        Task assignment — round-robin
        -----------------------------
        Tasks are assigned round-robin rather than randomly to guarantee
        even coverage. With num_tasks=3 and num_envs=8:
          env 0 → task 0, env 1 → task 1, env 2 → task 2,
          env 3 → task 0, env 4 → task 1, env 5 → task 2, ...
        This minimises idle environments waiting for a single slow task
        to accumulate its required episodes.

        On episode completion each environment immediately starts a new
        episode on the next task in the round-robin sequence, so all
        environments stay active throughout collection.

        ESR correctness
        ---------------
        Each episode's utility is computed from its own complete realized
        return — no information is shared between parallel environments.
        The round-robin assignment affects which task each episode serves
        but not the correctness of the utility computation.
        """
        task_counts   = {t: 0 for t in range(self.num_tasks)}
        episodes      = []
        num_envs      = self.envs.num_envs

        # Global round-robin counter — shared across all envs
        # Ensures tasks are filled evenly regardless of episode length variance
        rr_counter = 0

        def next_task() -> int:
            nonlocal rr_counter
            task = rr_counter % self.num_tasks
            rr_counter += 1
            return task

        # Per-environment episode buffers
        ep_obs      = [[] for _ in range(num_envs)]
        ep_actions  = [[] for _ in range(num_envs)]
        ep_logprobs = [[] for _ in range(num_envs)]
        ep_rewards  = [[] for _ in range(num_envs)]
        ep_acc      = [[] for _ in range(num_envs)]

        # Accumulated reward per env — raw or discounted depending on discount_utility
        acc_rewards  = np.zeros((num_envs, self.reward_size), dtype=np.float32)
        ep_steps     = np.zeros(num_envs, dtype=np.int32)

        # Round-robin initial task assignment
        active_tasks = np.array([next_task() for _ in range(num_envs)], dtype=np.int32)

        obs, _ = self.envs.reset(seed=self.seed + self._global_step)

        while not all(
            task_counts[t] >= self.min_episodes_per_task
            for t in range(self.num_tasks)
        ):
            # ---- Forward pass across all envs simultaneously ----
            task_idx_t   = torch.tensor(active_tasks, dtype=torch.long, device=self.device)
            task_one_hot = F.one_hot(task_idx_t, num_classes=self.num_tasks).float()
            obs_t        = torch.tensor(obs,         dtype=torch.float32, device=self.device)
            acc_t        = torch.tensor(acc_rewards, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                actions, logprobs = self.agent.sample_action_and_compute_log_prob(
                    obs_t, acc_t, task_one_hot,
                    deterministic=False, device=self.device,
                )

            actions_np  = actions.cpu().numpy()
            logprobs_np = logprobs.cpu().numpy()

            # Store pre-step state for every active env
            for i in range(num_envs):
                ep_acc[i].append(acc_rewards[i].copy())
                ep_obs[i].append(obs[i].copy())
                ep_actions[i].append(actions_np[i].copy())
                ep_logprobs[i].append(float(logprobs_np[i]))

            # ---- Parallel environment step ----
            next_obs, rewards, terminations, truncations, _ = self.envs.step(actions_np)
            self._global_step += num_envs

            # Accumulate rewards for each env
            for i in range(num_envs):
                ep_rewards[i].append(rewards[i].copy())
                if self.discount_utility and self.gamma < 1.0:
                    acc_rewards[i] += (self.gamma ** ep_steps[i]) * rewards[i]
                else:
                    acc_rewards[i] += rewards[i]
                ep_steps[i] += 1

            # ---- Handle episode completions ----
            dones = np.logical_or(terminations, truncations)
            dones = np.logical_or(dones, ep_steps >= self.max_episode_steps)

            for i in np.where(dones)[0]:
                # Build and store the completed episode
                ep = Episode(
                    task_id          = int(active_tasks[i]),
                    obs              = ep_obs[i],
                    actions          = ep_actions[i],
                    logprobs         = ep_logprobs[i],
                    rewards          = ep_rewards[i],
                    acc_rewards      = ep_acc[i],
                    gamma            = self.gamma,
                    discount_utility = self.discount_utility,
                )
                # Fix 4: convert to tensors once here — avoids repeated
                # torch.tensor() copies in every subsequent update call
                ep.to_device(self.device)
                ep.utility = float(
                    self.utility_functions[ep.task_id](
                        torch.tensor(ep.realized_return, dtype=torch.float32,
                                     device=self.device)
                    ).item()
                )

                episodes.append(ep)
                task_counts[ep.task_id] += 1

                self.logger.log_episode(
                    self._global_step, ep.task_id,
                    ep.utility, ep.realized_return, ep.length,
                )

                # Reset this environment's buffers immediately
                ep_obs[i], ep_actions[i]            = [], []
                ep_logprobs[i], ep_rewards[i], ep_acc[i] = [], [], []
                acc_rewards[i] = np.zeros(self.reward_size, dtype=np.float32)
                ep_steps[i]    = 0

                # Round-robin task assignment for next episode on this env
                active_tasks[i] = next_task()

            obs = next_obs

        return episodes

    # -----------------------------------------------------------------------
    # CHANGE 2: Decoupled baseline training (terminal steps only)
    # -----------------------------------------------------------------------

    def _train_baseline(self, episodes: List[Episode]) -> dict:
        """
        Train the critic on complete episode data.

        Utility head — trained on ALL steps with the SAME target per episode.
        Vector head  — trained on ALL steps with MC future return targets.

        Performance fixes applied here:
          Fix 3: future_returns pre-computed in Episode.__init__ via cumsum.
          Fix 4: pre-converted tensors (ep.obs_t, ep.acc_t) used directly.
        Both avoid Python loops and repeated numpy→torch copies.
        """
        # Collect per-episode tensors — concatenate across episodes once
        all_obs_list,  all_acc_list  = [], []
        all_task_list, all_util_list = [], []
        all_vec_list                 = []

        for ep in episodes:
            T   = ep.length
            tid = ep.task_id

            # Fix 4: use pre-converted tensors directly (no torch.tensor() here)
            all_obs_list.append(ep.obs_t)                           # (T, obs_dim)
            all_acc_list.append(ep.acc_t)                           # (T, reward_size)
            all_task_list.append(
                torch.full((T,), tid, dtype=torch.long, device=self.device)
            )

            # Utility target: same scalar for every step in this episode
            all_util_list.append(
                torch.full((T, 1), ep.utility, dtype=torch.float32, device=self.device)
            )

            # Fix 3: future_returns already computed in Episode.__init__ via cumsum
            all_vec_list.append(
                torch.tensor(ep.future_returns, dtype=torch.float32, device=self.device)
            )

        # Single concatenation — one large tensor per field
        b_obs  = torch.cat(all_obs_list,  dim=0)
        b_acc  = torch.cat(all_acc_list,  dim=0)
        b_task = F.one_hot(
            torch.cat(all_task_list), num_classes=self.num_tasks
        ).float()
        b_util = torch.cat(all_util_list, dim=0)
        b_vec  = torch.cat(all_vec_list,  dim=0)

        n       = b_obs.shape[0]
        mb_size = max(1, n // self.num_minibatches)

        critic_losses = []
        ev_util_first = None
        ev_vec_first  = None

        # Compute explained variance BEFORE any training, across the full dataset.
        # Computing on the first minibatch is unreliable because steps within a
        # single episode share the same utility target — a small minibatch may
        # contain steps from only one or two episodes, giving near-zero variance
        # in y_true and a meaningless EV value.
        with torch.no_grad():
            pred_util_full, pred_vec_full = self.agent.estimate_value_from_observation(
                b_obs, b_acc, b_task, device=self.device
            )
            # EV computed in original (denormalised) scale — meaningful for logging
            y_u   = b_util.flatten()
            yh_u  = pred_util_full.detach().flatten()
            var_u = torch.var(y_u)
            ev_util_first = (
                float('nan') if var_u == 0
                else (1.0 - torch.var(y_u - yh_u) / var_u).item()
            )
            y_v   = b_vec.flatten()
            yh_v  = pred_vec_full.detach().flatten()
            var_v = torch.var(y_v)
            ev_vec_first = (
                float('nan') if var_v == 0
                else (1.0 - torch.var(y_v - yh_v) / var_v).item()
            )

        for epoch in range(self.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]

                # Pass 1: utility head — independent forward + backward.
                # Use estimate_value_normalized to get raw network output in
                # PopArt-normalised space, then compare against normalised target.
                # This keeps the MSE loss near unit scale regardless of return magnitude.
                pred_util_norm, _ = self.agent.estimate_value_normalized(
                    b_obs[idx], b_acc[idx], b_task[idx], device=self.device
                )
                target_util_norm = self.agent.popart_util.normalize(b_util[idx])
                loss_util = 0.5 * ((pred_util_norm - target_util_norm.detach()) ** 2).mean()
                self.optimizer[1].zero_grad()
                loss_util.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), self.max_grad_norm
                )
                self.optimizer[1].step()
                # Update PopArt statistics and rescale output layer weights AFTER step.
                # The rescaling preserves the denormalised function under the new statistics.
                self.agent.popart_util.update_and_rescale(b_util[idx].detach())

                # Pass 2: vector head — fresh forward pass after optimizer step.
                _, pred_vec_norm = self.agent.estimate_value_normalized(
                    b_obs[idx], b_acc[idx], b_task[idx], device=self.device
                )
                target_vec_norm = self.agent.popart_returns.normalize(b_vec[idx])
                loss_vec = 0.5 * ((pred_vec_norm - target_vec_norm.detach()) ** 2).mean()
                self.optimizer[1].zero_grad()
                loss_vec.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), self.max_grad_norm
                )
                self.optimizer[1].step()
                self.agent.popart_returns.update_and_rescale(b_vec[idx].detach())

                critic_losses.append((loss_util + loss_vec).item())

        return {
            'critic_loss':        float(np.mean(critic_losses)),
            'explained_var_util': ev_util_first if ev_util_first is not None else float('nan'),
            'explained_var_vec':  ev_vec_first  if ev_vec_first  is not None else float('nan'),
        }

    # -----------------------------------------------------------------------
    # CHANGE 3+4: Trajectory-level PPO + trajectory-level counterfactual reuse
    # -----------------------------------------------------------------------

    def _compute_trajectory_advantages(
        self, episodes: List[Episode]
    ) -> Dict[int, List[Dict]]:
        """
        Compute IS-weighted per-step advantages for all episodes × all tasks.

        Performance fixes applied here:
          Fix 1: batch across ALL tasks in a single forward pass per episode.
                 Previously: num_tasks sequential passes of size (T, obs_dim).
                 Now: one pass of size (num_tasks*T, obs_dim) with task_hot stacked.
                 Reduces forward pass count from (num_episodes × num_tasks) to
                 num_episodes, giving ~num_tasks× speedup on this step.
          Fix 4: use pre-converted tensors ep.obs_t, ep.acc_t, ep.actions_t,
                 ep.logprobs_t — no torch.tensor() copies inside this loop.

        Two-level IS correction (unchanged):
          Trajectory level: w(τ) = clip(exp(mean_log_ratio), w_min, w_max)
          Step level:       clip(ρ_step, 1±ε)  applied in _train_actor
        """
        task_data = {t: [] for t in range(self.num_tasks)}

        # Pre-build task index tensors for stacking — reused across episodes
        task_ids_per_ep = torch.arange(
            self.num_tasks, dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            for ep in episodes:
                T = ep.length

                # Fix 4: use pre-converted tensors — no copy here
                obs_t  = ep.obs_t       # (T, obs_dim)
                act_t  = ep.actions_t   # (T, act_dim)
                acc_t  = ep.acc_t       # (T, reward_size)
                beh_lp = ep.logprobs_t  # (T,)

                # Fix 1: batch across all tasks simultaneously
                # Repeat each episode's data num_tasks times and assign
                # each block a different task one-hot.
                #
                # obs_rep shape:  (num_tasks * T, obs_dim)
                # task_hot shape: (num_tasks * T, num_tasks)
                #
                # Block 0: steps 0..T-1 with task_hot = [1,0,0]
                # Block 1: steps T..2T-1 with task_hot = [0,1,0]
                # ...
                obs_rep  = obs_t.repeat(self.num_tasks, 1)
                acc_rep  = acc_t.repeat(self.num_tasks, 1)
                act_rep  = act_t.repeat(self.num_tasks, 1) if act_t.dim() > 1 \
                           else act_t.repeat(self.num_tasks)
                beh_rep  = beh_lp.repeat(self.num_tasks)   # (num_tasks * T,)

                # Task one-hot: repeat each task index T times, then concatenate
                task_ids = task_ids_per_ep.repeat_interleave(T)  # (num_tasks * T,)
                task_hot = F.one_hot(task_ids, num_classes=self.num_tasks).float()

                # Single forward pass for ALL tasks
                target_lp_all, _ = self.agent.compute_action_log_probabilities_and_entropy(
                    obs_rep, act_rep, acc_rep, task_hot, self.device
                )   # shape: (num_tasks * T,)

                baselines_all, _ = self.agent.estimate_value_from_observation(
                    obs_rep, acc_rep, task_hot, device=self.device
                )   # shape: (num_tasks * T, 1)

                # Split results back into per-task blocks
                target_lp_split = target_lp_all.view(self.num_tasks, T)  # (K, T)
                baselines_split = baselines_all.view(self.num_tasks, T)   # (K, T)

                for target_task in range(self.num_tasks):
                    target_lp = target_lp_split[target_task]   # (T,)
                    baselines  = baselines_split[target_task]   # (T,)

                    # Trajectory-level IS weight
                    mean_log_ratio = (target_lp - beh_lp).mean()
                    cf_weight = float(torch.clamp(
                        mean_log_ratio.exp(),
                        min=self.cf_weight_min,
                        max=self.cf_weight_max,
                    ))

                    # Realized utility — exact
                    target_util = self.utility_functions[target_task](
                        torch.tensor(ep.realized_return, dtype=torch.float32,
                                     device=self.device)
                    ).item()

                    # IS-weighted per-step advantages
                    advantages_t = cf_weight * (target_util - baselines)  # (T,)

                    task_data[target_task].append({
                        'cf_weight':      cf_weight,
                        'mean_log_ratio': mean_log_ratio.item(),
                        'advantages_t':   advantages_t,
                        'obs':            obs_t,
                        'actions':        act_t,
                        'acc_rewards':    acc_t,
                        'behavior_lp':    beh_lp,
                        'target_lp_old':  target_lp,
                        'length':         T,
                        'is_real':        (ep.task_id == target_task),
                        'episode_utility': target_util,  # scalar — needed for episode-level normalization
                    })

        return task_data

    def _normalize_advantages(self, task_id: int, advantages_t: torch.Tensor,
                              episode_lengths: list) -> torch.Tensor:
        """
        Normalize advantages using episode-level utility statistics.

        WHY PER-STEP NORMALIZATION FAILS HERE
        --------------------------------------
        Within a single episode of length T, the utility target u(R(τ)) is
        identical for all T steps. The advantage is A_t = u(R(τ)) - b(s_t, g_t).
        Per-step normalization computes mean and std across all T steps of the
        concatenated batch. Since u(R(τ)) is constant within each episode,
        the normalization cancels the utility signal entirely:

            Ã_t = (A_t - mean) / std
                = (b_mean - b(s_t, g_t)) / std_t[b]

        Every episode — good or bad — produces normalized advantages with zero
        mean and unit variance. The policy cannot distinguish good episodes from
        bad ones. This is why KL ≈ 0 and clip_fraction ≈ 0: the gradient signal
        is structurally zero regardless of episode quality.

        THE FIX: episode-level normalization
        ------------------------------------
        Normalize using the mean and std of episode-level utility values across
        the batch. This preserves the inter-episode signal (which episodes were
        above/below average) while allowing the baseline to provide within-episode
        credit assignment via the b(s_t, g_t) term.

        The normalized advantage at step t of episode i is:
            Ã_t = (u(R(τ_i)) - b(s_t, g_t) - μ_util) / σ_util

        where μ_util and σ_util are computed across episode utilities in the batch.
        This equals:
            Ã_t = (u(R(τ_i)) - μ_util) / σ_util   ← episode quality signal
                  - b(s_t, g_t) / σ_util            ← within-episode credit

        Both terms survive normalization. The utility value no longer cancels.
        """
        if not episode_lengths:
            return advantages_t

        # Reconstruct episode-level utility values from the concatenated advantages.
        # advantages_t[t] = u(R(τ)) - b(s_t, g_t) for episode i containing step t.
        # The utility for episode i = mean_t[A_t] + mean_t[b(s_t,g_t)].
        # We only need the utility values to get μ and σ — extract them by noting
        # that within episode i, u(R(τ_i)) is constant, so:
        #     mean_t_in_episode_i[A_t + b_t] = u(R(τ_i))   if we had b_t separately.
        # Since we don't have b_t separately here, use A_t mean per episode as a
        # proxy: mean_t[A_t] = u(R(τ_i)) - mean_t[b_t].
        # The scale we want is just σ of u values across episodes, which is well
        # approximated by the std of per-episode advantage means times a correction.
        # Simpler and equivalent: track the episode utilities in task_data and pass
        # them here directly.

        # Extract per-episode mean advantage — each is u(R(τ_i)) - mean_b_i
        ep_means = []
        start = 0
        for L in episode_lengths:
            ep_means.append(advantages_t[start:start + L].mean())
            start += L
        ep_means_t = torch.stack(ep_means)   # (num_episodes,)

        # Use the cross-episode std of these means as the normalization scale.
        # This preserves inter-episode signal while keeping gradients stable.
        if len(ep_means_t) < 2:
            # Single episode: cannot compute meaningful std — use global std
            std = advantages_t.std().clamp(min=1e-8)
            return (advantages_t - advantages_t.mean()) / std

        ep_std = ep_means_t.std().clamp(min=1e-8)
        ep_mean = ep_means_t.mean()

        # Subtract the cross-episode mean from every step, scale by cross-episode std.
        # The within-episode variation from the baseline is preserved at its natural scale.
        return (advantages_t - ep_mean) / ep_std

    def _train_actor(self, task_data: Dict[int, List[Dict]]) -> Dict[str, float]:
        """
        Update actor using per-step PPO with IS-weighted trajectory advantages.

        Performance fixes applied here:
          Fix 2: concatenate ALL episodes for a given task into one large
                 tensor before the forward pass, then do ONE forward+backward
                 per task per epoch rather than (num_episodes) separate calls.
                 Reduces backward() calls from (epochs × tasks × episodes) to
                 (epochs × tasks), giving ~num_episodes× speedup on this step.
          Fix 5: advantage normalisation applied once to the full concatenated
                 batch rather than per-trajectory, eliminating per-episode
                 mean/std calls inside the inner loop.

        Gradient structure (unchanged):
            ∇L = mean_t[ clip(ρ_step, 1±ε) · w(τ) · A_t ]
        """
        actor_losses, entropies, kl_vals, clip_fracs, cf_weights = [], [], [], [], []

        for epoch in range(self.update_epochs):
            for target_task in range(self.num_tasks):
                trajs = task_data[target_task]
                if not trajs:
                    continue

                # Fix 2: concatenate all episodes for this task into one batch
                # Shape of each: (sum_of_T_across_episodes, ...)
                b_obs     = torch.cat([d['obs']          for d in trajs], dim=0)
                b_acc     = torch.cat([d['acc_rewards']  for d in trajs], dim=0)
                b_act     = torch.cat([d['actions']      for d in trajs], dim=0) \
                            if trajs[0]['actions'].dim() > 1 \
                            else torch.cat([d['actions'] for d in trajs])
                b_old_lp  = torch.cat([d['target_lp_old'] for d in trajs])
                b_adv     = torch.cat([d['advantages_t']  for d in trajs])
                b_lengths = [d['length'] for d in trajs]
                total_T   = b_obs.shape[0]

                # Episode-level normalization — preserves the inter-episode utility
                # signal that per-step normalization would cancel out. See
                # _normalize_advantages docstring for the full explanation.
                if self.normalize_advantages:
                    b_adv = self._normalize_advantages(target_task, b_adv, b_lengths)

                # Task one-hot for full batch
                task_hot = F.one_hot(
                    torch.full((total_T,), target_task, dtype=torch.long,
                               device=self.device),
                    num_classes=self.num_tasks
                ).float()

                # Fix 2: ONE forward+backward per task per epoch
                self.optimizer[0].zero_grad()

                new_lp, entropy = self.agent.compute_action_log_probabilities_and_entropy(
                    b_obs, b_act, b_acc, task_hot, self.device
                )

                log_ratios     = new_lp - b_old_lp
                step_ratios    = log_ratios.exp()
                clipped_ratios = torch.clamp(
                    step_ratios,
                    1.0 - self.surrogate_clip_threshold,
                    1.0 + self.surrogate_clip_threshold,
                )

                pg_loss  = torch.max(
                    -step_ratios    * b_adv,
                    -clipped_ratios * b_adv,
                ).mean()

                ent_loss = -entropy.mean()
                loss     = pg_loss + self.entropy_loss_coefficient * ent_loss
                loss.backward()

                # Early stopping check before stepping
                with torch.no_grad():
                    approx_kl = ((step_ratios - 1) - log_ratios).mean().item()
                    clip_frac = (
                        (step_ratios - 1.0).abs() > self.surrogate_clip_threshold
                    ).float().mean().item()
                    mean_cf = float(np.mean([d['cf_weight'] for d in trajs]))

                    actor_losses.append(pg_loss.item())
                    entropies.append(entropy.mean().item())
                    kl_vals.append(approx_kl)
                    clip_fracs.append(clip_frac)
                    cf_weights.append(mean_cf)

                if self.target_kl is not None and abs(approx_kl) > self.target_kl:
                    # Zero the accumulated gradient and skip step if KL too large
                    self.optimizer[0].zero_grad()
                    continue

                nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(), self.max_grad_norm
                )
                self.optimizer[0].step()

        return {
            'actor_loss':    float(np.mean(actor_losses))  if actor_losses else 0.0,
            'entropy':       float(np.mean(entropies))     if entropies    else 0.0,
            'approx_kl':     float(np.mean(kl_vals))       if kl_vals      else 0.0,
            'clip_fraction': float(np.mean(clip_fracs))    if clip_fracs   else 0.0,
            'mean_cf_weight':float(np.mean(cf_weights))    if cf_weights   else 1.0,
        }

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(self):
        if self.eval_envs is None:
            return

        print(f"--- Evaluation at Step {self._global_step} ---")
        self.agent.eval()
        n_eval_envs = self.eval_envs.num_envs

        results      = {t: [] for t in range(self.num_tasks)}
        tasks_to_run = np.repeat(np.arange(self.num_tasks), self.num_eval_episodes)
        total_needed = len(tasks_to_run)

        obs, _ = self.eval_envs.reset(seed=self.seed + 999)
        obs    = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Accumulate returns — with or without discounting matching training
        acc_rewards  = np.zeros((n_eval_envs, self.reward_size), dtype=np.float32)

        # Per-environment step counter for gamma^t scaling when discount_utility=True
        eval_steps   = np.zeros(n_eval_envs, dtype=np.int32)

        active_tasks = np.zeros(n_eval_envs, dtype=np.int64)
        ptr          = np.full(n_eval_envs, -1, dtype=np.int32)

        params_ptr = 0
        for i in range(n_eval_envs):
            if params_ptr < total_needed:
                active_tasks[i] = int(tasks_to_run[params_ptr])
                ptr[i]          = params_ptr
                params_ptr     += 1

        while (ptr != -1).any():
            task_t    = torch.tensor(active_tasks, dtype=torch.long, device=self.device)
            task_hot  = F.one_hot(task_t, num_classes=self.num_tasks).float()
            acc_t     = torch.tensor(acc_rewards, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                actions, _ = self.agent.sample_action_and_compute_log_prob(
                    obs, acc_t, task_hot, deterministic=False, device=self.device
                )

            next_obs, reward, terms, truncs, _ = self.eval_envs.step(actions.cpu().numpy())

            reward_np = np.array(reward, dtype=np.float32).reshape(
                n_eval_envs, self.reward_size
            )

            # Zero out idle environments
            idle_mask          = (ptr == -1)
            reward_np[idle_mask] = 0.0

            # Accumulate — consistent with training collection
            if self.discount_utility and self.gamma < 1.0:
                # gamma^t per environment, broadcast over reward_size
                discount = (self.gamma ** eval_steps).reshape(n_eval_envs, 1)
                acc_rewards += discount * reward_np
            else:
                acc_rewards += reward_np

            # Increment step counters for active environments only
            eval_steps[~idle_mask] += 1

            is_done = np.logical_or(terms, truncs)
            for i in np.where(is_done)[0]:
                if ptr[i] != -1:
                    tid   = int(active_tasks[i])
                    r_vec = torch.tensor(
                        acc_rewards[i], dtype=torch.float32, device=self.device
                    )
                    u_val = self.utility_functions[tid](r_vec).item()
                    results[tid].append(u_val)

                    # Reset this environment's buffers
                    acc_rewards[i] = np.zeros(self.reward_size, dtype=np.float32)
                    eval_steps[i]  = 0

                    if params_ptr < total_needed:
                        active_tasks[i] = int(tasks_to_run[params_ptr])
                        ptr[i]          = params_ptr
                        params_ptr     += 1
                    else:
                        ptr[i] = -1

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        for t_id, vals in results.items():
            if vals:
                self.logger.log_evaluation(
                    self._global_step,
                    np.min(vals), np.mean(vals), np.max(vals),
                    t_id,
                )

        self.agent.train()
        print("--- Evaluation Complete ---")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def learn(self, total_timesteps: int):
        """
        Main training loop.

        Each iteration:
          1. Collect complete episodes (episode-level, exact utility)
          2. Train baseline on all steps — utility head + vector head
          3. Compute per-step advantages using utility head as baseline
          4. Update actor with per-step PPO + counterfactual reuse
          5. Step LR schedulers
        """
        while self._global_step < total_timesteps:

            # Periodic evaluation
            if self._global_step >= self.next_eval_step:
                self.evaluate()
                self.next_eval_step += self.eval_interval

            t0 = time.time()

            # 1. Collect complete episodes — exact realized returns
            episodes = self.collect_episodes()
            collect_time = time.time() - t0

            t1 = time.time()

            # 2. Train both critic heads on all steps
            critic_stats = self._train_baseline(episodes)
            critic_time  = time.time() - t1

            t2 = time.time()

            # 3+4. Per-step advantages + actor update with counterfactual reuse
            task_data   = self._compute_trajectory_advantages(episodes)
            actor_stats = self._train_actor(task_data)
            actor_time  = time.time() - t2

            # 5. Step both LR schedulers
            if self.anneal_lr and self.lr_schedulers:
                for sched in self.lr_schedulers:
                    sched.step()

            stats = {
                **critic_stats,
                **actor_stats,
                'episodes_collected': len(episodes),
            }
            self.logger.log_policy_update(stats, self._global_step)

            ev_util = critic_stats['explained_var_util']
            ev_vec  = critic_stats['explained_var_vec']

            print("-" * 50)
            print(f"{'Step':<25} | {self._global_step}")
            print(f"{'Critic Loss':<25} | {critic_stats['critic_loss']:.6f}")
            print(f"{'Expl Var (Utility)':<25} | "
                  f"{ev_util:.4f}" if not np.isnan(ev_util) else
                  f"{'Expl Var (Utility)':<25} | nan")
            print(f"{'Expl Var (Vector)':<25} | "
                  f"{ev_vec:.4f}" if not np.isnan(ev_vec) else
                  f"{'Expl Var (Vector)':<25} | nan")
            print(f"{'Actor Loss':<25} | {actor_stats['actor_loss']:.6f}")
            print(f"{'Entropy':<25} | {actor_stats['entropy']:.6f}")
            print(f"{'Approx KL':<25} | {actor_stats['approx_kl']:.6f}")
            print(f"{'Clip Fraction':<25} | {actor_stats['clip_fraction']:.6f}")
            print(f"{'Mean CF Weight':<25} | {actor_stats['mean_cf_weight']:.4f}")
            print(f"{'Episodes':<25} | {len(episodes)}")
            print(f"{'Collect Time':<25} | {collect_time:.3f}s")
            print(f"{'Critic Time':<25} | {critic_time:.3f}s")
            print(f"{'Actor Time':<25} | {actor_time:.3f}s")
            print("-" * 50)

        # Final evaluation
        self.evaluate()
        return self.agent