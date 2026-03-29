from uuid import uuid4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from torch.utils.tensorboard import SummaryWriter


class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False, reward_size=1):
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self.reward_size = reward_size

    def log_rollout_step(self, infos, global_step, active_task_id=0, active_utility_val=0.0):
        if "episode" in infos:
            if 'dr' in infos['episode']:
                non_zero_rews = infos['episode']['dr'][infos['_episode']]
            elif 'r' in infos['episode']:
                scalar_ret = infos['episode']['r'][infos['_episode']]
                non_zero_rews = scalar_ret.reshape(-1, 1).repeat(1, self.reward_size)
            else:
                return

            print(f"step={global_step}, task={active_task_id}, utility={active_utility_val:.3f}, vec={non_zero_rews.mean(axis=0)}", flush=True)

            if self.use_tensorboard:
                self.writer.add_scalar("charts/episodic_length", infos['episode']['l'][infos['_episode']].mean(), global_step)
                self.writer.add_scalar("charts/episodic_utility", active_utility_val, global_step)
                if isinstance(active_task_id, (int, float, np.integer, torch.Tensor)):
                    if isinstance(active_task_id, torch.Tensor):
                        active_task_id = active_task_id.item()
                    self.writer.add_scalar(f"charts/utility_task_{active_task_id}", active_utility_val, global_step)
                for i in range(self.reward_size):
                    self.writer.add_scalar(f"charts/vec_reward_obj_{i}", non_zero_rews[:, i].mean(), global_step)

    def log_policy_update(self, stats, global_step):
        if self.use_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(f"losses/{k}", v, global_step)

    def log_evaluation(self, global_step, min_util, mean_util, max_util, task_id):
        print(f"EVAL step={global_step} | Task={task_id} | Min={min_util:.3f} Mean={mean_util:.3f} Max={max_util:.3f}", flush=True)
        if self.use_tensorboard:
            self.writer.add_scalar(f"eval/utility_min_task_{task_id}", min_util, global_step)
            self.writer.add_scalar(f"eval/utility_mean_task_{task_id}", mean_util, global_step)
            self.writer.add_scalar(f"eval/utility_max_task_{task_id}", max_util, global_step)


class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        eval_envs=None,
        utility_functions=None,
        env_is_discrete=False,
        reward_size=1,
        learning_rate=3e-4,
        num_rollout_steps=2048,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.001,
        value_function_loss_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=32,
        normalize_advantages=True,
        reward_rms=None,
        clip_value_function_loss=True,
        target_kl=None,
        anneal_lr=True,
        seed=1,
        logger=None,
        convex=False,
        scalar_reward=False,
        pareto_archive=None,
        policy_gradient_loss_coefficient=1.0,
        eval_interval=10000,
        num_eval_episodes=10,
        total_timesteps=1000000,   # PATCH 1+5: required for LR scheduler
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.envs = envs
        self.eval_envs = eval_envs
        self.reward_size = reward_size
        self.env_is_discrete = env_is_discrete

        if utility_functions is None:
            self.utility_functions = [lambda r: r.sum(-1)]
        else:
            self.utility_functions = utility_functions
        self.num_tasks = len(self.utility_functions)

        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.batch_size = num_envs * num_rollout_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_clip_threshold = surrogate_clip_threshold
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.policy_gradient_loss_coefficient = policy_gradient_loss_coefficient
        self.value_function_loss_coefficient = value_function_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.normalize_advantages = normalize_advantages
        self.reward_rms = reward_rms
        self.clip_value_function_loss = clip_value_function_loss
        self.target_kl = target_kl

        self.device = next(agent.parameters()).device
        self.logger = logger or PPOLogger(reward_size=reward_size)
        self.seed = seed
        self._global_step = 0

        # PATCH 1: store total_timesteps so LR scheduler can reference it
        self.total_timesteps = total_timesteps
        self.scalar_reward = scalar_reward

        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        self.anneal_lr = anneal_lr
        if self.anneal_lr:
            num_updates = self.total_timesteps // self.num_envs // self.num_rollout_steps
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer[0], start_factor=1.0, end_factor=0.0, total_iters=num_updates
            )
        else:
            self.lr_scheduler = None

    def _get_one_hot_task(self, task_idx_tensor, batch_size):
        return F.one_hot(task_idx_tensor.long(), num_classes=self.num_tasks).float()

    def _get_single_task_one_hot(self, task_idx, batch_size):
        task_one_hot = torch.zeros((batch_size, self.num_tasks), device=self.device)
        if self.num_tasks > 0:
            task_one_hot[:, task_idx] = 1.0
        return task_one_hot

    def evaluate(self):
        if self.eval_envs is None:
            return

        print(f"--- Starting Evaluation at Step {self._global_step} ---")
        self.agent.eval()
        n_eval_envs = self.eval_envs.num_envs

        results = {t_id: [] for t_id in range(self.num_tasks)}

        tasks_to_run = np.repeat(np.arange(self.num_tasks), self.num_eval_episodes)
        total_episodes_needed = len(tasks_to_run)

        obs, _ = self.eval_envs.reset(seed=self.seed + 1000 + int(self._global_step))
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # PATCH 3: Accumulate with gamma=1.0 in eval.
        # Utility functions (e.g. O1 = (shaping + terminal) / 300) require
        # raw undiscounted cumulative returns. Discounting with gamma=0.995
        # over ~200 steps reduces terminal +100 to ~60, distorting utilities.
        acc_rewards = torch.zeros((n_eval_envs, self.reward_size), device=self.device)

        active_tasks = torch.zeros(n_eval_envs, dtype=torch.long, device=self.device)
        env_task_ptr = np.full(n_eval_envs, -1, dtype=np.int32)

        params_ptr = 0
        for i in range(n_eval_envs):
            if params_ptr < total_episodes_needed:
                active_tasks[i] = int(tasks_to_run[params_ptr])
                env_task_ptr[i] = params_ptr
                params_ptr += 1

        while (env_task_ptr != -1).any():
            task_one_hot = self._get_one_hot_task(active_tasks, n_eval_envs)

            with torch.no_grad():
                action, _ = self.agent.sample_action_and_compute_log_prob(
                    obs, acc_rewards, task_one_hot, deterministic=False, device=self.device
                )

            next_obs, reward, terminations, truncations, infos = self.eval_envs.step(action.cpu().numpy())

            reward_tens = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(n_eval_envs, self.reward_size)

            # Mask idle environments
            idle_mask = torch.tensor(env_task_ptr == -1, device=self.device)
            reward_tens[idle_mask] = 0.0

            # PATCH 3: no gamma discounting in eval accumulator
            acc_rewards += reward_tens

            is_done = torch.logical_or(
                torch.tensor(terminations), torch.tensor(truncations)
            ).to(self.device)

            if is_done.any():
                done_indices = torch.where(is_done)[0]
                for idx in done_indices:
                    if env_task_ptr[idx.item()] != -1:
                        task_id = active_tasks[idx].item()
                        final_vec = acc_rewards[idx]
                        u_val = self.utility_functions[task_id](final_vec).item()
                        results[task_id].append(u_val)

                        acc_rewards[idx] = 0.0

                        if params_ptr < total_episodes_needed:
                            active_tasks[idx] = int(tasks_to_run[params_ptr])
                            env_task_ptr[idx.item()] = params_ptr
                            params_ptr += 1
                        else:
                            env_task_ptr[idx.item()] = -1

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        for t_id, vals in results.items():
            if len(vals) > 0:
                self.logger.log_evaluation(
                    self._global_step, np.min(vals), np.mean(vals), np.max(vals), t_id
                )

        self.agent.train()
        print("--- Evaluation Complete ---")

    def learn(self, total_timesteps):
        num_updates = total_timesteps // self.batch_size

        obs, _ = self.envs.reset(seed=self.seed)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        acc_rewards = torch.zeros((self.num_envs, self.reward_size), device=self.device)
        acc_gamma = torch.ones((self.num_envs, 1), device=self.device)
        active_task_indices = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

        done = torch.zeros(self.num_envs, device=self.device)
        truncated = torch.zeros(self.num_envs, device=self.device)

        self.next_eval_step = 0

        for update in range(num_updates):
            if self.anneal_lr and self.lr_scheduler:
                self.lr_scheduler.step()

            storage = self.collect_rollouts(
                obs, acc_rewards, acc_gamma, done, truncated, active_task_indices
            )

            obs = storage['next_obs']
            acc_rewards = storage['next_acc_rewards']
            acc_gamma = storage['next_acc_gamma']
            done = storage['next_done']
            truncated = storage['next_truncated']
            active_task_indices = storage['next_task_indices']

            update_stats = self.update(storage)
            self.logger.log_policy_update(update_stats, self._global_step)

        self.evaluate()
        return self.agent

    def _initialize_storage(self):
        collected_observations = torch.zeros(
            (self.num_rollout_steps, self.num_envs) + self.envs.single_observation_space.shape,
            device=self.device
        )
        action_shape = () if self.env_is_discrete else self.envs.single_action_space.shape
        actions = torch.zeros(
            (self.num_rollout_steps, self.num_envs) + action_shape, device=self.device
        )
        action_log_probabilities = torch.zeros(
            (self.num_rollout_steps, self.num_envs), device=self.device
        )
        rewards = torch.zeros(
            (self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device
        )
        is_episode_terminated = torch.zeros(
            (self.num_rollout_steps, self.num_envs), device=self.device
        )
        is_episode_truncated = torch.zeros(
            (self.num_rollout_steps, self.num_envs), device=self.device
        )
        observation_vector_values = torch.zeros(
            (self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device
        )
        acc_rewards_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device
        )
        acc_gamma_storage = torch.ones(
            (self.num_rollout_steps, self.num_envs, 1), device=self.device
        )
        task_id_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs), dtype=torch.long, device=self.device
        )
        behavior_util_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs, 1), device=self.device
        )
        return (
            collected_observations, actions, action_log_probabilities, rewards,
            is_episode_terminated, is_episode_truncated, observation_vector_values,
            acc_rewards_storage, acc_gamma_storage, task_id_storage, behavior_util_storage
        )

    def collect_rollouts(self, obs, acc_rewards, acc_gamma, done, truncated, active_task_indices):
        (
            collected_observations, actions, action_log_probabilities, rewards,
            is_episode_terminated, is_episode_truncated, observation_vector_values,
            acc_rewards_storage, acc_gamma_storage, task_id_storage, behavior_util_storage
        ) = self._initialize_storage()

        active_task_indices = active_task_indices.to(self.device)

        for step in range(self.num_rollout_steps):

            if self._global_step >= self.next_eval_step:
                self.evaluate()
                self.next_eval_step += self.eval_interval

            collected_observations[step] = obs
            acc_rewards_storage[step] = acc_rewards
            acc_gamma_storage[step] = acc_gamma
            is_episode_terminated[step] = done
            is_episode_truncated[step] = truncated
            task_id_storage[step] = active_task_indices

            task_one_hot = self._get_one_hot_task(active_task_indices, self.num_envs)

            with torch.no_grad():
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    obs, acc_rewards, task_one_hot, deterministic=False, device=self.device
                )
                _, vec_value = self.agent.estimate_value_from_observation(
                    obs, acc_rewards, task_one_hot, device=self.device
                )
                behavior_util, _ = self.agent.estimate_value_from_observation(
                    obs, acc_rewards, task_one_hot, device=self.device
                )

                observation_vector_values[step] = vec_value
                behavior_util_storage[step] = behavior_util

            actions[step] = action
            action_log_probabilities[step] = logprob

            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            self._global_step += self.num_envs

            if self.reward_rms is not None:
                rewards_reshaped = reward.reshape(-1, self.reward_size)
                self.reward_rms.update(rewards_reshaped)
                reward = self.reward_rms.normalize(rewards_reshaped).reshape(
                    self.num_envs, self.reward_size
                )

            reward_tens = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(
                self.num_envs, self.reward_size
            )
            rewards[step] = reward_tens

            acc_rewards = acc_rewards + acc_gamma * reward_tens
            acc_gamma = acc_gamma * self.gamma

            done = torch.tensor(terminations, dtype=torch.float32, device=self.device)
            truncated = torch.tensor(truncations, dtype=torch.float32, device=self.device)
            is_done = torch.logical_or(done, truncated)

            if "episode" in infos:
                if 'dr' in infos['episode']:
                    final_vec_ret = infos['episode']['dr'][infos['_episode']]
                elif 'r' in infos['episode']:
                    scalar_ret = infos['episode']['r'][infos['_episode']]
                    final_vec_ret = scalar_ret.reshape(-1, 1).repeat(1, self.reward_size)
                else:
                    final_vec_ret = None

                if final_vec_ret is not None:
                    finished_env_indices = np.where(infos['_episode'])[0]
                    if len(finished_env_indices) > 0:
                        finished_tasks = active_task_indices[finished_env_indices].cpu().numpy()
                        finished_vecs = torch.tensor(
                            final_vec_ret, device=self.device, dtype=torch.float32
                        )
                        for j, env_idx in enumerate(finished_env_indices):
                            task_idx = int(finished_tasks[j])
                            util_val = self.utility_functions[task_idx](finished_vecs[j]).item()
                            self.logger.log_rollout_step(
                                infos, self._global_step,
                                active_task_id=task_idx,
                                active_utility_val=util_val
                            )

            if is_done.any():
                mask = (~is_done).unsqueeze(1)
                acc_rewards = acc_rewards * mask
                acc_gamma = acc_gamma * mask
                acc_gamma[~mask.squeeze(1)] = 1.0

                new_tasks = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)
                active_task_indices = torch.where(is_done, new_tasks, active_task_indices)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        task_one_hot = self._get_one_hot_task(active_task_indices, self.num_envs)
        with torch.no_grad():
            _, next_aux_val = self.agent.estimate_value_from_observation(
                obs, acc_rewards, task_one_hot, device=self.device
            )

        storage = {
            'obs': collected_observations,
            'actions': actions,
            'logprobs': action_log_probabilities,
            'rewards': rewards,
            'dones': is_episode_terminated,
            'truncated': is_episode_truncated,
            'values': observation_vector_values,
            'acc_rewards_in': acc_rewards_storage,
            'acc_gamma_in': acc_gamma_storage,
            'task_ids': task_id_storage,
            'behavior_util': behavior_util_storage,
            'next_aux_val': next_aux_val,
            'next_obs': obs,
            'next_acc_rewards': acc_rewards,
            'next_acc_gamma': acc_gamma,
            'next_done': done,
            'next_truncated': truncated,
            'next_task_indices': active_task_indices,
        }
        return storage

    def update(self, storage):
        vec_returns = self.compute_vector_returns(
            storage['rewards'], storage['dones'], storage['truncated'],
            storage['next_aux_val'], storage['next_done']
        )

        b_obs = self._flatten(storage['obs'])
        b_acc = self._flatten(storage['acc_rewards_in'])
        b_gamma = self._flatten(storage['acc_gamma_in'])
        b_actions = self._flatten(storage['actions'])
        b_vec_returns = self._flatten(vec_returns)
        b_task_ids = self._flatten(storage['task_ids'])

        b_projected_total_vec = b_acc + b_gamma * b_vec_returns
        util_sample = self.utility_functions[0](b_projected_total_vec)
        print(f"UTILITY TARGETS: mean={util_sample.mean():.2f} "
            f"std={util_sample.std():.2f} "
            f"min={util_sample.min():.2f} "
            f"max={util_sample.max():.2f}")
        print(f"PROJECTED VEC shaping: mean={b_projected_total_vec[:,0].mean():.2f} "
            f"min={b_projected_total_vec[:,0].min():.2f} "
            f"max={b_projected_total_vec[:,0].max():.2f}")
        print(f"PROJECTED VEC terminal: mean={b_projected_total_vec[:,2].mean():.2f} "
            f"min={b_projected_total_vec[:,2].min():.2f} "
            f"max={b_projected_total_vec[:,2].max():.2f}")

        # =====================================================
        # PRE-CALCULATE ADVANTAGES (Using OLD Critic)
        # =====================================================
        all_task_advantages = []
        all_task_old_logprobs = []

        precalc_start_time = time.time()

        with torch.no_grad():
            for task_i in range(self.num_tasks):
                task_hot = self._get_single_task_one_hot(task_i, self.batch_size)

                target_util = self.utility_functions[task_i](b_projected_total_vec)
                if target_util.ndim == 1:
                    target_util = target_util.unsqueeze(1)

                baseline, _ = self.agent.estimate_value_from_observation(
                    b_obs, b_acc, task_hot, device=self.device
                )

                adv = target_util - baseline
                if self.normalize_advantages:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                all_task_advantages.append(adv)

                old_lp, _ = self.agent.compute_action_log_probabilities_and_entropy(
                    b_obs, b_actions, b_acc, task_hot, self.device
                )
                all_task_old_logprobs.append(old_lp)

        precalc_time = time.time() - precalc_start_time

        # =====================================================
        # PHASE 1: CRITIC UPDATE
        # =====================================================
        critic_loss_hist = []
        explained_var_util = []
        explained_var_vec = []

        critic_start_time = time.time()

        inds = np.arange(self.batch_size)
        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]

                mb_obs = b_obs[mb_inds]
                mb_acc = b_acc[mb_inds]
                mb_projected_vec = b_projected_total_vec[mb_inds]
                mb_vec_ret = b_vec_returns[mb_inds]
                mb_task_ids = b_task_ids[mb_inds]

                mb_task_hot_behavior = self._get_one_hot_task(mb_task_ids, len(mb_inds))

                mb_target_utils = torch.zeros(len(mb_inds), 1, device=self.device)
                unique_tasks = torch.unique(mb_task_ids)

                for t_id in unique_tasks:
                    mask = (mb_task_ids == t_id)
                    task_vecs = mb_projected_vec[mask]
                    task_utils = self.utility_functions[t_id.item()](task_vecs)
                    if len(task_utils.shape) == 1:
                        task_utils = task_utils.unsqueeze(1)
                    mb_target_utils[mask] = task_utils

                pred_utility, pred_vec = self.agent.estimate_value_from_observation(
                    mb_obs, mb_acc, mb_task_hot_behavior, device=self.device
                )

                loss_util = 0.5 * ((pred_utility - mb_target_utils) ** 2).mean()
                loss_vec = 0.5 * ((pred_vec - mb_vec_ret) ** 2).mean()
                total_critic_loss = loss_util + loss_vec

                if epoch == 0 and start == 0:
                    y_true = mb_target_utils.flatten()
                    y_pred = pred_utility.flatten()
                    var_y = torch.var(y_true)
                    ev_u = float('nan') if var_y == 0 else (
                        1 - torch.var(y_true - y_pred) / var_y
                    ).item()
                    explained_var_util.append(ev_u)

                    y_true_v = mb_vec_ret.flatten()
                    y_pred_v = pred_vec.flatten()
                    var_y_v = torch.var(y_true_v)
                    ev_v = float('nan') if var_y_v == 0 else (
                        1 - torch.var(y_true_v - y_pred_v) / var_y_v
                    ).item()
                    explained_var_vec.append(ev_v)

                self.optimizer[1].zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.optimizer[1].step()

                critic_loss_hist.append(total_critic_loss.item())

        critic_time = time.time() - critic_start_time

        # =====================================================
        # PHASE 2: ACTOR UPDATE (Vectorized Counterfactual)
        # =====================================================
        actor_start_time = time.time()

        actor_loss_hist = []
        entropy_hist = []
        kl_hist = []
        clip_frac_hist = []

        b_behavior_lp = self._flatten(storage['logprobs'])

        total_real_samples = self.batch_size
        inds = np.arange(total_real_samples)

        is_ratio_threshold = np.log(0.01)

        for _ in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, total_real_samples, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]

                mb_obs = b_obs[mb_inds]
                mb_acc = b_acc[mb_inds]
                mb_act = b_actions[mb_inds]
                mb_behavior_lp = b_behavior_lp[mb_inds]
                mb_behavior_id = b_task_ids[mb_inds]

                self.optimizer[0].zero_grad()

                processed_tasks = 0

                for t_idx in range(self.num_tasks):
                    mb_old_lp_target = all_task_old_logprobs[t_idx][mb_inds]
                    mb_adv_target = all_task_advantages[t_idx][mb_inds]

                    log_is_weight = mb_old_lp_target - mb_behavior_lp
                    is_plausible = (log_is_weight > is_ratio_threshold)
                    is_real = (mb_behavior_id == t_idx)
                    keep_mask = (is_real | is_plausible)

                    if not keep_mask.any():
                        continue

                    processed_tasks += 1

                    sub_obs = mb_obs[keep_mask]
                    sub_acc = mb_acc[keep_mask]
                    sub_act = mb_act[keep_mask]
                    sub_old_lp = mb_old_lp_target[keep_mask]
                    sub_adv = mb_adv_target[keep_mask]

                    sub_task_hot = self._get_single_task_one_hot(t_idx, sub_obs.shape[0])

                    new_lp, entropy = self.agent.compute_action_log_probabilities_and_entropy(
                        sub_obs, sub_act, sub_acc, sub_task_hot, self.device
                    )

                    logratio = new_lp - sub_old_lp
                    logratio = torch.clamp(logratio, max=20.0)
                    ratio = logratio.exp()

                    pg_loss1 = -sub_adv * ratio
                    pg_loss2 = -sub_adv * torch.clamp(
                        ratio,
                        1.0 - self.surrogate_clip_threshold,
                        1.0 + self.surrogate_clip_threshold
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    ent_loss = -entropy.mean()

                    loss = pg_loss + self.entropy_loss_coefficient * ent_loss
                    loss.backward()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = (
                            (ratio - 1.0).abs() > self.surrogate_clip_threshold
                        ).float().mean()

                        actor_loss_hist.append(pg_loss.item())
                        entropy_hist.append(entropy.mean().item())
                        kl_hist.append(approx_kl.item())
                        clip_frac_hist.append(clip_frac.item())

                if processed_tasks > 0:
                    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                    self.optimizer[0].step()

        actor_time = time.time() - actor_start_time

        metrics = {
            "actor_loss": np.mean(actor_loss_hist),
            "critic_loss": np.mean(critic_loss_hist),
            "entropy": np.mean(entropy_hist),
            "approx_kl": np.mean(kl_hist),
            "clip_fraction": np.mean(clip_frac_hist),
            "explained_var_utility": np.mean(explained_var_util) if explained_var_util else 0,
            "explained_var_vector": np.mean(explained_var_vec) if explained_var_vec else 0,
        }

        print("-" * 45)
        print(f"{'Metric':<25} | {'Value':<15}")
        print("-" * 45)
        for k, v in metrics.items():
            print(f"{k:<25} | {v: .6f}")
        print(f"{'Actor Time':<25} | {actor_time: .6f}")
        print(f"{'Critic Time':<25} | {critic_time: .6f}")
        print("-" * 45)

        return metrics

    def _flatten(self, tensor):
        return tensor.reshape((-1,) + tensor.shape[2:])

    def compute_vector_returns(self, rewards, dones, truncated, next_value, next_done):
        """
        PATCH 2: Correct truncation handling.
        - Terminated episodes (done=1, truncated=0): do NOT bootstrap
        - Truncated episodes (done=0, truncated=1): SHOULD bootstrap
          (episode cut by time limit, not a true terminal state)
        - next_done here is the termination flag for the state AFTER the rollout
        """
        T = self.num_rollout_steps
        returns = torch.zeros_like(rewards, device=self.device)

        # Bootstrap from next state unless truly terminated
        next_val = next_value * (1.0 - next_done).unsqueeze(1)

        for t in reversed(range(T)):
            # real_done: true terminal — mask out future returns
            real_done = dones[t] * (1.0 - truncated[t])
            cont_mask = (1.0 - real_done).unsqueeze(1)
            returns[t] = rewards[t] + self.gamma * cont_mask * next_val
            next_val = returns[t]

        return returns

    def calculate_policy_gradient_loss(self, minibatch_advantages, probability_ratio):
        unclipped = minibatch_advantages * probability_ratio
        clipped = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )
        return -torch.min(unclipped, clipped)

    def calculate_critic_loss(self, new_util_val, util_tgt, new_vec_val, vec_ret):
        util_loss = 0.5 * ((new_util_val - util_tgt) ** 2).mean()
        vec_loss = 0.5 * ((new_vec_val - vec_ret) ** 2).mean()
        return util_loss, vec_loss