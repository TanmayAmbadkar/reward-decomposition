from uuid import uuid4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class LinearLRSchedule:
    def __init__(self, optimizer, initial_lr, total_updates):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_updates = total_updates
        self.current_update = 0

    def step(self):
        self.current_update += 1
        frac = 1.0 - (self.current_update - 1.0) / self.total_updates
        lr = frac * self.initial_lr
        # Handle list of optimizers
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False, reward_size=1):
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self.reward_size = reward_size

    def log_rollout_step(self, infos, global_step, active_task_id=0, active_utility_val=0.0):
        if "episode" in infos:
            # infos['episode']['dr'] contains the undiscounted vector return
            non_zero_rews = infos['episode']['dr'][infos['_episode']] 
            
            # Print info
            print(f"step={global_step}, task={active_task_id}, utility={active_utility_val:.3f}, vec={non_zero_rews.mean(axis=0)}", flush=True)

            if self.use_tensorboard:
                self.writer.add_scalar("charts/episodic_length", infos['episode']['l'][infos['_episode']].mean(), global_step)
                self.writer.add_scalar("charts/episodic_utility", active_utility_val, global_step)
                
                # Log task-specific utility
                if isinstance(active_task_id, (int, float, np.integer, torch.Tensor)):
                     if isinstance(active_task_id, torch.Tensor):
                         active_task_id = active_task_id.item()
                     self.writer.add_scalar(f"charts/utility_task_{active_task_id}", active_utility_val, global_step)
                
                # Log raw physics (Vector Rewards)
                for i in range(self.reward_size):
                    self.writer.add_scalar(f"charts/vec_reward_obj_{i}", non_zero_rews[:, i].mean(), global_step)

    def log_policy_update(self, stats, global_step):
        if self.use_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(f"losses/{k}", v, global_step)

class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        utility_functions=None,  # List of functions: f(vector_return) -> scalar_utility
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
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.envs = envs
        self.reward_size = reward_size
        self.env_is_discrete = env_is_discrete
        
        # --- Multi-Task / Non-Linear Utility Setup ---
        if utility_functions is None:
            # Default to simply summing the vector rewards (Linear SER behavior)
            self.utility_functions = [lambda r: r.sum(-1)] 
        else:
            self.utility_functions = utility_functions
        self.num_tasks = len(self.utility_functions)
        # ---------------------------------------------

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
        self.anneal_lr = anneal_lr
        self.initial_lr = learning_rate
        self.lr_scheduler = None
        self.logger = logger or PPOLogger(reward_size=reward_size)
        self.seed = seed
        self._global_step = 0
        
        self.scalar_reward = scalar_reward 

    def create_lr_scheduler(self, num_updates):
        return LinearLRSchedule(self.optimizer, self.initial_lr, num_updates)

    def _get_one_hot_task(self, task_idx_tensor, batch_size):
        """
        Creates a one-hot tensor for the given task indices.
        task_idx_tensor: Tensor of shape (batch_size,) containing integer task indices.
        """
        task_one_hot = torch.zeros((batch_size, self.num_tasks), device=self.device)
        if self.num_tasks > 0:
            # task_idx_tensor should be long tensor on correct device
            # Ensure task_idx_tensor is long
            task_idx_tensor = task_idx_tensor.long()
            task_one_hot.scatter_(1, task_idx_tensor.unsqueeze(1), 1.0)
        return task_one_hot

    def _get_single_task_one_hot(self, task_idx, batch_size):
        """
        Helper for update loop where we want a batch of the SAME task index.
        """
        task_one_hot = torch.zeros((batch_size, self.num_tasks), device=self.device)
        if self.num_tasks > 0:
            task_one_hot[:, task_idx] = 1.0
        return task_one_hot

    def learn(self, total_timesteps):
        num_updates = total_timesteps // (self.num_rollout_steps * self.num_envs)
        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_updates)

        # Init Env
        obs, _ = self.envs.reset(seed=self.seed)
        obs = torch.Tensor(obs).to(self.device)
        
        # Init State Augmentation: Accumulated Rewards start at 0
        acc_rewards = torch.zeros((self.num_envs, self.reward_size), device=self.device)
        
        # Init Tasks: Randomly assign a task to each environment
        # Shape: (num_envs,)
        active_task_indices = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)
        
        done = torch.zeros(self.num_envs, device=self.device)
        truncated = torch.zeros(self.num_envs, device=self.device)

        for update in range(num_updates):
            if self.anneal_lr: self.lr_scheduler.step()

            # --- Rollout Collection ---
            # Collect data. active_task_indices is updated inside if envs finish.
            storage = self.collect_rollouts(obs, acc_rewards, done, truncated, active_task_indices)
            
            # Update state variables for next iteration
            obs = storage['next_obs']
            acc_rewards = storage['next_acc_rewards']
            done = storage['next_done']
            truncated = storage['next_truncated']
            active_task_indices = storage['next_task_indices'] # Carry over task IDs

            # --- Multi-Task Update ---
            # We use the collected data to update ALL tasks using Importance Sampling
            update_stats = self.update(storage)
            
            self.logger.log_policy_update(update_stats, self._global_step)

        return self.agent

    def _initialize_storage(self):
        collected_observations = torch.zeros((self.num_rollout_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        
        action_shape = () if self.env_is_discrete else self.envs.single_action_space.shape
        actions = torch.zeros((self.num_rollout_steps, self.num_envs) + action_shape).to(self.device)
        
        action_log_probabilities = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(self.device)
        is_episode_terminated = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        is_episode_truncated = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        
        # Store Vector Values (from Aux Head) for GAE calculation
        observation_vector_values = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(self.device)
        
        # Store Accumulated Rewards (History) for Non-Markovian policy input
        acc_rewards_storage = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(self.device)
        
        # Store Task IDs for Importance Sampling (Behavior Policy ID)
        task_id_storage = torch.zeros((self.num_rollout_steps, self.num_envs), dtype=torch.long).to(self.device)

        return (
            collected_observations, actions, action_log_probabilities, rewards, 
            is_episode_terminated, is_episode_truncated, observation_vector_values, acc_rewards_storage,
            task_id_storage
        )

    def collect_rollouts(self, obs, acc_rewards, done, truncated, active_task_indices):
        """
        Collects data. Handles per-env task sampling on episode end.
        """
        (
            collected_observations, actions, action_log_probabilities, rewards, 
            is_episode_terminated, is_episode_truncated, observation_vector_values, acc_rewards_storage,
            task_id_storage
        ) = self._initialize_storage()
        
        # Ensure indices are on device
        active_task_indices = active_task_indices.to(self.device)

        for step in range(self.num_rollout_steps):
            collected_observations[step] = obs
            acc_rewards_storage[step] = acc_rewards
            is_episode_terminated[step] = done
            is_episode_truncated[step] = truncated
            task_id_storage[step] = active_task_indices # Store which task generated this step

            # Create One-Hot batch for current set of tasks
            task_one_hot = self._get_one_hot_task(active_task_indices, self.num_envs)

            with torch.no_grad():
                # Sample Action
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    obs, acc_rewards, task_one_hot, deterministic=False, device=self.device
                )
                # Get Aux Value (Vector Returns) for GAE
                _, vec_value = self.agent.estimate_value_from_observation(
                    obs, acc_rewards, task_one_hot, device=self.device
                )
                observation_vector_values[step] = vec_value

            actions[step] = action
            action_log_probabilities[step] = logprob

            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            self._global_step += self.num_envs
            
            reward_tens = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(self.num_envs, self.reward_size)
            rewards[step] = reward_tens
            
            # --- Non-Markovian History Update ---
            # Add current reward to history
            acc_rewards = acc_rewards + reward_tens
            
            done = torch.tensor(terminations, dtype=torch.float32, device=self.device)
            truncated = torch.tensor(truncations, dtype=torch.float32, device=self.device)
            is_done = torch.logical_or(done, truncated)
            
            # Log info and Resample Task if episode finished
            if "episode" in infos:
                # Calculate the utility of the COMPLETED episode for logging
                final_vec_ret = infos['episode']['dr'][infos['_episode']] # (num_dones, rew_size)
                
                # Get task IDs for environments that actually finished
                finished_env_indices = np.where(infos['_episode'])[0]
                
                if len(finished_env_indices) > 0:
                    finished_tasks = active_task_indices[finished_env_indices]
                    
                    # Log for the first finished environment to avoid spamming scalars
                    first_idx = finished_env_indices[0]
                    first_task = active_task_indices[first_idx].item()
                    first_vec = final_vec_ret[0] 
                    
                    # Calculate scalar utility for that specific task
                    first_vec_tensor = torch.tensor(first_vec, device=self.device)
                    util_val = self.utility_functions[first_task](first_vec_tensor).item()
                    
                    self.logger.log_rollout_step(infos, self._global_step, first_task, util_val)

            # If episode done:
            # 1. Reset Accumulated Reward History to 0 for the NEXT step.
            # 2. Sample NEW Task ID for that specific environment
            if is_done.any():
                acc_rewards = acc_rewards * (~is_done.unsqueeze(1))
                
                # Resample tasks for done environments
                new_tasks = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)
                active_task_indices = torch.where(is_done, new_tasks, active_task_indices)
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        # Estimate next value for GAE (Auxiliary Head)
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
            'task_ids': task_id_storage, # Stores the task ID active at each step
            'next_aux_val': next_aux_val,
            'next_obs': obs,
            'next_acc_rewards': acc_rewards,
            'next_done': done,
            'next_truncated': truncated,
            'next_task_indices': active_task_indices
        }
        return storage

    def update(self, storage):
        """
        Updates ALL tasks using the collected rollout.
        Uses Importance Sampling for off-policy tasks.
        """
        # 1. Calculate Vector Advantages/Returns (GAE) for the Auxiliary Head
        # This is task-agnostic (Environment Physics)
        vec_advantages, vec_returns = self.compute_advantages(
            storage['rewards'],
            storage['values'],
            storage['dones'],
            storage['truncated'],
            storage['next_aux_val'],
            storage['next_done'],
            storage['next_truncated']
        )
        
        # Flatten common data
        b_obs = self._flatten(storage['obs'])
        b_acc = self._flatten(storage['acc_rewards_in'])
        b_actions = self._flatten(storage['actions'])
        b_behavior_logp = self._flatten(storage['logprobs'])
        b_behavior_task_ids = self._flatten(storage['task_ids']) 
        b_vec_returns = self._flatten(vec_returns)
        
        # Stats aggregators
        total_pol_loss = 0
        total_val_loss = 0
        total_ent = 0
        
        # --- Multi-Task Loop ---
        # We iterate over every task we want to learn (0..N)
        for task_i in range(self.num_tasks):
            # Create TARGET Task ID batch (All same task)
            b_target_task_one_hot = self._get_single_task_one_hot(task_i, b_obs.shape[0])
            
            # Calculate Targets for Task i
            with torch.no_grad():
                # A. Apply Task i's Utility Function to the TOTAL Return (History + Future)
                # Correct logic: V(s, g_acc) = u(g_acc + G_future)
                total_episode_return = b_acc + b_vec_returns
                b_utility_targets = self.utility_functions[task_i](total_episode_return)
                
                if len(b_utility_targets.shape) == 1: 
                    b_utility_targets = b_utility_targets.unsqueeze(1)
                
                # B. Get current Value estimate (Main Head) for Task i
                curr_util_val, _ = self.agent.estimate_value_from_observation(b_obs, b_acc, b_target_task_one_hot, self.device)
                
                # C. Calculate Scalar Advantage
                b_advantages = b_utility_targets - curr_util_val
                
                if self.normalize_advantages:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # PPO Mini-batch Update for Task i
            inds = np.arange(self.batch_size)
            for _ in range(self.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = inds[start:end]
                    
                    # Slices
                    mb_obs = b_obs[mb_inds]
                    mb_acc = b_acc[mb_inds]
                    mb_task = b_target_task_one_hot[mb_inds] # The task we are TRAINING
                    mb_act = b_actions[mb_inds]
                    mb_adv = b_advantages[mb_inds]
                    mb_util_tgt = b_utility_targets[mb_inds]
                    mb_vec_ret = b_vec_returns[mb_inds]
                    mb_old_logp = b_behavior_logp[mb_inds] 
                    
                    # Forward Pass (Current Policy for Task i)
                    new_logp, entropy = self.agent.compute_action_log_probabilities_and_entropy(
                        mb_obs, mb_act, mb_acc, mb_task, self.device
                    )
                    new_util_val, new_vec_val = self.agent.estimate_value_from_observation(
                        mb_obs, mb_acc, mb_task, self.device
                    )
                    
                    # Importance Sampling Ratio
                    logratio = new_logp - mb_old_logp
                    ratio = logratio.exp()
                    
                    # Compute Losses
                    pg_loss = self.calculate_policy_gradient_loss(mb_adv.squeeze(), ratio)
                    util_loss, vec_loss = self.calculate_critic_loss(new_util_val, mb_util_tgt, new_vec_val, mb_vec_ret)
                    
                    loss = self.policy_gradient_loss_coefficient * pg_loss - \
                           self.entropy_loss_coefficient * entropy.mean() + \
                           self.value_function_loss_coefficient * (util_loss + vec_loss)
                    
                    # Optimization Step
                    # Handle multiple optimizers (Actor, Critic)
                    if isinstance(self.optimizer, list):
                        for opt in self.optimizer:
                            opt.zero_grad()
                    else:
                        self.optimizer.zero_grad()
                        
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    
                    if isinstance(self.optimizer, list):
                        for opt in self.optimizer:
                            opt.step()
                    else:
                        self.optimizer.step()
                    
                    total_pol_loss += pg_loss.item()
                    total_val_loss += (util_loss + vec_loss).item()
                    total_ent += entropy.mean().item()

        num_updates = self.num_tasks * self.update_epochs * self.num_minibatches
        return {
            "policy_loss": total_pol_loss / num_updates,
            "value_loss": total_val_loss / num_updates,
            "entropy": total_ent / num_updates
        }

    def _flatten(self, tensor):
        return tensor.reshape((-1,) + tensor.shape[2:])

    def compute_advantages(
        self, rewards, values, is_observation_terminal, is_observation_truncated,
        next_value, is_next_observation_terminal, is_next_observation_truncated
    ):
        """
        Standard GAE, but operates on Vector Rewards/Values.
        Returns: (Vector Advantages, Vector Returns)
        """
        T = self.num_rollout_steps
        advantages = torch.zeros_like(rewards).to(self.device)
        gae_running = torch.zeros_like(next_value).to(self.device)
        
        # Reward Normalization (Optional but recommended for Aux Head)
        if self.reward_rms is not None:
             rewards_reshaped = rewards.cpu().numpy().reshape(-1, self.reward_size)
             self.reward_rms.update(rewards_reshaped)
             rewards = torch.tensor(
                self.reward_rms.normalize(rewards.cpu().numpy()),
                dtype=torch.float32
             ).to(self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                cont = 1 - is_next_observation_terminal.unsqueeze(1)
                mask_trunc = is_next_observation_truncated.bool() & (~is_next_observation_terminal.bool())
                gae_running[mask_trunc] = 0
                bootstrap = next_value.clone()
                bootstrap[is_next_observation_terminal.bool()] = 0
            else:
                cont = 1 - is_observation_terminal[t + 1].unsqueeze(1)
                mask_trunc = is_observation_truncated[t + 1].bool() & (~is_observation_terminal[t + 1].bool())
                gae_running[mask_trunc] = 0
                bootstrap = values[t + 1].clone()
                bootstrap[is_observation_terminal[t + 1].bool()] = 0
            
            delta = rewards[t] + self.gamma * cont * bootstrap - values[t]
            gae_running = delta + self.gamma * self.gae_lambda * cont * gae_running
            advantages[t] = gae_running

        returns = advantages + values
        return advantages, returns

    def calculate_policy_gradient_loss(self, minibatch_advantages, probability_ratio):
        """
        Standard PPO Clipped Loss.
        """
        unclipped_pg_obj = minibatch_advantages * probability_ratio
        clipped_pg_obj = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )
        policy_gradient_loss = -torch.min(unclipped_pg_obj, clipped_pg_obj).mean()
        return policy_gradient_loss

    def calculate_critic_loss(self, new_util_val, util_tgt, new_vec_val, vec_ret):
        """
        Combined Critic Loss:
        1. Utility Head (Main): MSE against u(G)
        2. Returns Head (Aux): MSE against G
        """
        # Head 1: Utility (Scalar) -> Fits u(G)
        util_loss = 0.5 * ((new_util_val - util_tgt) ** 2).mean()
        
        # Head 2: Returns (Vector) -> Fits G
        vec_loss = 0.5 * ((new_vec_val - vec_ret) ** 2).mean()
        
        return util_loss, vec_loss