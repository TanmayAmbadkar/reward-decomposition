from uuid import uuid4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from torch.utils.tensorboard import SummaryWriter


class PPOLogger:
    """Logger for PPO training, supporting console output and TensorBoard.

    Attributes:
        use_tensorboard (bool): Whether to log metrics to TensorBoard.
        writer (SummaryWriter): TensorBoard writer instance (if enabled).
        reward_size (int): Dimension of the reward vector.
    """

    def __init__(self, run_name: Optional[str] = None, use_tensorboard: bool = False, reward_size: int = 1):
        """Initializes the PPOLogger.

        Args:
            run_name (str, optional): Name of the run for TensorBoard logging.
                Defaults to a generated UUID if None.
            use_tensorboard (bool, optional): Whether to enable TensorBoard logging.
                Defaults to False.
            reward_size (int, optional): Dimension of the reward vector. Defaults to 1.
        """
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self.reward_size = reward_size

    def log_rollout_step(self, infos: Dict[str, Any], global_step: int, active_task_id: Union[int, torch.Tensor] = 0, active_utility_val: float = 0.0):
        """Logs metrics for a single rollout step when an episode finishes.

        Args:
            infos (dict): Info dictionary returned by the environment, containing episode stats.
            global_step (int): Current global training step.
            active_task_id (int or torch.Tensor, optional): ID of the active task. Defaults to 0.
            active_utility_val (float, optional): Utility value achieved in the episode. Defaults to 0.0.
        """
        if "episode" in infos:
            # [MODIFIED] Added fallback for environments without 'dr' (dense/discounted return)
            if 'dr' in infos['episode']:
                non_zero_rews = infos['episode']['dr'][infos['_episode']]
            elif 'r' in infos['episode']:
                # Fallback to scalar return, expanded to vector size
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

    def log_policy_update(self, stats: Dict[str, float], global_step: int):
        """Logs statistics from a policy update.

        Args:
            stats (dict): Dictionary of training statistics (losses, entropy, etc.).
            global_step (int): Current global training step.
        """
        if self.use_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(f"losses/{k}", v, global_step)
    
    def log_evaluation(self, global_step: int, min_util: float, mean_util: float, max_util: float, task_id: int):
        """Logs evaluation metrics for a specific task.

        Args:
            global_step (int): Current global training step.
            min_util (float): Minimum utility achieved during evaluation.
            mean_util (float): Mean utility achieved during evaluation.
            max_util (float): Maximum utility achieved during evaluation.
            task_id (int): ID of the task being evaluated.
        """
        print(f"EVAL step={global_step} | Task={task_id} | Min={min_util:.3f} Mean={mean_util:.3f} Max={max_util:.3f}", flush=True)
        if self.use_tensorboard:
            self.writer.add_scalar(f"eval/utility_min_task_{task_id}", min_util, global_step)
            self.writer.add_scalar(f"eval/utility_mean_task_{task_id}", mean_util, global_step)
            self.writer.add_scalar(f"eval/utility_max_task_{task_id}", max_util, global_step)

class PPO:
    """Proximal Policy Optimization (PPO) agent for multi-objective reinforcement learning.

    This implementation supports:
    - Multi-objective rewards (vectorized rewards).
    - Multiple utility functions (tasks).
    - Vectorized evaluation.
    - Native PyTorch learning rate scheduling.
    - TensorBoard logging.

    Attributes:
        agent (nn.Module): The actor-critic agent network.
        optimizer (torch.optim.Optimizer): Optimizer for the agent.
        envs (gym.vector.VectorEnv): Vectorized training environments.
        eval_envs (gym.vector.VectorEnv): Vectorized evaluation environments.
        utility_functions (list): List of utility functions (callables) mapping reward vectors to scalar utility.
        reward_size (int): Dimension of the reward vector.
        num_rollout_steps (int): Number of steps to run for each environment per update.
        num_envs (int): Number of parallel environments.
        device (torch.device): Device to run computations on (inferred from agent).
        logger (PPOLogger): Logger instance.
    """

    def __init__(
        self,
        agent: nn.Module,
        optimizer: torch.optim.Optimizer,
        envs: Any,
        eval_envs: Optional[Any] = None,  
        utility_functions: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        env_is_discrete: bool = False,
        reward_size: int = 1,
        learning_rate: float = 3e-4,
        num_rollout_steps: int = 2048,
        num_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        surrogate_clip_threshold: float = 0.2,
        entropy_loss_coefficient: float = 0.001,
        value_function_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        num_minibatches: int = 32,
        normalize_advantages: bool = True,
        reward_rms: Optional[Any] = None,
        clip_value_function_loss: bool = True,
        target_kl: Optional[float] = None,
        anneal_lr: bool = True,
        seed: int = 1,
        logger: Optional[PPOLogger] = None,
        convex: bool = False,
        scalar_reward: bool = False,
        pareto_archive: Optional[Any] = None,
        policy_gradient_loss_coefficient: float = 1.0,
        eval_interval: int = 10000,
        num_eval_episodes: int = 10,
        total_timesteps: int = 1000000 # [ADDED] Required for native LR scheduler
    ):
        """Initializes the PPO agent.

        Args:
            agent (nn.Module): The actor-critic agent network.
            optimizer (torch.optim.Optimizer): Optimizer for the agent.
            envs (gym.vector.VectorEnv): Vectorized training environments.
            eval_envs (gym.vector.VectorEnv, optional): Vectorized evaluation environments. Defaults to None.
            utility_functions (list, optional): List of utility functions. Defaults to None (sum of rewards).
            env_is_discrete (bool, optional): Whether the environment action space is discrete. Defaults to False.
            reward_size (int, optional): Dimension of the reward vector. Defaults to 1.
            learning_rate (float, optional): Learning rate (used if optimizer needs it, though optimizer is passed in). Defaults to 3e-4.
            num_rollout_steps (int, optional): Steps per environment per update. Defaults to 2048.
            num_envs (int, optional): Number of parallel environments. Defaults to 1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            gae_lambda (float, optional): GAE lambda parameter. Defaults to 0.95.
            surrogate_clip_threshold (float, optional): PPO clipping threshold. Defaults to 0.2.
            entropy_loss_coefficient (float, optional): Coefficient for entropy loss. Defaults to 0.001.
            value_function_loss_coefficient (float, optional): Coefficient for value function loss. Defaults to 0.5.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0.5.
            update_epochs (int, optional): Number of epochs to update policy per rollout. Defaults to 10.
            num_minibatches (int, optional): Number of minibatches per update. Defaults to 32.
            normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
            reward_rms (object, optional): Running mean standard deviation for reward normalization. Defaults to None.
            clip_value_function_loss (bool, optional): Whether to clip value function loss. Defaults to True.
            target_kl (float, optional): Target KL divergence for early stopping. Defaults to None.
            anneal_lr (bool, optional): Whether to anneal learning rate. Defaults to True.
            seed (int, optional): Random seed. Defaults to 1.
            logger (PPOLogger, optional): Logger instance. Defaults to None.
            convex (bool, optional): Unused parameter. Defaults to False.
            scalar_reward (bool, optional): Whether to treat reward as scalar. Defaults to False.
            pareto_archive (object, optional): Unused parameter. Defaults to None.
            policy_gradient_loss_coefficient (float, optional): Coefficient for policy gradient loss. Defaults to 1.0.
            eval_interval (int, optional): Steps between evaluations. Defaults to 10000.
            num_eval_episodes (int, optional): Number of episodes per task for evaluation. Defaults to 10.
            total_timesteps (int, optional): Total training timesteps. Defaults to 1000000.
        """
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

        self.scalar_reward = scalar_reward
        
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        # [MODIFIED] Use Native Torch Scheduler
        self.anneal_lr = anneal_lr
        
        if self.anneal_lr:
            self.num_updates = self.total_timesteps // self.num_envs // self.num_rollout_steps
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.num_updates
            )
        else:
            self.lr_scheduler = None


    # [DELETED] create_lr_scheduler removed

    def _get_one_hot_task(self, task_idx_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Creates a one-hot encoding for a batch of task indices.

        Args:
            task_idx_tensor (torch.Tensor): Tensor of task indices.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: One-hot encoded task tensor of shape (batch_size, num_tasks).
        """
        return F.one_hot(task_idx_tensor.long(), num_classes=self.num_tasks).float()

    def _get_single_task_one_hot(self, task_idx: int, batch_size: int) -> torch.Tensor:
        """Creates a one-hot encoding for a single task index repeated across a batch.

        Args:
            task_idx (int): The task index to encode.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: One-hot encoded task tensor of shape (batch_size, num_tasks).
        """
        task_one_hot = torch.zeros((batch_size, self.num_tasks), device=self.device)
        if self.num_tasks > 0:
            task_one_hot[:, task_idx] = 1.0
        return task_one_hot

    def evaluate(self) -> None:
        """Evaluates the agent on all tasks using vectorized environments.

        This method runs evaluation episodes for each task in parallel chunks to save time.
        It logs the minimum, mean, and maximum utility achieved for each task.
        """
        """
        [MODIFIED] Vectorized Evaluation
        Evaluates all tasks in parallel chunks to save time.
        """
        if self.eval_envs is None:
            return

        print(f"--- Starting Evaluation at Step {self._global_step} ---")
        self.agent.eval()
        n_eval_envs = self.eval_envs.num_envs
        
        # Dictionary to store results: task_id -> list of utilities
        results = {t_id: [] for t_id in range(self.num_tasks)}
        
        # We process tasks in chunks based on available eval envs
        tasks_to_run = np.repeat(np.arange(self.num_tasks), self.num_eval_episodes)
        total_episodes_needed = len(tasks_to_run)
        
        obs, _ = self.eval_envs.reset(seed=self.seed + 1000 + int(self._global_step))
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        acc_rewards = torch.zeros((n_eval_envs, self.reward_size), device=self.device)
        acc_gamma = torch.ones((n_eval_envs, 1), device=self.device)
        
        active_tasks = torch.zeros(n_eval_envs, dtype=torch.long, device=self.device)
        env_task_ptr = np.full(n_eval_envs, -1, dtype=np.int32)
        
        # Fill initially
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
                    obs, acc_rewards, task_one_hot, deterministic=True, device=self.device
                )
            
            next_obs, reward, terminations, truncations, infos = self.eval_envs.step(action.cpu().numpy())
            
            reward_tens = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(n_eval_envs, self.reward_size)
            
            # Mask out idle environments
            idle_mask = torch.tensor(env_task_ptr == -1, device=self.device).unsqueeze(1)
            reward_tens[idle_mask.squeeze(1)] = 0.0

            acc_rewards += acc_gamma * reward_tens
            acc_gamma *= self.gamma
            
            is_done = torch.logical_or(torch.tensor(terminations), torch.tensor(truncations)).to(self.device)
            
            if is_done.any():
                done_indices = torch.where(is_done)[0]
                for idx in done_indices:
                    if env_task_ptr[idx.item()] != -1:
                        task_id = active_tasks[idx].item()
                        final_vec = acc_rewards[idx]
                        u_val = self.utility_functions[task_id](final_vec).item()
                        results[task_id].append(u_val)
                        
                        acc_rewards[idx] = 0
                        acc_gamma[idx] = 1.0
                        
                        if params_ptr < total_episodes_needed:
                            active_tasks[idx] = int(tasks_to_run[params_ptr])
                            env_task_ptr[idx.item()] = params_ptr
                            params_ptr += 1
                        else:
                            env_task_ptr[idx.item()] = -1 
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        
        for t_id, vals in results.items():
            if len(vals) > 0:
                self.logger.log_evaluation(self._global_step, np.min(vals), np.mean(vals), np.max(vals), t_id)
        
        self.agent.train()
        print("--- Evaluation Complete ---")

    def learn(self, total_timesteps: int) -> nn.Module:
        """Trains the agent for a specified number of timesteps.

        Args:
            total_timesteps (int): Total number of environment steps to train for.

        Returns:
            nn.Module: The trained agent.
        """
        # [MODIFIED] Simplified scheduler logic (no need to create it here)
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

            storage = self.collect_rollouts(obs, acc_rewards, acc_gamma, done, truncated, active_task_indices)

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

    def _initialize_storage(self) -> Tuple[torch.Tensor, ...]:
        """Initializes storage tensors for a rollout.

        Returns:
            tuple: A tuple containing initialized tensors for observations, actions, logprobs,
            rewards, dones, truncateds, values, accumulated rewards, accumulated gamma,
            task IDs, and behavior utility.
        """
        collected_observations = torch.zeros((self.num_rollout_steps, self.num_envs) + self.envs.single_observation_space.shape, device=self.device)

        action_shape = () if self.env_is_discrete else self.envs.single_action_space.shape
        actions = torch.zeros((self.num_rollout_steps, self.num_envs) + action_shape, device=self.device)

        action_log_probabilities = torch.zeros((self.num_rollout_steps, self.num_envs), device=self.device)
        rewards = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device)
        is_episode_terminated = torch.zeros((self.num_rollout_steps, self.num_envs), device=self.device)
        is_episode_truncated = torch.zeros((self.num_rollout_steps, self.num_envs), device=self.device)

        observation_vector_values = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device)

        acc_rewards_storage = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size), device=self.device)
        acc_gamma_storage = torch.ones((self.num_rollout_steps, self.num_envs, 1), device=self.device)

        task_id_storage = torch.zeros((self.num_rollout_steps, self.num_envs), dtype=torch.long, device=self.device)

        behavior_util_storage = torch.zeros((self.num_rollout_steps, self.num_envs, 1), device=self.device)

        return (
            collected_observations, actions, action_log_probabilities, rewards,
            is_episode_terminated, is_episode_truncated, observation_vector_values,
            acc_rewards_storage, acc_gamma_storage, task_id_storage, behavior_util_storage
        )

    def collect_rollouts(self, obs: torch.Tensor, acc_rewards: torch.Tensor, acc_gamma: torch.Tensor, done: torch.Tensor, truncated: torch.Tensor, active_task_indices: torch.Tensor) -> Dict[str, Any]:
        """Collects rollouts by interacting with the environment.

        Args:
            obs (torch.Tensor): Current observations.
            acc_rewards (torch.Tensor): Accumulated rewards so far.
            acc_gamma (torch.Tensor): Accumulated gamma so far.
            done (torch.Tensor): Whether the previous step was a termination.
            truncated (torch.Tensor): Whether the previous step was a truncation.
            active_task_indices (torch.Tensor): Indices of active tasks for each environment.

        Returns:
            dict: A dictionary containing collected rollout data and next state information.
        """
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

            # [MODIFIED] Normalize rewards HERE before accumulation
            if self.reward_rms is not None:
                rewards_reshaped = reward.reshape(-1, self.reward_size)
                self.reward_rms.update(rewards_reshaped)
                # We update reward variable with normalized values for storage and accumulation
                reward = self.reward_rms.normalize(rewards_reshaped).reshape(self.num_envs, self.reward_size)

            reward_tens = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(self.num_envs, self.reward_size)
            rewards[step] = reward_tens

            acc_rewards = acc_rewards + acc_gamma * reward_tens
            acc_gamma = acc_gamma * self.gamma

            done = torch.tensor(terminations, dtype=torch.float32, device=self.device)
            truncated = torch.tensor(truncations, dtype=torch.float32, device=self.device)
            is_done = torch.logical_or(done, truncated)

            if "episode" in infos:
                # Same logging logic with fallback
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
                        finished_vecs = torch.tensor(final_vec_ret, device=self.device, dtype=torch.float32)

                        for j, env_idx in enumerate(finished_env_indices):
                            task_idx = int(finished_tasks[j])
                            util_val = self.utility_functions[task_idx](finished_vecs[j]).item()
                            self.logger.log_rollout_step(
                                infos, self._global_step, active_task_id=task_idx, active_utility_val=util_val
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
            'next_task_indices': active_task_indices
        }
        return storage

    def update(self, storage: Dict[str, Any]) -> Dict[str, float]:
        """Updates the agent's policy and value function using the collected rollout data.

        Args:
            storage (dict): Dictionary containing collected rollout data.

        Returns:
            dict: Dictionary of training metrics (losses, entropy, etc.).
        """
        # 1. Calculate Monte Carlo Vector Returns
        vec_returns = self.compute_vector_returns(
            storage['rewards'], storage['dones'], storage['truncated'],
            storage['next_aux_val'], storage['next_done']
        )

        # Flatten data
        b_obs = self._flatten(storage['obs'])
        b_acc = self._flatten(storage['acc_rewards_in'])
        b_gamma = self._flatten(storage['acc_gamma_in'])
        b_actions = self._flatten(storage['actions'])
        b_vec_returns = self._flatten(vec_returns)
        b_task_ids = self._flatten(storage['task_ids'])

        b_projected_total_vec = b_acc + b_gamma * b_vec_returns

        # =====================================================
        # STEP 0: PRE-CALCULATE ADVANTAGES (Using OLD Critic)
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

        precalc_end_time = time.time()
        precalc_time = precalc_end_time - precalc_start_time
        
        # =====================================================
        # PHASE 1: CRITIC UPDATE (On-Policy) — unchanged
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

                # [FIX] Explicitly define the "Behavior Task" conditioning.
                # The Vector Head predicts physics (env dynamics). It MUST see the 
                # task context that generated these dynamics (the behavior task).
                mb_task_hot_behavior = self._get_one_hot_task(mb_task_ids, len(mb_inds))

                # Calculate Targets for the *actual* tasks (On-Policy Utility Target)
                mb_target_utils = torch.zeros(len(mb_inds), 1, device=self.device)
                unique_tasks = torch.unique(mb_task_ids)
                
                for t_id in unique_tasks:
                    mask = (mb_task_ids == t_id)
                    task_vecs = mb_projected_vec[mask]
                    task_utils = self.utility_functions[t_id.item()](task_vecs)
                    if len(task_utils.shape) == 1: task_utils = task_utils.unsqueeze(1)
                    mb_target_utils[mask] = task_utils

                # Forward pass using BEHAVIOR conditioning
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
                    ev_u = float('nan') if var_y == 0 else (1 - torch.var(y_true - y_pred) / var_y).item()
                    explained_var_util.append(ev_u)

                    y_true_v = mb_vec_ret.flatten()
                    y_pred_v = pred_vec.flatten()
                    var_y_v = torch.var(y_true_v)
                    ev_v = float('nan') if var_y_v == 0 else (1 - torch.var(y_true_v - y_pred_v) / var_y_v).item()
                    explained_var_vec.append(ev_v)

                self.optimizer[1].zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.optimizer[1].step()

                critic_loss_hist.append(total_critic_loss.item())

        critic_end_time = time.time()
        critic_time = critic_end_time - critic_start_time
       
        # =====================================================
        # PHASE 2: ACTOR UPDATE (Vectorized Counterfactual)
        # =====================================================

        actor_start_time = time.time()

        actor_loss_hist = []
        entropy_hist = []
        kl_hist = []
        clip_frac_hist = []
        
        # [SETUP] 1. Get the log_prob of the action under the behavior policy (flattened)
        # This represents pi_behavior(a|s), needed for the IS ratio.
        b_behavior_lp = self._flatten(storage['logprobs'])

        # 2. Flatten & Repeat Data for Super Batch
        M = self.num_tasks
        mk_reps = lambda x: (M,) + (1,) * (x.ndim - 1)

        super_obs = b_obs.repeat(*mk_reps(b_obs))
        super_acc = b_acc.repeat(*mk_reps(b_acc))
        super_act = b_actions.repeat(*mk_reps(b_actions))
        
        # [NEW] Repeat behavior logprobs to align with the super batch
        super_behavior_lp = b_behavior_lp.repeat(*mk_reps(b_behavior_lp))
        
        # Track the Behavior Task ID for every sample in the Super Batch
        super_behavior_task_ids = b_task_ids.repeat(M)

        # 3. Construct Target Task IDs
        # Pattern: [0,0... (B times), 1,1... (B times), ...]
        task_indices = torch.arange(self.num_tasks, device=self.device).repeat_interleave(self.batch_size)
        super_task_hot = self._get_one_hot_task(task_indices, self.batch_size * self.num_tasks)
        
        super_adv = torch.cat(all_task_advantages, dim=0)
        super_old_lp = torch.cat(all_task_old_logprobs, dim=0)

        total_samples = self.batch_size * M
        inds = np.arange(total_samples)
        
        for _ in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, total_samples, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]

                mb_obs = super_obs[mb_inds]
                mb_acc = super_acc[mb_inds]
                mb_task = super_task_hot[mb_inds]
                mb_act = super_act[mb_inds]
                mb_adv = super_adv[mb_inds]
                
                # mb_old_lp is LogProb(Action | Target Task)
                mb_old_lp = super_old_lp[mb_inds] 
                
                # [NEW] mb_behavior_lp is LogProb(Action | Behavior Task)
                mb_behavior_lp = super_behavior_lp[mb_inds]
                
                mb_target_id = task_indices[mb_inds]       
                mb_behavior_id = super_behavior_task_ids[mb_inds]

                # =========================================================
                # 1. Dynamic IS-Based Plausibility Mask
                # =========================================================
                # Calculate Log Importance Sampling Weight: log( P_target / P_behavior )
                # If 0: policies are identical. If negative: target is less likely.
                log_is_weight = mb_old_lp - mb_behavior_lp
                
                # Threshold: Keep if action is at least 1% as likely under 
                # the target policy as it was under the behavior policy.
                # log(0.01) approx -4.6
                is_ratio_threshold = np.log(0.01) 
                
                is_plausible = (log_is_weight > is_ratio_threshold)
                
                if self.num_tasks == 1:
                    is_real = torch.ones_like(mb_target_id, dtype=torch.bool)
                else:
                    is_real = (mb_target_id == mb_behavior_id)
                
                keep_mask = (is_real | is_plausible).float()
                
                # Check if we have valid samples to avoid division by zero
                valid_samples = keep_mask.sum()
                
                if valid_samples == 0:
                    continue # Skip this minibatch if no samples are valid

                # =========================================================
                # 2. Compute New LogProbs
                # =========================================================
                new_lp, entropy = self.agent.compute_action_log_probabilities_and_entropy(
                    mb_obs, mb_act, mb_acc, mb_task, self.device
                )

                logratio = new_lp - mb_old_lp
                
                # [SAFETY] Clamp logratio to prevent e^100 explosion in counterfactuals
                if (logratio > 20.0).any():
                    # Optional: Log this occurrence if debugging
                    pass 
                
                logratio = torch.clamp(logratio, max=20.0) 
                ratio = logratio.exp()

                # Metrics (Apply mask to metrics to get accurate reporting)
                with torch.no_grad():
                    # Only calculate KL on kept samples
                    approx_kl = (((ratio - 1) - logratio) * keep_mask).sum() / valid_samples
                    clip_frac = ((((ratio - 1.0).abs() > self.surrogate_clip_threshold).float()) * keep_mask).sum() / valid_samples

                # Loss Calculation
                pg_loss1 = -mb_adv.squeeze() * ratio
                pg_loss2 = -mb_adv.squeeze() * torch.clamp(ratio, 1.0 - self.surrogate_clip_threshold, 1.0 + self.surrogate_clip_threshold)
                pg_loss_elementwise = torch.max(pg_loss1, pg_loss2)
                ent_loss_elementwise = -entropy 
                
                total_loss_elementwise = pg_loss_elementwise + self.entropy_loss_coefficient * ent_loss_elementwise

                # =========================================================
                # 3. Apply Mask and NORMALIZE (Mean over valid)
                # =========================================================
                masked_loss = total_loss_elementwise * keep_mask
                
                # Divide by valid_samples to get the MEAN loss over valid data
                actor_loss = masked_loss.sum() / (valid_samples + 1e-8)

                self.optimizer[0].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                self.optimizer[0].step()

                actor_loss_hist.append(actor_loss.item())
                entropy_hist.append(entropy.mean().item()) 
                kl_hist.append(approx_kl.item())
                clip_frac_hist.append(clip_frac.item())

        actor_end_time = time.time()
        actor_time = actor_end_time - actor_start_time

        # self.entropy_loss_coefficient *= self.gamma

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
    
    def _flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flattens a tensor by combining the first two dimensions.

        Args:
            tensor (torch.Tensor): Input tensor of shape (T, B, ...).

        Returns:
            torch.Tensor: Flattened tensor of shape (T*B, ...).
        """
        return tensor.reshape((-1,) + tensor.shape[2:])

    def compute_vector_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor, truncated: torch.Tensor, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> torch.Tensor:
        """Computes Monte Carlo vector returns (discounted cumulative rewards).

        Args:
            rewards (torch.Tensor): Rewards tensor.
            dones (torch.Tensor): Termination flags.
            truncated (torch.Tensor): Truncation flags.
            next_value (torch.Tensor): Value estimate for the next state (bootstrap).
            next_done (torch.Tensor): Termination flag for the next state.

        Returns:
            torch.Tensor: Computed vector returns.
        """
        T = self.num_rollout_steps
        returns = torch.zeros_like(rewards, device=self.device)
        
        # Bootstrap from next_value if the episode didn't end naturally
        # If next_done is True, next_value is irrelevant (masked out)
        next_val = next_value * (1.0 - next_done).unsqueeze(1)

        for t in reversed(range(T)):
            
            real_done = dones[t] * (1 - truncated[t])
            cont_mask = (1.0 - real_done).unsqueeze(1)
            
            returns[t] = rewards[t] + self.gamma * cont_mask * next_val
            next_val = returns[t]
            
        return returns

    def calculate_policy_gradient_loss(self, minibatch_advantages: torch.Tensor, probability_ratio: torch.Tensor) -> torch.Tensor:
        """Calculates the PPO policy gradient loss.

        Args:
            minibatch_advantages (torch.Tensor): Advantages for the minibatch.
            probability_ratio (torch.Tensor): Ratio of new to old probabilities.

        Returns:
            torch.Tensor: Policy gradient loss.
        """
        unclipped_pg_obj = minibatch_advantages * probability_ratio
        clipped_pg_obj = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )
        policy_gradient_loss = -torch.min(unclipped_pg_obj, clipped_pg_obj)
        return policy_gradient_loss

    def calculate_critic_loss(self, new_util_val: torch.Tensor, util_tgt: torch.Tensor, new_vec_val: torch.Tensor, vec_ret: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the critic loss (utility and vector value loss).

        Args:
            new_util_val (torch.Tensor): Predicted utility values.
            util_tgt (torch.Tensor): Target utility values.
            new_vec_val (torch.Tensor): Predicted vector values.
            vec_ret (torch.Tensor): Target vector returns.

        Returns:
            tuple: Utility loss and vector loss.
        """
        util_loss = 0.5 * ((new_util_val - util_tgt) ** 2).mean()
        vec_loss = 0.5 * ((new_vec_val - vec_ret) ** 2).mean()
        return util_loss, vec_loss