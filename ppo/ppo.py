from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.dirichlet import Dirichlet
from morl_baselines.common.pareto import ParetoArchive


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
        for param_group in self.optimizer[0].param_groups:
            param_group["lr"] = min(param_group["lr"] * 0.99, 0.00005)
        for param_group in self.optimizer[1].param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False, reward_size = 1):
        self.use_tensorboard = use_tensorboard
        self.global_steps = []
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")
        self.reward_size = reward_size

    def log_rollout_step(self, infos, global_step):
        self.global_steps.append(global_step)
        if "episode" in infos:
            non_zero_rews = infos['episode']['dr'][infos['_episode']]
            non_zero_lens = infos['episode']['l'][infos['_episode']]
            non_zero_comps = []
            for i in range(self.reward_size):
                non_zero_comps.append(infos['episode'][f'dr'][infos['_episode']][:,i].mean())
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

    def log_policy_update(self, update_results, global_step):
        if self.use_tensorboard:
            self.writer.add_scalar(
                "losses/policy_loss", update_results["policy_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/value_loss", update_results["value_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/entropy_loss", update_results["entropy_loss"], global_step
            )

            # self.writer.add_scalar(
            #     "losses/kl_divergence", update_results["old_approx_kl"], global_step
            # )
            self.writer.add_scalar(
                "losses/kl_divergence", update_results["approx_kl"], global_step
            )
            self.writer.add_scalar(
                "losses/clipping_fraction",
                update_results["clipping_fractions"],
                global_step,
            )
            self.writer.add_scalar(
                "losses/explained_variance",
                update_results["explained_variance"],
                global_step,
            )

    def write_video(self, frames):
    
        frames = frames.reshape(1, *frames.shape)
        frames = torch.Tensor(frames).permute([0, 1, 4, 2, 3])
        self.writer.add_video(tag = "video", vid_tensor = frames[:,::3], fps = 24)

class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        env_is_discrete = False,
        reward_size = 1,
        learning_rate=3e-4,
        num_rollout_steps=2048,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.001,
        policy_gradient_loss_coefficient=2.0,
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
        scalar_reward = False,
        pareto_archive = ParetoArchive(),
    ):
        """
        Proximal Policy Optimization (PPO) algorithm implementation.

        This class implements the PPO algorithm, a policy gradient method for reinforcement learning.
        It's designed to be more stable and sample efficient compared to traditional policy gradient methods.

        Args:
            agent: The agent (policy/value network) to be trained.
            optimizer: The optimizer for updating the agent's parameters.
            envs: The vectorized environment(s) to train on.

            # Core PPO parameters
            learning_rate (float): Learning rate for the optimizer.
            num_rollout_steps (int): Number of steps to run for each environment per update.
            num_envs (int): Number of parallel environments.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.
            entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
            value_function_loss_coefficient (float): Coefficient for the value function loss.
            max_grad_norm (float): Maximum norm for gradient clipping.

            # Training process parameters
            update_epochs (int): Number of epochs to update the policy for each rollout.
            num_minibatches (int): Number of minibatches to use per update.
            normalize_advantages (bool): Whether to normalize advantages.
            clip_value_function_loss (bool): Whether to clip the value function loss.
            target_kl (float or None): Target KL divergence for early stopping.

            # Learning rate schedule
            anneal_lr (bool): Whether to use learning rate annealing.

            # Reproducibility and tracking
            seed (int): Random seed for reproducibility of environment initialisation.
            logger (PPOLogger): A logger instance for logging. if None is passed, a default logger is created.

        The PPO algorithm works by collecting a batch of data from the environment,
        then performing multiple epochs of optimization on this data. It uses a surrogate
        objective function and a value function, both clipped to prevent too large policy updates.

        KL Divergence Approach:
        This implementation uses a fixed KL divergence threshold (`target_kl`) for early stopping.
        If `target_kl` is set (not None), the policy update will stop early if the approximate
        KL divergence exceeds this threshold. This acts as a safeguard against too large policy
        updates, helping to maintain the trust region.

        - If `target_kl` is None, no early stopping based on KL divergence is performed.
        - A smaller `target_kl` (e.g., 0.01) results in more conservative updates, potentially
          leading to more stable but slower learning.
        - A larger `target_kl` (e.g., 0.05) allows for larger policy updates, potentially
          leading to faster but possibly less stable learning.
        - Common values for `target_kl` range between 0.01 and 0.05. The original paper (https://arxiv.org/abs/1707.06347)
          settled on a target range of (0.003 to 0.03)

        The optimal `target_kl` can depend on the specific environment and problem. It's often
        beneficial to monitor the KL divergence during training and adjust `target_kl` based on
        the stability and speed of learning.
        """
        self.agent = agent
        self.envs = envs
        self.env_is_discrete = env_is_discrete
        self.optimizer = optimizer
        self.seed = seed

        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.batch_size = num_envs * num_rollout_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches
        self.reward_size = reward_size
        self.scalar_reward = scalar_reward

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
        self._global_step = 0
        self.logger = logger or PPOLogger()
        self.convex = convex
        self.beta = 0.9
        self.log_barrier_t = 20.0
        self.pareto_archive = pareto_archive

    def create_lr_scheduler(self, num_policy_updates):
        return LinearLRSchedule(self.optimizer, self.initial_lr, num_policy_updates)

    def learn(self, total_timesteps):
        """
        Train the agent using the PPO algorithm.

        This method runs the full training loop for the specified number of timesteps,
        collecting rollouts from the environment and updating the policy accordingly.

        Args:
            total_timesteps (int): The total number of environment timesteps to train for.

        Returns:
            agent: The trained agent (policy/value network).

        Process:
            1. Initialize the learning rate scheduler if annealing is enabled.
            2. Reset the environment to get initial observations.
            3. For each update iteration:
                a. Collect rollouts from the environment.
                b. Compute advantages and returns.
                c. Update the policy and value function multiple times on the collected data.
                d. Log relevant metrics and training progress.
            4. Return the trained agent.

        Notes:
            - The actual number of environment steps may be slightly less than `total_timesteps`
              due to the integer division when calculating the number of updates.
            - Early stopping based on KL divergence may occur if `target_kl` is set.
        """
        num_policy_updates = total_timesteps // (self.num_rollout_steps * self.num_envs)

        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_policy_updates)

        next_observation, is_next_observation_terminal, is_next_observation_truncated = self._initialize_environment()
        # weights = torch.ones(self.num_envs, self.reward_size).to(self.device)

        # Initialize logging variables
        self._global_step = 0

        current_weights = None
        for update in range(num_policy_updates):
            if self.anneal_lr:
                self.lr_scheduler.step()

            (
                batch_observations,
                batch_log_probabilities,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_values,
                batch_weights,
                next_observation,
                is_next_observation_terminal,
                is_next_observation_truncated,
                current_weights,
            ) = self.collect_rollouts(next_observation, is_next_observation_terminal, is_next_observation_truncated, current_weights)

            update_results = self.update_policy(
                batch_observations,
                batch_log_probabilities,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_values,
                batch_weights,
            )

            self.logger.log_policy_update(update_results, self._global_step)


        print(f"Training completed. Total steps: {self._global_step}")

        return self.agent  # Return the trained agent

    def _initialize_environment(self):
        """
        Initialize the environment for the start of training, resets the vectorized environments
        to their initial states and prepares the initial observation and termination flag for the agent to begin
        interacting with the environment.

        Returns:
            tuple: A tuple containing:
                - initial_observation (torch.Tensor): The initial observation from the
                environment
                - is_initial_observation_terminal (torch.Tensor): A tensor of zeros
                indicating that the initial state is not terminal.

        Note:
            The method uses the seed set during the PPO initialization to ensure
            reproducibility of the environment's initial state across different runs.
        """
        initial_observation, _ = self.envs.reset(seed=self.seed)
        initial_observation = torch.Tensor(initial_observation).to(self.device)
        is_initial_observation_terminal = torch.zeros(self.num_envs).to(self.device)
        is_initial_observation_truncated = torch.zeros(self.num_envs).to(self.device)
        return initial_observation, is_initial_observation_terminal, is_initial_observation_truncated

    def collect_rollouts(self, next_observation, is_next_observation_terminal, is_next_observation_truncated, current_weights):
        """
        Collect a set of rollout data by interacting with the environment. A rollout is a sequence of observations,
        actions, and rewards obtained by running the current policy in the environment. The collected data is crucial
        for the subsequent policy update step in the PPO algorithm.

        This method performs multiple timesteps across all (parallel) environments, collecting data which
        will be used to update the policy.

        The collected data represents a fixed number of timesteps (num_rollout_steps * num_envs)
        of interaction, which forms the basis for a single PPO update iteration. This may
        include partial trajectories from multiple episodes across the parallel environments.

        The method uses the current policy to select actions, executes these actions in the
        environment, and stores the resulting observations, rewards, and other relevant
        information. It also computes advantages and returns using Generalized Advantage
        Estimation (GAE), which are crucial for the PPO algorithm.

        Args:
            next_observation (torch.Tensor): The starting observation for this rollout.
            is_next_observation_terminal (torch.Tensor): Boolean tensor indicating whether
                the starting state is terminal.

        Returns:
            tuple: A tuple containing:
                - batch_observations (torch.Tensor): Flattened tensor of all observations.
                - batch_log_probabilities (torch.Tensor): Log probabilities of the actions taken.
                - batch_actions (torch.Tensor): Flattened tensor of all actions taken.
                - batch_advantages (torch.Tensor): Computed advantage estimates.
                - batch_returns (torch.Tensor): Computed returns (sum of discounted rewards).
                - batch_values (torch.Tensor): Value function estimates for each state.
                - next_observation (torch.Tensor): The final observation after collecting rollouts.
                - is_next_observation_terminal (torch.Tensor): Whether the final state is terminal.

        Note:
            This method updates the global step during the rollout collection process.
        """
        (
            collected_observations,
            actions,
            action_log_probabilities,
            rewards,
            is_episode_terminated,
            is_episode_truncated,
            observation_values,
            rollout_weights

        ) = self._initialize_storage()    
        if current_weights is None:
            if self.reward_size != 1:
                weights = torch.distributions.uniform.Uniform(low=0, high=1).sample((self.num_envs, self.reward_size)).to(self.device).type(torch.float32)
               
                if self.convex:
                    dist = Dirichlet(torch.ones(self.reward_size, device=self.device))
                    weights = dist.sample((self.num_envs, )).type(torch.float32)
            else:
                weights = torch.ones(self.num_envs, 1).to(self.device).type(torch.float32)
               
        else:
            weights = current_weights

        for step in range(self.num_rollout_steps):
            # Store current observation
            collected_observations[step] = next_observation
            is_episode_terminated[step] = is_next_observation_terminal
            is_episode_truncated[step] = is_next_observation_truncated

            change_weights = torch.logical_or(is_next_observation_terminal.type(torch.bool), is_next_observation_truncated.type(torch.bool))
            if self.reward_size != 1 and change_weights.any():
                # print(weights)   
                weights[change_weights] = torch.distributions.uniform.Uniform(low=0, high=1).sample(weights[change_weights].shape).to(self.device).type(torch.float32)
                if self.convex:
                    dist = Dirichlet(torch.ones(self.reward_size, device=self.device))
                    weights[change_weights] = dist.sample((sum(change_weights), )).type(torch.float32)
                
                
            with torch.no_grad():
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    next_observation, weights, deterministic=False,
                )
                value = self.agent.estimate_value_from_observation(next_observation, weights)

                observation_values[step] = value
            
            clipped_action = action.cpu().numpy()
            actions[step] = torch.as_tensor(clipped_action).to(self.device)
            action_log_probabilities[step] = logprob
            next_observation, reward, terminations, truncations, infos = self.envs.step(
                # np.clip(action.cpu().numpy(), self.envs.single_action_space.low, self.envs.single_action_space.high)
                clipped_action
            )
            self._global_step += self.num_envs
            if self.scalar_reward:
                rewards[step] = torch.sum(weights * torch.as_tensor(reward, device=self.device), dim=-1, keepdim=True)
            else:
                rewards[step] = torch.as_tensor(reward, device=self.device).reshape(self.num_envs, self.reward_size)
                
            # print(clipped_action, rewards[step])
            is_next_observation_terminal = terminations
            is_next_observation_truncated = truncations
            rollout_weights[step] = weights
            
            next_observation, is_next_observation_terminal, is_next_observation_truncated = (
                torch.as_tensor(
                    next_observation, dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    is_next_observation_terminal,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    is_next_observation_truncated,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            
            self.logger.log_rollout_step(infos, self._global_step)
            if "episode" in infos:
                for curr_ret in infos['episode']['dr'][infos['_episode']]:
                    self.pareto_archive.add(tuple(weights[infos['_episode']].detach().cpu().numpy()), curr_ret)
                
                
                self.envs.disc_episode_returns[infos['_episode']] = 0
            

        # Estimate the value of the next state (the state after the last collected step) using the current policy
        # This value will be used in the GAE calculation to compute advantages
        with torch.no_grad():
            if self.scalar_reward:
                next_value = self.agent.estimate_value_from_observation(
                    next_observation, weights
                ).reshape(self.num_envs, 1)
            else:
                next_value = self.agent.estimate_value_from_observation(
                    next_observation, weights
                ).reshape(self.num_envs, self.reward_size)

            advantages, returns = self.compute_advantages(
                rewards[:self.num_rollout_steps],
                observation_values[:self.num_rollout_steps],
                is_episode_terminated[:self.num_rollout_steps],
                is_episode_truncated[:self.num_rollout_steps],
                next_value,
                is_next_observation_terminal,
                is_next_observation_truncated,
            )

        # Flatten the batch for easier processing in the update step
        (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
            batch_weights
        ) = self._flatten_rollout_data(
            collected_observations,
            action_log_probabilities,
            actions,
            advantages,
            returns,
            observation_values,
            rollout_weights
        )

        # Return the collected and computed data for the policy update step
        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
            batch_weights,
            next_observation,
            is_next_observation_terminal,
            is_next_observation_truncated,
            weights
        )

    def _initialize_storage(self):
        collected_observations = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
            + self.envs.single_observation_space.shape
        ).to(self.device)
        action_space = () if self.env_is_discrete else self.envs.single_action_space.shape
        actions = torch.zeros(
            (self.num_rollout_steps, self.num_envs, )
        + action_space
        ).to(self.device)
        action_log_probabilities = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
        ).to(self.device)
        if self.scalar_reward:
            rewards = torch.zeros((self.num_rollout_steps, self.num_envs, 1)).to(self.device)
        else:
            rewards = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(self.device)
        is_episode_terminated = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        is_episode_truncated = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        if self.scalar_reward:
            observation_values = torch.zeros((self.num_rollout_steps, self.num_envs, 1)).to(
                self.device
            )
        else:
            observation_values = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(
                self.device
            )
        rollout_weights = torch.zeros((self.num_rollout_steps, self.num_envs, self.reward_size)).to(
            self.device
        )

        return (
            collected_observations,
            actions,
            action_log_probabilities,
            rewards,
            is_episode_terminated,
            is_episode_truncated,
            observation_values,
            rollout_weights
        )

    def compute_advantages(
        self,
        rewards,                       # shape: [T, num_envs]
        values,                        # shape: [T, num_envs]
        is_observation_terminal,       # shape: [T, num_envs]
        is_observation_truncated,      # shape: [T, num_envs]
        next_value,                    # shape: [num_envs]
        is_next_observation_terminal,  # shape: [num_envs]
        is_next_observation_truncated  # shape: [num_envs]
    ):
        """
        Compute advantages with an elementwise handling of terminal and truncated states.
        
        For each environment and time step:
        - If the next state is terminal: we set the continuation factor to 0 and use a bootstrap value of 0.
        - If the next state is truncated (and not terminal): we still bootstrap from the next value but reset
            the advantage accumulator.
        """
        T = self.num_rollout_steps
        # Initialize advantages with the same shape as rewards.
        advantages = torch.zeros_like(rewards).to(self.device)
        # Initialize gae_running as a tensor with shape [num_envs]
        gae_running = torch.zeros_like(next_value).to(self.device)
        reward_size = 1 if self.scalar_reward else self.reward_size
        if self.reward_rms is not None:
            self.reward_rms.update(rewards.cpu().numpy().reshape(-1, reward_size))
            
            # 2. Normalize the returns
            rewards = torch.tensor(
                self.reward_rms.normalize(rewards.cpu().numpy()),
                dtype=torch.float32
            ).to(self.device)
            # Iterate backwards over the rollout steps.
        for t in reversed(range(T)):
            if t == T - 1:
                # For the final step, use the provided next_value and flags.
                # For each environment: if next state is terminal, no bootstrapping.
                cont = 1 - is_next_observation_terminal
                # If truncated (and not terminal), reset gae_running.
                mask_trunc = is_next_observation_truncated.bool() & (~is_next_observation_terminal.bool())
                # gae_running = torch.where(mask_trunc, torch.zeros_like(gae_running), gae_running)
                gae_running[mask_trunc] = 0
                # Bootstrap value: use next_value if not terminal, else 0.
                bootstrap = next_value.clone()
                bootstrap[is_next_observation_terminal.bool()] = 0
            else:
                # For intermediate steps, use flags from the next time step (t+1).
                # cont = torch.ones_like(is_next_observation_terminal)
                cont = 1 - is_observation_terminal[t + 1]
                
                mask_trunc = is_observation_truncated[t + 1].bool() & (~is_observation_terminal[t + 1].bool())
                gae_running[mask_trunc] = 0
                bootstrap = values[t + 1].clone()
                bootstrap[is_observation_terminal[t + 1].bool()] = 0
            
            # Compute the TD error (delta) elementwise.
            delta = rewards[t] + self.gamma * cont.reshape(-1, 1) * bootstrap  - values[t]
            gae_running = delta + self.gamma * self.gae_lambda * cont.reshape(-1, 1) * gae_running
            advantages[t] = gae_running

        returns = advantages + values
        return advantages, returns



    def _flatten_rollout_data(
        self,
        collected_observations,
        action_log_probabilities,
        actions,
        advantages,
        returns,
        observation_values,
        weights,
    ):
        batch_observations = collected_observations.reshape(
            (-1,) + self.envs.single_observation_space.shape
        )
        batch_log_probabilities = action_log_probabilities.reshape(-1)
        
        action_space = () if self.env_is_discrete else self.envs.single_action_space.shape
        batch_actions = actions.reshape((-1,) + action_space)
        if self.scalar_reward:
            batch_advantages = advantages.reshape(-1, 1)
            batch_returns = returns.reshape(-1, 1)
            batch_values = observation_values.reshape(-1, 1)
        else:
            batch_advantages = advantages.reshape(-1, self.reward_size)
            batch_returns = returns.reshape(-1, self.reward_size)
            batch_values = observation_values.reshape(-1, self.reward_size)
        batch_weights = weights.reshape(-1, self.reward_size)

        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
            batch_weights
        )

    def update_policy(
        self,
        collected_observations,
        collected_action_log_probs,
        collected_actions,
        computed_advantages,
        computed_returns,
        previous_value_estimates,
        collected_weights,
    ):
        """
        PPO update with multi-objective PCGrad for the actor (plus entropy),
        followed by a separate critic update.
        """
        self.noise_level = 0.1
        self.lambda_diversity = 0.01
        self.diversity_scale = 0.0

        batch_size = self.num_rollout_steps * self.num_envs
        assert collected_observations.shape[0] == batch_size
        batch_indices = np.arange(batch_size)
        clipping_fractions = []

        for _ in range(self.update_epochs):
            np.random.shuffle(batch_indices)
            value_function_losses, all_policy_losses, entropy_losses, kl_div = [], [], [], []
            
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                idx = batch_indices[start:end]

                # Unpack minibatch data
                obs_mb = collected_observations[idx]
                old_logp = collected_action_log_probs[idx]
                acts_mb = collected_actions[idx]
                adv_mb = computed_advantages[idx]      # [N, d]
                w_mb = collected_weights[idx]         # [N, d]

                # --- Get new policy and value estimates ---
                new_logp, entropy = self.agent.compute_action_log_probabilities_and_entropy(
                    obs_mb, acts_mb, w_mb
                )
                new_value = self.agent.estimate_value_from_observation(obs_mb, w_mb)
                # 3) Importance sampling ratio & KL estimates
                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                with torch.no_grad():
                    old_kl = (-log_ratio).mean()
                    new_kl = ((ratio - 1) - log_ratio).mean()
                    # clipping fraction
                    clipping_fractions.append(
                        ((ratio - 1.0).abs() > self.surrogate_clip_threshold)
                        .float()
                        .mean()
                        .item()
                    )
                
                # --- Critic Update (Separate) ---
                value_function_loss = self.calculate_value_function_loss(
                    new_value,
                    computed_returns, # Use full returns_mb
                    previous_value_estimates,
                    idx,
                    (w_mb != 0).float() # Use mask
                ) * self.value_function_loss_coefficient
                
                self.optimizer[1].zero_grad()
                value_function_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.optimizer[1].step()

                # --- Actor Update with Diversity Loss ---
                self.optimizer[0].zero_grad()
                
                # 1. Calculate the standard, weighted PPO policy loss (your current method)
                if self.normalize_advantages:
                    adv_mb = (adv_mb - adv_mb.mean(dim=0)) / (adv_mb.std(dim=0) + 1e-8)
                
                # weighted_adv_mb = adv_mb * w_mb
                # Summing the losses from each objective's weighted advantage
                # Note: Your calculate_policy_gradient_loss should handle the sum over objectives
                policy_losses = self.calculate_policy_gradient_loss(
                    adv_mb,  # Use the weights to scale the advantages
                    (new_logp - old_logp).exp().reshape(-1, 1)
                )
                
                policy_losses = policy_losses * w_mb
                policy_losses = policy_losses.mean(dim = 0)  # Sum over the reward dimensions

                # 2. Calculate the Diversity Penalty
                # a. Sample distractor weights on the fly
                noise = torch.randn_like(w_mb) * self.noise_level
                distractor_noisy = w_mb + noise
                distractor_clipped = torch.clamp(distractor_noisy, min=0)
                distractor_weights_mb = distractor_clipped / (torch.sum(distractor_clipped, dim=1, keepdim=True) + 1e-8)

                # b. Get action distributions
                current_action_dist = self.agent.get_action_distribution(torch.hstack([obs_mb, w_mb]))
                with torch.no_grad():
                    distractor_action_dist = self.agent.get_action_distribution(torch.hstack([obs_mb, distractor_weights_mb]))

                # c. Calculate the actual KL divergence
                actual_kl_divergence = torch.distributions.kl.kl_divergence(
                    current_action_dist, distractor_action_dist
                ).mean()

                # d. Calculate the TARGET KL divergence based on weight distance
                with torch.no_grad():
                    # Calculate the L1 distance between the preference vectors
                    weight_distance = torch.abs(w_mb - distractor_weights_mb).sum(dim=1).mean()
                    # Set the target KL. self.diversity_scale is a new hyperparameter (e.g., 0.5)
                    target_kl_divergence = self.diversity_scale * weight_distance

                # 3. Construct the Final Loss
                # The diversity loss is now the squared error between the actual and target KL.
                # We want to MINIMIZE this error, so we ADD it to the loss.
                diversity_loss = self.lambda_diversity * (target_kl_divergence - actual_kl_divergence).pow(2)

                ent_loss = self.entropy_loss_coefficient * entropy.mean()
                
                total_actor_loss = sum(policy_losses) - ent_loss + diversity_loss
                
                # 4. Perform a single backward pass and step for the actor.
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                self.optimizer[0].step()
                
                value_function_losses.append(value_function_loss.item())
                all_policy_losses.append(sum(policy_losses).item())
                entropy_losses.append(entropy.mean().item())
                kl_div.append(new_kl.item())

        # === Diagnostics ===
        predicted, actual = previous_value_estimates.cpu().numpy(), computed_returns.cpu().numpy()
        var_r = np.var(actual)
        explained_var = (
            np.nan if var_r == 0 else 1 - np.var(actual - predicted) / var_r
        )

        return {
            "policy_loss": np.mean(all_policy_losses),
            # "policy_loss": policy_losses,
            "value_loss": np.mean(value_function_losses),
            "entropy_loss": np.mean(entropy_losses),
            # "old_approx_kl": old_kl.item(),
            "approx_kl": np.mean(kl_div),
            "clipping_fractions": np.mean(clipping_fractions),
            "explained_variance": explained_var,
        }


    def calculate_policy_gradient_loss(self, minibatch_advantages, probability_ratio):
        """
        Calculate the policy gradient loss using the PPO clipped objective, which is designed to
        improve the stability of policy updates. It uses a clipped surrogate objective
        that limits the incentive for the new policy to deviate too far from the old policy.

        Args:
            minibatch_advantages (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the advantage estimates for each sample in the minibatch.
            probability_ratio (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the ratio of probabilities under the new and old policies for each action.

        Returns:
            torch.Tensor: A scalar tensor containing the computed policy gradient loss.

        The PPO loss is defined as:
        L^CLIP(θ) = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        Where:
        - r_t(θ) is the probability ratio
        - A_t is the advantage estimate
        - ε is the surrogate_clip_threshold
        """

        # L^PG(θ) = r_t(θ) * A_t
        # This is the standard policy gradient objective. It encourages
        # the policy to increase the probability of actions that led to higher
        # advantages (i.e., performed better than expected).
        unclipped_pg_obj = minibatch_advantages * probability_ratio

        # L^CLIP(θ) = clip(r_t(θ), 1-ε, 1+ε) * A_t
        # This limits how much the policy can change for each action.
        # If an action's probability increased/decreased too much compared to
        # the old policy, we clip it. This prevents drastic policy changes,
        # promoting more stable learning.
        clipped_pg_obj = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )

        policy_gradient_loss = -torch.min(unclipped_pg_obj, clipped_pg_obj)

        return policy_gradient_loss

    def calculate_value_function_loss(
        self, new_value, computed_returns, previous_value_estimates, minibatch_indices, mask
    ):
        """
        Calculate the value function loss, optionally with clipping, for the value function approximation.
        It uses either a simple MSE loss or a clipped version similar to the policy loss clipping
        in PPO. When clipping is enabled, it uses the maximum of clipped and unclipped losses.
        The clipping helps to prevent the value function from changing too much in a single update.


        Args:
            new_value (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the new value estimates for the sampled states.
            computed_returns (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed returns for each step in the rollout.
            previous_value_estimates (torch.Tensor): Tensor of shape (batch_size,)
                containing the value estimates from the previous iteration.
            minibatch_indices (np.array): Array of indices for the current minibatch.

        Returns:
            torch.Tensor: A scalar tensor containing the computed value function loss.

        The value function loss is defined as:
        If clipping is enabled:
        L^VF = 0.5 * E[max((V_θ(s_t) - R_t)^2, (clip(V_θ(s_t) - V_old(s_t), -ε, ε) + V_old(s_t) - R_t)^2)]
        If clipping is disabled:
        L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]

        Where:
        - V_θ(s_t) is the new value estimate
        - R_t is the computed return
        - V_old(s_t) is the old value estimate
        - ε is the surrogate_clip_threshold
        """

        if self.clip_value_function_loss:
            # L^VF_unclipped = (V_θ(s_t) - R_t)^2
            # This is the standard MSE loss, pushing the value estimate
            # towards the actual observed returns.
            unclipped_vf_loss = (new_value - mask * computed_returns[minibatch_indices]) ** 2

            # V_clipped = V_old(s_t) + clip(V_θ(s_t) - V_old(s_t), -ε, ε)
            # This limits how much the value estimate can change from its
            # previous value, promoting stability in learning.
            clipped_value_diff = torch.clamp(
                new_value - previous_value_estimates[minibatch_indices],
                -self.surrogate_clip_threshold,
                self.surrogate_clip_threshold,
            )
            clipped_value = (
                previous_value_estimates[minibatch_indices] + clipped_value_diff
            )

            # L^VF_clipped = (V_clipped - R_t)^2
            # This loss encourages updates within the clipped range, preventing drastic changes to the value function.
            clipped_vf_loss = (clipped_value - mask * computed_returns[minibatch_indices]) ** 2

            # L^VF = max(L^VF_unclipped, L^VF_clipped)
            # By taking the maximum, we choose the more pessimistic (larger) loss.
            # This ensures we don't ignore large errors outside the clipped range
            # while still benefiting from clipping's stability.
            v_loss_max = torch.max(unclipped_vf_loss, clipped_vf_loss)

            # The 0.5 factor simplifies the gradient of the squared error loss,
            # as it cancels out with the 2 from the derivative of x^2.
            value_function_loss = 0.5 * v_loss_max.mean()
        else:
            # If not clipping, use simple MSE loss
            # L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]
            # Intuition: Without clipping, we directly encourage the value function
            # to predict the observed returns as accurately as possible.
            value_function_loss = (
                0.5 * ((new_value - mask * computed_returns[minibatch_indices]) ** 2).mean()
            )

        return value_function_loss