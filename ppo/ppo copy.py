from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy


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
        for param_group in self.optimizer.param_groups:
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
            non_zero_rews = infos['episode']['r'][np.nonzero(infos['episode']['r'])[0]]
            non_zero_lens = infos['episode']['l'][np.nonzero(infos['episode']['l'])[0]]
            non_zero_comps = []
            for i in range(self.reward_size):
                non_zero_comps.append(infos['episode'][f'r_{i}'][np.nonzero(infos['episode'][f'r_{i}'])[0]])
            print(
                f"global_step={global_step}, episodic_return={non_zero_rews.mean()}",
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

            self.writer.add_scalar(
                "losses/kl_divergence", update_results["old_approx_kl"], global_step
            )
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


class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        reward_size = 1,
        learning_rate=3e-4,
        num_rollout_steps=2048,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.01,
        value_function_loss_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=32,
        normalize_advantages=True,
        clip_value_function_loss=True,
        target_kl=None,
        anneal_lr=True,
        seed=1,
        logger=None,
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
        self.advantage_weights_network = nn.Sequential(copy.deepcopy(agent.critic), nn.Softmax())
        
        self.envs = envs
        self.optimizer = optimizer
        self.optimizer.add_param_group({'params': self.advantage_weights_network.parameters()})
        self.seed = seed

        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.batch_size = num_rollout_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches
        self.reward_size = reward_size

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_clip_threshold = surrogate_clip_threshold
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_function_loss_coefficient = value_function_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.normalize_advantages = normalize_advantages
        self.clip_value_function_loss = clip_value_function_loss
        self.target_kl = target_kl

        self.device = next(agent.parameters()).device

        self.anneal_lr = anneal_lr
        self.initial_lr = learning_rate

        self.lr_scheduler = None
        self._global_step = 0
        self.logger = logger or PPOLogger()

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
        num_policy_updates = int(np.round(total_timesteps / (self.num_rollout_steps)))

        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_policy_updates)

        next_observation, is_next_observation_terminal = self._initialize_environment()

        # Initialize logging variables
        self._global_step = 0

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
                next_observation,
                is_next_observation_terminal,
            ) = self.collect_rollouts(next_observation, is_next_observation_terminal)

            update_results = self.update_policy(
                batch_observations,
                batch_log_probabilities,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_values,
            )

            self.logger.log_policy_update(update_results, self._global_step)
            print(self.agent.actor_logstd)

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
        # print(initial_observation)
        initial_observation = torch.Tensor(initial_observation).to(self.device)
        is_initial_observation_terminal = torch.zeros(self.num_envs).to(self.device)
        return initial_observation, is_initial_observation_terminal

    def collect_rollouts(self, next_observation, is_next_observation_terminal):
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
            observation_values,
        ) = self._initialize_storage()

        for step in range(self.num_rollout_steps//self.num_envs):
            # Store current observation
            collected_observations[step] = next_observation
            is_episode_terminated[step] = is_next_observation_terminal

            with torch.no_grad():
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    next_observation
                )
                value = self.agent.estimate_value_from_observation(next_observation)

                observation_values[step] = value
            actions[step] = action
            action_log_probabilities[step] = logprob

            # Execute the environment and store the data
            next_observation, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            self._global_step += self.num_envs
            rewards[step] = torch.as_tensor(reward, device=self.device)
            is_next_observation_terminal = np.logical_or(terminations, truncations)

            next_observation, is_next_observation_terminal = (
                torch.as_tensor(
                    next_observation, dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    is_next_observation_terminal,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

            self.logger.log_rollout_step(infos, self._global_step)

        # Estimate the value of the next state (the state after the last collected step) using the current policy
        # This value will be used in the GAE calculation to compute advantages
        with torch.no_grad():
            next_value = self.agent.estimate_value_from_observation(
                next_observation
            )

            advantages, returns = self.compute_advantages(
                rewards,
                observation_values,
                is_episode_terminated,
                next_value,
                is_next_observation_terminal,
            )

        # Flatten the batch for easier processing in the update step
        (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
        ) = self._flatten_rollout_data(
            collected_observations,
            action_log_probabilities,
            actions,
            advantages,
            returns,
            observation_values,
        )


        # Return the collected and computed data for the policy update step
        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
            next_observation,
            is_next_observation_terminal,
        )

    def _initialize_storage(self):
        collected_observations = torch.zeros(
            (self.num_rollout_steps//self.num_envs, self.num_envs)
            + self.envs.observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.num_rollout_steps//self.num_envs, self.num_envs)
            + self.envs.action_space.shape
        ).to(self.device)
        action_log_probabilities = torch.zeros(
            (self.num_rollout_steps//self.num_envs, self.num_envs)
        ).to(self.device)
        rewards = torch.zeros((self.num_rollout_steps//self.num_envs, self.num_envs, self.reward_size)).to(self.device)
        is_episode_terminated = torch.zeros((self.num_rollout_steps//self.num_envs, self.num_envs)).to(
            self.device
        )
        observation_values = torch.zeros((self.num_rollout_steps//self.num_envs, self.num_envs, self.reward_size)).to(self.device)

        return (
            collected_observations,
            actions,
            action_log_probabilities,
            rewards,
            is_episode_terminated,
            observation_values,
        )

    def compute_advantages(
        self,
        rewards,                        # shape [num_steps, batch_size, reward_size]
        values,                         # shape [num_steps, batch_size, reward_size]
        is_observation_terminal,        # shape [num_steps, batch_size]
        next_value,                     # shape [batch_size, reward_size]
        is_next_observation_terminal,   # shape [batch_size]
    ):
        """
        Computes generalized advantage estimates (GAE) and returns
        for a multi-dimensional (vector) reward.

        Args:
            rewards:  (T, B, R) float tensor of rewards for T steps, B envs, R dims
            values:   (T, B, R) float tensor of value predictions
            is_observation_terminal: (T, B) 1 if terminal at step t, else 0
            next_value: (B, R) value prediction for t=T (bootstrap)
            is_next_observation_terminal: (B,) terminal indicator for the next obs after last step
        Returns:
            advantages: (T, B, R) GAE estimates for each dimension
            returns:    (T, B, R) GAE-based returns for each dimension
        """

        # Allocate storage
        advantages = torch.zeros_like(rewards).to(self.device)  # (T,B,R)
        returns = torch.zeros_like(rewards).to(self.device)      # (T,B,R)

        # We'll accumulate GAE in a "running" variable with shape (B,R)
        gae_running_value = torch.zeros_like(next_value)  # (B,R)

        for t in reversed(range(self.num_rollout_steps//self.num_envs)):
            if t == self.num_rollout_steps//self.num_envs - 1:
                # At the very last step in the rollout, compare to 'next_value'
                #   and the 'is_next_observation_terminal'
                next_nonterminal = 1.0 - is_next_observation_terminal  # shape (B,)
                next_values = next_value  # shape (B,R)
            else:
                # Otherwise, GAE looks one step ahead in the buffer
                next_nonterminal = 1.0 - is_observation_terminal[t + 1]  # shape (B,)
                next_values = values[t + 1]  # shape (B,R)

            # Expand next_nonterminal from shape (B,) -> (B,1) to broadcast over reward dimensions
            next_nonterminal = next_nonterminal.unsqueeze(-1)  # (B,1)

            # "delta" = one-step TD error for each reward dimension
            #   rewards[t]          => shape (B,R)
            #   next_values         => shape (B,R)
            #   values[t]           => shape (B,R)
            delta = (
                rewards[t]
                + self.gamma * next_values * next_nonterminal
                - values[t]
            )
            # Accumulate GAE(λ)
            gae_running_value = delta + (
                self.gamma * self.gae_lambda * next_nonterminal * gae_running_value
            )
            advantages[t] = gae_running_value  # store it

        # Finally, returns = V + advantage
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
    ):
        batch_observations = collected_observations.reshape(
            (-1,) + self.envs.single_observation_space.shape
        )
        batch_log_probabilities = action_log_probabilities.reshape(-1)
        batch_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        batch_advantages = advantages.reshape(-1, self.reward_size)
        batch_returns = returns.reshape(-1, self.reward_size)
        batch_values = observation_values.reshape(-1, self.reward_size)

        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
        )

    def update_policy(
        self,
        collected_observations,
        collected_action_log_probs,
        collected_actions,
        computed_advantages,
        computed_returns,
        previous_value_estimates,
    ):
        """
        Update the policy and value function using the collected rollout data.

        This method implements the core PPO algorithm update step. It performs multiple
        epochs of updates on minibatches of the collected rollout data, optimizing both the policy
        and value function.

        Args:
            collected_observations (torch.Tensor): Tensor of shape (batch_size, *obs_shape)
                containing the observations from the rollout.
            collected_action_log_probs (torch.Tensor): Tensor of shape (batch_size,)
                containing the log probabilities of the actions taken during the rollout.
            collected_actions (torch.Tensor): Tensor of shape (batch_size, *action_shape)
                containing the actions taken during the rollout.
            computed_advantages (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed advantages for each step in the rollout.
            computed_returns (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed returns (sum of discounted rewards) for each step.
            previous_value_estimates (torch.Tensor): Tensor of shape (batch_size,)
                containing the value estimates from the previous iteration.

        Returns:
            dict: A dictionary containing various statistics about the update process:
                - policy_loss: The final policy gradient loss.
                - value_loss: The final value function loss.
                - entropy_loss: The entropy loss, encouraging exploration.
                - old_approx_kl: The old approximate KL divergence.
                - approx_kl: The new approximate KL divergence.
                - clipping_fraction: The fraction of policy updates that were clipped.
                - explained_variance: A measure of how well the value function explains
                                    the observed returns.

        The method performs the following key steps:
        1. Iterates over the data for multiple epochs, shuffling at each epoch.
        2. For each minibatch:
            a. Computes new action probabilities and values.
            b. Calculates the policy ratio and clipped policy objective.
            c. Computes the value function loss, optionally with clipping.
            d. Calculates the entropy bonus to encourage exploration.
            e. Combines losses and performs a gradient update step.
        3. Optionally performs early stopping based on KL divergence.
        4. Computes and returns various statistics about the update process.

        This implementation uses the PPO clipped objective, which helps to constrain
        the policy update and improve training stability. It also uses advantage
        normalization and gradient clipping for further stability.
        """
        # Prepare for minibatch updates
        batch_size = self.num_rollout_steps
        batch_indices = np.arange(batch_size)

        # Track clipping for monitoring policy update magnitude
        clipping_fractions = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_indices)

            # Minibatch updates help stabilize training and can be more compute-efficient
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Get updated action probabilities and values for the current policy
                current_policy_log_probs, entropy = (
                    self.agent.compute_action_log_probabilities_and_entropy(
                        collected_observations[minibatch_indices],
                        collected_actions[minibatch_indices],
                    )
                )
                new_value = self.agent.estimate_value_from_observation(
                    collected_observations[minibatch_indices]
                )

                # Calculate the probability ratio for importance sampling
                # This allows us to use old trajectories to estimate the new policy's performance)
                # collected_action_log_probs = collected_action_log_probs.reshape(len(minibatch_indices), -1)
                # print(collected_action_log_probs.shape)
                log_probability_ratio = (
                    current_policy_log_probs
                    - collected_action_log_probs[minibatch_indices]
                )
                probability_ratio = log_probability_ratio.exp()

                # Estimate KL divergence for early stopping
                # This helps prevent the new policy from diverging too much from the old policy
                # approx_kl http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    old_approx_kl = (-log_probability_ratio).mean()
                    approx_kl = ((probability_ratio - 1) - log_probability_ratio).mean()

                    # Track the fraction of updates being clipped
                    # High clipping fraction might indicate too large policy updates
                    clipping_fractions += [
                        (
                            (probability_ratio - 1.0).abs()
                            > self.surrogate_clip_threshold
                        )
                        .float()
                        .mean()
                        .item()
                    ]
                
                minibatch_advantages = computed_advantages[minibatch_indices]

                # In update_policy()
                # if self.normalize_advantages:
                #     # Normalize per objective
                #     minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean(dim=0)) / \
                #                         (minibatch_advantages.std(dim=0) + 1e-8)
                # print("BEFORE", computed_advantages[minibatch_indices][0, :])

                policy_gradient_loss = self.calculate_policy_gradient_loss(
                    collected_observations[minibatch_indices], minibatch_advantages, probability_ratio
                )
                value_function_loss = self.calculate_value_function_loss(
                    new_value,
                    computed_returns,
                    previous_value_estimates,
                    minibatch_indices,
                )
                # Entropy encourages exploration by penalizing overly deterministic policies
                entropy_loss = entropy.mean()

                # Combine losses: minimize policy and value losses, maximize entropy
                loss = (
                    policy_gradient_loss
                    - self.entropy_loss_coefficient
                    * entropy_loss  # subtraction here to maximise entropy (exploration)
                    + value_function_loss * self.value_function_loss_coefficient
                )

                # Perform backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping helps prevent too large policy updates
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping based on KL divergence, if enabled, done at epoch level for stability
            # This provides an additional safeguard against too large policy updates
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        predicted_values, actual_returns = (
            previous_value_estimates.cpu().numpy(),
            computed_returns.cpu().numpy(),
        )
        observed_return_variance = np.var(actual_returns)
        # explained variance measures how well the value function predicts the actual returns
        explained_variance = (
            np.nan
            if observed_return_variance == 0
            else 1
            - np.var(actual_returns - predicted_values) / observed_return_variance
        )

        return {
            "policy_loss": policy_gradient_loss.item(),
            "value_loss": value_function_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipping_fractions": np.mean(clipping_fractions),
            "explained_variance": explained_variance,
        }
    def calculate_policy_gradient_loss(self, collected_observations, minibatch_advantages, probability_ratio):
        if self.normalize_advantages:
            # First per-dimension
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean(dim = 0)) / (minibatch_advantages.std(dim = 0) + 1e-8)
            # Then scalarized
            # scalarized = (advantages * weights).sum(dim=1)
            # scalarized = (scalarized - scalarized.mean()) / (scalarized.std() + 1e-8)

        # print(minibatch_advantages[0, :])
        unclipped_pg_obj = probability_ratio.reshape(-1, 1) * minibatch_advantages 
        clipped_pg_obj = torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold
        ).reshape(-1, 1) * minibatch_advantages 
        return torch.mean(-torch.minimum(unclipped_pg_obj, clipped_pg_obj), dim = 0).mean()

    def calculate_value_function_loss(self, new_value, computed_returns, previous_value_estimates, minibatch_indices):
        # Extract the minibatch data for returns and old values.
        old_values = previous_value_estimates[minibatch_indices]  # shape: (minibatch_size, reward_size)
        returns = computed_returns[minibatch_indices]             # shape: (minibatch_size, reward_size)
        
        old_values = old_values.sum(dim = 1)
        returns = returns.sum(dim = 1)
        new_value = new_value.sum(dim = 1)
        if self.clip_value_function_loss:
            # Clip the new_value based on the old value estimates
            clipped_values = old_values + torch.clamp(
                new_value - old_values,
                -self.surrogate_clip_threshold,
                self.surrogate_clip_threshold
            )
            # Compute element-wise squared errors and take the maximum for clipping
            v_loss = 0.5 * torch.maximum((new_value - returns) ** 2, (clipped_values - returns) ** 2)
        else:
            v_loss = 0.5 * (new_value - returns) ** 2

        # Return the mean loss over all samples and reward dimensions
        return v_loss.mean()
