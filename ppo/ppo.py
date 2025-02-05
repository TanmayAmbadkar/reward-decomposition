
from ppo.actor_critic import ActorCritic
from ppo.dataset import RecurrentPPODataset

from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, BatchSampler, RandomSampler
import copy

class PPO:
    """
    Implementation of PPO based on:
    https://github.com/vwxyzjn/cleanrl
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/    
    """

    def __init__(self, 
        actor_critic: ActorCritic,
        recurrent: bool,
        num_steps: int,
        num_envs: int,
        reward_size: int,
        obs_size_actor: int,
        obs_size_critic: int,
        action_size: int,
        learning_rate: float,
        num_traj_minibatch: int,
        bptt_len: int,
        update_epochs: int,
        use_gae: bool,
        gae_lambda: float,
        gamma: float,
        clip_vloss: bool,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        use_norm_adv: bool,
        max_grad_norm: float,
        target_kl: Union[float, None], # not used when None
        learning_rate_gamma: Union[float, None], # not used when None
        min_seq_size: Union[int, None], # not used when None
        device: str):
        """
        actor_critic: instance of ActorCritic class
        recurrent: use recurrent PPO update step (training on entire trajectories)
        num_steps: number of steps to run in each environment per policy rollout
        num_envs: number of parallel game environments
        observation_size: shape of input to actor/critic
        action_size: shape of output of actor
        learning_rate: learning rate of the optimizer
        num_traj_minibatch: number of trajectories in each minibatch during update
        bptt_len: number of steps to consider when calculating loss using bptt
        update_epochs: number of epochs to update the policy
        use_gae: use GAE for advantage computation
        gae_lambda: lambda for the general advantage estimation
        gamma: discount factor gamma
        clip_vloss: use a clipped loss for the value function
        clip_coef: surrogate clipping coefficient
        ent_coef: entropy coefficient
        vf_coef: value function coefficient
        use_norm_adv: use advantages normalization
        max_grad_norm: maximum norm for the gradient clipping
        target_kl: if not None, target KL divergence threshold
        learning_rate_gamma: if not None, multiplicative factor of learning rate decay
        min_seq_size: minimum length of sequence during training, all shorter sequences are discarted.
            (only used for recurrent PPO).
        """

        self._actor_critic = actor_critic
        self._weight_advantage = torch.nn.Sequential(copy.deepcopy(self._actor_critic._critic), torch.nn.Softmax())
        self._recurrent = recurrent
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._reward_size = reward_size
        self._obs_size_actor = obs_size_actor
        self._obs_size_critic = obs_size_critic
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._num_traj_minibatch = num_traj_minibatch
        self._bptt_len = bptt_len
        self._update_epochs = update_epochs
        self._use_gae = use_gae
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._clip_vloss = clip_vloss
        self._clip_coef = clip_coef
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self._use_norm_adv = use_norm_adv
        self._max_grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._min_seq_size = min_seq_size
        self._device = device

        # ignore bptt parameter if using non-recurrent PPO
        if not self._recurrent:
            self._bptt_len = self._num_steps

        # batch size for flat trajectories
        self._batch_size = int(self._num_envs * self._num_steps)
        
        # optimizer
        # self.weights = torch.nn.Parameter(
        #     torch.ones(self._reward_size, device=self._device) / self._reward_size
        # )
        
        # Update optimizer to include weights
        self._optimizer = torch.optim.Adam(
            list(self._actor_critic.parameters()) + list(self._weight_advantage.parameters()),  # Include weights
            lr=self._learning_rate, 
            eps=1e-5
        )
        if learning_rate_gamma is None:
            learning_rate_gamma = 1
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, learning_rate_gamma, last_epoch=-1)

        # storage
        # TODO add option to save collected info in cpu and then transfer only batches to GPU during training
        self._obs_actor = torch.zeros((self._num_steps, self._num_envs, self._obs_size_actor)).to(self._device)
        self._obs_critic = torch.zeros((self._num_steps, self._num_envs, self._obs_size_critic)).to(self._device)
        self._actions = torch.zeros((self._num_steps, self._num_envs, self._action_size)).to(self._device)
        self._logprobs = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._rewards = torch.zeros((self._num_steps, self._num_envs, self._reward_size)).to(self._device)
        self._dones = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._values = torch.zeros((self._num_steps, self._num_envs, self._reward_size)).to(self._device)
        self._advantages = torch.zeros((self._num_steps, self._num_envs, self._reward_size)).to(self._device)
        self._returns = torch.zeros((self._num_steps, self._num_envs, self._reward_size)).to(self._device)
        self._step = 0

    def _reset_storage(self):
        self._obs_actor[:] = 0.
        self._obs_critic[:] = 0.
        self._actions[:] = 0.
        self._logprobs[:] = 0.
        self._rewards[:] = 0.
        self._dones[:] = 0.
        self._values[:] = 0.
        self._advantages[:] = 0.
        self._returns[:] = 0.
        self._step = 0

    def act(self, obs_actor, obs_critic, actor_state=None, critic_state=None):
        with torch.no_grad():
            actions, logprobs, _, actor_state = self._actor_critic.get_action(obs_actor, actor_state)
            values, critic_state = self._actor_critic.get_value(obs_critic, critic_state)
            
        return actions, logprobs, values, actor_state, critic_state

    def set_step(self, obs_actor, obs_critic, actions, logprobs, rewards, dones, values):
        self._obs_actor[self._step] = obs_actor
        self._obs_critic[self._step] = obs_critic
        self._actions[self._step] = actions
        self._logprobs[self._step] = logprobs.flatten()
        self._rewards[self._step] = rewards
        self._dones[self._step] = dones.flatten()
        self._values[self._step] = values
        # increment step
        self._step += 1

    def compute_returns(self, obs_critic, dones, critic_state=None):
        # bootstrap value if not done
        with torch.no_grad():
            value, _ = self._actor_critic.get_value(
                obs_critic,
                critic_state)

            value = value.reshape(1, -1)
            if self._use_gae:
                self._advantages = torch.zeros((self._num_steps, self._num_envs, self._reward_size)).to(self._device)
                lastgaelam = 0
                for t in reversed(range(self._num_steps)):
                    if t == self._num_steps - 1:
                        nextnonterminal = 1.0 - dones.float()
                        nextvalues = value
                    else:
                        nextnonterminal = 1.0 - self._dones[t + 1]
                        nextvalues = self._values[t + 1]
                    delta = self._rewards[t] + self._gamma * nextvalues * nextnonterminal - self._values[t]
                    self._advantages[t] = lastgaelam = delta + self._gamma * self._gae_lambda * nextnonterminal * lastgaelam
                self._returns = self._advantages + self._values
            else:
                self._returns = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
                for t in reversed(range(self._num_steps)):
                    if t == self._num_steps - 1:
                        nextnonterminal = 1.0 - dones.float()
                        next_return = value
                    else:
                        nextnonterminal = 1.0 - self._dones[t + 1]
                        next_return = self._returns[t + 1]
                    self._returns[t] = self._rewards[t] + self._gamma * nextnonterminal * next_return
                self._advantages = self._returns - self._values

    def update(self, initial_actor_state=None, initial_critic_state=None, calc_extra_loss=None):

        # Preprocess collected data, create a dataset and specify sampling strategy
        if self._recurrent:
            # Recurrent approach:
            # 1. Split obs and other collected data into lists of single trajectories
            #    (split sequences everywhere there's a done)
            # 2. Select random trajectories and build batches of size `num_traj_minibatch`
            # 3. Get a batch of trajectories and join them by padding the trajectories
            # 4. After inference, apply mask to output to select only valid (non-padded) steps
            # Advantages of this approach:
            # - Fast inference of entire trajectories
            # - Memory footprint is limited to tensor of size `num_traj_minibatch*num_steps`
            #   (upper bound, as collected trajectories can be smaller)
            obs_actor, traj_mask, initial_actor_state, initial_critic_state,\
            [obs_critic, actions, logprobs, advantages, returns, values] \
                = self.split_sequences(
                    self._obs_actor, self._dones, initial_actor_state, initial_critic_state,
                    [self._obs_critic, self._actions, self._logprobs, self._advantages, self._returns, self._values])

            # has to be the same order as below
            # print(obs_actor.shape, obs_critic.shape, a26ctions.shape, logprobs.shape, advantages.shape, returns.shape, values.shape)
            dataset = RecurrentPPODataset(obs_actor, obs_critic, actions, logprobs, advantages, returns, values)
            sampler = BatchSampler(RandomSampler(dataset), self._num_traj_minibatch, False)
        else:
            # Non-recurrent approach:
            # 1. Reshape experience into batch of individual steps
            # 2. Select batches of steps of size `num_traj_minibatch*num_steps`
            #    (approximately same number of steps in each batch as the recurrent approach)
            obs_actor = self._obs_actor.reshape((-1, self._obs_size_actor))
            obs_critic = self._obs_critic.reshape((-1, self._obs_size_critic))
            actions = self._actions.reshape((-1, self._action_size))
            logprobs = self._logprobs.reshape(-1)
            advantages = self._advantages.reshape(-1, self._reward_size)
            returns = self._returns.reshape(-1, self._reward_size)
            values = self._values.reshape(-1, self._reward_size)
            # force states to None
            initial_actor_state = None
            initial_critic_state = None

            # has to be the same order as below
            # print(obs_actor.shape, obs_critic.shape actions.shape, logprobs.shape, advantages.shape, returns.shape, values.shape)95
            dataset = TensorDataset(obs_actor, obs_critic, actions, logprobs, advantages, returns, values)
            sampler = BatchSampler(RandomSampler(dataset), self._num_traj_minibatch, False)

        # clear variable names to avoid bugs
        del obs_actor, obs_critic, actions, logprobs, advantages, returns, values

        losses = 0.
        v_losses = 0.
        pg_losses = 0.
        entropy_losses = 0.
        extra_losses = 0.

        for epoch in range(self._update_epochs):
            for batch_indices in sampler:

                # get batch data
                # has to have same order as above
                obs_actor_b, obs_critic_b, actions_b, logprobs_b, advantages_b, returns_b, values_b \
                    = dataset[batch_indices]
                actor_state = initial_actor_state[..., batch_indices, :] if torch.is_tensor(initial_actor_state) else None
                critic_state = initial_critic_state[..., batch_indices, :] if torch.is_tensor(initial_critic_state) else None

                for subseq_index in range(self._num_steps//self._bptt_len):
                    if self._recurrent:
                        start_step = subseq_index*self._bptt_len
                        end_step = (subseq_index+1)*self._bptt_len
                        # stop if sequence size of batch is reached
                        if start_step >= obs_actor_b.shape[0]:
                            break
                    else:
                        start_step = 0
                        end_step = -1
                    obs_actor = obs_actor_b[..., start_step:end_step, :]
                    obs_critic = obs_critic_b[..., start_step:end_step, :]
                    actions = actions_b[..., start_step:end_step, :]
                    logprobs = logprobs_b[..., start_step:end_step]
                    advantages = advantages_b[..., start_step:end_step, :]
                    returns = returns_b[..., start_step:end_step, :]
                    values = values_b[..., start_step:end_step, :]
                    if self._recurrent:
                        traj_mask_sub = traj_mask[start_step:end_step, batch_indices]

                    _, newlogprob, entropy, actor_state = self._actor_critic.get_action(
                        obs=obs_actor,
                        hidden_states=actor_state,
                        action=actions)

                    newvalue, critic_state = self._actor_critic.get_value(
                        obs=obs_critic,
                        hidden_states=critic_state)

                    if self._recurrent:
                        # apply mask to select only unpadded data
                        if calc_extra_loss is not None:
                            obs_actor = self.apply_mask(obs_actor, traj_mask_sub, seq_dim=0, batch_dim=1)
                            obs_critic = self.apply_mask(obs_critic, traj_mask_sub, seq_dim=0, batch_dim=1)
                            actions = self.apply_mask(actions, traj_mask_sub, seq_dim=0, batch_dim=1)
                        logprobs = self.apply_mask(logprobs, traj_mask_sub, seq_dim=0, batch_dim=1)
                        newlogprob = self.apply_mask(newlogprob, traj_mask_sub, seq_dim=0, batch_dim=1)
                        advantages = self.apply_mask(advantages, traj_mask_sub, seq_dim=0, batch_dim=1)
                        returns = self.apply_mask(returns, traj_mask_sub, seq_dim=0, batch_dim=1)
                        values = self.apply_mask(values, traj_mask_sub, seq_dim=0, batch_dim=1)
                        newvalue = self.apply_mask(newvalue, traj_mask_sub, seq_dim=0, batch_dim=1)
                        entropy = self.apply_mask(entropy, traj_mask_sub, seq_dim=0, batch_dim=1)

                    # extra loss summed to PPO loss
                    if calc_extra_loss is not None:
                        extra_loss = calc_extra_loss(traj_mask_sub, obs_actor, obs_critic, actions, logprobs, advantages, returns, values)
                    else:
                        extra_loss = 0.

                    logratio = newlogprob - logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        # clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                    # if self._use_norm_adv:
                    #     advantages = (advantages - advantages.sum(dim = 1).mean()) / (advantages.sum(dim = 1).std(unbiased=False) + 1e-6)
                    # 
                    weights = self._weight_advantage(obs_critic)
                    weighted_adv = (advantages * weights).sum(dim=1)
                    if self._use_norm_adv:
                        weighted_adv = (weighted_adv - weighted_adv.mean()) / (weighted_adv.std() + 1e-8)
        
                    # Add: Weighted advantage calculation
                    # weights = F.softmax(self.weights, dim=0)  # Ensure convex combination
                    # weighted_advantages = torch.sum(advantages * weights, dim=1)  # Shape: (batch_size,)
                    pg_loss1_per_obj = ratio * weighted_adv
                    pg_loss2_per_obj = torch.clamp(ratio, 1-self._clip_coef, 1+self._clip_coef) * weighted_adv
                    pg_loss = -torch.min(pg_loss1_per_obj, pg_loss2_per_obj).mean()
                    # pg_loss = (pg_loss_per_obj * weights).mean()  # Weighted mean

                    # Value loss
                    newvalue = newvalue.view(-1, self._reward_size)
                    if self._clip_vloss:
                        v_loss_unclipped = (newvalue - returns).pow(2)
                        v_clipped = values + torch.clamp(newvalue - values, -self._clip_coef, self._clip_coef)
                        v_loss_clipped = (v_clipped - returns).pow(2)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * (newvalue - returns).pow(2).mean()

                    # entropy_loss = entropy.mean()
                    # print(entropy.shape)
                    entropy_loss = (entropy.unsqueeze(1) * weights).mean() 
                    loss = pg_loss + self._ent_coef * entropy_loss + v_loss * self._vf_coef + extra_loss

                    self._optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._max_grad_norm)
                    grads = []
                    for param in self._actor_critic.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)

                    # Project conflicting gradients (pseudo-code)
                    # project_conflicting_gradients(grads)
                    torch.nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), 
                        self._max_grad_norm
                    )

                    # Update gradients
                    idx = 0
                    for param in self._actor_critic.parameters():
                        if param.grad is not None:
                            param.grad = grads[idx:idx+param.numel()].view_as(param.grad)
                            idx += param.numel()
                    self._optimizer.step()
                    # detach tensors that we want to use in the next iteration
                    actor_state = actor_state.detach() if torch.is_tensor(actor_state) else None
                    critic_state = critic_state.detach() if torch.is_tensor(critic_state) else None

                    losses += loss.item()
                    v_losses += self._vf_coef*v_loss.item()
                    pg_losses += pg_loss.item()
                    entropy_losses += self._ent_coef*entropy_loss.item()
                    if calc_extra_loss is not None:
                        extra_losses += extra_loss.item()

            if self._target_kl is not None and approx_kl > self._target_kl:
                break

        # reset storage
        self._reset_storage()
        self._scheduler.step()

        return losses, v_losses, pg_losses, entropy_losses, extra_losses, self._scheduler.get_last_lr()[0]

    def split_sequences(self, obs, dones, actor_states, critic_states, tensors):
        """
        Splits a batch of sequences containing several trajectories into several
        sequences that contain only one trajectory.
        This is accomplished by splitting the input sequence everywhere where
        there's a done.

        Original sequences: [[a1, a2, a3, a4 | a5, a6],
                             [b1, b2 | b3, b4, b5 | b6]]

        Split sequences:    [[a1, a2, a3, a4],
                             [a5, a6],
                             [b1, b2],
                             [b3, b4, b5],
                             [b6]]

        These split sequences can be then joined together into a batch by
        padding them. This enables a computationally efficient way to process
        entire batches of sequences, without having to iterate over the
        individual steps.

        Attention: this method can lead to very large batch sizes. This becomes
        a problem if input contains many small trajectories. For this reason,
        trajectories shorter than `_min_seq_size` are removed.
        """

        # input must be (seq_len, batch_size, obs_size)
        assert obs.ndim == 3
        seq_len = obs.shape[0]
        batch_size = obs.shape[1]

        # SPLIT TRAJECTORIES
        # clone, as this tensor will be modified
        split_indices = dones.clone()
        # force split at the start of trajectory
        split_indices[0] = 1
        # transpose and flatten tensors to reorder entries
        # required later for splitting the observations correctly
        # for tensor with indices (sequence, batch):
        #   old shape: [obs(0,0), obs(0,1), ...]
        #   new shape: [obs(0,0), obs(1,0), ...]
        # causes data to be copied
        obs = obs.movedim(0, 1).flatten(0, 1)
        dones = dones.movedim(0, 1).flatten(0, 1)
        split_indices = split_indices.movedim(0, 1).flatten(0, 1)
        # get indices (split requires tensor in CPU)
        split_indices = torch.argwhere(split_indices).flatten().cpu()
        # split trajectories at dones or start of trajectory
        obs_split = list(torch.tensor_split(obs, split_indices, dim=0))
        # remove first element if empty
        if obs_split[0].numel() == 0:
            obs_split = obs_split[1:]

        # REMOVE SHORT TRAJECTORIES
        # get start and end of each trajectory
        # for this we need to add the end of last trajectory to split_indices
        start_end_indices = torch.cat((split_indices, torch.tensor([batch_size*seq_len])))
        # get original length of each trajectory
        traj_lengths = torch.diff(start_end_indices)
        # select trajectories to remove (smaller than x steps)
        remove_mask = traj_lengths < self._min_seq_size
        remove_indices = torch.argwhere(remove_mask).flatten()
        keep_indices = torch.argwhere(torch.logical_not(remove_mask)).flatten()
        # update trajectory lengths
        traj_lengths = traj_lengths[keep_indices]
        # filter trajectories only if needed
        if torch.any(remove_mask):
            # reverse indices otherwise only the first index is correctly removed
            for index in reversed(remove_indices):
                del obs_split[index]
            assert len(obs_split) > 0

        # OTHER TENSORS
        # apply same operations as above to the other tensors
        tensors_split = []
        for t in tensors:
            t = t.movedim(0, 1).flatten(0, 1)
            t_split = list(torch.tensor_split(t, split_indices, dim=0))
            if t_split[0].numel() == 0:
                t_split = t_split[1:]
            if torch.any(remove_mask):
                # reverse indices otherwise only the first index is correctly removed
                for index in reversed(remove_indices):
                    del t_split[index]
                assert len(t_split) > 0
            tensors_split.append(t_split)

        # GENERATE HIDDEN STATES
        # get indices of started trajectories
        # (the ones that have an initial hidden state)
        started_traj_indices = seq_len*torch.arange(batch_size)
        # batches corresponding to started trajectories
        started_traj_batches = torch.argwhere(torch.isin(start_end_indices, started_traj_indices)).flatten()
        # generate mask for started trajectories that have a done
        # (these should be discarded)
        started_done_traj_mask = torch.logical_not(dones[started_traj_indices] > 0.5).cpu()
        # generate mask for new hidden states only with relevant batches
        new_batches_mask = torch.isin(keep_indices, started_traj_batches[started_done_traj_mask])
        # generate mask for hidden states of started trajectories
        started_batches_mask = torch.isin(started_traj_batches, keep_indices)
        started_batches_mask = torch.logical_and(started_batches_mask, started_done_traj_mask)
        # insert existing hidden states of started trajectories
        if actor_states != None:
            h_actor = self._actor_critic.init_hidden(len(obs_split))
            h_actor[..., new_batches_mask, :] = actor_states[..., started_batches_mask, :]
            actor_states = h_actor
        if critic_states != None:
            h_critic = self._actor_critic.init_hidden(len(obs_split))
            h_critic[..., new_batches_mask, :] = critic_states[..., started_batches_mask, :]
            critic_states = h_critic

        # GENERATE MASK
        # get mask of valid trajectories
        traj_mask = traj_lengths > torch.arange(0, seq_len).unsqueeze(1)

        print("Num trajectories before/after split: ", batch_size, "/", len(obs_split))

        return obs_split, traj_mask, actor_states, critic_states, tensors_split

    def apply_mask(self, tensor, mask, seq_dim=0, batch_dim=1):
        """
        TODO
        Joins tensors together by selecting only unpadded steps. Result is a
        flattened tensor of valid steps .
        """
        assert seq_dim < batch_dim  # otherwise algorithm does not work
        # create array with full size of output
        # (as if short trajectories were not removed)
        shape = list(tensor.shape)
        shape[seq_dim] = mask.shape[0]
        shape[batch_dim] = mask.shape[1]
        tensor_full = torch.zeros(shape, device=tensor.device)
        # move data
        tensor_full = tensor_full.movedim(seq_dim, 0).movedim(batch_dim, 1)
        tensor = tensor.movedim(seq_dim, 0).movedim(batch_dim, 1)
        tensor_full[:tensor.shape[0]] = tensor
        # apply mask to unpad outputs and join them
        tensor_full = tensor_full.movedim(0, 1)[mask.movedim(0, 1)]
        # move dimension back
        tensor_full = tensor_full.movedim(0, seq_dim)
        return tensor_full

    def state_dict(self):
        return {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._scheduler.load_state_dict(state_dict["scheduler"])

def project_conflicting_gradients(grads):
    for i in range(len(grads)):
        for j in range(i+1, len(grads)):
            if grads[i].dot(grads[j]) < 0:
                grads[i] -= grads[i].dot(grads[j]) * grads[j] / (grads[j].norm()**2 + 1e-8)
    return grads