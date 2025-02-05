
import copy
import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    """
    Supports any combination of actor/critic architecture (recurrent or
    non-recurrent).
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            recurrent_actor: bool,
            recurrent_critic: bool,
            actor_output_size: int,
            device: str):

        super().__init__()

        self._actor = actor
        self._critic = critic
        self._recurrent_actor = recurrent_actor
        self._recurrent_critic = recurrent_critic
        self._actor_output_size = actor_output_size

        self._log_std = torch.nn.Parameter(torch.zeros(actor_output_size, device=device))

    def get_action(self, obs, hidden_states, action=None):
        if self._recurrent_actor:
            assert hidden_states != None
            action_mean, hidden_states = self._actor(obs, hidden_states)
        else:
            action_mean = self._actor(obs)
            hidden_states = None

        probs = torch.distributions.Normal(action_mean, torch.exp(self._log_std))
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), hidden_states

    def get_action_inference(self, obs, hidden_states):
        if self._recurrent_actor:
            assert hidden_states != None
            action, hidden_states = self._actor(obs, hidden_states)
        else:
            action = self._actor(obs)
            hidden_states = None
        return action, hidden_states

    def get_value(self, obs, hidden_states):
        if self._recurrent_critic:
            assert hidden_states != None
            value, hidden_states = self._critic(obs, hidden_states)
        else:
            value = self._critic(obs)
            hidden_states = None
        return value, hidden_states

    def init_hidden(self, batch_size=None):
        return self._actor.init_hidden(batch_size)

    def clamp_std(self, min_std, max_std):
        self._log_std.detach().clamp_(np.log(min_std), np.log(max_std))

    @property
    def policy(self):
        return copy.deepcopy(self._actor)

    @property
    def log_std(self):
        return self._log_std