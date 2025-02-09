"""Implementation of a synchronous (for loop) vectorization method of any environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Sequence

import time
from collections import deque

import numpy as np

from gymnasium import Env, Space
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from gymnasium.vector.vector_env import ArrayType, VectorEnv, VectorWrapper






__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="sync")
        >>> envs
        SyncVectorEnv(Pendulum-v1, num_envs=2)
        >>> envs = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> envs
        SyncVectorEnv(num_envs=2)
        >>> obs, infos = envs.reset(seed=42)
        >>> obs
        array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
        >>> infos
        {}
        >>> _ = envs.action_space.seed(42)
        >>> actions = envs.action_space.sample()
        >>> obs, rewards, terminates, truncates, infos = envs.step(actions)
        >>> obs
        array([[-0.1878752 ,  0.98219293,  0.7695615 ],
               [ 0.6102389 ,  0.79221743, -0.8498053 ]], dtype=float32)
        >>> rewards
        array([-2.96562607, -0.99902063])
        >>> terminates
        array([False, False])
        >>> truncates
        array([False, False])
        >>> infos
        {}
        >>> envs.close()
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]] | Sequence[Callable[[], Env]],
        copy: bool = True,
        observation_mode: str | Space = "same",
        reward_size: int = 1
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.
            observation_mode: Defines how environment observation spaces should be batched. 'same' defines that there should be ``n`` copies of identical spaces.
                'different' defines that there can be multiple observation spaces with the same length but different high/low values batched together. Passing a ``Space`` object
                allows the user to set some custom observation space mode not covered by 'same' or 'different.'
        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.copy = copy
        self.env_fns = env_fns
        self.observation_mode = observation_mode
        self.reward_size = reward_size

        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]

        # Define core attributes using the sub-environments
        # As we support `make_vec(spec)` then we can't include a `spec = self.envs[0].spec` as this doesn't guarantee we can actual recreate the vector env.
        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.render_mode = self.envs[0].render_mode

        self.single_action_space = self.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if isinstance(observation_mode, tuple) and len(observation_mode) == 2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = self.envs[0].observation_space
                self.observation_space = batch_space(
                    self.single_observation_space, self.num_envs
                )
            elif observation_mode == "different":
                self.single_observation_space = self.envs[0].observation_space
                self.observation_space = batch_differing_spaces(
                    [env.observation_space for env in self.envs]
                )
            else:
                raise ValueError(
                    f"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {observation_mode}"
                )

        # check sub-environment obs and action spaces
        for env in self.envs:
            if observation_mode == "same":
                assert (
                    env.observation_space == self.single_observation_space
                ), f"SyncVectorEnv(..., observation_mode='same') however the sub-environments observation spaces are not equivalent. single_observation_space={self.single_observation_space}, sub-environment observation_space={env.observation_space}. If this is intentional, use `observation_mode='different'` instead."
            else:
                assert is_space_dtype_shape_equiv(
                    env.observation_space, self.single_observation_space
                ), f"SyncVectorEnv(..., observation_mode='different' or custom space) however the sub-environments observation spaces do not share a common shape and dtype, single_observation_space={self.single_observation_space}, sub-environment observation space={env.observation_space}"

            assert (
                env.action_space == self.single_action_space
            ), f"Sub-environment action space doesn't make the `single_action_space`, action_space={env.action_space}, single_action_space={self.single_action_space}"

        # Initialise attributes used in `step` and `reset`
        self._observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs, self.reward_size), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Returns a tuple of np random seeds for the wrapped envs."""
        return self.get_attr("np_random_seed")

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Returns a tuple of the numpy random number generators for the wrapped envs."""
        return self.get_attr("np_random")

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets each of the sub-environments and concatenate the results together.

        Args:
            seed: Seeds used to reset the sub-environments, either
                * ``None`` - random seeds for all environment
                * ``int`` - ``[seed, seed+1, ..., seed+n]``
                * List of ints - ``[1, 2, 3, ..., n]``
            options: Option information used for each sub-environment

        Returns:
            Concatenated observations and info from each sub-environment
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        observations, infos = [], {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            env_obs, env_info = env.reset(seed=single_seed, options=options)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        return deepcopy(self._observations) if self.copy else self._observations, infos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                env_obs, env_info = self.envs[i].reset()

                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    env_obs,
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the rendered frames from the environments."""
        return tuple(env.render() for env in self.envs)

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Calls a sub-environment method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...] | Any):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

    def close_extras(self, **kwargs: Any):
        """Close the environments."""
        if hasattr(self, "envs"):
            [env.close() for env in self.envs]


"""Wrapper that tracks the cumulative rewards and episode lengths."""

__all__ = ["RecordEpisodeStatistics"]



class RecordEpisodeStatistics(VectorWrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of any episode within the vectorized env, the statistics of the episode
    will be added to ``info`` using the key ``episode``, and the ``_episode`` key
    is used to indicate the environment index which has a terminated or truncated episode.

        >>> infos = {  # doctest: +SKIP
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward for each done sub-environment>",
        ...         "l": "<array of episode length for each done sub-environment>",
        ...         "t": "<array of elapsed time since beginning of episode for each done sub-environment>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes

    Example:
        >>> from pprint import pprint
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> envs = RecordEpisodeStatistics(envs)
        >>> obs, info = envs.reset(123)
        >>> _ = envs.action_space.seed(123)
        >>> end = False
        >>> while not end:
        ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        ...     end = term.any() or trunc.any()
        ...
        >>> envs.close()
        >>> pprint(info) # doctest: +SKIP
        {'_episode': array([ True, False, False]),
         '_final_info': array([ True, False, False]),
         '_final_observation': array([ True, False, False]),
         'episode': {'l': array([11,  0,  0], dtype=int32),
                     'r': array([11.,  0.,  0.], dtype=float32),
                     't': array([0.007812, 0.      , 0.      ], dtype=float32)},
         'final_info': array([{}, None, None], dtype=object),
         'final_observation': array([array([ 0.11448676,  0.9416149 , -0.20946532, -1.7619033 ], dtype=float32),
               None, None], dtype=object)}
    """

    def __init__(
        self,
        env: VectorEnv,
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key to save the data
        """
        super().__init__(env)
        self._stats_key = stats_key

        self.episode_count = 0

        self.episode_start_times: np.ndarray = np.zeros(())
        self.episode_returns: np.ndarray = np.zeros(())
        self.episode_lengths: np.ndarray = np.zeros((), dtype=int)
        self.prev_dones: np.ndarray = np.zeros((), dtype=bool)

        self.time_queue = deque(maxlen=buffer_length)
        self.return_queue = deque(maxlen=buffer_length)
        self.length_queue = deque(maxlen=buffer_length)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_start_times = np.full(self.num_envs, time.perf_counter())
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.prev_dones = np.zeros(self.num_envs, dtype=bool)

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        assert isinstance(
            infos, dict
        ), f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."

        self.episode_returns[self.prev_dones] = 0
        self.episode_lengths[self.prev_dones] = 0
        self.episode_start_times[self.prev_dones] = time.perf_counter()
        self.episode_returns[~self.prev_dones] += np.sum(rewards, axis = 1)[~self.prev_dones]
        self.episode_lengths[~self.prev_dones] += 1

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)

        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(
                    f"Attempted to add episode stats when they already exist, info keys: {list(infos.keys())}"
                )
            else:
                episode_time_length = np.round(
                    time.perf_counter() - self.episode_start_times, 6
                )
                infos[self._stats_key] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones

            for i in np.where(dones):
                self.time_queue.extend(episode_time_length[i])
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

