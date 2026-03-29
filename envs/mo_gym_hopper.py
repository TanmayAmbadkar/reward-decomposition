import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHopperEnv(HopperEnv, EzPickle):
    """
    Multi-objective Hopper with survival bonus removed.

    Reward vector (3-dim, no survival bonus):
      R[0] = x_velocity              (forward speed, can be negative)
      R[1] = 10 * z_distance         (jump height, can be negative)
      R[2] = neg_energy_cost         (control cost, always <= 0)

    Without the survival bonus, each objective is a pure signal:
      - A fallen hopper produces low/negative R[0] and R[1]
      - Survival is implicitly encoded rather than artificially inflating all components
      - R[2] reflects pure energy expenditure independent of episode length

    The original mo-hopper-v5 scalar reward is recovered by:
      r = R[0] + R[2] + survival_bonus (not included here)

    2-objective version: set cost_objective=False, R[2] is added to other objectives.
    """

    def __init__(self, cost_objective=True, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, cost_objective, **kwargs)
        self._cost_objective = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.reward_dim,),
            dtype=np.float32,
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        x_velocity      = info["x_velocity"]
        height          = 10.0 * info["z_distance_from_origin"]
        neg_energy_cost = info["reward_ctrl"] / self._ctrl_cost_weight

        # No survival bonus added — pure objective signals only
        if self._cost_objective:
            vec_reward = np.array(
                [x_velocity, height, neg_energy_cost], dtype=np.float32
            )
        else:
            # 2-obj: fold cost into the other objectives
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward += neg_energy_cost
        
        return observation, vec_reward, terminated, truncated, info
