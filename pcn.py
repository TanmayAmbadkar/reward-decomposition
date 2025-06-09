import mo_gymnasium as mo_gym
import numpy as np
import torch  # noqa: F401

from morl_baselines.multi_policy.morld.morld import MORLD
import envs

def main():
    gamma = 0.99

    env = mo_gym.make("mo-hopper-v6")
    eval_env = mo_gym.make("mo-hopper-v6")

    algo = MORLD(
        env=env,
        exchange_every=int(5e4),
        pop_size=6,
        policy_name="MOSAC",
        scalarization_method="ws",
        evaluation_mode="ser",
        gamma=gamma,
        log=True,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=True,
        sharing_mechanism=[],
        weight_adaptation_method=None,
        seed=0,
    )

    algo.train(
        eval_env=eval_env,
        total_timesteps=2e6,
        ref_point=np.array([-100.0, -100.0, -100.0]),
        known_pareto_front=None,
    )


if __name__ == "__main__":
    main()