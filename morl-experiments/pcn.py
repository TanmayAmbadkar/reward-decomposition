import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)
from morl_baselines.multi_policy.capql.capql import CAPQL


# from gymnasium.wrappers.record_video import RecordVideo


def main(algo: str, gpi_pd: bool, g: int, timesteps_per_iter: int = 15000):
    def make_env(record_episode_statistics: bool = False):
        env = mo_gym.make("mo-hopper-2obj-v5", cost_objective=False, max_episode_steps=500)
        if record_episode_statistics:
            env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env(record_episode_statistics=True)
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = CAPQL(
        env,
        
        learning_rate = 3e-4,
        gamma= 0.99,
        tau = 0.005,
        buffer_size = 1000000,
        net_arch = [256, 256],
        batch_size = 128,
        num_q_nets = 2,
        alpha = 0.2,
        learning_starts = 1000,
        gradient_updates = 1,
        project_name = "MORL-Baselines",
        experiment_name = "CAPQL",
        log= True,
        device = "auto",
    )

    agent.train(
        total_timesteps=10 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=np.array([-100.0, -100.0]),
        known_pareto_front=None,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )


if __name__ == "__main__":
    fire.Fire(main)