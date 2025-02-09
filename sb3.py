from stable_baselines3 import PPO
from envs.lunar_lander import LunarLander
from envs.utils import SyncVectorEnv, RecordEpisodeStatistics

env = SyncVectorEnv(
    [
        lambda: LunarLander(continuous = True),
        lambda: LunarLander(continuous = True),
        lambda: LunarLander(continuous = True),
    ],
    reward_size=8
)

env = RecordEpisodeStatistics(env)

env.reset()
for _ in range(1000):
    # print(state)
    _, _, _, _, info = env.step(env.action_space.sample())
    print(info)