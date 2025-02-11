from stable_baselines3 import PPO
from envs.lunar_lander import LunarLander
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(env_id = "LunarLander-v3", n_envs = 4)

model = PPO("MlpPolicy", env, verbose = 1, device = "cpu")
model.learn(total_timesteps = 1000000)