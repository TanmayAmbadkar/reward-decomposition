from stable_baselines3 import PPO
from envs.lunar_lander import LunarLander

env = LunarLander(scalar_reward=True)

model = PPO("MlpPolicy", env, verbose = 1, device = "cpu")
model.learn(total_timesteps = 1000000)