import gymnasium as gym
from envs.lunar_lander import LunarLander, demo_heuristic_lander
from envs.bipedal_walker import BipedalWalker, demo_heuristic_walker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main_ppo import load_and_evaluate_model
from ppo.agent import ContinuousAgent
from envs.utils import SyncVectorEnv
import torch

env =SyncVectorEnv(
[
    lambda: gym.wrappers.TimeLimit(LunarLander(continuous = True), max_episode_steps = 500)
],
reward_size = 8
)


eval_agent = ContinuousAgent(env, reward_size=8)
eval_agent.load_state_dict(torch.load("runs/LunarLander__main_ppo__2025-02-19 21:43:35.089738__0/main_ppo.rl_model"))
eval_agent.eval()
env = LunarLander(continuous = True)

obs, _ = env.reset()
episodic_returns = []
value_function = []
done = False
trunc = False
weight = np.zeros(8)
weight[[5, 6]] = 1
while not done and not trunc:
    action, value = eval_agent.predict(obs, weight, deterministic=True)
    obs, rew, done, trunc, infos = env.step(action[0])
    episodic_returns.append(rew)
    value_function.append(value[0])

total_reward = np.array(episodic_returns).sum(axis = 0)
rewards = np.array(episodic_returns)

# rewards = np
print(total_reward)
print(rewards[-2])
print(rewards[-1])
rewards = np.array(rewards)

df = pd.DataFrame(rewards, columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig("RewardDecom.png")

plt.figure(figsize = (10, 5))
sns.barplot(df.sum())
plt.savefig("BarPlot.png")


# df = pd.DataFrame(np.array(value_function), columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


# plt.figure(figsize = (10, 5))
# df.iloc[:-1].plot()
# plt.savefig("ValueDecom.png")

# plt.figure(figsize = (10, 5))
# sns.barplot(df.sum())
# plt.savefig("ValueBarPlot.png")




