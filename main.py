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
import imageio
import os
env =SyncVectorEnv(
    [
        lambda: gym.wrappers.TimeLimit(LunarLander(continuous = True, scalar_reward=False, render_mode="rgb_array"), max_episode_steps = 500),
    ],
    reward_size = 8
)


eval_agent = ContinuousAgent(env, reward_size=8)
eval_agent.load_state_dict(torch.load("runs/LunarLander__main_ppo__2025-02-24 20:00:10.501470__100/main_ppo.rl_model"))
eval_agent.eval()
env = LunarLander(continuous = True)

episodic_returns = []
value_function = []
done = False
trunc = False
weight = np.zeros(8)
# weight[[0, 1]] = 0
weight[0] = 1
weight[1] = -1
weight[2] = 0
weight[[3, 4]] = 1
weight[[5, 6]] = 1
weight[-1] = 1
frames = []

episode_len = 0
for i in range(5):
    done = False
    trunc = False
    obs, _ = env.reset()
    while not done and not trunc:
        action, value = eval_agent.predict(obs, weight, deterministic=True)
        obs, rew, done, trunc, infos = env.step(action[0])
        if i == 0:
            episodic_returns.append(rew)
            value_function.append(value[0])
        frames.append(env.render())



if not os.path.exists(f"results/{weight}"):
    os.mkdir(f"results/{weight}")
gif_name = f"results/{weight}/run.gif"
imageio.mimsave(gif_name, frames, fps=30)

total_reward = np.array(episodic_returns).sum(axis = 0)
rewards = np.array(episodic_returns)

# rewards = np
# print(total_reward)
# print(rewards[-2])
# print(rewards[-1])
rewards = np.array(rewards)

df = pd.DataFrame(rewards, columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/ActualRewardDecom.png")

plt.figure(figsize = (10, 5))
sns.barplot(df.sum())
plt.savefig(f"results/{weight}/BarPlot.png")


df = pd.DataFrame(weight * rewards, columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/WeightedRewardDecom.png")


plt.figure(figsize = (10, 5))
for col in df.columns:
    df[col] = df[col] * (0.99 ** df.index)
df.cumsum().iloc[::-1].reset_index().iloc[:, 1:].plot()
plt.savefig(f"results/{weight}/RewardDecomCumSum.png")


df = pd.DataFrame(np.array(value_function), columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/ValueDecom.png")

plt.figure(figsize = (10, 5))
# sns.barplot(df.sum())
sns.barplot(df.iloc[0])
plt.savefig(f"results/{weight}/ValueBarPlot.png")



