import gymnasium as gym
from envs.lunar_lander import LunarLander, demo_heuristic_lander
from envs.bipedal_walker import BipedalWalker, demo_heuristic_walker
from envs.crafter_env import CrafterEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main_ppo import load_and_evaluate_model
from ppo.agent import ContinuousAgent, CNNDiscreteAgent
from envs.utils import SyncVectorEnv
import torch
import imageio
import os
env =SyncVectorEnv(
    [
        lambda: gym.wrappers.TimeLimit(CrafterEnv(scalar_reward = False, render_mode = "rgb_array"), max_episode_steps = 10000),
    ],
    reward_size = 39
)


eval_agent = CNNDiscreteAgent(env, reward_size=39).to("cuda")
eval_agent.load_state_dict(torch.load("runs/Crafter__main_ppo__2025-02-26 20:47:44.430902__100/main_ppo.rl_model"))
eval_agent.eval()
# env = LunarLander(continuous = True)

episodic_returns = []
value_function = []
done = False
trunc = False
weight = np.zeros(39)
weight[[1, 2, 3, 4, 5, 6, 7, 8]] = 1# # weight[[0, 1]] = 0
# weight[0] = 0.9
# weight[1] = 0.7
# weight[2] = 0
# weight[[3, 4]] = 0.5
# weight[[5, 6]] = 1
# weight[-1] = 1
frames = []

episode_len = 0
for i in range(5):
    done = False
    trunc = False
    obs, _ = env.reset()
    while not done and not trunc:
        action, value = eval_agent.predict(obs, weight, deterministic=True, device = "cuda")
        obs, rew, done, trunc, infos = env.step(action)
        if i == 0:
            episodic_returns.append(rew[0])
            value_function.append(value[0])
        # print(obs.shape)
        frames.append(obs[0])


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

df = pd.DataFrame(rewards, columns = [
    'health',
    'achievement',
    'food',
    'drink',
    'energy',
    'sapling',
    'wood',
    'stone',
    'coal',
    'iron',
    'diamond',
    'wood_pickaxe',
    'stone_pickaxe',
    'iron_pickaxe',
    'wood_sword',
    'stone_sword',
    'iron_sword',
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up'
]
)


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/ActualRewardDecom.png")

plt.figure(figsize = (10, 5))
sns.barplot(df.sum())
plt.savefig(f"results/{weight}/BarPlot.png")


df = pd.DataFrame(weight * rewards, columns = [
    'health',
    'achievement',
    'food',
    'drink',
    'energy',
    'sapling',
    'wood',
    'stone',
    'coal',
    'iron',
    'diamond',
    'wood_pickaxe',
    'stone_pickaxe',
    'iron_pickaxe',
    'wood_sword',
    'stone_sword',
    'iron_sword',
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up'
]
)


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/WeightedRewardDecom.png")


plt.figure(figsize = (10, 5))
for col in df.columns:
    df[col] = df[col] * (0.99 ** df.index)
df.cumsum().iloc[::-1].reset_index().iloc[:, 1:].plot()
plt.savefig(f"results/{weight}/RewardDecomCumSum.png")


df = pd.DataFrame(np.array(value_function), columns = [
    'health',
    'achievement',
    'food',
    'drink',
    'energy',
    'sapling',
    'wood',
    'stone',
    'coal',
    'iron',
    'diamond',
    'wood_pickaxe',
    'stone_pickaxe',
    'iron_pickaxe',
    'wood_sword',
    'stone_sword',
    'iron_sword',
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up'
]
)


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig(f"results/{weight}/ValueDecom.png")

plt.figure(figsize = (10, 5))
# sns.barplot(df.sum())
sns.barplot(df.iloc[0])
plt.savefig(f"results/{weight}/ValueBarPlot.png")



