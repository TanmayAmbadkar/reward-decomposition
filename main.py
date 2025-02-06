import gymnasium as gym
from envs.lunar_lander import LunarLander, demo_heuristic_lander
from envs.bipedal_walker import BipedalWalker, demo_heuristic_walker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = BipedalWalker()

total_reward, rewards = demo_heuristic_walker(env)
# rewards = np
print(total_reward)
print(rewards[-2])
print(rewards[-1])
rewards = np.array(rewards)

df = pd.DataFrame(rewards, columns = ["forward", "head_str", "joint1", "joint2", "joint3", "joint4", "Final"])


plt.figure(figsize = (10, 5))
df.iloc[:-1].plot()
plt.savefig("RewardDecom.png")

plt.figure(figsize = (10, 5))
sns.barplot(df.sum())
plt.savefig("BarPlot.png")




