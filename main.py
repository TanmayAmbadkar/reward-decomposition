import gymnasium as gym
from envs.lunar_lander import LunarLander, demo_heuristic_lander
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
env = LunarLander()

total_reward, rewards = demo_heuristic_lander(env)
# rewards = np
print(total_reward)
rewards = np.array(rewards)

df = pd.DataFrame(rewards, columns = ["distance", "Speed", "Tilt", "Leg 1", "Leg 2", "main engine", "side engine", "success"])


plt.figure(figsize = (10, 5))
df.plot()
plt.savefig("RewardDecom.png")

plt.figure(figsize = (10, 5))
sns.barplot(df.sum())
plt.savefig("BarPlot.png")




