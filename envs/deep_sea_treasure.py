
import gymnasium as gym

env = gym.make("Ant-v5", render_mode="rgb_array")

state, info = env.reset()
print(info)
while True:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

    print(info)