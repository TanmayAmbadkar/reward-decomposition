import mo_gymnasium
env = mo_gymnasium.make("mo-lunar-lander-v3")

print(env.action_space, env.observation_space)
