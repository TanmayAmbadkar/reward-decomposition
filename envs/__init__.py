from envs.crafter_env import CrafterEnv
from gymnasium.envs.registration import register

register(
    id='mo-hopper-v6',
    entry_point='envs.mo_gym_hopper:MOHopperEnv',
    max_episode_steps=500,
    kwargs={
        'cost_objective': False,
    }
)

register(
    id='deep-sea-treasure-v1',
    entry_point='envs.dst:DeepSeaTreasureEnv',
    max_episode_steps=100,
    kwargs={
        'cost_objective': False,
    }
)
