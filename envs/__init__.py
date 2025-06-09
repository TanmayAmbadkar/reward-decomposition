from envs.bipedal_walker import BipedalWalker
from envs.lunar_lander import LunarLander
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
