"""
Multi-Objective Lunar Lander
============================
Subclasses the gymnasium LunarLander source directly so that we can access
m_power and s_power at each step before they are collapsed into a scalar.

Reward vector: [shaping_reward, fuel_cost, terminal_reward]

  shaping_reward  – delta-shaping signal encoding position accuracy,
                    velocity, angle, and leg-contact bonuses.
                    Equals (shaping_t - shaping_{t-1}) on non-terminal steps,
                    0 on terminal steps.

  fuel_cost       – negative engine firing cost each step.
                    = -(m_power * 0.30 + s_power * 0.03)
                    0 on terminal steps (engine cost is NOT applied at
                    termination in the base env — reward is overridden to ±100).

  terminal_reward – 0 on every non-terminal step.
                    +100 on successful landing (lander comes to rest).
                    -100 on crash (body contact or out-of-viewport).

This decomposition is exact: sum(reward_vector) == scalar_reward from the
base environment at every step.

Usage
-----
    # Discrete actions (default)
    env = MOLunarLanderEnv(continuous=False)

    # Continuous actions
    env = MOLunarLanderEnv(continuous=True)

    obs, _ = env.reset(seed=0)
    while True:
        action = env.action_space.sample()
        obs, reward_vec, terminated, truncated, info = env.step(action)
        # reward_vec: np.ndarray shape (3,)
        #   [shaping_reward, fuel_cost, terminal_reward]
        if terminated or truncated:
            break
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed. Run: pip install swig && pip install "gymnasium[box2d]"'
    ) from e


# ---------------------------------------------------------------------------
# Constants (kept identical to gymnasium source)
# ---------------------------------------------------------------------------
FPS = 50
SCALE = 30.0
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_RANDOM = 1000.0
LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40
SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = 4
VIEWPORT_W = 600
VIEWPORT_H = 400


# ---------------------------------------------------------------------------
# Contact detector (unchanged from gymnasium source)
# ---------------------------------------------------------------------------
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


# ---------------------------------------------------------------------------
# Multi-Objective Lunar Lander
# ---------------------------------------------------------------------------
class MOLunarLanderEnv(gym.Env, EzPickle):
    """
    Multi-Objective Lunar Lander.

    Reward vector (3-dim):
      [0] shaping_reward  - delta shaping (position/velocity/angle/leg contact)
      [1] fuel_cost       - engine firing cost, always <= 0
      [2] terminal_reward - 0 mid-episode, +100 landing, -100 crash

    The scalar sum of the reward vector equals the base LunarLander scalar
    reward exactly at every timestep.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        continuous: bool = False,
        render_mode=None,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self, continuous, render_mode, gravity,
            enable_wind, wind_power, turbulence_power,
        )

        assert -12.0 < gravity < 0.0, (
            f"gravity must be between -12 and 0 (got {gravity})"
        )
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.continuous = continuous
        self.render_mode = render_mode

        # Internal Box2D state
        self.screen = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander = None
        self.particles = []
        self.prev_shaping = None
        self.game_over = False

        # Observation space (identical to base env v3)
        low = np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action space
        if self.continuous:
            self.action_space = spaces.Box(-1.0, +1.0, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

        # Multi-objective reward space: [shaping_reward, fuel_cost, terminal_reward]
        self.reward_space = spaces.Box(
            low=np.array([-np.inf, -0.33, -100.0], dtype=np.float32),
            high=np.array([np.inf, 0.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # World construction helpers (from gymnasium source, unchanged)
    # ------------------------------------------------------------------

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    # ------------------------------------------------------------------
    # Reset (from gymnasium source, with MO bookkeeping)
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()

        # Recreate world to avoid Box2D bug #728
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        # Take one dummy step to produce a valid first observation
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    # ------------------------------------------------------------------
    # Step with exact vector reward decomposition
    # ------------------------------------------------------------------

    def step(self, action):
        assert self.lander is not None, "Call reset() before step()."

        # Wind
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(action), f"Invalid action {action!r}"

        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Main engine
        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
            else:
                m_power = 1.0

            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (
                self.lander.position[0] + ox,
                self.lander.position[1] + oy,
            )
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                    impulse_pos, True,
                )
            self.lander.ApplyLinearImpulse(
                (
                    -ox * MAIN_ENGINE_POWER * m_power,
                    -oy * MAIN_ENGINE_POWER * m_power,
                ),
                impulse_pos,
                True,
            )

        # Side engines
        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            else:
                direction = action - 2
                s_power = 1.0

            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (
                    -ox * SIDE_ENGINE_POWER * s_power,
                    -oy * SIDE_ENGINE_POWER * s_power,
                ),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = np.array(
            [
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                self.lander.angle,
                20.0 * self.lander.angularVelocity / FPS,
                1.0 if self.legs[0].ground_contact else 0.0,
                1.0 if self.legs[1].ground_contact else 0.0,
            ],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Exact reward decomposition
        # ------------------------------------------------------------------
        shaping = (
            -100 * np.sqrt(state[0] ** 2 + state[1] ** 2)
            - 100 * np.sqrt(state[2] ** 2 + state[3] ** 2)
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )

        terminated = False
        terminal_reward = 0.0

        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            terminal_reward = -100.0
        elif not self.lander.awake:
            terminated = True
            terminal_reward = +100.0

        if terminated:
            # Base env sets reward = ±100 outright; shaping delta and fuel
            # cost are NOT included on terminal steps.
            shaping_reward = 0.0
            fuel_cost = 0.0
        else:
            shaping_reward = (
                (shaping - self.prev_shaping)
                if self.prev_shaping is not None
                else 0.0
            )
            fuel_cost = -(m_power * 0.30 + s_power * 0.03)

        self.prev_shaping = shaping

        reward_vector = np.array(
            [shaping_reward, fuel_cost, terminal_reward], dtype=np.float32
        )

        info = {
            "vector_reward": reward_vector,
            "m_power": m_power,
            "s_power": s_power,
        }

        if self.render_mode == "human":
            self.render()

        # truncated is always False here; time-limit wrapping handled externally
        return state, reward_vector, terminated, False, info

    # ------------------------------------------------------------------
    # Render (from gymnasium source, unchanged)
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed. Run: pip install "gymnasium[box2d]"'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = obj.color1

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = [(c[0] * SCALE, c[1] * SCALE) for c in p]
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=int(f.shape.radius * SCALE),
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, obj.color1, path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(self.surf, obj.color2, True, path)

        for x in [self.helipad_x1, self.helipad_x2]:
            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            flagy2 = flagy1 + 50
            pygame.draw.line(
                self.surf, (255, 255, 255), (x, flagy1), (x, flagy2), width=1
            )
            pygame.draw.polygon(
                self.surf,
                (204, 204, 0),
                [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
            )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# ---------------------------------------------------------------------------
# Sanity check: verify sum(reward_vector) == base scalar reward at every step
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check vector sum against base env scalar reward step-by-step",
    )
    args = parser.parse_args()

    print(f"MOLunarLanderEnv  continuous={args.continuous}")
    mo_env = MOLunarLanderEnv(continuous=args.continuous)

    if args.verify:
        base_env = gym.make("LunarLander-v3", continuous=args.continuous)
        rng = np.random.default_rng(0)
        mismatches = 0
        N_EPISODES = 10

        for ep in range(N_EPISODES):
            mo_env.reset(seed=ep)
            base_env.reset(seed=ep)
            ep_vec = np.zeros(3)

            for _ in range(500):
                if args.continuous:
                    action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
                else:
                    action = int(rng.integers(0, 4))

                _, rvec, term_mo, _, _ = mo_env.step(action)
                _, rscalar, term_base, _, _ = base_env.step(action)
                ep_vec += rvec

                if not np.isclose(float(np.sum(rvec)), rscalar, atol=1e-4):
                    print(
                        f"  MISMATCH ep={ep}: "
                        f"vec_sum={float(np.sum(rvec)):.5f}  "
                        f"scalar={rscalar:.5f}  vec={rvec}"
                    )
                    mismatches += 1

                if term_mo or term_base:
                    break

            print(
                f"  ep={ep}  "
                f"[shaping={ep_vec[0]:.2f}  fuel={ep_vec[1]:.2f}  "
                f"terminal={ep_vec[2]:.2f}]  sum={ep_vec.sum():.2f}"
            )

        base_env.close()
        print(f"\nTotal step-level mismatches: {mismatches}")

    else:
        obs, _ = mo_env.reset(seed=42)
        print(f"obs shape:    {obs.shape}")
        print(f"action space: {mo_env.action_space}")
        print(f"reward space: {mo_env.reward_space}")

        total = np.zeros(3)
        steps = 0
        for _ in range(500):
            action = mo_env.action_space.sample()
            obs, rvec, term, trunc, info = mo_env.step(action)
            total += rvec
            steps += 1
            if term or trunc:
                break

        print(f"\nEpisode finished: {steps} steps")
        print(
            f"Cumulative vector reward: "
            f"shaping={total[0]:.2f}  fuel={total[1]:.2f}  "
            f"terminal={total[2]:.2f}  scalar_equiv={total.sum():.2f}"
        )

    mo_env.close()