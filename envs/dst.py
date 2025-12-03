import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DeepSeaTreasureEnv(gym.Env):
    """
    Deep Sea Treasure environment aligned to the provided pixel-art layout.

    Grid size: 10 rows x 11 cols
    Start: top-left (0, 0)

    Rewards (vector):
      [time_penalty, treasure_value]
      time_penalty = -1 per step
      treasure_value = value at collected chest, otherwise 0

    Actions:
      0: Up
      1: Down
      2: Left
      3: Right

    Movement into invalid (rock/out-of-bounds) keeps agent in place.
    Episode ends when a treasure is collected or max_steps exceeded.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Grid
        self.rows = 10    # 0..9
        self.cols = 11    # 0..10

        # Spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.rows, self.cols])

        # Reward space for informational use
        self.reward_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        # Treasure locations based on the image
        # Format: (row, col) : value
        # Rows are 0-indexed from top
        self.treasures = {
            (1, 0): 18.0,
            (1, 1): 26.0,
            (1, 2): 31.0,
            (4, 3): 44.0,
            (4, 4): 48.2,
            (4, 5): 56.0,
            (7, 6): 72.0,
            (7, 7): 76.3,
            (8, 9): 90.0,
            (9, 10): 100.0
        }

        # Sea floor depth: maximum allowed row index (inclusive) per column
        # Any row index > sea_floor_depth[col] is rock and invalid to occupy
        # These values were chosen to reproduce the stepped cliff from the image
        self.sea_floor_depth = {
            0: 1,   # columns 0..2 are shallow shelf
            1: 1,
            2: 1,
            3: 4,   # middle shelf
            4: 4,
            5: 4,
            6: 7,   # lower shelves
            7: 7,
            8: 8,   # near-bottom shelf for the 90 chest
            9: 8,
            10: 9   # final deep column for the 100 chest
        }

        # State
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.current_step = 0
        return self.agent_pos, {}

    def step(self, action):
        self.current_step += 1
        row, col = int(self.agent_pos[0]), int(self.agent_pos[1])

        new_row, new_col = row, col
        if action == 0:   # Up
            new_row = row - 1
        elif action == 1: # Down
            new_row = row + 1
        elif action == 2: # Left
            new_col = col - 1
        elif action == 3: # Right
            new_col = col + 1
        else:
            # invalid action index, ignore move
            pass

        # Validate move: bounds then sea floor collision
        is_valid_move = False
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            max_depth = self.sea_floor_depth.get(new_col, -1)
            # valid if the target row is not below the sea floor for that column
            if new_row <= max_depth:
                is_valid_move = True

        if is_valid_move:
            self.agent_pos = np.array([new_row, new_col], dtype=np.int32)
        # else remain in place

        current_pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        treasure_value = float(self.treasures.get(current_pos_tuple, 0.0))

        reward_vector = np.array([treasure_value, -1.0], dtype=np.float32)

        terminated = treasure_value > 0.0
        truncated = self.current_step >= self.max_steps

        info = {"vector_reward": reward_vector}

        # Returning vector reward in the reward slot for MORL usage.
        # If you plan to plug this into scalar RL algorithms, wrap or convert as needed.
        return self.agent_pos, reward_vector, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        grid = np.full((self.rows, self.cols), " ", dtype=object)

        # Draw rocks below the allowed depth
        for c in range(self.cols):
            depth = self.sea_floor_depth.get(c, -1)
            if depth < self.rows - 1:
                # rows > depth are rock
                grid[depth + 1 :, c] = "#"

        # Draw treasures
        for (r, c), v in self.treasures.items():
            grid[r, c] = "T"

        # Draw agent
        r, c = int(self.agent_pos[0]), int(self.agent_pos[1])
        if grid[r, c] == "T":
            grid[r, c] = "X"
        else:
            grid[r, c] = "S"

        # Print
        print("-" * (self.cols * 2 + 1))
        for r in range(self.rows):
            print("|" + "|".join(grid[r]) + "|")
        print("-" * (self.cols * 2 + 1))


# Quick test when run as main
if __name__ == "__main__":
    env = DeepSeaTreasureEnv(render_mode="human")
    obs, _ = env.reset()
    print("Initial State:", obs)
    env.render()

    # Move down to first shelf treasure at (1,0)
    print("\nStep: Down")
    obs, reward, term, trunc, info = env.step(1)
    env.render()
    print(f"State: {obs}, Reward: {reward}, Done: {term}, Info: {info}")

    # Try to dive below the allowed depth in column 0
    print("\nAttempt invalid dive (Down twice)")
    env.reset()
    env.step(1)  # to (1,0)
    obs, reward, term, trunc, info = env.step(1)  # try to (2,0); should be blocked
    env.render()
    print(f"After invalid dive attempt Agent Pos: {obs} (should be [1 0])")
