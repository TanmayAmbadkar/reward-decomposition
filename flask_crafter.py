import os
import time
import threading
import io
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym
from flask import Flask, render_template, Response, request, jsonify
from envs.crafter_env import CrafterEnv
from envs.utils import SyncVectorEnv
from ppo.agent import CNNDiscreteAgent

app = Flask(__name__)

# List of reward component keys (39 components)
reward_keys = [
    'health', 'achievement', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone',
    'coal', 'iron', 'diamond', 'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
    'wood_sword', 'stone_sword', 'iron_sword', 'collect_coal', 'collect_diamond',
    'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone', 'collect_wood',
    'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe',
    'make_wood_sword', 'place_furnace', 'place_plant', 'place_stone', 'place_table',
    'wake_up'
]

# Initialize global weight vector (defaulting to zeros, with some default ones)
weight = np.zeros(39)
weight[[1, 2, 3, 4, 5, 6, 7]] = 1  # example: set some default values

# Locks for thread-safe access to shared variables
weights_lock = threading.Lock()
episode_reward_lock = threading.Lock()

# Global variable to store accumulated rewards for the current episode.
# It is a list of reward vectors (one per timestep).
current_episode_reward = [np.zeros(39)]

# --- Initialize the environment and the agent ---
# Create a vectorized environment for the agent.
env = SyncVectorEnv(
    [
        lambda: gym.wrappers.TimeLimit(
            CrafterEnv(scalar_reward=False, render_mode="rgb_array"),
            max_episode_steps=10000
        ),
    ],
    reward_size=39
)

eval_agent = CNNDiscreteAgent(env, reward_size=39).to("cuda")
model_path = "runs/Crafter__main_ppo__2025-03-03 14:07:30.609268__100/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))
eval_agent.eval()

# Create a separate environment for rendering
env_render = CrafterEnv(scalar_reward=False, render_mode="rgb_array")

@app.route('/')
def index():
    # Pass current weights as a dictionary (key: value) for slider initialization.
    with weights_lock:
        weight_dict = {key: float(weight[i]) for i, key in enumerate(reward_keys)}
    return render_template('crafter.html', weights=weight_dict)

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global weight
    try:
        new_weights = np.zeros(39)
        for i, key in enumerate(reward_keys):
            new_weights[i] = float(request.form.get(key, 0))
        with weights_lock:
            weight = new_weights
        # Return updated weights as a dictionary.
        return jsonify({"status": "success", "weights": {k: float(weight[i]) for i, k in enumerate(reward_keys)}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def simulation_generator():
    """Generator that steps through the Crafter environment, accumulates rewards,
    and yields rendered frames."""
    global weight, current_episode_reward
    while True:
        # Reset accumulated rewards at the start of an episode.
        with episode_reward_lock:
            current_episode_reward = [np.zeros(39)]
        done = False
        trunc = False
        obs, _ = env_render.reset()
        while not (done or trunc):
            with weights_lock:
                current_weight = np.copy(weight)
            # Predict an action using the agent (note: agent expects a batch)
            action, _ = eval_agent.predict(obs, current_weight, deterministic=True, device="cuda")
            print(action)
            obs, rew, done, trunc, infos = env_render.step(action[0])
            with episode_reward_lock:
                current_episode_reward.append(rew)
            # Render the frame from the environment.
            frame = env_render.render(size = (256, 256))
            # resized_frame = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            ret, jpg = cv2.imencode('.jpg', frame,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if ret:
                frame_bytes = jpg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

@app.route('/render_feed')
def render_feed():
    return Response(simulation_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_plot():
    """Generates a PNG plot of the accumulated reward per component for the current episode."""
    with episode_reward_lock:
        data = np.array(current_episode_reward)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data)
    ax.set_title("Accumulated Reward per Component (Current Episode)")
    ax.set_ylabel("Accumulated Reward")
    ax.legend(reward_keys, loc='upper left', fontsize='xx-small')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@app.route('/plot_feed')
def plot_feed():
    png = generate_plot()
    return Response(png, mimetype='image/png')


def reward_plot_gen():
    """Generator that continuously yields updated reward plot frames."""
    while True:
        png = generate_plot()  # Generate the current reward plot as PNG
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + png + b'\r\n')
        time.sleep(0.03)  # Adjust the refresh rate as needed

@app.route('/reward_plot_stream')
def reward_plot_stream():
    return Response(reward_plot_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
