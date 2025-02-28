import os
import cv2
import time
import threading
import numpy as np
import torch
import io
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, Response, jsonify
from envs.lunar_lander import LunarLander  # Ensure this is on your PYTHONPATH
from ppo.agent import ContinuousAgent
from envs.utils import SyncVectorEnv
import gymnasium as gym

# Set up paths for templates.
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

app = Flask(__name__, template_folder=TEMPLATE_PATH)

# Global weights vector (initialized as desired) and lock.
weights = np.array([0.9, 0.7, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
weights_lock = threading.Lock()

# Global variable for the accumulated reward for the current episode.
current_episode_reward = [np.zeros(8), ]
episode_reward_lock = threading.Lock()

# --- Initialize the agent ---
# Create a vectorized environment for the agent.
env_agent = SyncVectorEnv(
    [
        lambda: gym.wrappers.TimeLimit(
            LunarLander(continuous=True, scalar_reward=False, render_mode="rgb_array"),
            max_episode_steps=500
        ),
    ],
    reward_size=8
)
eval_agent = ContinuousAgent(env_agent, reward_size=8)
model_path = "runs/LunarLander__main_ppo__2025-02-26 20:34:33.618312__100/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))
eval_agent.eval()

# Create a separate (non-vectorized) environment for rendering.
env_render = LunarLander(continuous=True, scalar_reward=False, render_mode="rgb_array")

@app.route('/')
def index():
    # Pass current weights to the template for slider initialization.
    return render_template('index.html', weights=weights.tolist())

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global weights
    # Get form data sent via AJAX.
    data = request.form
    try:
        new_weights = np.array([
            float(data.get("w_distance", 0.9)),
            float(data.get("w_speed", 0.7)),
            float(data.get("w_tilt", 0.0)),
            float(data.get("w_leg1", 0.5)),
            float(data.get("w_leg2", 0.5)),
            float(data.get("w_main_engine", 1.0)),
            float(data.get("w_side_engine", 1.0)),
            float(data.get("w_success", 1.0)),
        ])
        with weights_lock:
            weights = new_weights
        return jsonify({"status": "success", "weights": weights.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def simulation_generator():
    """Generator that steps through the simulation, accumulates weighted rewards,
    resets the accumulator at the start of each episode, and yields rendered frames."""
    global weights, current_episode_reward
    while True:
        # Reset the accumulated reward at the start of each episode.
        with episode_reward_lock:
            current_episode_reward = [np.zeros(8), ]
        done = False
        trunc = False
        obs, _ = env_render.reset()
        while not (done or trunc):
            with weights_lock:
                current_weights = np.copy(weights)
            # Agent expects a batch; use the first action.
            action, _ = eval_agent.predict(obs, current_weights, deterministic=True)
            obs, rew, done, trunc, infos = env_render.step(action[0])
            # Compute the weighted reward for this step.
            step_reward = current_weights * rew
            with episode_reward_lock:
                current_episode_reward.append(step_reward)
            frame = env_render.render()
            yield frame
            time.sleep(0.03)

def frame_gen(generator_func, *args, **kwargs):
    """Encodes frames from the generator as PNG and yields them in a multipart response."""
    get_frame = generator_func(*args, **kwargs)
    while True:
        frame = next(get_frame, None)
        if frame is None:
            continue
        ret, png = cv2.imencode('.png', frame)
        if not ret:
            continue
        frame_bytes = png.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/render_feed')
def render_feed():
    return Response(frame_gen(simulation_generator),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_plot():
    """Generates a PNG bar plot of the accumulated reward per component for the current episode."""
    with episode_reward_lock:
        data = np.array(current_episode_reward)
    fig, ax = plt.subplots()
    components = ["Distance", "Speed", "Tilt", "Leg 1", "Leg 2", "Main Engine", "Side Engine", "Success"]
    # for i in range(8):
    ax.plot(data, label = components)
    ax.set_title("Accumulated Reward per Component (Current Episode)")
    ax.set_ylabel("Accumulated Reward")
    buf = io.BytesIO()
    ax.legend()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@app.route('/plot_feed')
def plot_feed():
    png = generate_plot()
    return Response(png, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
