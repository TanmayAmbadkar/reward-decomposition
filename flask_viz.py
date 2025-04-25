import os
import cv2
import time
import threading
import numpy as np
import torch
import io
import json
import argparse
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, Response, jsonify
import mo_gymnasium as mo_gym
from io import BytesIO
import base64
import subprocess

# PPO agent classes
from ppo.agent import DiscreteAgent, ContinuousAgent

# Set up paths for templates.
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

# Parse command‐line for a config JSON
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True,
                    help="Path to JSON config with env_name, continuous, reward_size, reward_labels, model_path")
args = parser.parse_args()

# Load config
with open(args.config) as f:
    cfg = json.load(f)

ENV_NAME       = cfg['env_name']
CONTINUOUS     = cfg['continuous']      # True or False
REWARD_SIZE    = cfg['reward_size']     # int
REWARD_LABELS  = cfg['reward_labels']   # list of strings, len == REWARD_SIZE
MODEL_PATH     = cfg['model_path']      # str
INITIAL_WEIGHTS = cfg.get('initial_weights', [1.0]*REWARD_SIZE)
# Will hold our TensorBoard subprocess handle
TB_PROCESS = None

# Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Global weights & lock
weights      = np.array(INITIAL_WEIGHTS, dtype=float)
weights_lock = threading.Lock()

# For live‐plotting the in‐flight episode
current_episode_reward = [np.zeros(REWARD_SIZE)]
episode_reward_lock    = threading.Lock()

# Recorded trajectory storage
recorded_trajectory     = []
record_lock             = threading.Lock()
recording_in_progress   = False
recording_lock          = threading.Lock()

# --- Initialize the agent ---
# Vectorized environment for inference
env_agent = mo_gym.make(ENV_NAME, render_mode="rgb_array")
env_render = mo_gym.make(ENV_NAME, render_mode = "rgb_array")
AgentClass = ContinuousAgent if CONTINUOUS else DiscreteAgent
eval_agent = AgentClass(env_agent, reward_size=REWARD_SIZE).to("cuda")
eval_agent.load_state_dict(torch.load(MODEL_PATH + "main_ppo.rl_model"))
eval_agent.eval()

    # Pass current weights to the template for slider initialization.
@app.route('/')
def index():
    return render_template('index_main.html', weights=weights.tolist(),
                           reward_labels=[(i, label) for i, label in enumerate(REWARD_LABELS)])

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global weights
    data = request.form
    try:
        new = [float(data.get(f"w_{i}", INITIAL_WEIGHTS[i])) for i in range(REWARD_SIZE)]
        with weights_lock:
            weights[:] = new
        return jsonify(status="success", weights=weights.tolist())
    except Exception as e:
        return jsonify(status="error", message=str(e)), 400
    
    
def simulation_generator():
    """Generator that steps through the simulation, accumulates weighted rewards,
    resets the accumulator at the start of each episode, and yields rendered frames."""
    global weights, current_episode_reward
    while True:
        # Reset the accumulated reward at the start of each episode.
        with episode_reward_lock:
            current_episode_reward = [np.zeros(2), ]
        done = False
        trunc = False
        obs, _ = env_render.reset()
        while not (done or trunc):
            with weights_lock:
                current_weights = np.copy(weights)
            # Agent expects a batch; use the first action.
            action, _ = eval_agent.predict(obs, current_weights, deterministic=True, device = "cuda")
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
    with episode_reward_lock:
        data = np.array(current_episode_reward)
    fig, ax = plt.subplots()
    ax.plot(data, label=REWARD_LABELS)
    ax.set_title("Accumulated Reward per Component (Current Episode)")
    ax.set_ylabel("Accumulated Reward")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
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

def generate_partial_plot_uri(step: int) -> str:
    with record_lock:
        data = np.array([ e['reward_components'] for e in recorded_trajectory ])
    cum = data.cumsum(axis=0)
    fig, ax = plt.subplots()
    for idx, lbl in enumerate(REWARD_LABELS):
        ax.plot(cum[:, idx], label=lbl)
    ax.plot([step, step], [cum.min(), cum.max()], 'k--', lw=1)
    ax.set_title(f"Cumulative Reward up to step {step}")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.route('/play')
def play():
    """
    Streams the simulation as MJPEG and records the trajectory.
    Ensures recording_in_progress is always cleared, even if the client
    stops the stream early.
    """
    def generate():
        global recorded_trajectory, recording_in_progress, record_lock, weights_lock, recording_lock, weights

        # Prevent two concurrent plays
        with recording_lock:
            if recording_in_progress:
                return
            recording_in_progress = True
            recorded_trajectory = []

        try:
            
            env_render = mo_gym.make("deep-sea-treasure-v0", render_mode = "rgb_array")
            obs, _ = env_render.reset()
            done = trunc = False

            while not (done or trunc):
                with weights_lock:
                    w = weights.copy()
                action, value = eval_agent.predict(obs, w, deterministic=True, device="cuda")
                next_obs, rew, done, trunc, _ = env_render.step(action[0])

                ret, png = cv2.imencode('.png', env_render.render())
                if not ret:
                    obs = next_obs
                    continue

                # record
                b64 = base64.b64encode(png.tobytes()).decode('ascii')
                with record_lock:
                    recorded_trajectory.append({
                        'state':            np.round(obs, 3).tolist(),
                        'action':           int(action[0]),
                        'reward_components': (w * rew).tolist(),
                        'weights':          w.tolist(),
                        'value':            np.round(value[0], 3).tolist(),
                        'frame_b64':        b64
                    })

                # stream
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' +
                    png.tobytes() +
                    b'\r\n'
                )

                time.sleep(0.03)
                obs = next_obs
                
        except Exception as e:
            print("Error during simulation:", e)
            # Handle any exceptions that occur during the simulation
            # You can log the error or take appropriate action here

        finally :
            print("In Finally")
            # ALWAYS clear the flag, even if client aborted
            with recording_lock:
                recording_in_progress = False

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/scrub')
def scrub():
    # block while recording
    global recorded_trajectory, recording_in_progress, record_lock, weights_lock, recording_lock, weights
    
    with recording_lock:
        if recording_in_progress:
            return jsonify({'error':'episode still running'}), 409

    step = request.args.get('step', type=int)
    if step is None:
        return jsonify({'error':'must provide ?step=N'}), 400

    with record_lock:
        if step < 0 or step >= len(recorded_trajectory):
            return jsonify({'error':'step out of range'}), 400
        entry = recorded_trajectory[step]

    # 1) frame data URI
    frame_uri = f"data:image/png;base64,{entry['frame_b64']}"
    # 2) cumulative‐reward line plot up to step
    plot_uri  = generate_partial_plot_uri(step)

    # 3) per‐step bar plot of reward components
    # ------------------------------------------------
    labels = ["Treasure", "Time"]
    values = entry['reward_components']
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(f"Reward Components at Step {step}")
    ax.set_ylabel("Reward")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    bar_b64 = base64.b64encode(buf.read()).decode("ascii")
    bar_uri = f"data:image/png;base64,{bar_b64}"
    # ------------------------------------------------

    return jsonify({
        'step':              step,
        'state':             entry['state'],
        'action':            entry['action'],
        'reward_components': entry['reward_components'],
        'weights':           entry['weights'],
        'value':             entry['value'],
        'frame':             frame_uri,
        'plot':              plot_uri,
        'bar_plot':          bar_uri
    })

@app.route('/record_status')
def record_status():
    """
    Returns whether a recording is in progress, and how many steps have been recorded so far.
    Front-end polls this to know when to re-enable scrubbing.
    """
    # read the “in progress” flag
    
    global recorded_trajectory, recording_in_progress, record_lock, weights_lock, recording_lock, weights
    with recording_lock:
        in_progress = recording_in_progress
        print("RECORD STATUS", in_progress)
    # read the current recorded length
    with record_lock:
        length = len(recorded_trajectory)
    return jsonify({
        "recording": in_progress,
        "length":    length
    })
    
@app.route('/recompute_action', methods=['POST'])
def recompute_action():
    """
    Given a scrub step and a new weight vector, return the action
    the agent would now take in that recorded state—without stepping
    the environment.
    """
    # 1) Parse JSON body
    data = request.get_json(force=True)
    step       = data.get('step')
    new_weights = data.get('weights')

    # 2) Ensure recording has finished
    with recording_lock:
        if recording_in_progress:
            return jsonify({'error': 'episode still running'}), 409

    # 3) Validate inputs
    if step is None or new_weights is None:
        return jsonify({'error': 'must provide "step" and "weights" in JSON'}), 400
    try:
        step = int(step)
        w_arr = np.array(new_weights, dtype=float)
    except Exception:
        return jsonify({'error': '"step" must be int and "weights" a list of floats'}), 400

    # 4) Grab the recorded observation
    with record_lock:
        if step < 0 or step >= len(recorded_trajectory):
            return jsonify({'error': 'step out of range'}), 400
        obs = np.array(recorded_trajectory[step]['state'])

    # 5) Compute the new action
    #    Note: agent.predict expects a batch, so wrap obs
    action_batch, value = eval_agent.predict(
        obs[np.newaxis, ...],
        w_arr,
        deterministic=True,
        device="cuda"
    )
    action0 = action_batch[0]
    # If discrete, convert to int; if continuous, to list
    try:
        action_serialized = int(action0)
    except (TypeError, ValueError):
        action_serialized = action0.tolist()

    value_seralized = np.round(value[0], 3)
    return jsonify({'action': action_serialized,
                    'value':  value_seralized.tolist()})
    
@app.route('/launch_tensorboard', methods=['POST'])
def launch_tensorboard():
    """
    Launch (or re‐use) a TensorBoard server pointing at TENSORBOARD_LOGDIR.
    Returns JSON with status and the port.
    """
    global TB_PROCESS

    # 1) If already running and still alive, just return status
    if TB_PROCESS is not None and TB_PROCESS.poll() is None:
        return jsonify({
            "status": "already_running",
            "port":   6006
        })

    # 2) Make sure the logdir exists
    if not os.path.isdir(MODEL_PATH):
        return jsonify({
            "status":  "error",
            "message": f"logdir not found: {MODEL_PATH}"
        }), 400

    # 3) Start TensorBoard
    try:
        TB_PROCESS = subprocess.Popen([
            "tensorboard",
            f"--logdir={MODEL_PATH}",
            f"--port={6006}",
            "--host=0.0.0.0"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
    except Exception as e:
        return jsonify({
            "status":  "error",
            "message": f"failed to launch TensorBoard: {e}"
        }), 500

    return jsonify({
        "status": "started",
        "port":   6006,
        "url":   "127.0.0.1:6006"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5043, debug=False)
