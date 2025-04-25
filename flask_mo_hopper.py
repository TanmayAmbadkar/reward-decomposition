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
from ppo.agent import DiscreteAgent, ContinuousAgent
from envs.utils import SyncVectorEnv
import mo_gymnasium as mo_gym
import base64
import traceback
from io import BytesIO

# Set up paths for templates.
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

app = Flask(__name__, template_folder=TEMPLATE_PATH)

# Global weights vector (initialized as desired) and lock.
weights = np.array([1, 0, 0])
weights_lock = threading.Lock()

# Global variable for the accumulated reward for the current episode.
current_episode_reward = [np.zeros(3), ]
episode_reward_lock = threading.Lock()

# --- Initialize the agent ---
# Create a vectorized environment for the agent.
env_agent = mo_gym.make("mo-hopper-v5", render_mode = "rgb_array")
eval_agent = ContinuousAgent(env_agent, reward_size=3).to("cuda")
model_path = "runs/mo-hopper-v5__main_ppo__2025-04-24 19:48:25.226526__1/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))
eval_agent.eval()

# Create a separate (non-vectorized) environment for rendering.
# env_render = mo_gym.make("mo-hopper-v5", render_mode = "rgb_array")

recorded_trajectory = []
record_lock = threading.Lock()# signal whether a /play episode is currently running
recording_in_progress = False
recording_lock = threading.Lock()


@app.route('/')
def index():
    # Pass current weights to the template for slider initialization.
    return render_template('mo_hopper.html', weights=weights.tolist())

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global weights
    # Get form data sent via AJAX.
    data = request.form
    try:
        new_weights = np.array([
            float(data.get("w_forward", 1)),
            float(data.get("w_jump", 0)),
            float(data.get("w_control", 0.0)),
        ])
        print(new_weights)
        with weights_lock:
            weights = new_weights
        return jsonify({"status": "success", "weights": weights.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# def simulation_generator():
#     """Generator that steps through the simulation, accumulates weighted rewards,
#     resets the accumulator at the start of each episode, and yields rendered frames."""
#     global weights, current_episode_reward
#     while True:
#         # Reset the accumulated reward at the start of each episode.
#         with episode_reward_lock:
#             current_episode_reward = [np.zeros(3), ]
#         done = False
#         trunc = False
#         obs, _ = env_render.reset()
#         while not (done or trunc):
#             with weights_lock:
#                 current_weights = np.copy(weights)
#             # Agent expects a batch; use the first action.
#             print(obs, current_weights)
#             action, _ = eval_agent.predict(obs, current_weights, deterministic=True, device = "cuda")
#             obs, rew, done, trunc, infos = env_render.step(action[0])
#             # Compute the weighted reward for this step.
#             step_reward = current_weights * rew
#             with episode_reward_lock:
#                 current_episode_reward.append(step_reward)
#             frame = env_render.render()
#             yield frame
#             time.sleep(0.03)

# def frame_gen(generator_func, *args, **kwargs):
#     """Encodes frames from the generator as PNG and yields them in a multipart response."""
#     get_frame = generator_func(*args, **kwargs)
#     while True:
#         frame = next(get_frame, None)
#         if frame is None:
#             continue
#         ret, png = cv2.imencode('.png', frame)
#         if not ret:
#             continue
#         frame_bytes = png.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/render_feed')
# def render_feed():
#     return Response(frame_gen(simulation_generator),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_plot():
    """Generates a PNG bar plot of the accumulated reward per component for the current episode."""
    with episode_reward_lock:
        data = np.array(current_episode_reward)
    fig, ax = plt.subplots()
    components = ["Forward", "Jump", "Energy", ]
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
    """
    Build a cumulative-reward line plot up to `step` and return it
    as a 'data:image/png;base64,…' URI.
    """
    
    global recorded_trajectory, recording_in_progress, record_lock, weights_lock, recording_lock, weights
    with record_lock:
        # gather reward vectors up to and including `step`
        data = np.array([ e['reward_components'] 
                          for e in recorded_trajectory ])  # shape (step+1, 4)

    # cumulative sum along time
    cum = data.cumsum(axis=0)

    labels = ["Forward", "Jump", "Energy", ]
    fig, ax = plt.subplots()
    for idx, lbl in enumerate(labels):
        ax.plot(cum[:, idx], label=lbl)
    ax.plot([step, step], [cum.min(), cum.max()], 'k--', lw=1)
    ax.set_title("Cumulative Reward up to step {}".format(step))
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
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
            
            env_render = mo_gym.make("mo-hopper-v5", render_mode = "rgb_array")
            obs, _ = env_render.reset()
            done = trunc = False

            while not (done or trunc):
                with weights_lock:
                    w = weights.copy()
                
                action, _ = eval_agent.predict(obs, w, deterministic=True, device="cuda")
                next_obs, rew, done, trunc, _ = env_render.step(action[0])

                ret, png = cv2.imencode('.png', env_render.render())
                if not ret:
                    obs = next_obs
                    continue

                # record
                b64 = base64.b64encode(png.tobytes()).decode('ascii')
                with record_lock:
                    recorded_trajectory.append({
                        'state':            obs.tolist(),
                        'action':           action[0].tolist(),
                        'reward_components': (w * rew).tolist(),
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
            print(traceback.format_exc())
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

    labels = ["Forward", "Jump", "Energy", ]
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
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5043, debug=False)
