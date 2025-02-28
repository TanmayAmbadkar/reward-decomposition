import streamlit as st
import numpy as np
import torch
import time
from envs.lunar_lander import LunarLander  # Ensure this is on your PYTHONPATH
from ppo.agent import ContinuousAgent
from streamlit_autorefresh import st_autorefresh
from envs.utils import SyncVectorEnv
import gymnasium as gym

# --- Initialize session state keys if not already present ---
if "env" not in st.session_state:
    st.session_state.env = SyncVectorEnv(
        [
            lambda: gym.wrappers.TimeLimit(
                LunarLander(continuous=True, scalar_reward=False, render_mode="rgb_array"),
                max_episode_steps=500
            ),
        ],
        reward_size=8
    )
    st.session_state.agent = ContinuousAgent(st.session_state.env, reward_size=8)
    model_path = "runs/LunarLander__main_ppo__2025-02-26 20:34:33.618312__100/main_ppo.rl_model"
    st.session_state.agent.load_state_dict(torch.load(model_path))
    st.session_state.agent.eval()
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.frame = None

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = True

if "simulation_thread" not in st.session_state:
    st.session_state.simulation_thread = None

# --- Sidebar: Weight Sliders ---
st.sidebar.header("Weight Components")
_ = st.sidebar.slider("Distance",    0.0, 1.0, 0.9, step=0.1, key="w_distance")
_ = st.sidebar.slider("Speed",       0.0, 1.0, 0.7, step=0.1, key="w_speed")
_ = st.sidebar.slider("Tilt",        0.0, 1.0, 0.0, step=0.1, key="w_tilt")
_ = st.sidebar.slider("Leg 1",       0.0, 1.0, 0.5, step=0.1, key="w_leg1")
_ = st.sidebar.slider("Leg 2",       0.0, 1.0, 0.5, step=0.1, key="w_leg2")
_ = st.sidebar.slider("Main Engine", 0.0, 1.0, 1.0, step=0.1, key="w_main_engine")
_ = st.sidebar.slider("Side Engine", 0.0, 1.0, 1.0, step=0.1, key="w_side_engine")
_ = st.sidebar.slider("Success",     0.0, 1.0, 1.0, step=0.1, key="w_success")

def get_weights():
    return np.array([
        st.session_state.w_distance,
        st.session_state.w_speed,
        st.session_state.w_tilt,
        st.session_state.w_leg1,
        st.session_state.w_leg2,
        st.session_state.w_main_engine,
        st.session_state.w_side_engine,
        st.session_state.w_success,
    ])

st.sidebar.write("Current Weights:", get_weights())

# --- Main Page ---
st.title("Lunar Lander Simulation")
st.write("Use the sliders on the left to adjust the weight components in real time.")

# --- Controls to Start/Stop Simulation ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Simulation"):
        st.session_state.run_simulation = True
with col2:
    if st.button("Stop Simulation"):
        st.session_state.run_simulation = False

# --- Simulation Step ---
def simulation_step():
    if st.session_state.run_simulation:
        weights = get_weights()
        action, _ = st.session_state.agent.predict(st.session_state.obs, weights, deterministic=True)
        print(action)
        st.session_state.obs, reward, done, trunc, info = st.session_state.env.step(action)
        frame = st.session_state.env.render()
        # If the environment returns a batch of frames, take the first one.
        if isinstance(frame, (list, tuple, np.ndarray)):
            if isinstance(frame, np.ndarray) and frame.ndim == 4:
                frame = frame[0]
            elif isinstance(frame, (list, tuple)):
                frame = frame[0]
        st.session_state.frame = frame
        if done or trunc:
            st.session_state.obs, _ = st.session_state.env.reset()

# Run one simulation step on every script rerun.
simulation_step()

# Display the simulation frame.
if st.session_state.frame is not None:
    st.image(st.session_state.frame, channels="RGB", use_column_width=True)
else:
    st.text("Waiting for simulation frame...")

# Auto-refresh the app so the displayed frame updates every 50ms.
st_autorefresh(interval=50, limit=None, key="simulationrefresh")

st.write("Close the browser window (or press the X button) to terminate the program.")
