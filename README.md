# Reward Decomposition with PPO for Multi-Objective RL

This repository implements Proximal Policy Optimization (PPO) tailored for **Multi-Objective Reinforcement Learning (MORL)** using a **Reward Decomposition** framework. The agent is designed to handle vectorized rewards and optimize for diverse utility functions, supporting both discrete and continuous action spaces.

## 🚀 Key Features

- **Multi-Objective PPO**: Specialized PPO implementation that process vectorized rewards rather than traditional scalar rewards.
- **Reward Decomposition**: Supports training agents where the final reward is a composition of multiple objective components.
- **Utility Functions**: Provides a suite of utility functions (Linear, Threshold, Ratio, Product, etc.) for environments like Deep Sea Treasure (DST) and Fruit Tree Navigation (FTN).
- **Interactive Visualization**: A Flask-based web interface to visualize agent behavior, adjust weights in real-time, and analyze reward trajectories.
- **Robust Training**: Integration with TensorBoard for monitoring losses, entropy, and multi-objective returns.

---

## 🛠️ Installation

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate reward-decomposition
```

### Using Pip
```bash
pip install -r requirements.txt
```

---

## 📈 Running Experiments

The main entry point for training is `main_ppo.py`. 

### Basic Usage
```bash
python main_ppo.py --env_id fruit-tree-5 --total_timesteps 2000000
```

### Key Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--env_id` | `str` | `"fruit-tree-max"` | Environment identifier from Gymnasium or MO-Gymnasium. |
| `--num_envs` | `int` | `4` | Number of parallel vectorized environments. |
| `--env_is_discrete` | `bool` | `True` | Whether the environment has a discrete action space. |
| `--total_timesteps` | `int` | `500000` | Total training steps across all environments. |
| `--scalar_reward` | `bool` | `False` | If True, uses the environment's default scalarization instead of vector rewards. |
| `--learning_rate` | `float` | `0.0003` | Learning rate for the Adam optimizer. |
| `--gamma` | `float` | `0.995` | Discount factor for GAE and value estimation. |
| `--eval_interval` | `int` | `10000` | How often (in timesteps) to run evaluation. |
| `--use_tensorboard`| `bool` | `True` | Enable logging to TensorBoard. |

---

## 🌍 Supported Environments

The project includes specialized support and utility functions for several Multi-Objective benchmarks:

- **Deep Sea Treasure (DST)**: `deep-sea-treasure-1`, `deep-sea-treasure-3`.
- **Fruit Tree Navigation (FTN)**: `fruit-tree-dist`, `fruit-tree-5`.
- **MuJoCo MORL**: `mo-hopper-v5`, `mo-walker2d-v5`, `mo-halfcheetah-v5`.

Different "tasks" are defined for these environments, mapping vector rewards to scalar utilities via `utility_functions.py`.

---

## 🌐 Interactive Visualization

You can visualize the trained models using the Flask-based dashboard located in `flask_viz.py`. This tool allows you to:
- Watch the agent interact with the environment in real-time.
- Dynamically adjust objective weights using sliders.
- View live reward plots for each objective component.

### Launching the Dashboard
```bash
python flask_viz.py --config config.json
```
Make sure `config.json` points to your trained model path and defines the reward labels.

---

## 📂 Project Structure

- `ppo/`: Core logic including `agent.py` (Actor-Critic architectures) and `ppo.py` (The training algorithm).
- `environments/` & `envs/`: Custom and wrapper environments for MORL tasks.
- `utility_functions.py`: Implementation of various scalarization and utility mappings.
- `main_ppo.py`: Script to start training runs.
- `flask_viz.py`: The visualization server.
- `eval_morl.py`: Dedicated evaluation scripts for multi-objective benchmarks.

---

## 📊 Logging and Monitoring

Training metrics are logged to the `runs/` directory. You can monitor progress by running:
```bash
tensorboard --logdir runs
```
Log includes:
- **Losses**: Policy gradient, Value function, and Entropy losses.
- **Performance**: Mean utility, total return per objective, and episode lengths.
- **Evaluation**: Periodic evaluation results across multiple task configurations.
