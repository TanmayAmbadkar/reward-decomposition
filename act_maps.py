import os
import imageio
import torch
import numpy as np
import numpy as np
import torch
import io
import matplotlib.pyplot as plt
from ppo.agent import DiscreteAgent, ContinuousAgent
import mo_gymnasium as mo_gym
from morl_baselines.common.performance_indicators import hypervolume
from tqdm import tqdm
from PIL import Image

env_agent = mo_gym.make("mo-hopper-v5")
eval_agent = ContinuousAgent(env_agent, reward_size=3).to("cuda")
model_path = "runs/mo-hopper-v5__main_ppo__2025-05-05 15:58:29.869162__1/main_ppo.rl_model"
eval_agent.load_state_dict(torch.load(model_path))

# Ensure the output directory exists
os.makedirs("act_maps", exist_ok=True)

# Prepare to hook into the actor network to capture activations
activations = {}
hooks = []

# Register a forward hook on every Linear layer of the actor
for name, module in eval_agent.actor.named_modules():
    if isinstance(module, torch.nn.Linear):
        activations[name] = []
        def make_hook(n):
            def hook(module, inp, out):
                # store a copy of the activation
                activations[n].append(out.detach().cpu())
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

# Function to record one trajectory's activations for a given weight
def record_trajectory_activations(weight, max_steps=200):
    # Clear previous activations
    for key in activations:
        activations[key].clear()
    obs, _ = env_agent.reset()
    w = weight.to("cuda")
    done = trunc = False
    steps = 0
    while not done and not trunc:
        # Forward pass through the agent, recording activations via hooks
        action, _ = eval_agent.predict(obs, w, deterministic=True, device="cuda")
        obs, rew, done, trunc, _ = env_agent.step(action[0])
        steps += 1

def create_activation_gif(idx, output_dir="act_maps", fps=10, size=1024):
    """
    Create and save a 1024x1024 GIF of activation maps for trajectory idx.
    
    Args:
        idx (int): index/name for the GIF file
        activations (dict): layer_name -> list of activations per timestep
        output_dir (str): directory to save GIF
        fps (int): frames per second
        size (int): desired width and height of output images (pixels)
    """
    os.makedirs(output_dir, exist_ok=True)
    layer_names = list(activations.keys())
    T = len(activations[layer_names[0]])
    frames = []
    
    for t in range(T):
        # 1) concatenate activations
        vecs = [activations[name][t].flatten().numpy() for name in layer_names]
        concat_vec = np.concatenate(vecs)
        # 2) pad to square
        L = concat_vec.size
        side = int(np.ceil(np.sqrt(L)))
        pad = side*side - L
        arr = np.pad(concat_vec, (0, pad), mode="constant")
        img = arr.reshape(side, side)
        # 3) normalize to [0,255]
        norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_uint8 = (norm * 255).astype(np.uint8)
        # 4) upscale to size x size
        pil = Image.fromarray(img_uint8, mode='L')
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        frames.append(np.array(pil))
    
    # 5) save GIF
    gif_path = f"{output_dir}/{idx}.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved {gif_path}")
    
# Generate GIFs for a sequence of weight vectors
num_weights = 10
for idx in range(num_weights):
    w = torch.tensor([1.0 - idx/(num_weights-1), idx/(num_weights-1), 0.0])
    record_trajectory_activations(w)
    create_activation_gif(idx)

# Remove hooks to clean up
for h in hooks:
    h.remove()

# List saved GIFs
print("Generated GIFs:", sorted(os.listdir("act_maps")))
