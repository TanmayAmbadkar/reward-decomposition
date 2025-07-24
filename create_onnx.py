import torch
import torch.nn as nn
import numpy as np
import json
import os
import mo_gymnasium as mo_gym

# Important: Make sure the ppo.agent module is importable from where you run this script.
from ppo.agent_onnx import ContinuousAgent 

def export_policy_to_onnx(config_path: str, output_path: str = "policy_model.onnx"):
    """
    Loads a trained MOPPO agent, isolates its actor network, and exports it to
    the ONNX format for client-side inference.

    Args:
        config_path (str): Path to the JSON config file used for training.
        output_path (str): Path to save the output .onnx file.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)

    # --- 1. Load Configuration ---
    ENV_NAME = cfg['env_name']
    REWARD_SIZE = cfg['reward_size']
    MODEL_PATH = cfg['model_path']

    print(f"Environment: {ENV_NAME}, Reward Size: {REWARD_SIZE}")
    print(f"Model Path: {MODEL_PATH}")

    # --- 2. Instantiate the Agent and Load Weights ---
    # Create a dummy environment to get observation and action space info
    try:
        env = mo_gym.make(ENV_NAME)
    except Exception as e:
        print(f"Error creating environment '{ENV_NAME}'. Make sure it's registered.")
        print(f"Original error: {e}")
        return

    # Instantiate the agent class
    eval_agent = ContinuousAgent(env, reward_size=REWARD_SIZE).to("cpu")

    # Load the trained state dictionary
    model_file = os.path.join(MODEL_PATH, "converted_ppo.rl_model")
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found at '{model_file}'")
        return
        
    print(f"Loading trained weights from: {model_file}")
    eval_agent.load_state_dict(torch.load(model_file))
    eval_agent.eval() # Set the agent to evaluation mode

    # --- 3. Prepare for ONNX Export ---
    # We are only exporting the ACTOR part of the agent.
    actor_model = eval_agent.actor
    print("\nActor Network Architecture:")
    print(actor_model)

    # The ONNX exporter needs a "dummy" input to trace the network's forward pass.
    # The shape must match the actor's expected input shape.
    # Your actor takes a concatenated state and preference vector.
    obs_shape = env.observation_space.shape
    input_size = obs_shape[0] + REWARD_SIZE
    
    # Create a dummy input tensor of shape [batch_size, input_size]
    dummy_input = torch.randn(1, input_size, device="cpu")
    print(f"\nCreated dummy input with shape: {dummy_input.shape}")

    # --- 4. Export the Model to ONNX ---
    try:
        torch.onnx.export(
            actor_model,              # The model to export (specifically the actor)
            dummy_input,              # A dummy input to trace the model
            output_path,              # Where to save the model
            export_params=True,       # Store the trained weights in the model file
            opset_version=11,         # A stable ONNX version
            do_constant_folding=True, # A performance optimization
            input_names=['input'],    # A name for the model's input tensor
            output_names=['output'],  # A name for the model's output tensor
            dynamic_axes={'input' : {0 : 'batch_size'},    # Allow for variable batch size
                          'output' : {0 : 'batch_size'}}
        )
        print(f"\nSUCCESS: Model successfully exported to '{output_path}'")
    except Exception as e:
        print(f"\nERROR: Failed to export ONNX model. Error: {e}")

if __name__ == '__main__':
    # Create a dummy config file for demonstration if it doesn't exist
    if not os.path.exists('config.json'):
        print("Creating a dummy 'config.json'. Please edit it with your actual paths.")
        dummy_cfg = {
            "env_name": "mo-hopper-2obj-v5", # Make sure this matches your environment
            "continuous": True,
            "reward_size": 2,
            "reward_labels": ["Forward Speed", "Jumping Height"],
            "model_path": "./models/", # IMPORTANT: Change this to your model directory
            "initial_weights": [0.5, 0.5]
        }
        with open('config.json', 'w') as f:
            json.dump(dummy_cfg, f, indent=4)
            
    # Create a dummy model directory and file for demonstration
    # if not os.path.exists('./models/'):
    #     os.makedirs('./models/')
    #     # This part is just to make the script runnable out-of-the-box.
    #     # In your case, you will already have your trained model.
    #     print("Creating a dummy agent and model file 'models/main_ppo.rl_model'.")
    #     temp_env = mo_gym.make("mo-hopper-2obj-v5")
    #     temp_agent = ContinuousAgent(temp_env, reward_size=2)
    #     torch.save(temp_agent.state_dict(), './models/main_ppo.rl_model')


    # Run the export process
    export_policy_to_onnx(config_path='config.json')
