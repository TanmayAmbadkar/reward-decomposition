
1. Create environment using environment.yml file (recommended) or requirements.txt
2. Run main_ppo.py with the following example command
```bash
$ python main_ppo.py --env_id mo-hopper-v5 --total_timesteps 2000000
```
The rest of the parameters with their type are given below

- **env_id**: str = "mo-hopper-v5" - Environment identifier
- **env_is_discrete**: bool = False - Whether environment has discrete action space
- **num_envs**: int = 4 - Number of parallel environments
- **convex**: bool = True - Use convex reward decomposition
- **scalar_reward**: bool = False - Use scalar reward instead of vector reward
- **total_timesteps**: int = 5000000 - Total training timesteps
- **num_rollout_steps**: int = 2048 - Steps per rollout
- **update_epochs**: int = 10 - Number of update epochs per rollout
- **num_minibatches**: int = 32 - Number of minibatches per update
- **learning_rate**: float = 0.0003 - Learning rate for optimizer
- **gamma**: float = 0.995 - Discount factor for training
- **eval_gamma**: float = 0.99 - Discount factor for evaluation
- **gae_lambda**: float = 0.95 - GAE lambda parameter
- **surrogate_clip_threshold**: float = 0.2 - PPO clipping parameter
- **entropy_loss_coefficient**: float = 0.0000 - Entropy loss weight
- **policy_gradient_loss_coefficient**: float = 1.0 - Policy gradient loss weight
- **value_function_loss_coefficient**: float = 0.5 - Value function loss weight
- **normalize_advantages**: bool = True - Normalize advantages
- **normalize_observations**: bool = True - Normalize observations
- **normalize_rewards**: bool = True - Normalize rewards
- **clip_value_function_loss**: bool = False - Clip value function loss
- **max_grad_norm**: float = 0.5 - Maximum gradient norm for clipping
- **target_kl**: float = None - Target KL divergence threshold
- **anneal_lr**: bool = False - Anneal learning rate
- **rpo_alpha**: float = None - RPO alpha parameter
- **seed**: int = 1 - Random seed
- **torch_deterministic**: bool = True - Use deterministic PyTorch operations
- **capture_video**: bool = False - Capture video recordings
- **use_tensorboard**: bool = True - Enable TensorBoard logging
- **save_model**: bool = True - Save trained model

