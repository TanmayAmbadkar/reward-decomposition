import numpy as np
import gymnasium as gym
from pysr import PySRRegressor

# Create the CartPole environment.
env = gym.make("CartPole-v1")
gamma = 0.99       # Discount factor
learning_rate = 0.01  # A step size for our “policy gradient” update.
state_size = env.observation_space.shape[0]  # For CartPole, 4 dimensions.
n_actions = env.action_space.n               # Typically 2 actions.

# We'll represent our policy with one symbolic model per action.
# Each model outputs a "logit" (unnormalized score) for its action given the state.
policy_models = [
    PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "abs", "sqrt"],
        population_size=50,
        model_selection="best",
        maxsize=20,
        loss="L2DistLoss()",
        verbosity=0,
        warm_start = True
    )
    for _ in range(n_actions)
]

# Dummy initialization (each model takes a state and returns 0 as initial logit).
dummy_state = np.zeros((1, state_size))
dummy_target = np.array([0.0])
for model in policy_models:
    model.fit(dummy_state, dummy_target)

def get_policy_logits(state):
    """
    Given a state, return a vector of logits (one per action).
    """
    state = np.array(state).reshape(1, -1)
    logits = np.array([model.predict(state)[0] for model in policy_models])
    return logits

def get_action_probs(state):
    """
    Compute softmax over logits to obtain probabilities.
    """
    logits = get_policy_logits(state)
    # For numerical stability subtract the max logit.
    # exp_logits = np.exp(logits - np.max(logits))
    probs = np.nan_to_num(logits, 0)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

def select_action(state):
    """
    Sample an action according to the policy's probability distribution.
    """
    probs = get_action_probs(state)
    return np.random.choice(n_actions, p=probs)

def update_policy(episode_transitions):
    """
    Given a list of transitions (state, action, reward), compute the return G for each time step,
    then update the symbolic policy models to push the logit for the taken action in the direction of G.
    
    Here we "simulate" a policy gradient update by creating target logit values:
      target = current logit + learning_rate * G
    for the taken action.
    """
    # First, compute the returns for the episode.
    returns = []
    G = 0
    # Process transitions in reverse order.
    for (_, _, reward) in reversed(episode_transitions):
        G = reward + gamma * G
        returns.insert(0, G)
    
    # For each action, gather states where that action was taken and compute targets.
    # For transitions where an action was not taken, we use the current logit as the target.
    # We'll update each model separately.
    # First, collect all states for the episode.
    states = np.array([s for (s, _, _) in episode_transitions])
    # Get current logits for all transitions.
    current_logits = np.array([get_policy_logits(s) for s in states])
    # Build target matrix (shape: [num_transitions, n_actions]) that initially equals current_logits.
    targets = current_logits.copy()
    
    # For each transition, modify the target for the taken action.
    for i, (s, a, _) in enumerate(episode_transitions):
        targets[i, a] += learning_rate * returns[i]
    
    # Now, update each policy model using the states and the corresponding target for that action.
    for a in range(n_actions):
        # Find indices where we have an update for action a.
        indices = [i for i, (_, act, _) in enumerate(episode_transitions) if act == a]
        if indices:
            X = states[indices]  # Shape: (num_samples, state_size)
            y = targets[indices, a]  # The desired logit for action a.
            policy_models[a].fit(X, y)

# --- Training Loop using a REINFORCE-like update ---
num_episodes = 100
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()  # Gymnasium returns (observation, info)
    done = False
    episode_transitions = []  # Will store tuples of (state, action, reward)
    total_reward = 0

    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_transitions.append((state, action, reward))
        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    update_policy(episode_transitions)
    print(f"Episode {episode+1}, Reward: {total_reward}")

# --- (Optional) Inspect the learned symbolic policy expressions ---
for a, model in enumerate(policy_models):
    print(f"\nLearned policy expression for action {a} (logit):")
    print(model.sympy())

env.close()
