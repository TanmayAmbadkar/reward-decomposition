import gymnasium as gym
from pysr import PySRRegressor
import numpy as np

env = gym.make("LunarLander-v3", continuous = True)

states, actions, next_states = [], [], []

state, info = env.reset()
for i in range(100000):
    
    action = env.action_space.sample()
    next_state, rew, done, trunc, info = env.step(action)

    states.append(state)
    actions.append(action)
    next_states.append(next_state)
    state = next_state
    if done or trunc:
        state, info = env.reset()

model = PySRRegressor(
    maxsize=15,  # Increase complexity allowed
    niterations=200,  # Increase the number of iterations
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "sqrt",
        "square",
        "cube",
        "log",      # Added log operator
        "tanh"      # Added tanh operator
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # You might consider a custom loss that penalizes multi-step error as well.
)


y = np.array(next_states)
x = np.array(states)
u = np.array(actions)

from sklearn.preprocessing import StandardScaler

# Combine state and action data for scaling.
inp = np.hstack([x, u])
inp_scaler = StandardScaler()
y_scaler = StandardScaler()

inp_scaled = inp_scaler.fit_transform(inp)
y_scaled = y_scaler.fit_transform(y)

model.fit(inp_scaled, y_scaled)


# inp = np.hstack([x, u])

# model.fit(inp, y)
print(model)

next_states = []
pred_next_states = []

state, info = env.reset()
pred_state = state  # initial predicted state equals the actual state

for i in range(1000):
    action = env.action_space.sample()
    next_state, rew, done, trunc, info = env.step(action)
    
    # Create the input vector by concatenating the predicted state and the action.
    input_vector = np.concatenate([pred_state, action], axis=None).reshape(1, -1)
    
    # Scale the input before feeding it to the model.
    input_scaled = inp_scaler.transform(input_vector)
    
    # Get the prediction (in the scaled space)
    pred_scaled = model.predict(input_scaled)
    
    # Inverse-transform the prediction to get it back into the original scale.
    pred_state = y_scaler.inverse_transform(pred_scaled).flatten()
    
    next_states.append(next_state)
    pred_next_states.append(pred_state)
    
    state = next_state
    if done or trunc:
        state, info = env.reset()
        pred_state = state

        next_states = np.array(next_states)
        pred_next_states = np.array(pred_next_states).reshape(len(next_states), -1)
        
        plt.figure()
        plt.plot(next_states[:, 0], next_states[:, 1], color="green", label="True")
        plt.plot(pred_next_states[:, 0], pred_next_states[:, 1], color="blue", label="Predicted")
        plt.legend()
        plt.savefig(f"xy_{i}.png")

        plt.figure()
        plt.plot(next_states[:, 2], next_states[:, 3], color="green", label="True")
        plt.plot(pred_next_states[:, 2], pred_next_states[:, 3], color="blue", label="Predicted")
        plt.legend()
        plt.savefig(f"vxvy_{i}.png")
        
        print("R2 score:", r2_score(next_states, pred_next_states))
        print("MSE:", mean_squared_error(next_states, pred_next_states))
        
        next_states = []
        pred_next_states = []

