import numpy as np
import gymnasium as gym
from keras.models import load_model
from problem import  m, n, dead_cells

model = load_model('trained_model_15x15.keras')

# Model is evaluated on 15x15 by default. 15x15 model is loaded by default

# Evaluate model on 20x20
# n, m = 20, 20
# dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
#               (15, 16), (16, 16), (15, 17), (16, 17)]
# model = load_model('trained_model_20x20.keras')


# Evaluate model on 20x20
# n, m = 25, 25
# dead_cells = [(5, 5), (5, 6), (6, 5), (6, 6), (5, 18), (5, 19), (6, 18), (6, 19), (18, 5), (19, 5), (18, 6), (19, 6),
#               (18, 18), (18, 19), (19, 18), (19, 19), (7, 7), (7, 6), (7, 5), (7, 18), (7, 19), (18, 7), (19, 7),
#               (5, 7), (6, 7), (5, 17), (6, 17), (7, 17), (17, 5), (17, 6), (17, 7), (17, 17), (17, 18), (17, 19),
#               (18, 17), (19, 17)]
# model = load_model('trained_model_25x25_2.keras')

gym.envs.registration.register(
    id="WindFarm-v0",
    entry_point="reinforcement_learning_environment.envs:WindFarmEnv",
    max_episode_steps=500,
)
# Create and wrap the custom environment
env = gym.make('WindFarm-v0', dead_cells=dead_cells, x_size=m, y_size=n, render_mode="human")
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n


# Show the model architecture
model.summary()

steps_per_episode = 200
state, info = env.reset()
state = np.reshape(state, [1, state_size])

action_mask = info["action_mask"]
action = 0
for time in range(steps_per_episode):
    env.render()

    q_values = model.predict(state)[0]
    action_mask[len(action_mask) // 2:] = False
    q_values = np.where(action_mask, q_values, -np.inf)

    # Choose action without exploration (epsilon=0)
    action = np.argmax(q_values)

    # Take the chosen action and observe the next state and reward
    next_state, reward, done, _, info = env.step(action)
    action_mask = info["action_mask"]
    next_state = np.reshape(next_state, [1, state_size])

    # Update the current state
    state = next_state
print(info["best_solution"])
print(info["best_fitness_value"])
