import numpy as np
import gymnasium as gym
from keras.models import load_model

n, m = 20, 20
dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
              (15, 16), (16, 16), (15, 17), (16, 17)]

gym.envs.registration.register(
    id="WindFarm-v0",
    entry_point="reinforcement_learning_environment.envs:WindFarmEnv",
    max_episode_steps=500,
)
# Create and wrap the custom environment
env = gym.make('WindFarm-v0', dead_cells=dead_cells, x_size=m, y_size=n, render_mode="human")
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

model = load_model('trained_model071223105036.keras')

# Show the model architecture
model.summary()

play_episodes = 1
steps_per_episode = 20
for episode in range(play_episodes):
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
        # If the episode is done, break from the loop
        if done or time == steps_per_episode - 1 or sum(action_mask) == 0:
            print("episode: {}/{}".format(episode, play_episodes))
            print(info["best_solution"])
            print(info["best_fitness_value"])
            break
