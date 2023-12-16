import numpy as np
import gymnasium as gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from datetime import datetime

# n,m = 25,25 dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),
# (18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,
# 6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]

n, m = 20, 20
dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
              (15, 16), (16, 16), (15, 17), (16, 17)]


# Define the Deep Q Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(np.arange(self.action_size),
                                    p=np.concatenate((action_mask[:len(action_mask) // 2] * 0.5 / np.sum(
                                        action_mask[:len(action_mask) // 2]),
                                                      action_mask[len(action_mask) // 2:] * 0.5 / np.sum(
                                                          action_mask[len(action_mask) // 2:]))))
        act_values = self.model.predict(state)[0]
        act_values = np.where(action_mask, act_values, -np.inf)
        return np.argmax(act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


gym.envs.registration.register(
    id="WindFarm-v0",
    entry_point="reinforcement_learning_environment.envs:WindFarmEnv",
    max_episode_steps=500,
)

# Create and wrap the custom environment
env = gym.make('WindFarm-v0', dead_cells=dead_cells, x_size=m, y_size=n)
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

# Initialize DQN agent
agent = DQNAgent(state_size, action_size)

# Training parameters
batch_size = 32
episodes = 1000
steps_per_episode = 50

# Training loop
for episode in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    action_mask = info["action_mask"]
    for time in range(steps_per_episode):  # You can adjust the maximum number of steps per episode
        env.render()

        # Choose action
        action = agent.act(state, action_mask)

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _, info = env.step(action)
        action_mask = info["action_mask"]
        next_state = np.reshape(next_state, [1, state_size])

        # Store the experience in the replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update the current state
        state = next_state
        # If the episode is done, break from the loop
        # print(info)
        # t.sleep(2)
        if done or time == steps_per_episode - 1 or sum(action_mask) == 0:
            print("episode: {}/{}".format(episode, episodes))
            break

    # Train the agent using experience replay
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

agent.model.save(f'trained_model{datetime.now().strftime("%d%m%y%H%M%S")}.keras')
print("Trained model saved to:", f'trained_model{datetime.now().strftime("%d%m%y%H%M%S")}.keras')

play_episodes = 1
for episode in range(play_episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    action_mask = info["action_mask"]
    agent.epsilon = 0
    for time in range(steps_per_episode):
        env.render()

        # Choose action
        action = agent.act(state, action_mask)

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

# Close the environment after training
env.close()
