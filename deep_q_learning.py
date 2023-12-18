import numpy as np
import gymnasium as gym
from collections import deque
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, dead_cells, WT_list_length
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import time as t
from datetime import datetime
import matplotlib.pyplot as plt


# Model is trained on 15x15 by default

# Train model on 20x20
# n, m = 20, 20
# dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
#               (15, 16), (16, 16), (15, 17), (16, 17)]

# Train model on 25x25
# n, m = 25, 25
# dead_cells = [(5, 5), (5, 6), (6, 5), (6, 6), (5, 18), (5, 19), (6, 18), (6, 19), (18, 5), (19, 5), (18, 6), (19, 6),
#               (18, 18), (18, 19), (19, 18), (19, 19), (7, 7), (7, 6), (7, 5), (7, 18), (7, 19), (18, 7), (19, 7),
#               (5, 7), (6, 7), (5, 17), (6, 17), (7, 17), (17, 5), (17, 6), (17, 7), (17, 17), (17, 18), (17, 19),
#               (18, 17), (19, 17)]

# Define the Deep Q Network
class DQNAgent:
    """
    A Deep Q Network agent for reinforcement learning.
    The agent can remember past actions, choose the next action, and update its model based on the rewards received.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the neural network model.
        :return: The compiled model.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the agent's memory.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state.
        :param done: Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, action_mask):
        """
        Choose the next action.
        :param state: The current state.
        :param action_mask: A mask that indicates the valid actions.
        :return: The chosen action.
        """

        # randomly choose an action with probability epsilon
        if np.random.rand() <= self.epsilon:
            # add or remove a turbine with equal probability
            if np.sum(action_mask[:len(action_mask)//2]) == 0 or np.sum(action_mask[len(action_mask)//2:]) == 0:
                return np.random.choice(np.arange(self.action_size), p=action_mask / np.sum(action_mask))
            return np.random.choice(np.arange(self.action_size),
                                    p=np.concatenate((action_mask[:len(action_mask)//2] / np.sum(action_mask[:len(action_mask)//2]) * 0.5,
                                                      action_mask[len(action_mask)//2:] / np.sum(action_mask[len(action_mask)//2:]) * 0.5)))
        # choose the best action based on the current state
        act_values = self.model.predict(state)[0]
        act_values = np.where(action_mask, act_values, -np.inf)
        return np.argmax(act_values)

    def replay(self, batch_size):
        """
        Train the model using a batch of experiences from the memory.
        :param batch_size: The size of the batch to use for training.
        """
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        total_loss = 0  # Initialize total loss for the batch
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        average_loss = total_loss / len(minibatch)
        return average_loss


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
episodes = 40
steps_per_episode = 300
all_rewards = []
all_losses = []

# Training loop
for episode in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    action_mask = info["action_mask"]
    total_reward = 0
    # action mask is used to mask out invalid actions
    action_mask = info["action_mask"]
    for time in range(steps_per_episode):
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
        total_reward += reward

        if done or time == steps_per_episode - 1 or sum(action_mask) == 0:
            break

    if len(agent.memory) > batch_size:
        average_loss = agent.replay(batch_size)
        all_losses.append(average_loss)  # Record loss
    all_rewards.append(total_reward)  # Record reward

    print(f"Episode: {episode}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

# Calculate metrics
total_reward = np.sum(all_rewards)
average_reward = np.mean(all_rewards)
max_reward = np.max(all_rewards)
min_reward = np.min(all_rewards)
cumulative_rewards = np.cumsum(all_rewards)

total_loss = np.sum(all_losses)
average_loss = np.mean(all_losses)
max_loss = np.max(all_losses)
min_loss = np.min(all_losses)
std_dev_rewards = np.std(all_rewards)
std_dev_losses = np.std(all_losses)

# Display metrics
print("Reward Metrics:")
print(f"Total Reward: {total_reward}")
print(f"Average Reward: {average_reward}")
print(f"Max Reward: {max_reward}")
print(f"Min Reward: {min_reward}")


# Plotting the cumulative reward
plt.figure(figsize=(6, 5))
plt.plot(cumulative_rewards)
plt.title('Cumulative Reward Over Time')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.show()

# First plot: Rewards per Step
plt.figure(figsize=(6, 5))
plt.plot(all_rewards)
plt.title('Rewards per Training Episode')
plt.xlabel('Training Episode')
plt.ylabel('Reward')
plt.show()

# Second plot: Loss per Training Step
plt.figure(figsize=(6, 5))
plt.plot(all_losses)
plt.title('Loss per Training Episode')
plt.xlabel('Training Episode')
plt.ylabel('Loss')
plt.show()

agent.model.save(f'trained_model_{m}x{n}.keras')
print("Trained model saved to:", f'trained_model_{m}x{n}.keras')

# Close the environment after training
env.close()
