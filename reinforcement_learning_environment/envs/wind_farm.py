import math

from problem import spacing_distance, MAX_WT_number, objective_function, WT_list, WT_max_number, \
    WT_list_length
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class WindFarmEnv(gym.Env):
    """
    A custom environment for reinforcement learning that simulates a wind farm.
    The agent's task is to place wind turbines on a grid, maximizing the power output
    while minimizing the cost and ensuring the turbines are spaced correctly.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000000}

    def __init__(self, render_mode=None, x_size=20, y_size=20, dead_cells=None):
        """
        Initialize the environment.
        :param render_mode: Mode to render the environment. Can be 'human' or 'rgb_array'.
        :param x_size: Width of the grid.
        :param y_size: Height of the grid.
        :param dead_cells: List of cells that are not available for placing turbines.
        """
        self.dead_cells = [] if dead_cells is None else dead_cells
        self.initial_dead_cells = dead_cells.copy()
        self._grid_state = np.zeros((x_size, y_size), dtype=int)
        self._solution = []
        self._best_solution = []
        self.fitness_value = 0.003
        self.best_fitness_value = 0.003
        self.x_size = x_size
        self.y_size = y_size
        self.window_size = 512  # The size of the PyGame window
        for cell in self.dead_cells:
            self._grid_state[cell[0]][cell[1]] = 3

        self.observation_space = spaces.Box(low=0, high=3, shape=(x_size, x_size), dtype=int)

        self.action_space = spaces.Discrete(x_size * y_size * 2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Get the current state of the environment.
        :return: The current state of the environment.
        """
        return self._grid_state

    def _get_info(self):
        """
        Get additional information about the environment.
        :return: A dictionary containing additional information about the environment.
        """
        return {
            "num_of_turbines": len(self._solution),
            "solution": self._solution,
            "best_fitness_value": self.best_fitness_value,
            "best_solution": self._best_solution,
            "grid_state": self._grid_state,
            "action_mask": np.concatenate(((self._grid_state == 0).flatten(order='F'), (self._grid_state == 1).flatten(order='F')))
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        :param seed: Seed for the random number generator. Not used here.
        :param options: Additional options for resetting the environment. Not used here.
        :return: The initial state of the environment and additional information.
        """
        # Resetting the environment to its initial state

        self._grid_state = np.zeros((self.x_size, self.y_size), dtype=int)
        self.dead_cells = self.initial_dead_cells.copy()
        for cell in self.dead_cells:
            self._grid_state[cell[0]][cell[1]] = 3
        self._solution = []
        self._best_solution = []
        self.fitness_value = 0.003
        self.best_fitness_value = 0.003

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.
        :param action: The action to execute.
        :return: The new state, reward, done (whether the episode is finished), info (additional information).
        """

        # Executing the action in the environment

        # update grid state when adding a turbine
        def _update_grid_state_add(cell):
            for dx in list(range(-spacing_distance, spacing_distance + 1)):
                for dy in list(range(-spacing_distance, spacing_distance + 1)):
                    if (self.x_size > cell[0] + dx >= 0 and self.y_size > cell[1] + dy >= 0
                            and self._grid_state[cell[0] + dx, cell[1] + dy] != 3):
                        self._grid_state[cell[0] + dx, cell[1] + dy] = 2
            self._grid_state[cell[0], cell[1]] = 1

        # update grid state when removing a turbine
        def _update_grid_state_remove(cell):
            self._grid_state = np.zeros((self.x_size, self.y_size), dtype=int)
            self.dead_cells.append(cell)
            for cell in self.dead_cells:
                self._grid_state[cell[0]][cell[1]] = 3
            for cell in self._solution:
                _update_grid_state_add((math.floor(cell[0]), math.floor(cell[1])))

        action_type = action // (self.x_size * self.y_size)  # 0: add, 1: remove
        action = action % (self.x_size * self.y_size)
        grid_x = action % self.x_size
        grid_y = action // self.x_size
        if action_type == 0:
            # check if the cell is available for placing a turbine
            if self._grid_state[grid_x, grid_y] == 0:
                _update_grid_state_add((grid_x, grid_y))
                self._solution.append((grid_x+0.5, grid_y+0.5))
                new_fitness_value, _, satisfies = objective_function(self._solution, self.x_size, self.y_size)
                new_fitness_value = 0.003 if math.isnan(new_fitness_value) else new_fitness_value
                # calculate reward
                reward = ((self.fitness_value - new_fitness_value) * 10000) + 10
                self.fitness_value = new_fitness_value
                if self.fitness_value < self.best_fitness_value:
                    self.best_fitness_value = self.fitness_value
                    self._best_solution = self._solution.copy()
            else:
                reward = 0
        else:
            # check if the cell is occupied by a turbine
            if self._grid_state[grid_x, grid_y] == 1:
                self._solution.remove((grid_x + 0.5, grid_y + 0.5))
                _update_grid_state_remove((grid_x, grid_y))
                new_fitness_value, _, satisfies = objective_function(self._solution, self.x_size, self.y_size)
                new_fitness_value = 0.003 if math.isnan(new_fitness_value) else new_fitness_value
                # calculate reward
                reward = ((self.fitness_value - new_fitness_value) * 10000) - 10
                self.fitness_value = new_fitness_value
                if self.fitness_value < self.best_fitness_value:
                    self.best_fitness_value = self.fitness_value
                    self._best_solution = self._solution.copy()
            else:
                reward = 0

        # check if the episode is finished. The episode is finished if no turbine can be added or removed.
        terminated = all([self._grid_state[x, y] == 3 for x in range(self.x_size) for y in range(self.y_size)])
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def render(self):
        """
        Render the environment.
        :return: The rendered environment in case of rgb_array mode.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render one frame of the environment.
        :return: The rendered frame.
        """
        # Rendering one frame of the environment
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.x_size, self.window_size / self.y_size
        )
        colors = {0: (255, 255, 255), 2: (0, 255, 255), 1: (0, 0, 255), 3: (169, 169, 169)}

        for x in range(self._grid_state.shape[0]):
            for y in range(self._grid_state.shape[1]):
                value = self._grid_state[x, y]
                color = colors.get(int(value), (0, 0, 0))
                pygame.draw.rect(canvas,
                                 color,
                                 pygame.Rect(x * pix_square_size[0], y * pix_square_size[1],
                                             pix_square_size[0], pix_square_size[1])
                                 )

        # Finally, add some gridlines
        for x in range(self.y_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size[1] * x),
                (self.window_size, pix_square_size[1] * x),
                width=3,
            )
        for x in range(self.x_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size[0] * x, 0),
                (pix_square_size[0] * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close the environment and clean up resources.
        """
        # Closing the environment and cleaning up resources
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
