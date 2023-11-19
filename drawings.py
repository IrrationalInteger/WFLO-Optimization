
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Draws a live simulation for genetic algorithm
def draw_simulation_genetic(x_range):
    fig, ax = plt.subplots()
    ax.scatter(x_range, [0] * len(x_range))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Values')

    plt.show(block=False)  # Use non-blocking show
    return ax

# Updates the live simulation for genetic algorithm
def update_plot_genetic( ax, x_value, y_values, max_y=None, min_y=None):

    # Normalize y-values
    normalized_y = (np.array(y_values) - min(y_values)) / (max(y_values) - min(y_values))

    # Create a scatter plot
    scatter_plot = ax.scatter([x_value] * len(y_values), y_values, c=normalized_y, cmap='viridis')

    # Update max and min y-values

    current_max_y = max(max_y,max(scatter_plot.get_offsets()[:, 1])) if max_y is not None else max(scatter_plot.get_offsets()[:, 1])
    current_min_y = min(min_y,min(scatter_plot.get_offsets()[:, 1])) if min_y is not None else min(scatter_plot.get_offsets()[:, 1])
    # Update y-axis limits based on the new data and historical max/min values
    ax.set_ylim(current_min_y, current_max_y)

    # Store and set x-axis limits
    x_limits = ax.get_xlim()

    # Set x-axis limits back to the initial values
    ax.set_xlim(x_limits)

    ax.relim()  # Recalculate the data limits for proper autoscaling
    ax.autoscale_view(scalex=False)  # Autoscale the y-axis only

    plt.draw()
    plt.pause(0.1)

    return current_max_y, current_min_y  # Return updated max and min y-values
# Draws the final solution for genetic algorithm
def draw_solution_genetic(solution, fitness, dead_cells, m, n):
    # Create a white grid
    global grid
    grid = np.ones((m, n, 4))
    global ax1, cax
    # Create a figure with two subplots (side by side)
    fig, ax1 = plt.subplots(1, 1)

    # Create the initial plots with a color map
    cax = ax1.imshow(grid, cmap='bwr', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(m))
    ax1.invert_yaxis()

    # Function to add gridlines to an axis
    def add_gridlines(ax):
        ax.set_xticks(np.arange(-.5, m, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", length=0)

    # Add gridlines
    add_gridlines(ax1)
    ax1.set_title("Best Solution"
                  "\nFitness:" + str(round(fitness, 8)))

    plt.show(block=False)
    # Reset the grid to all white with full opacity
    grid[:, :, :3] = 1  # All pixels white
    grid[:, :, 3] = 0  # Full opacity (no transparency)
    for coord in dead_cells:
        y, x = int(coord[0]), int(coord[1])
        if 0 <= x < n and 0 <= y < m:
            grid[x, y, :3] = [0.2, 0.2, 0.2]
            grid[x, y, 3] = 1

    # Apply the new coordinates for blue
    for coord in solution:
        y, x = int(coord[0]), int(coord[1])
        if 0 <= x < n and 0 <= y < m:
            grid[x, y, :3] = [0, 0, 1]
            grid[x, y, 3] = 1

    cax.set_data(grid)  # Update plot data # Update plot data

# Plots the number of turbines against the power and objective function values
def draw_number_of_turbines_against_power_and_objective(power_data, objective_data):
    # Clean from float.inf
    temp_objective_vs_N = [t for t in objective_data if t != float('inf')]
    temp_power_vs_N = [t for t in power_data if t != float('inf')]
    bounds = [min(temp_objective_vs_N), max(temp_objective_vs_N), min(temp_power_vs_N), max(temp_power_vs_N)]
    objective_data = [(index, t) for index, t in enumerate(objective_data) if t != float('inf')]
    power_data = [(index, t) for index, t in enumerate(power_data) if t != float('inf')]

    power_x, power_y = zip(*power_data)
    objective_x, objective_y = zip(*objective_data)

    # Create a figure and a subplot
    fig, ax1 = plt.subplots()

    # Plot the first line with 'power_y' on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of turbines')
    ax1.set_ylabel('Power', color=color)
    ax1.plot(power_x, power_y, color=color, label='Power')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bounds[2], bounds[3])  # Set the limits of the left y-axis

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Objective', color=color)
    ax2.plot(objective_x, objective_y, color=color, label='Objective',
             linestyle='--')  # We use a dashed line for the second line
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bounds[0], bounds[1])  # Set the limits of the right y-axis

    # Set the limits of the x-axis
    ax1.set_xlim(0, len(power_data))

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Show the plot
    plt.show()


# Plots the number of generated solutions against the objective function
def draw_iterations_against_solution(objective_data, optimal):
    # Create a figure and a subplot
    fig, ax1 = plt.subplots()
    annealing_iterations = len(objective_data)
    # Plot the first line with 'power_y' on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Optimal Objective Function' if optimal else "Generated Objective Function", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    print(objective_data)
    ax1.set_ylim(min(objective_data), max(objective_data))  # Set the limits of the left y-axis
    objective_data = [(index, t) for index, t in enumerate(objective_data)]
    objective_x, objective_y = zip(*objective_data)
    ax1.plot(objective_x, objective_y, color=color, label='Objective')

    # Set the limits of the x-axis
    ax1.set_xlim(0, annealing_iterations)

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Show the plot
    plt.show()
