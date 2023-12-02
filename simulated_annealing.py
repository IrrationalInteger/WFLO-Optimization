import math
import time
import random
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from drawings import draw_iterations_against_solution, draw_number_of_turbines_against_power_and_objective

matplotlib.use('TkAgg')
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, WT_list, WT_max_number, dead_cells


# Adds a new turbine while respecting the spacing distance and dead cells
def add_new_WT(solution, exclusion_list, m, n):
    for i in range(len(solution)):
        solution[i] = (solution[i][0] - 0.5, solution[i][1] - 0.5)

    def is_valid(x, y):
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx, y + dy) in solution:
                    return False
        return True

    i_max = m * n
    i = 0
    while i < i_max:
        x = random.randint(0, m - 1)
        y = random.randint(0, n - 1)
        new_tuple = (x, y)
        if new_tuple not in exclusion_list and is_valid(x, y):
            solution.append((x, y))
            break
        i += 1
    for i in range(len(solution)):
        solution[i] = (solution[i][0] + 0.5, solution[i][1] + 0.5)


# Generates a new solution from the previous one by randomly adding, removing, or moving a single wind turbine while
# respecting the spacing distance and the dead cells
def generate_neighbour_solution(solution, exclusion_list, m, n):
    op = random.randint(1, 2) if len(solution) == MAX_WT_number else random.randint(0, 1) if len(
        solution) == 1 else random.randint(0, 2)
    solution = solution.copy()
    if op == 0:  # op = 0, add a WT at random location
        add_new_WT(solution, exclusion_list, m, n)
    elif op == 1:  # op = 1, change the location of one WT ????
        solution.pop(random.randint(0, len(solution) - 1))
        add_new_WT(solution, exclusion_list, m, n)
    else:  # op = 2, remove a random WT
        solution.pop(random.randint(0, len(solution) - 1))
    return solution


# Linear scheduling formula
def calculate_T_linear(T, step):
    return T - step


# Geometric scheduling formula
def calculate_T_geometric(T, factor):
    return factor * T


# SA parameters
T_initial = 500  # Initial temperature of annealing
T_final = 0  # Final temperature of annealing
iteration_per_T = 2  # The number of solutions generated per temperature
i_max = 500  # Artificial stopping condition
factor = 1  # Factor used for decreasing temperature. Used as step for linear and factor for geometric.
fitness_value_scaling_factor = 10000000  # Scaling of fitness for temperature
calculate_T = calculate_T_linear  # Choice of scheduling function


# Performs simulated annealing using the given parameters
def simulated_annealing(visualise):
    start = time.perf_counter()
    current_solution = WT_list  # Initial solution set to the randomized layout
    current_fitness, __, _ = objective_function(current_solution, m, n)  # Initial fitness
    best_solution = current_solution  # Best solution so far
    best_fitness = current_fitness  # Best fitness so far
    T_current = T_initial  # Temperature init
    objective_vs_N = [float('inf')] * (WT_max_number + 1)  # Objective vs Number of Turbines for plotting
    power_vs_N = [float('inf')] * (WT_max_number + 1)  # Power vs Number of Turbines for plotting
    objective_vs_I = []  # Objective vs iterations for plotting
    optimal_objective_vs_I = []  # Optimal Objective vs iterations for plotting
    i = 0
    probability = 0
    if visualise:
        draw_simulation_annealing()
        time.sleep(
            3)  # Delay to allow grid to properly initialize. May need to rerun code multiple times for it to work
    while T_current > T_final and i <= i_max:
        for j in range(iteration_per_T):
            # Generate new solution
            new_solution = generate_neighbour_solution(current_solution, dead_cells, m, n)
            new_fitness, new_power, satisfies = objective_function(new_solution, m, n)
            objective_vs_N[len(current_solution)] = new_fitness if objective_vs_N[
                                                                       len(current_solution)] > new_fitness else \
            objective_vs_N[len(current_solution)]
            power_vs_N[len(current_solution)] = new_power if power_vs_N[len(current_solution)] > new_power else \
            power_vs_N[len(current_solution)]
            # Calculate the delta in fitness
            objective_function_change = new_fitness - current_fitness
            if objective_function_change < 0:  # If negative, then improvement: keep this solution
                current_solution = new_solution
                current_fitness = new_fitness
            else:  # Else randomize a number from 0 to 1
                random_number = random.uniform(0, 1)
                # Scale fitness to range of temperature
                objective_function_change_scaled = objective_function_change * fitness_value_scaling_factor
                # Boltzmann-Gibbs calculation for probability
                probability = math.exp(-objective_function_change_scaled / T_current)
                if random_number < probability:  # If greater than random_number then keep
                    current_solution = new_solution
                    current_fitness = new_fitness
            if current_fitness < best_fitness and satisfies:  # Accept the new solution as optimal iff it has better fitness and also satisifies the power constraint
                best_solution = current_solution
                best_fitness = current_fitness
            if visualise:
                ax2.set_title("Best Solution"
                              "\nFitness:" + str(round(best_fitness, 8)))
                # Draw new generation
                plt.pause(0.1)  # Pause to view the updated plot
                ax1.set_title('Generated Solution'
                              '\nTemperature:' + str(T_current) +
                              "\nIteration:" + str(j) +
                              "\nProbability:" + str(round(probability, 3)) +
                              "\nFitness:" + str(round(current_fitness, 8)))
                update_grid_annealing(grid1, cax1, current_solution, best_solution, True)
                update_grid_annealing(grid2, cax2, None, best_solution, False)

            print(current_fitness)
            print(current_solution)
            objective_vs_I.append(current_fitness)
            optimal_objective_vs_I.append(best_fitness)
        T_current = calculate_T(T_current, factor)
        i = i + 1
        print(f"T_current : {T_current}")
    if (visualise):
        draw_number_of_turbines_against_power_and_objective(power_vs_N, objective_vs_N)
        draw_iterations_against_solution(objective_vs_I, False)
        draw_iterations_against_solution(optimal_objective_vs_I, True)
    end = time.perf_counter()
    return best_solution, best_fitness, objective_vs_N, power_vs_N, objective_vs_I, optimal_objective_vs_I, end - start


# Draws a live simulation for simulated annealing
def draw_simulation_annealing():
    # Create a white grid
    global grid1
    grid1 = np.ones((n, m, 4))
    global grid2
    grid2 = np.ones((n, m, 4))
    global ax1, ax2, cax1, cax2
    # Create a figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Create the initial plots with a color map
    cax1 = ax1.imshow(grid1, cmap='bwr', vmin=0, vmax=1)
    cax2 = ax2.imshow(grid2, cmap='bwr', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(n))
    ax2.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(m))
    ax2.set_yticks(np.arange(m))
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    print(np.arange(m)[::-1])

    # Function to add gridlines to an axis
    def add_gridlines(ax):
        ax.set_xticks(np.arange(-.5, m, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", length=0)

    # Add gridlines
    add_gridlines(ax1)
    add_gridlines(ax2)
    ax1.set_title('Generated Solution\nTemperature:\nIteration:')
    ax2.set_title('Best Solution')

    plt.show(block=False)


# Updates the live simulation for simulated annealing
def update_grid_annealing(grid, cax, coords_red, coords_blue, blue_trans):
    # Reset the grid to all white with full opacity
    grid[:, :, :3] = 1  # All pixels white
    grid[:, :, 3] = 0  # Full opacity (no transparency)
    for coord in dead_cells:
        y, x = int(coord[0]), int(coord[1])
        if 0 <= x < n and 0 <= y < m:
            grid[x, y, :3] = [0.2, 0.2, 0.2]
            grid[x, y, 3] = 1
    # Define the RGBA values for grey and blue with 50% transparency
    grey_color = np.array([0.5, 0.5, 0.5, 0.8])
    red_color = np.array([1, 0, 0, 0.4])
    blue_color = np.array([0, 0, 1, 0.6])

    # Create a copy of the grid representing the current state
    current_colors = np.copy(grid)

    # Apply the new coordinates for grey
    if (coords_red != None):
        for coord in coords_red:
            y, x = int(coord[0]), int(coord[1])
            if 0 <= x < n and 0 <= y < m:
                grid[x, y, :3] = red_color[:3]
                grid[x, y, 3] = 1

    # Apply the new coordinates for blue
    if (coords_blue != None):
        for coord in coords_blue:
            y, x = int(coord[0]), int(coord[1])
            if 0 <= x < n and 0 <= y < m:
                grid[x, y, 3] = 1

                if (np.array_equal(grid[x, y, :3], red_color[:3])):
                    grid[x, y, :3] = [1, 0, 0]
                    grid[x, y, 3] = 0.5

                else:
                    if (blue_trans):
                        grid[x, y, 3] = 0.3
                    grid[x, y, :3] = blue_color[:3]

    cax.set_data(grid)  # Update plot data # Update plot data


# Test case 1 is run by default

# Uncomment this block for test case 2
# n,m = 15,15
# dead_cells = [(2,2),(12,2),(2,12),(12,12)]
# T_initial = 1000
# factor = 0.95
# calculate_T = calculate_T_geometric

# Uncomment this block for test case 3
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# T_initial = 500
# factor = 1
# calculate_T = calculate_T_linear

# Uncomment this block for test case 4
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# T_initial = 1000
# factor = 0.95
# calculate_T = calculate_T_geometric

# Uncomment this block for test case 5
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# T_initial = 500
# factor = 1
# calculate_T = calculate_T_linear

# Uncomment this block for test case 6
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# T_initial = 1000
# factor = 0.95
# calculate_T = calculate_T_geometric


if __name__ == "__main__":
    simulated_annealing(True)
