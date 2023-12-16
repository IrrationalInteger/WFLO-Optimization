import concurrent.futures
import math
import random
import time
from multiprocessing import Manager
import matplotlib
from problem import spacing_distance, objective_function, m, n, dead_cells
from drawings import draw_iterations_against_solution, \
    draw_simulation_population, update_plot_population, draw_solution_population
from functions import generate_random_tuples

matplotlib.use('TkAgg')

# Moth flame parameters
population_size = 50  # Number of moths and flames per iteration
max_iterations = 100  # Maximum number of allowed iterations
b = 0.3  # Spiral size coefficient
lower_bound = -1  # Lower bound for spiral coefficient
upper_bound = 1  # Upper bound for spiral coefficient


def calculate_bound_linear(t):
    return t - 1 / max_iterations


# Geometric scheduling formula
def calculate_bound_geometric(t):
    return 2 ** (1 / 99) * t


calculate_lower_bound = calculate_bound_linear  # Choice of scheduling function


def add_new_WT(solution, exclusion_list, m_inner, n_inner):
    for i in range(len(solution)):
        solution[i] = (solution[i][0] - 0.5, solution[i][1] - 0.5)

    def is_valid(x_inner, y_inner):
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if (x_inner + dx, y_inner + dy) in solution:
                    return False
        return True

    i_max = m_inner * n_inner
    i = 0
    while i < i_max:
        x = random.randint(0, m_inner - 1)
        y = random.randint(0, n_inner - 1)
        new_tuple = (x, y)
        if new_tuple not in exclusion_list and is_valid(x, y):
            solution.append((x, y))
            break
        i += 1
    for i in range(len(solution)):
        solution[i] = (solution[i][0] + 0.5, solution[i][1] + 0.5)


def init_moth():
    solution = generate_random_tuples(random.randint(1, 5), dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution, n, m)
    solution.sort(key=lambda x: (x[0], x[1]))
    return solution, fitness


def init_population():
    population = []
    population_fitness = []
    flames = []
    flames_fitness = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(init_moth) for _ in range(population_size)]
        for f in concurrent.futures.as_completed(results):
            population.append(f.result()[0])
            population_fitness.append(f.result()[1])
            flames.append(f.result()[0])
            flames_fitness.append(f.result()[1][0])

    return population, population_fitness, flames, flames_fitness


def calculate_position(moth, flame):
    if not moth:
        return []

    error_cells = []

    def calculate_error_cells(x_inner, y_inner, j):
        moth.pop(j)
        error_cells_new = []
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if (x_inner + dx, y_inner + dy) in moth:
                    error_cells_new.append((x_inner + dx, y_inner + dy))
        # remove error cells from moth
        for error_cell in error_cells_new:
            moth[moth.index(error_cell)] = (-100, -100)
            error_cells.append(error_cell)
        # add (x,y) to its position in j
        moth.insert(j, (x_inner, y_inner))
        return error_cells

    i = 0
    t = random.uniform(lower_bound, upper_bound)
    # Find the minimum length of the two lists
    min_length = min(len(moth), len(flame))

    # Perform element-wise position calculation up to the minimum length
    while i < min_length:

        x = abs(flame[i][0] - moth[i][0]) * math.exp(b * t) * math.cos(
            2 * math.pi * t) + flame[i][0]
        y = abs(flame[i][1] - moth[i][1]) * math.exp(b * t) * math.cos(
            2 * math.pi * t) + flame[i][1]
        if x > m - 0.5:
            x = 14.5
        if y > n - 0.5:
            y = 14.5
        if x < 0.5:
            x = 0.5
        if y < 0.5:
            y = 0.5

        moth[i] = (x, y)

        calculate_error_cells(math.ceil(x) - 0.5, math.ceil(y) - 0.5, i)
        i += 1

    number_of_differences = int((abs(len(flame) - len(moth)) * math.exp(b * t * 2) * math.cos(
        2 * math.pi * t * 2)))
    # Check if the first list is longer
    if len(moth) > min_length:
        # Handle the remaining tuples in the first list
        for i in range(min_length, number_of_differences):
            moth[random.randint(0, len(moth) - 1)] = (-100, -100)

    # Check if the second list is longer
    elif len(flame) > min_length:
        # Handle the remaining tuples in the second list
        for i in range(min_length, number_of_differences):
            add_new_WT(moth, dead_cells, m, n)
    return moth, error_cells


def square_tuple(t):
    return tuple(math.ceil(x) + 0.5 for x in t)


def update_moth(i, population, population_fitness, flames, number_of_flames, lookup_table_dead_space_offset):
    moth = population[i]

    def is_valid(x, y):
        for dx_inner in list(range(-spacing_distance, spacing_distance + 1)):
            for dy_inner in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx_inner, y + dy_inner) in moth:
                    return False
        return True

    moth, error_cells = calculate_position(population[i],
                                           flames[(i // (len(population) // number_of_flames)) % number_of_flames])

    moth = [moth[j] for j in range(len(moth)) if moth[j][0] != -100]

    for cell in error_cells:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if is_valid(cell[0] + dx, cell[1] + dy) and (int(cell[0] + dx), int(cell[1] + dy)) not in dead_cells:
                    moth.append((cell[0] + dx, cell[1] + dy))
                    break
    if len(moth) == 0:
        add_new_WT(moth, dead_cells, m, n)

    moth = list(map(lambda x: (math.ceil(x[0]) - 0.5, math.ceil(x[1]) - 0.5), moth))

    population_fitness[i] = objective_function(moth, n, m)
    moth.sort(key=lambda x: (x[0], x[1]))

    print("fitness: ", population_fitness[i][0])

    print("Num of Turbines for particle " + str(i), len(moth))
    print("particle: ", moth)
    population[i] = moth
    return i


def moth_flame(visualise):
    global lower_bound, ax
    start = time.perf_counter()
    population, population_fitness, flames, flames_fitness = init_population()
    best_fitness = float('inf')
    best_population = []
    number_of_flames = population_size
    lookup_table_dead_space_offset_x = [x for x in range(-m, m + 1)]
    lookup_table_dead_space_offset_y = [x for x in range(-n, n + 1)]
    lookup_table_dead_space_offset = [(x, y) for x in lookup_table_dead_space_offset_x for y in
                                      lookup_table_dead_space_offset_y]
    lookup_table_dead_space_offset.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))
    optimal_objective_vs_I = []  # Optimal Objective vs iterations for plotting
    if visualise:
        num_of_generations = []  # Num of generations used for plotting
        for i in range(1, max_iterations + 1):
            num_of_generations.append(i)
        ax = draw_simulation_population(num_of_generations)
        time.sleep(3)
    for i in range(population_size):
        if population_fitness[i][0] < best_fitness and population_fitness[i][2]:
            best_fitness = population_fitness[i][0]
            best_population = population[i].copy()
    with Manager() as manager:
        population = manager.list(population)
        population_fitness = manager.list(population_fitness)
        flames = manager.list(flames)
        flames_fitness = manager.list(flames_fitness)
        for i in range(max_iterations):
            print("iteration: ", i)
            combined_data = list(zip(flames_fitness, flames))

            # Sort the combined data based on fitness values
            sorted_data = sorted(combined_data, key=lambda x: x[0])

            # Unpack the sorted data back into separate arrays
            flames_fitness, flames = zip(*sorted_data)
            flames_fitness = list(flames_fitness)
            flames = list(flames)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [
                    executor.submit(update_moth, i, population, population_fitness, flames, number_of_flames,
                                    lookup_table_dead_space_offset) for i in
                    range(population_size)]
                for f in concurrent.futures.as_completed(results):
                    print("finished: ", f.result())

            for j in range(population_size):
                if population_fitness[j][0] < best_fitness and population_fitness[j][2]:
                    best_fitness = population_fitness[j][0]
                    best_population = population[j].copy()
            number_of_flames = math.floor(((len(population) - i) * (len(population) - 1) / max_iterations) + 0.5)

            if number_of_flames < 1:
                number_of_flames = 1

            for j in range(len(population)):
                flame_index = (j // (len(population) // number_of_flames)) % number_of_flames
                if population_fitness[j][0] < flames_fitness[flame_index]:
                    flames_fitness[flame_index] = population_fitness[j][0]
                    flames[flame_index] = population[j].copy()
            flames = flames[:number_of_flames]
            flames_fitness = flames_fitness[:number_of_flames]
            print("Best Fitness: ", best_fitness)
            assert (len(flames) == number_of_flames)
            lower_bound = calculate_lower_bound(lower_bound)
            if visualise:
                fitness_values = []
                for fitness in population_fitness:
                    fitness_values.append(fitness[0])
                myMax, myMin = update_plot_population(ax, i, fitness_values, None if i == 0 else myMax,
                                                      None if i == 0 else myMin)
            optimal_objective_vs_I.append(best_fitness)

    if visualise:
        draw_solution_population(best_population, best_fitness, dead_cells, m, n)
        draw_iterations_against_solution(optimal_objective_vs_I, True)
    end = time.perf_counter()
    return best_population, best_fitness, end - start


# Test case 1 is run by default

# Uncomment this block for test case 2
# n, m = 20, 20
# dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
#               (15, 16), (16, 16), (15, 17), (16, 17)]
# calculate_lower_bound = calculate_bound_linear

# Uncomment this block for test case 3
# n, m = 25, 25
# dead_cells = [(5, 5), (5, 6), (6, 5), (6, 6), (5, 18), (5, 19), (6, 18), (6, 19), (18, 5), (19, 5), (18, 6), (19, 6),
#               (18, 18), (18, 19), (19, 18), (19, 19), (7, 7), (7, 6), (7, 5), (7, 18), (7, 19), (18, 7), (19, 7),
#               (5, 7), (6, 7), (5, 17), (6, 17), (7, 17), (17, 5), (17, 6), (17, 7), (17, 17), (17, 18), (17, 19),
#               (18, 17), (19, 17)]
# calculate_lower_bound = calculate_bound_linear

# Uncomment this block for test case 4
# n, m = 15, 15
# dead_cells = [(2, 2), (12, 2), (2, 12), (12, 12)]  # no turbines can be placed in these cells
# calculate_lower_bound = calculate_bound_geometric

# Uncomment this block for test case 5
# n, m = 20, 20
# dead_cells = [(3, 2), (4, 2), (3, 3), (4, 3), (15, 2), (16, 2), (15, 3), (16, 3), (3, 16), (4, 16), (3, 17), (4, 17),
#               (15, 16), (16, 16), (15, 17), (16, 17)]
# calculate_lower_bound = calculate_bound_geometric

# Uncomment this block for test case 6
# n, m = 25, 25
# dead_cells = [(5, 5), (5, 6), (6, 5), (6, 6), (5, 18), (5, 19), (6, 18), (6, 19), (18, 5), (19, 5), (18, 6), (19, 6),
#               (18, 18), (18, 19), (19, 18), (19, 19), (7, 7), (7, 6), (7, 5), (7, 18), (7, 19), (18, 7), (19, 7),
#               (5, 7), (6, 7), (5, 17), (6, 17), (7, 17), (17, 5), (17, 6), (17, 7), (17, 17), (17, 18), (17, 19),
#               (18, 17), (19, 17)]
# calculate_lower_bound = calculate_bound_geometric

if __name__ == '__main__':
    bpopulation, bfitness, time = moth_flame(True)
    print("best population: ", bpopulation)
    print("best fitness: ", bfitness)
    print("time: ", time)
