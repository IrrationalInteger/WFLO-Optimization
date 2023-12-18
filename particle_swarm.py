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

# Particle swarm parameters
population_size = 50  # number of particles per iteration
w = 0.792  # Weight of inertia
c1 = 1.4944  # Cognitive factor
c2 = 1.4944  # Social factor
max_iterations = 100  # Maximum number of allowed iterations
neighbourhood_size = 2  # Neighbourhood size for each particle to consider the global best


def add_new_WT(solution, exclusion_list, m_inner, n_inner):
    for i in range(len(solution)):
        solution[i] = (solution[i][0] - 0.5, solution[i][1] - 0.5)

    def is_valid(x_inner,
                 y_inner):
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

# Initialize a new particle in the population
def init_particle():
    solution = generate_random_tuples(random.randint(1, 5), dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution, n, m)
    solution.sort(key=lambda x: (x[0], x[1]))
    return solution, fitness

# Initialize the population
def init_population():
    population = []
    population_fitness = []
    velocity_vector = [[] for _ in range(population_size)]
    pbest_position = []
    pbest_fitness = []
    gbest_position = [[] for _ in range(population_size)]
    gbest_fitness = [float('inf') for _ in range(population_size)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(init_particle) for _ in range(population_size)]
        for f in concurrent.futures.as_completed(results):
            population.append(f.result()[0])
            population_fitness.append(f.result()[1])
            pbest_position.append(f.result()[0])
            pbest_fitness.append(f.result()[1][0])
    for i in range(population_size):
        neighbours = [(x + population_size) % population_size for x in
                      range(-neighbourhood_size, neighbourhood_size + 1)]
        best_fitness_index = min(neighbours, key=lambda x: population_fitness[x][0])
        gbest_position[i] = population[best_fitness_index].copy()
        gbest_fitness[i] = population_fitness[best_fitness_index][0]
    return population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness

# Subtract two solutions from each other with a custom operator
def subtract_solutions(list1, list2):
    result = []

    # Find the minimum length of the two lists
    min_length = min(len(list1), len(list2))

    # Perform element-wise subtraction up to the minimum length
    for i in range(min_length):
        tuple1 = list1[i]
        tuple2 = list2[i]
        result.append((tuple1[0] - tuple2[0], tuple1[1] - tuple2[1]))

    # Check if the first list is longer
    if len(list1) > min_length:
        # Handle the remaining tuples in the first list
        for i in range(min_length, len(list1)):
            result.append(('+', (list1[i][0], list1[i][1])))

    # Check if the second list is longer
    elif len(list2) > min_length:
        # Handle the remaining tuples in the second list
        for i in range(min_length, len(list2)):
            result.append(('-', (list2[i][0], list2[i][1])))

    return result

# Add two solutions to each other with a custom operator
def add_velocity(list1, list2):
    result = []
    list1_new = [list1[i] for i in range(len(list1)) if list1[i][0] != '+' and list1[i][0] != '-']
    list2_new = [list2[i] for i in range(len(list2)) if list2[i][0] != '+' and list2[i][0] != '-']
    # Find the minimum length of the two lists
    min_length = min(len(list1_new), len(list2_new))

    # Perform element-wise addition up to the minimum length
    for i in range(min_length):
        tuple1 = list1_new[i]
        tuple2 = list2_new[i]
        result.append((tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))

    # append the remaining tuples in the first list
    for i in range(min_length, len(list1_new)):
        result.append(list1_new[i])
    for i in range(min_length, len(list2_new)):
        result.append(list2_new[i])

    # reappend the removed special operators
    for i in range(len(list1)):
        if list1[i][0] == '+' or list1[i][0] == '-':
            result.append(list1[i])
    for i in range(len(list2)):
        if list2[i][0] == '+' or list2[i][0] == '-':
            result.append(list2[i])

    return result

# Multiply a solution by a scalar with a custom operator
def multiply_velocity(list_of_tuples, scalar):
    if not list_of_tuples:
        return []
    result = []

    for t in list_of_tuples:
        # Check if the tuple is special (e.g., special_value_list1 or special_value_list2)
        if t[0] in ('+', '-'):
            result.append(t)  # Add special tuple without modification
        else:
            # Normal multiplication for regular tuples
            result.append((int(t[0] * scalar), int(t[1] * scalar)))

    if scalar < 1:
        # Delete a percentage of the tuples randomly based on the scalar value
        deleted_indices = random.sample(range(len(result)), k=len(result) - int(len(result) * scalar))
        result_copy = []
        for i, item in enumerate(result):
            if i in deleted_indices:
                if item[0] == '+' or item[0] == '-':
                    continue
                else:
                    result_copy.append((0, 0))
            else:
                result_copy.append(item)
        result = result_copy

    if scalar > 1:
        # Append a number of tuples of the type (0, '+') equal to half the length of the list
        for _ in range(max(int(len(result) * (scalar - 1)), 4)):
            result.append(('+', 0))
    return result

# Update the position of each particle
def update_particle(i, population, velocity_vector, pbest_position, gbest_position, population_fitness, pbest_fitness,
                    lookup_table_dead_space_offset):
    particle = population[i]
    error_cells = []

    # Aggregates all the turbines that violate constraints in a lookup table
    def calculate_error_cells(x, y, j_inner):
        particle.pop(j_inner)
        error_cells_new = []
        for \
                dx_inner in list(range(-spacing_distance, spacing_distance + 1)):
            for dy_inner in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx_inner, y + dy_inner) in particle:
                    error_cells_new.append((x + dx_inner, y + dy_inner))
        # remove error cells from particle
        for error_cell in error_cells_new:
            particle[particle.index(error_cell)] = (-100, -100)
            error_cells.append(error_cell)
        # add (x,y) to its position in j
        particle.insert(j_inner, (x, y))
        return error_cells

    def is_valid(x, y):
        for dx_inner in list(range(-spacing_distance, spacing_distance + 1)):
            for dy_inner in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx_inner, y + dy_inner) in particle:
                    return False
        return True

    print("particle: ", i)
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)

    step1_result = multiply_velocity(velocity_vector[i], w)
    step2_result = subtract_solutions(pbest_position[i], particle)
    step3_result = multiply_velocity(step2_result, c1 * r1)
    step4_result = add_velocity(step1_result, step3_result)
    step5_result = subtract_solutions(gbest_position[i], particle)
    step6_result = multiply_velocity(step5_result, c2 * r2)
    final_result = add_velocity(step4_result, step6_result)
    velocity_vector[i] = final_result

    # Apply all the special operators while fixing constraints
    for j in range(len(velocity_vector[i])):
        if velocity_vector[i][j][0] == 0 and velocity_vector[i][j][1] == 0:
            continue
        if velocity_vector[i][j][0] == '+' and velocity_vector[i][j][1] == 0:
            add_new_WT(particle, dead_cells, m, n)
            continue
        if velocity_vector[i][j][0] == '+':
            new_wt = velocity_vector[i][j][1]
            if new_wt not in particle:
                particle.append(new_wt)
                calculate_error_cells(new_wt[0], new_wt[1], len(particle) - 1)
            continue
        if velocity_vector[i][j][0] == '-' and velocity_vector[i][j][1] == 0:
            particle[random.randint(0, len(particle) - 1)] = (-100, -100)
            continue
        if velocity_vector[i][j][0] == '-':
            if velocity_vector[i][j][1] in particle:
                particle.remove(velocity_vector[i][j][1])
            continue
        if particle[j][0] == -100 and particle[j][1] == -100:
            continue
        particle[j] = (((particle[j][0] + velocity_vector[i][j][0] + m - 0.5) % m) + 0.5,
                       ((particle[j][1] + velocity_vector[i][j][1] + n - 0.5) % n) + 0.5)
        if (int(particle[j][0]), int(particle[j][1])) in dead_cells:
            error_cells.append((particle[j][0], particle[j][1]))
            particle[j] = (-100, -100)
            continue
        calculate_error_cells(particle[j][0], particle[j][1], j)
    particle = [particle[j] for j in range(len(particle)) if particle[j][0] != -100]
    for cell in error_cells:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if is_valid(cell[0] + dx, cell[1] + dy) and (int(cell[0] + dx), int(cell[1] + dy)) not in dead_cells:
                    particle.append((cell[0] + dx, cell[1] + dy))
                    break
    if len(particle) == 0:
        add_new_WT(particle, dead_cells, m, n)
    population_fitness[i] = objective_function(particle, n, m)
    length = 0
    for vel in range(len(velocity_vector[i])):
        if velocity_vector[i][vel][0] == '+' or velocity_vector[i][vel][0] == '-':
            break
        length += 1
    velocity_vector_values = velocity_vector[i][:length]
    if len(velocity_vector_values) > len(particle):
        velocity_vector_values = velocity_vector_values[:len(particle)]
    velocity_vector_symbols = velocity_vector[i][length:]
    velocity_vector_symbols = [(velocity_vector_symbols[j][0], 0) for j in range(len(velocity_vector_symbols))]
    plus_count = 0
    minus_count = 0
    for j in range(len(velocity_vector_symbols)):
        if velocity_vector_symbols[j][0] == '+':
            plus_count += 1
        elif velocity_vector_symbols[j][0] == '-':
            minus_count += 1
    number_of_symbols_to_remove = min(plus_count, minus_count)
    for j in range(number_of_symbols_to_remove):
        velocity_vector_symbols.remove(('+', 0))
    for j in range(number_of_symbols_to_remove):
        velocity_vector_symbols.remove(('-', 0))
    velocity_vector[i] = velocity_vector_values + velocity_vector_symbols
    print("fitness: ", population_fitness[i][0])
    if population_fitness[i][0] < pbest_fitness[i]:
        pbest_position[i] = particle.copy()
        pbest_fitness[i] = population_fitness[i][0]

    print("Num of Turbines for particle " + str(i), len(particle))
    print("particle: ", particle)
    particle.sort(key=lambda x: (x[0], x[1]))
    population[i] = particle
    return i

# Performs particle swarm algorithm using the given parameters
def particle_swarm(visualise):
    global ax
    start = time.perf_counter()
    population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness = init_population() # Initialize a population
    best_fitness = float('inf') # Record the best fitness during running
    best_population = []  # Record the best particle during running
    # Create the lookup table early and pass it to save on CPU
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
        time.sleep(3) # Delay to allow grid to properly initialize. May need to rerun code multiple times for it to work
    for i in range(population_size):
        if population_fitness[i][0] < best_fitness and population_fitness[i][2]:
            best_fitness = population_fitness[i][0]
            best_population = population[i].copy()
    # Use manager for lists across processes
    with Manager() as manager:
        population = manager.list(population)
        population_fitness = manager.list(population_fitness)
        velocity_vector = manager.list(velocity_vector)
        pbest_position = manager.list(pbest_position)
        gbest_position = manager.list(gbest_position)
        pbest_fitness = manager.list(pbest_fitness)
        gbest_fitness = manager.list(gbest_fitness)
        for i in range(max_iterations):
            print("iteration: ", i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [
                    executor.submit(update_particle, i, population, velocity_vector, pbest_position, gbest_position,
                                    population_fitness, pbest_fitness, lookup_table_dead_space_offset) for i in
                    range(population_size)]
                for f in concurrent.futures.as_completed(results):
                    print("finished: ", f.result())
            # Calculate global bests according to the neighbourhood size
            for k in range(population_size):
                neighbours = [(x + population_size) % population_size for x in
                              range(-neighbourhood_size, neighbourhood_size + 1)]
                best_fitness_index = min(neighbours, key=lambda x: population_fitness[x][0])
                if population_fitness[best_fitness_index][0] < gbest_fitness[k]:
                    gbest_position[k] = population[best_fitness_index].copy()
                    gbest_fitness[k] = population_fitness[best_fitness_index][0]
            # Record the best fitness that satisfies the power constraint
            for j in range(population_size):
                if population_fitness[j][0] < best_fitness and population_fitness[j][2]:
                    best_fitness = population_fitness[j][0]
                    best_population = population[j].copy()
            print("best fitness: ", best_fitness)
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
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# w = 0.792
# c1 = 1.4944
# c2 = 1.4944
# neighbourhood_size = 2

# Uncomment this block for test case 3
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# w = 0.792
# c1 = 1.4944
# c2 = 1.4944
# neighbourhood_size = 2

# Uncomment this block for test case 4
# n,m = 15,15
# dead_cells = [(2,2),(12,2),(2,12),(12,12)] # no turbines can be placed in these cells
# w = 0.792
# c1 = 1.4944
# c2 = 1.4944
# neighbourhood_size = 25

# Uncomment this block for test case 5
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# w = 0.792
# c1 = 1.4944
# c2 = 1.4944
# neighbourhood_size = 25

# Uncomment this block for test case 6
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# w = 0.792
# c1 = 1.4944
# c2 = 1.4944
# neighbourhood_size = 25


if __name__ == '__main__':
    bpopulation, bfitness, time = particle_swarm(True)
    print("best population: ", bpopulation)
    print("best fitness: ", bfitness)
    print("time: ", time)
