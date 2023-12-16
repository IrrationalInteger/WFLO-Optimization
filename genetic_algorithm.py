import concurrent.futures
import math
import time
import random
import matplotlib
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, dead_cells, WT_list_length
from drawings import draw_iterations_against_solution, draw_solution_population, \
    draw_simulation_population, update_plot_population
from functions import generate_random_tuples

matplotlib.use('TkAgg')

# GA parameters
population_size = 50  # Population size (number of chromosomes per generation)
population = []  # Current generation population
population_fitness = []  # Current generation population fitness
survivor_percentage = 10  # Percentage of chromosomes that survive till next generation
crossover_percentage = 80  # Percentage of crossed over chromosomes
mutation_percentage = 10  # Percentage of mutated chromosomes
max_generations = 300  # Maximum number of allowed generations
do_uniform = True  # Specifies if uniform crossover can be chosen randomly


# Adds a new turbine while respecting the spacing distance and dead cells
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


# Generates a new solution from the previous one by randomly adding, removing, or moving a single wind turbine while
# respecting the spacing distance and the dead cells
def generate_neighbour_solution(solution, exclusion_list, m_inner, n_inner, num_of_genes):
    op = random.randint(1, 2) if len(solution) == MAX_WT_number else random.randint(0, 1) if len(
        solution) == 1 else random.randint(0, 2)
    solution = solution.copy()
    if op == 0:  # op = 0, add a WT at random location
        for _ in range(num_of_genes):
            add_new_WT(solution, exclusion_list, m_inner, n_inner)
    elif op == 1:  # op = 1, change the location of one WT ????
        for _ in range(num_of_genes):
            solution.pop(random.randint(0, len(solution) - 1))
            add_new_WT(solution, exclusion_list, m_inner, n_inner)
    else:  # op = 2, remove a random WT
        max_iterations = min_inner(num_of_genes, len(solution) - 1)
        for _ in range(max_iterations):
            solution.pop(random.randint(0, len(solution) - 1))
    return solution


#  Create a new chromosome and calculate its fitness
def init_chromosome():
    solution = generate_random_tuples(WT_list_length, dead_cells, m, n, spacing_distance)
    fitness_inner = objective_function(solution, n, m)
    return solution, fitness_inner


#  Generate the initial population randomly and calculate their fitness's
def init_population():
    global population
    global population_fitness
    population = []
    population_fitness = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(init_chromosome) for _ in range(population_size)]
        for f in concurrent.futures.as_completed(results):
            population.append(f.result()[0])
            population_fitness.append(f.result()[1])

    combined = list(zip(population, population_fitness))

    combined.sort(key=lambda pair: pair[1][0])

    # Update the original lists in place
    for i, (pop, fitness_inner) in enumerate(combined):
        population[i] = pop
        population_fitness[i] = fitness_inner


# Finds the best N/2 elites that satisfy the power constraint if they exist and the rest of the elites are chosen as
# the best fitness chromosomes
def elite_chromosomes(new_population, new_fitness):
    elite_population = []
    half_num_of_elites = math.floor(((survivor_percentage / 2) * population_size / 100) + 0.5)
    num_of_elites = half_num_of_elites * 2
    chromosomes_added = 0
    for y in range(len(population)):
        if population_fitness[y][2]:
            new_population.append(population[y])
            elite_population.append(population[y])
            new_fitness.append(population_fitness[y])
            chromosomes_added += 1
            if chromosomes_added == half_num_of_elites:
                break
    for i in range(len(population)):
        if population[i] not in elite_population:
            new_population.append(population[i])
            new_fitness.append(population_fitness[i])
            chromosomes_added += 1
            if chromosomes_added == num_of_elites:
                break


# Mutates a given chromosome
def mutate(chromosome_inner):
    chromosome_inner = chromosome_inner.copy()
    num_of_genes = 1  # or for multiple gene mutations : max(4,math.floor((0.25*len(chromosome)) + 0.5))
    chromosome_inner = generate_neighbour_solution(chromosome_inner, dead_cells, m, n, num_of_genes)
    fitness_inner = objective_function(chromosome_inner, m, n)
    return chromosome_inner, fitness_inner


# Performs uniform crossover by randomly swapping some genes (turbines) between the parents and fixing the spacing
# distance violations using a lookup table
def uniform_crossover(parents_pair, lookup_table_dead_space_offset):
    parent1, parent2 = parents_pair
    child1 = []
    child2 = []
    child1_error = []
    child2_error = []
    lookup_table_1 = [[0 for _ in range(n)] for _ in range(m)]
    lookup_table_2 = [[0 for _ in range(n)] for _ in range(m)]
    parent_min_length = min_inner(len(parent1), len(parent2))
    parent_max_length = max_inner(len(parent1), len(parent2))
    swap = [random.randint(0, 1) for _ in range(parent_max_length)]

    def add_dead_space(cell_inner, lookup_table):
        for dx_inner in list(range(-spacing_distance, spacing_distance + 1)):
            for dy_inner in list(range(-spacing_distance, spacing_distance + 1)):
                if m > math.floor(cell_inner[0] + dx_inner) >= 0 and n > math.floor(cell_inner[1] + dy_inner) >= 0:
                    lookup_table[math.floor(cell_inner[0] + dx_inner)][math.floor(cell_inner[1] + dy_inner)] = 1

    for i in range(parent_min_length):
        if swap[i] == 0:
            if lookup_table_1[math.floor(parent1[i][0])][math.floor(parent1[i][1])] == 0:
                child1.append(parent1[i])
                add_dead_space(parent1[i], lookup_table_1)
            else:
                child1_error.append(parent1[i])
            if lookup_table_2[math.floor(parent2[i][0])][math.floor(parent2[i][1])] == 0:
                child2.append(parent2[i])
                add_dead_space(parent2[i], lookup_table_2)
            else:
                child2_error.append(parent2[i])
        else:
            if lookup_table_1[math.floor(parent2[i][0])][math.floor(parent2[i][1])] == 0:
                child1.append(parent2[i])
                add_dead_space(parent2[i], lookup_table_1)
            else:
                child1_error.append(parent2[i])
            if lookup_table_2[math.floor(parent1[i][0])][math.floor(parent1[i][1])] == 0:
                child2.append(parent1[i])
                add_dead_space(parent1[i], lookup_table_2)
            else:
                child2_error.append(parent1[i])

    for i in range(parent_min_length, parent_max_length):
        parent = parent1 if parent_max_length == len(parent1) else parent2
        if swap[i] == 0:
            if lookup_table_1[math.floor(parent[i][0])][math.floor(parent[i][1])] == 0:
                child1.append(parent[i])
                add_dead_space(parent[i], lookup_table_1)
            else:
                child1_error.append(parent[i])
        else:
            if lookup_table_2[math.floor(parent[i][0])][math.floor(parent[i][1])] == 0:
                child2.append(parent[i])
                add_dead_space(parent[i], lookup_table_2)
            else:
                child2_error.append(parent[i])

    for cell in child1_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_1[math.floor(cell[0] + dx)][math.floor(cell[1] + dy)] == 0:
                    child1.append((cell[0] + dx, cell[1] + dy))
                    add_dead_space((cell[0] + dx, cell[1] + dy), lookup_table_1)
                    break

    for cell in child2_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_2[math.floor(cell[0] + dx)][math.floor(cell[1] + dy)] == 0:
                    child2.append((cell[0] + dx, cell[1] + dy))
                    add_dead_space((cell[0] + dx, cell[1] + dy), lookup_table_2)
                    break

    return child1, child2, objective_function(child1, m, n), objective_function(
        child2, m, n)


# Performs one point crossover at a randomly selected divider between the parents and swapping all the genes (
# turbines) after it and fixing the spacing distance violations using a lookup table
def one_point_crossover(parents_pair, lookup_table_dead_space_offset):
    parent1, parent2 = parents_pair
    child1 = []
    child2 = []
    child1_error = []
    child2_error = []
    lookup_table_1 = [[0 for _ in range(n)] for _ in range(m)]
    lookup_table_2 = [[0 for _ in range(n)] for _ in range(m)]
    crossover_y = random.randint(0, 1)
    crossover_point = random.randint(1, m - 1) if crossover_y == 0 else random.randint(1, n - 1)

    def add_dead_space(cell_inner, lookup_table):
        for dx_inner in list(range(-spacing_distance, spacing_distance + 1)):
            for dy_inner in list(range(-spacing_distance, spacing_distance + 1)):
                if m > math.floor(cell_inner[0] + dx_inner) >= 0 and n > math.floor(cell_inner[1] + dy_inner) >= 0:
                    lookup_table[math.floor(cell_inner[0] + dx_inner)][math.floor(cell_inner[1] + dy_inner)] = 1

    for cell in parent1:
        if cell[crossover_y] < crossover_point:
            if lookup_table_1[math.floor(cell[0])][math.floor(cell[1])] == 0:
                child1.append(cell)
                add_dead_space(cell, lookup_table_1)
            else:
                child1_error.append(cell)
        else:
            if lookup_table_2[math.floor(cell[0])][math.floor(cell[1])] == 0:
                child2.append(cell)
                add_dead_space(cell, lookup_table_2)
            else:
                child2_error.append(cell)
    for cell in parent2:
        if cell[crossover_y] < crossover_point:
            if lookup_table_2[math.floor(cell[0])][math.floor(cell[1])] == 0:
                child2.append(cell)
                add_dead_space(cell, lookup_table_2)
            else:
                child2_error.append(cell)
        else:
            if lookup_table_1[math.floor(cell[0])][math.floor(cell[1])] == 0:
                child1.append(cell)
                add_dead_space(cell, lookup_table_1)
            else:
                child1_error.append(cell)

    for cell in child1_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_1[math.floor(cell[0] + dx)][math.floor(cell[1] + dy)] == 0:
                    child1.append((cell[0] + dx, cell[1] + dy))
                    add_dead_space((cell[0] + dx, cell[1] + dy), lookup_table_1)
                    break

    for cell in child2_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_2[math.floor(cell[0] + dx)][math.floor(cell[1] + dy)] == 0:
                    child2.append((cell[0] + dx, cell[1] + dy))
                    add_dead_space((cell[0] + dx, cell[1] + dy), lookup_table_2)
                    break
    if len(child1) == 0:
        child1 = parent1.copy()
    if len(child2) == 0:
        child2 = parent2.copy()
    return child1, child2, objective_function(child1, m, n), objective_function(
        child2, m, n)


# Performs rank selection across a number of parents in a given population
def rank_selection(num_of_parents, population_inner):
    if num_of_parents % 2 != 0:
        num_of_parents += 1
    # Calculate the weights
    probabilities = [x for x in range(1, len(population_inner) + 1)]
    probabilities.reverse()

    # Initialize a list to store pairs of parents for crossover
    selected_parents = []

    # Perform rank-based selection
    for _ in range(num_of_parents):
        new_parent = random.choices(population_inner, weights=probabilities, k=1)
        selected_parents.append(new_parent[0])

    # Create pairs of parents for crossover
    parent_pairs = []
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        while parent1 == parent2:
            parent2 = random.choices(population_inner, weights=probabilities, k=1)[0]
        parent_pairs.append((parent1, parent2))

    return parent_pairs


# Performs crossover on the population using the given crossover percentage
def crossover_chromosomes(population_inner, lookup_table_dead_space_offset):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        num_of_children = math.floor((crossover_percentage * population_size / 100) + 0.5)
        parent_pairs = rank_selection(num_of_children, population_inner)
        results = [
            executor.submit(one_point_crossover if random.randint(0,
                                                                  1) == 0 else uniform_crossover if do_uniform == True else one_point_crossover,
                            parent_pairs[i],
                            lookup_table_dead_space_offset) for i in range(len(parent_pairs))]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[2])
            new_population.append(f.result()[1])
            new_fitness.append(f.result()[3])
    if num_of_children % 2 != 0:
        new_population.pop()
        new_fitness.pop()
    return new_population, new_fitness


# Performs mutation on the population using the given mutation percentage
def mutate_chromosomes(population_inner):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        num_of_mutants = math.floor((mutation_percentage * population_size / 100) + 0.5)
        probabilities = [x ** 2 for x in range(1, len(population_inner) + 1)]
        results = [
            executor.submit(mutate, random.choices(population_inner, weights=probabilities, k=1)[0]) for
            _ in range(num_of_mutants)]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[1])
    return new_population, new_fitness


# Generates a new population based on the previous population using elitism, crossover, and mutation
def generate_population(lookup_table_dead_space_offset):
    new_population = [] * population_size
    new_fitness = [] * population_size
    elite_chromosomes(new_population, new_fitness)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_mutate = executor.submit(mutate_chromosomes, population)
        result_crossover = executor.submit(crossover_chromosomes, population,
                                           lookup_table_dead_space_offset)
        results = [result_mutate, result_crossover]
        for f in concurrent.futures.as_completed(results):
            new_population.extend(f.result()[0])
            new_fitness.extend(f.result()[1])

    combined = list(zip(new_population, new_fitness))
    combined.sort(key=lambda pair: pair[1][0])
    # Update the original lists in place
    for i, (pop, fitness_inner) in enumerate(combined):
        new_population[i] = pop
        new_fitness[i] = fitness_inner
    return new_population, new_fitness


# Performs genetic algorithm using the given parameters
def genetic_algorithm(visualise):
    start = time.perf_counter()
    num_of_generations = []  # Num of generations used for plotting
    for i in range(1, max_generations + 1):
        num_of_generations.append(i)
    global population, ax, max_inner, min_inner
    global population_fitness
    init_population()  # Initialize a population
    best_chromosome_yet = population[0]  # Record the best chromosome during running
    best_fitness_yet = population_fitness[0]  # Record the best fitness during running

    # Create the lookup table early and pass it to save on CPU
    lookup_table_dead_space_offset_x = [x for x in range(-m, m + 1)]
    lookup_table_dead_space_offset_y = [x for x in range(-n, n + 1)]
    lookup_table_dead_space_offset = [(x, y) for x in lookup_table_dead_space_offset_x for y in
                                      lookup_table_dead_space_offset_y]
    lookup_table_dead_space_offset.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))

    optimal_objective_vs_I = []  # Optimal Objective vs iterations for plotting
    if visualise:
        ax = draw_simulation_population(num_of_generations)
        time.sleep(
            1)  # Delay to allow grid to properly initialize. May need to rerun code multiple times for it to work
    for i in range(max_generations):
        new_population, new_fitness = generate_population(
            lookup_table_dead_space_offset)  # Generate new population each iteration
        if visualise:
            fitness_values = []
            for fitness_inner in new_fitness:
                fitness_values.append(fitness_inner[0])
            max_inner, min_inner = update_plot_population(ax, i, fitness_values, None if i == 0 else max_inner, None if i == 0 else min_inner)
        population = new_population
        population_fitness = new_fitness
        for j in range(len(population)):
            if population_fitness[j][2]:
                if population_fitness[j][0] < best_fitness_yet[0]:
                    best_fitness_yet = population_fitness[j]
                    best_chromosome_yet = population[j]
                break
        print(f'Generation: {i}')
        print(population_fitness)
        print(best_fitness_yet)
        optimal_objective_vs_I.append(best_fitness_yet[0])

    if visualise:
        draw_solution_population(best_chromosome_yet, best_fitness_yet[0], dead_cells, m, n)
        draw_iterations_against_solution(optimal_objective_vs_I, True)

    end = time.perf_counter()
    return best_chromosome_yet, best_fitness_yet, end - start


# Test case 1 is run by default

# Uncomment this block for test case 2
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10
# do_uniform = True

# Uncomment this block for test case 3
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10
# do_uniform = True

# Uncomment this block for test case 4
# n,m = 15,15
# dead_cells = [(2,2),(12,2),(2,12),(12,12)] # no turbines can be placed in these cells'
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10
# do_uniform = False

# Uncomment this block for test case 5
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10
# do_uniform = False

# Uncomment this block for test case 6
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10
# do_uniform = False


if __name__ == '__main__':
    chromosome, fitness, runtime = genetic_algorithm(True)
    print(chromosome)
    print(fitness)
    print(runtime)
