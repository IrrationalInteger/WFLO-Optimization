
# Adds a new turbine while respecting the spacing distance and dead cells
import concurrent.futures
import math
import time
import random
import matplotlib
from multiprocessing import Manager
import os
import psutil
import numpy as np
from matplotlib import pyplot as plt

from drawings import draw_number_of_turbines_against_power_and_objective, draw_iterations_against_solution, \
    draw_solution
from functions import generate_random_tuples

matplotlib.use('TkAgg')
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, WT_list, WT_max_number, dead_cells, \
    WT_list_length


def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is Windows only, on Unix use ps.nice(19)
    # p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    p.nice(19)

def add_new_WT(solution, exclusion_list, m , n):
  for i in range(len(solution)):
    solution[i] = (solution[i][0] - 0.5, solution[i][1] - 0.5)
  def is_valid(x, y):
        for dx in list(range(-spacing_distance,spacing_distance+1)):
            for dy in list(range(-spacing_distance,spacing_distance+1)):
                if (x + dx, y + dy) in solution:
                    return False
        return True
  i_max = m * n
  i = 0
  while i<i_max:
    x = random.randint(0, m-1)
    y = random.randint(0, n-1)
    new_tuple = (x, y)
    if new_tuple not in exclusion_list and is_valid(x,y):
      solution.append((x,y))
      break
    i += 1
  for i in range(len(solution)):
    solution[i] = (solution[i][0] + 0.5, solution[i][1] + 0.5)


# Generates a new solution from the previous one by randomly adding, removing, or moving a single wind turbine while
# respecting the spacing distance and the dead cells
def generate_neighbour_solution(solution, exclusion_list, m , n, num_of_genes):
  op = random.randint(1, 2) if len(solution) == MAX_WT_number else random.randint(0, 1) if len(solution)==1 else random.randint(0, 2)
  solution = solution.copy()
  if op==0: # op = 0, add a WT at random location
    for _ in range(num_of_genes):
        add_new_WT(solution, exclusion_list, m , n)
  elif op==1: # op = 1, change the location of one WT ????
    for _ in range(num_of_genes):
        solution.pop(random.randint(0,len(solution)-1))
        add_new_WT(solution, exclusion_list, m , n)
  else: # op = 2, remove a random WT
    max_iterations = min(num_of_genes, len(solution)-1)
    for _ in range(max_iterations):
        solution.pop(random.randint(0,len(solution)-1))
  return solution


def calculate_objective_function(solution, explored_chromosomes):
    solution = tuple(solution)
    if solution in explored_chromosomes:
        return explored_chromosomes[solution]
    else:
        fitness = objective_function(solution, m, n)
        explored_chromosomes[solution] = fitness
        return fitness

#  Create a new chromosome and calculate its fitness
def init_chromosome():
    solution = generate_random_tuples(WT_list_length, dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution ,n,m)
    return solution,fitness

#  Generate the initial population randomly and calculate their fitnesses
def init_population():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit( init_chromosome) for _ in range(population_size)]
        for f in concurrent.futures.as_completed(results):
            population.append(f.result()[0])
            population_fitness.append(f.result()[1])

    combined = list(zip(population, population_fitness))

    combined.sort(key=lambda pair: pair[1][0])

    # Update the original lists in place
    for i, (pop, fitness) in enumerate(combined):
        population[i] = pop
        population_fitness[i] = fitness

def elite_chromosomes(new_population,new_fitness):
    if elitism:
        elite_population = []
        half_num_of_elites = math.floor(((survivor_percentage/2)*population_size/100)+0.5)
        num_of_elites = half_num_of_elites*2
        chromosomes_added = 0
        y = 0
        for y in range(len(population)):
            if population_fitness[y][2]:
                new_population.append(population[y])
                elite_population.append(population[y])
                new_fitness.append(population_fitness[y])
                chromosomes_added += 1
                if chromosomes_added == half_num_of_elites:
                    break
        i = 0
        for i in range(len(population)):
            if population[i] not in elite_population:
                new_population.append(population[i])
                new_fitness.append(population_fitness[i])
                chromosomes_added += 1
                if chromosomes_added == num_of_elites:
                    break
    else:
        #later REMEMBER DONT DO THIS
        pass


def mutate(chromosome, explored_chromosomes):
    chromosome = chromosome.copy()
    num_of_genes = 1 # Mutate a fourth of the genes   max(4,math.floor((0.25*len(chromosome)) + 0.5))
    chromosome = generate_neighbour_solution(chromosome, dead_cells, m, n, num_of_genes)
    fitness = calculate_objective_function(chromosome, explored_chromosomes)
    return chromosome, fitness


def uniform_crossover(parents_pair, lookup_table_dead_space_offset, explored_chromosomes):
    parent1, parent2 = parents_pair
    child1 = []
    child2 = []
    child1_error = []
    child2_error = []
    lookup_table_1 = [[0 for j in range(n)] for i in range(m)]
    lookup_table_2 = [[0 for j in range(n)] for i in range(m)]
    parent_min_length = min(len(parent1), len(parent2))
    parent_max_length = max(len(parent1), len(parent2))
    swap = [random.randint(0,1) for _ in range(parent_max_length)]
    def add_dead_space(cell,lookup_table):
        for dx in list(range(-spacing_distance,spacing_distance+1)):
            for dy in list(range(-spacing_distance,spacing_distance+1)):
                if m > math.floor(cell[0]+dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                    lookup_table[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] = 1
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
                if lookup_table_1[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] == 0:
                    child1.append((cell[0]+dx, cell[1]+dy))
                    add_dead_space((cell[0]+dx, cell[1]+dy), lookup_table_1)
                    break

    for cell in child2_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_2[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] == 0:
                    child2.append((cell[0]+dx, cell[1]+dy))
                    add_dead_space((cell[0]+dx, cell[1]+dy), lookup_table_2)
                    break

    return child1,child2,calculate_objective_function(child1, explored_chromosomes),calculate_objective_function(child2, explored_chromosomes)


def one_point_crossover(parents_pair, lookup_table_dead_space_offset, explored_chromosomes):
    parent1, parent2 = parents_pair
    child1 = []
    child2 = []
    child1_error = []
    child2_error = []
    lookup_table_1 = [[0 for j in range(n)] for i in range(m)]
    lookup_table_2 = [[0 for j in range(n)] for i in range(m)]
    crossover_y = random.randint(0, 1)
    crossover_point = random.randint(1, m - 1) if crossover_y == 0 else random.randint(1, n - 1)

    def add_dead_space(cell, lookup_table):
        for dx in list(range(-spacing_distance,spacing_distance+1)):
            for dy in list(range(-spacing_distance,spacing_distance+1)):
                if m > math.floor(cell[0]+dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                    lookup_table[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] = 1
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
                if lookup_table_1[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] == 0:
                    child1.append((cell[0]+dx, cell[1]+dy))
                    add_dead_space((cell[0]+dx, cell[1]+dy), lookup_table_1)
                    break

    for cell in child2_error:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if lookup_table_2[math.floor(cell[0]+dx)][math.floor(cell[1]+dy)] == 0:
                    child2.append((cell[0]+dx, cell[1]+dy))
                    add_dead_space((cell[0]+dx, cell[1]+dy), lookup_table_2)
                    break
    if len(child1) == 0:
        child1 = parent1.copy()
    if len(child2) == 0:
        child2 = parent2.copy()
    return child1,child2,calculate_objective_function(child1, explored_chromosomes),calculate_objective_function(child2, explored_chromosomes)


def rank_selection(num_of_parents, population, population_fitness):
    if num_of_parents % 2 != 0:
        num_of_parents += 1
    # Calculate the weights
    probabilities = [x for x in range(1, len(population) + 1)]
    probabilities.reverse()

    # Initialize a list to store pairs of parents for crossover
    selected_parents = []

    # Perform rank-based selection
    for _ in range(num_of_parents):
        new_parent = random.choices(population, weights=probabilities, k=1)
        selected_parents.append(new_parent[0])

    # Create pairs of parents for crossover
    parent_pairs = []
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        while parent1 == parent2:
            parent2 = random.choices(population, weights=probabilities, k=1)[0]
        parent_pairs.append((parent1, parent2))

    return parent_pairs
def crossover_chromosomes(population, population_fitness,lookup_table_dead_space_offset, explored_chromosomes):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=None, initializer=limit_cpu) as executor:
        num_of_children = math.floor((crossover_percentage * population_size / 100) + 0.5)
        parent_pairs = rank_selection(num_of_children, population, population_fitness)
        results = [executor.submit(one_point_crossover if random.randint(0, 1) == 0 else uniform_crossover, parent_pairs[i], lookup_table_dead_space_offset, explored_chromosomes) for i in range(len(parent_pairs))]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[2])
            new_population.append(f.result()[1])
            new_fitness.append(f.result()[3])
    if num_of_children % 2 != 0:
        new_population.pop()
        new_fitness.pop()
    return new_population, new_fitness

def mutate_chromosomes(population, explored_chromosomes):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=None, initializer=limit_cpu) as executor:
        num_of_mutants = math.floor((mutation_percentage * population_size / 100) + 0.5)
        probabilities = [x ** 2 for x in range(1, len(population) + 1)]
        results = [executor.submit(mutate, random.choices(population, weights=probabilities, k=1)[0], explored_chromosomes) for i in range(num_of_mutants)]
        #results = [executor.submit(mutate, population[len(population)-i-1]) for i in range(num_of_mutants)]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[1])
    return new_population, new_fitness

def generate_population(lookup_table_dead_space_offset, explored_chromosomes):
    new_population = []*population_size
    new_fitness = []*population_size
    elite_chromosomes(new_population,new_fitness)
    with concurrent.futures.ProcessPoolExecutor(max_workers=None, initializer=limit_cpu) as executor:
        result_mutate = executor.submit(mutate_chromosomes, population, explored_chromosomes)
        result_crossover = executor.submit(crossover_chromosomes, population,population_fitness,lookup_table_dead_space_offset, explored_chromosomes)
        results = [result_mutate, result_crossover]
        for f in concurrent.futures.as_completed(results):
            new_population.extend(f.result()[0])
            new_fitness.extend(f.result()[1])

    combined = list(zip(new_population, new_fitness))
    combined.sort(key=lambda pair: pair[1][0])
    # Update the original lists in place
    for i, (pop, fitness) in enumerate(combined):
        new_population[i] = pop
        new_fitness[i] = fitness
    return new_population, new_fitness


# GA parameters
population_size = 50 # Population size (number of chromosomes per generation)
population = []
population_fitness = []
survivor_percentage = 10 # Percentage of chromosomes that survive till next generation
crossover_percentage = 80 # Percentage of crossed over chromosomes
mutation_percentage = 10 # Percentage of mutated chromosomes
max_generations = 200 # Maximum number of allowed generations

selection_strategy = 'rank' # Strategy of parent selection
crossover_strategy = 'uniform' # Strategy of crossover
elitism = True # Preserve the best layout from one generation to the next
def genetic_algorithm(visualise, explored_chromosomes):
    start = time.perf_counter()
    global population
    global population_fitness
    init_population()
    best_chromosome_yet = population[0]
    best_fitness_yet = population_fitness[0]
    lookup_table_dead_space_offset_x = [x for x in range(-m, m + 1)]
    lookup_table_dead_space_offset_y = [x for x in range(-n, n + 1)]
    lookup_table_dead_space_offset = [(x, y) for x in lookup_table_dead_space_offset_x for y in lookup_table_dead_space_offset_y]
    lookup_table_dead_space_offset.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))
    optimal_objective_vs_I = []  # Optimal Objective vs iterations for plotting

    for i in range(max_generations):
        new_population, new_fitness = generate_population(lookup_table_dead_space_offset, explored_chromosomes)
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
        draw_solution(best_chromosome_yet,dead_cells,m,n)
        draw_iterations_against_solution(optimal_objective_vs_I, True)
    end = time.perf_counter()

    return best_chromosome_yet, best_fitness_yet, end-start


# Test case 1 is run by default

# Uncomment this block for test case 2
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10

# Uncomment this block for test case 3
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# survivor_percentage = 10
# crossover_percentage = 80
# mutation_percentage = 10

# Uncomment this block for test case 4
# n,m = 15,15
# dead_cells = [(2,2),(12,2),(2,12),(12,12)] # no turbines can be placed in these cells
# T_initial = 1000
# factor = 0.95
# calculate_T = calculate_T_geometric

# Uncomment this block for test case 5
# n,m = 20,20
# dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
# T_initial = 500
# factor = 1
# calculate_T = calculate_T_linear

# Uncomment this block for test case 6
# n,m = 25,25
# dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
# T_initial = 1000
# factor = 0.95
 # calculate_T = calculate_T_geometric


def multiple_genetic(num_of_times_to_run):
    with Manager() as manager:
        best_fitnesses = []
        run_time = []
        explored_chromosomes = manager.dict()
        for _ in range(num_of_times_to_run):
            _, best_fitness_yet, time_taken = genetic_algorithm(False,explored_chromosomes)
            best_fitnesses.append(best_fitness_yet[0])
            run_time.append(time_taken)
        # best_fitnesses = np.array(best_fitnesses)
        # run_time = np.array(run_time)
        # # average run time
        # average_run_time = np.mean(run_time)
        # # average best fitness
        # average_best_fitness = np.mean(best_fitnesses)
        # # standard deviation of best fitness
        # std_best_fitness = np.std(best_fitnesses)
        # # best best fitness
        # best_best_fitness = np.min(best_fitnesses)
        # # worst best fitness
        # worst_best_fitness = np.max(best_fitnesses)
        # # coefficient of variation
        # coefficient_of_variation = std_best_fitness / average_best_fitness
        # #print results
        # print(f"Average run time : {average_run_time}")
        # print(f"Average best fitness : {average_best_fitness}")
        # print(f"Standard deviation of best fitness : {std_best_fitness}")
        # print(f"Best best fitness : {best_best_fitness}")
        # print(f"Worst best fitness : {worst_best_fitness}")
        # print(f"Coefficient of variation : {coefficient_of_variation}")
        print(f"Best fitnesses : {best_fitnesses}")
        print(f"Run time : {run_time}")


if __name__ == '__main__':
    multiple_genetic(5)
