
# Adds a new turbine while respecting the spacing distance and dead cells
import concurrent.futures
import math
import time
import random
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from functions import generate_random_tuples

matplotlib.use('TkAgg')
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, WT_list, WT_max_number, dead_cells, \
    WT_list_length


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
def generate_neighbour_solution(solution, exclusion_list, m , n):
  op = random.randint(1, 2) if len(solution) == MAX_WT_number else random.randint(0, 1) if len(solution)==1 else random.randint(0, 2)
  solution = solution.copy()
  if op==0: # op = 0, add a WT at random location
    add_new_WT(solution, exclusion_list, m , n)
  elif op==1: # op = 1, change the location of one WT ????
    solution.pop(random.randint(0,len(solution)-1))
    add_new_WT(solution, exclusion_list, m , n)
  else: # op = 2, remove a random WT
    solution.pop(random.randint(0,len(solution)-1))
  return solution


# GA parameters
population_size = 30 # Population size (number of chromosomes per generation)
population = []
population_fitness = []
survivor_percentage = 20 # Percentage of chromosomes that survive till next generation
crossover_percentage = 70 # Percentage of crossed over chromosomes
mutation_percentage = 10 # Percentage of mutated chromosomes
max_generations = 500 # Maximum number of allowed generations
selection_strategy = 'rank' # Strategy of parent selection
crossover_strategy = 'uniform' # Strategy of crossover
elitism = True # Preserve the best layout from one generation to the next


dead_space_list = [x for x in range(-spacing_distance, spacing_distance+1)]
#dead_space_list = [(x, y) for x in dead_space_list for y in dead_space_list]
#dead_space_list = sorted(dead_space_list, key=lambda x: math.sqrt(abs(x[0])**2 + abs(x[1])**2))

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
        num_of_elites = math.floor((survivor_percentage*population_size/100)+0.5)
        for i in range(num_of_elites):
            new_population.append(population[i])
            new_fitness.append(population_fitness[i])
    else:
        #later REMEMBER DONT DO THIS
        pass


def mutate(chromosome):
    chromosome = chromosome.copy()
    num_of_genes = max(4,math.floor((0.25*len(chromosome)) + 0.5))  # Mutate a fourth of the genes
    for _ in range(num_of_genes):
        chromosome = generate_neighbour_solution(chromosome,dead_cells,m,n)
    fitness = objective_function(chromosome, m, n)
    return chromosome,fitness


def uniform_crossover(parents_pair, lookup_table_dead_space_offset):
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

    return child1,child2,objective_function(child1,m,n),objective_function(child2,m,n)


def one_point_crossover(parents_pair, lookup_table_dead_space_offset):
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
    return child1,child2,objective_function(child1,m,n),objective_function(child2,m,n)


def rank_selection(num_of_parents, population, population_fitness):
    if num_of_parents % 2 != 0:
        num_of_parents += 1
    # Calculate the weights
    probabilities = [x for x in range(1, len(population) + 1)]
    probabilities.reverse()

    print(probabilities)
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
        parent_pairs.append((parent1, parent2))

    return parent_pairs
def crossover_chromosomes(population, population_fitness,lookup_table_dead_space_offset):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        num_of_children = math.floor((crossover_percentage * population_size / 100) + 0.5)
        parent_pairs = rank_selection(num_of_children, population, population_fitness)
        results = [executor.submit(one_point_crossover, parent_pairs[i],lookup_table_dead_space_offset) for i in range(len(parent_pairs))]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[2])
            new_population.append(f.result()[1])
            new_fitness.append(f.result()[3])
    if num_of_children % 2 != 0:
        new_population.pop()
        new_fitness.pop()
    return new_population, new_fitness

def mutate_chromosomes(population):
    new_population = []
    new_fitness = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        num_of_mutants = math.floor((mutation_percentage * population_size / 100) + 0.5)
        #probabilities = [x ** 2 for x in range(1, len(population) + 1)]
        #results = [executor.submit(mutate, random.choices(population, weights=probabilities, k=1)[0]) for i in range(num_of_mutants)]
        results = [executor.submit(mutate, population[len(population)-i-1]) for i in range(num_of_mutants)]
        for f in concurrent.futures.as_completed(results):
            new_population.append(f.result()[0])
            new_fitness.append(f.result()[1])
    return new_population, new_fitness

def generate_population(lookup_table_dead_space_offset):
    new_population = []*population_size
    new_fitness = []*population_size
    elite_chromosomes(new_population,new_fitness)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_mutate = executor.submit(mutate_chromosomes, population)
        result_crossover = executor.submit(crossover_chromosomes, population,population_fitness,lookup_table_dead_space_offset)
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


def genetic_algorithm():
    global population
    global population_fitness
    init_population()
    lookup_table_dead_space_offset_x = [x for x in range(-m, m + 1)]
    lookup_table_dead_space_offset_y = [x for x in range(-n, n + 1)]
    lookup_table_dead_space_offset = [(x, y) for x in lookup_table_dead_space_offset_x for y in lookup_table_dead_space_offset_y]
    lookup_table_dead_space_offset.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))
    for i in range(max_generations):
        new_population, new_fitness = generate_population(lookup_table_dead_space_offset)
        population = new_population
        population_fitness = new_fitness
        print(f'Generation: {i}')
        print(population_fitness)
    return population[0], population_fitness[0]

if __name__ == '__main__':
    best_population, best_fitness = genetic_algorithm()
    print(best_population)
    print(best_fitness)
