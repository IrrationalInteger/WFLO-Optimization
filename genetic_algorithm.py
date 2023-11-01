
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
  #op = 0
  #print(op)
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
population_size = 50 # Population size (number of chromosomes per generation)
population = []
population_fitness = []
survivor_percentage = 10 # Percentage of chromosomes that survive till next generation
crossover_percentage = 60 # Percentage of crossed over chromosomes
mutation_percentage = 30 # Percentage of mutated chromosomes
max_generations = 200 # Maximum number of allowed generations
selection_strategy = 'tour' # Strategy of parent selection
crossover_strategy = 'uniform' # Strategy of crossover
elitism = True # Preserve the best layout from one generation to the next

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

    combined.sort(key=lambda pair: pair[1],reverse=True)

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

def crossover_chromosomes(new_population,new_fitness):

def mutate_chromosome(new_population,new_fitness):



def generate_population():
    new_population = []*population_size
    new_fitness = []*population_size


    with concurrent.futures.ProcessPoolExecutor() as executor:




if __name__ == '__main__':
    init_population()
    print(population)
    print(population_fitness)

def genetic_algorithm():
    pass
