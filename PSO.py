import concurrent.futures
import math
import time
import random
from multiprocessing import Manager

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from drawings import draw_number_of_turbines_against_power_and_objective, draw_iterations_against_solution, \
    draw_solution
from functions import generate_random_tuples

matplotlib.use('TkAgg')
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, WT_list, WT_max_number, dead_cells, \
    WT_list_length

# PSO parameters
population_size = 100
w = 0.5
c1 = 0.8
c2 = 0.9
max_iterations = 100
v_max = 6


def init_particle():
    solution = generate_random_tuples(WT_list_length, dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution, n, m)
    solution = transform_to_binary(solution)
    return solution, fitness


def init_population():
    population = []
    population_fitness = []
    velocity_vector = [[0 for _ in range(m * n)] for _ in range(population_size)]
    pbest_position = []
    pbest_fitness = []
    gbest_position = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(init_particle) for _ in range(population_size)]
        for f in concurrent.futures.as_completed(results):
            population.append(f.result()[0])
            population_fitness.append(f.result()[1])
            pbest_position.append(f.result()[0])
            pbest_fitness.append(f.result()[1])
    for i in range(population_size):
        gbest_position[i] = population[min((i, (i+1) % population_size, (i-1) % population_size),
                                           key=lambda x: population_fitness[x][0])]
    return population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness


# function that transforms from list of tuples into list of 0s and 1s
def transform_to_binary(solution):
    binary_solution = []
    for i in range(m):
        for j in range(n):
            if solution[i][j] == 1:
                binary_solution.append(1)
            else:
                binary_solution.append(0)
    return binary_solution


# function that transforms from list of 0s and 1s into list of tuples
def transform_to_tuples(solution):
    tuples_solution = []
    for i in range(0, len(solution)):
        if solution[i] == 1:
            tuples_solution.append((i // n, i % m))
    return tuples_solution


def update_particle(i, population, velocity_vector, pbest_position, gbest_position, population_fitness, pbest_fitness):
    def is_valid(index):
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if population[i][index + dx + dy * n]:
                    return False
        return True
    for j in range(m * n):
        r1 = random.random()
        r2 = random.random()
        velocity_vector[i][j] = (w * velocity_vector[i][j] + c1 * r1 * (pbest_position[i][j] - population[i][j])
                                 + c2 * r2 * (gbest_position[i][j] - population[i][j]))
        velocity_vector[i][j] = np.clip(velocity_vector[i][j], -v_max, v_max)
        # apply sigmoid function on velocity vector
        velocity_normalised = 1 / (1 + math.exp(-velocity_vector[i][j]))
        if random.random() < velocity_normalised and is_valid(j):
            population[i][j] = 1
        else:
            population[i][j] = 0
    population_fitness[i] = objective_function(transform_to_tuples(population[i]), n, m)
    if population_fitness[i][0] < pbest_fitness[i]:
        pbest_position[i] = population[i].copy()
        pbest_fitness[i] = population_fitness[i][0]
    for j in range(i-1, i+1):
        j = j % population_size
        gbest_position[j] = population[
            min((j, (j + 1) % population_size, (j - 1) % population_size), key=lambda x: population_fitness[x])].copy()

def PSO():
    start = time.perf_counter()
    population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness = init_population()
    best_fitness = float('inf')
    best_population = []
    for i in range(population_size):
        if population_fitness[i][0] < best_fitness and population_fitness[i][2]:
            best_fitness = population_fitness[i][0]
            best_population = population[i].copy()
    with Manager() as manager:
        population = manager.list(population)
        population_fitness = manager.list(population_fitness)
        velocity_vector = manager.list(velocity_vector)
        pbest_position = manager.list(pbest_position)
        gbest_position = manager.list(gbest_position)
        pbest_fitness = manager.list(pbest_fitness)
        for i in range(max_iterations):
            print("iteration: ", i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(update_particle, i, population, velocity_vector, pbest_position, gbest_position, population_fitness, pbest_fitness)
                           for i in range(population_size)]
                for f in concurrent.futures.as_completed(results):
                    pass
    end = time.perf_counter()
    return best_population, best_fitness, end - start

