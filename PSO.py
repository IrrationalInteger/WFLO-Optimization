import concurrent.futures
import math
import time
import random
from multiprocessing import Manager

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from drawings import draw_number_of_turbines_against_power_and_objective, draw_iterations_against_solution, \
    draw_simulation_genetic, update_plot_genetic
from functions import generate_random_tuples

matplotlib.use('TkAgg')
from problem import spacing_distance, MAX_WT_number, objective_function, m, n, WT_list, WT_max_number, dead_cells, \
    WT_list_length

# PSO parameters
population_size = 100
w = 0.792
c1 = 1.4944
c2 = 1.4944
max_iterations = 300
neighbourhood_size = 1

def init_particle():
    solution = generate_random_tuples(int(random.uniform(1,math.ceil(m / spacing_distance + 1) * math.ceil(n / spacing_distance + 1))), dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution, n, m)
    solution = transform_to_binary(solution)
    return solution, fitness


def init_population():
    population = []
    population_fitness = []
    velocity_vector = [[random.uniform(-1.5,1.5) for _ in range(m*n)] for i in range(population_size)]
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
        neighbours = [(x+population_size) % population_size for x in range(-neighbourhood_size, neighbourhood_size+1)]
        best_fitness_index = min(neighbours, key=lambda x: population_fitness[x][0])
        gbest_position[i] = population[best_fitness_index].copy()
        gbest_fitness[i] = population_fitness[best_fitness_index][0]
    return population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness


# function that transforms from list of tuples into list of 0s and 1s
def transform_to_binary(solution):
    binary_solution = [-0.5 for _ in range(m * n)]
    for i in range(0, len(solution)):
        binary_solution[math.floor(solution[i][0]) + math.floor(solution[i][1]) * n] = 0.5
    return binary_solution


# function that transforms from list of 0s and 1s into list of tuples
def transform_to_tuples(solution):
    tuples_solution = []
    for i in range(0, len(solution)):
        if solution[i] > 0:
            tuples_solution.append(((i % m) + 0.5, (i // m) + 0.5))
    return tuples_solution

def update_particle(i, population, velocity_vector, pbest_position, gbest_position, population_fitness, pbest_fitness,
                    dead_cells_binary):
    particle = population[i]
    particle_velocity = velocity_vector[i]

    def is_valid(index):
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if 0 <= index + dx + dy * n <= m*n - 1 and particle[index + dx + dy * n] > 0:
                    return False
        return True

    print("particle: ", i)
    for j in range(m * n):
        if(dead_cells_binary[j] == 0.5) or not is_valid(j):
            continue
        r1 = random.random()
        r2 = random.random()
        #print(str(velocity_vector[i])+"sdklb")
        particle_velocity[j] = (w * particle_velocity[j] + c1 * r1 * (pbest_position[i][j] - particle[j])
                                 + c2 * r2 * (gbest_position[i][j] - particle[j]))


        # if(is_valid(j)):
        #     continue

        particle[j] += particle_velocity[j]
        if particle[j] >1.5:
            particle[j] = 1.5
        elif particle[j] <-1.5:
            particle[j] = -1.5


    empty  = all(element <= 0 for element in particle)
    particle[0] = 0.5 if empty else particle[0]

    population_fitness[i] = objective_function(transform_to_tuples(particle), n, m)
    print("fitness: ", population_fitness[i][0])
    if population_fitness[i][0] < pbest_fitness[i]:
        pbest_position[i] = particle.copy()
        pbest_fitness[i] = population_fitness[i][0]

    print("Num of Turbines for particle "+str(i), sum(1 for element in particle if element > 0))

    population[i] = particle
    velocity_vector[i] = particle_velocity
    return i


def PSO(visualise):
    start = time.perf_counter()
    population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness = init_population()
    dead_cells_binary = transform_to_binary(dead_cells)
    best_fitness = float('inf')
    best_population = []
    if(visualise):
        num_of_generations = []  # Num of generations used for plotting
        for i in range(1, max_iterations + 1):
            num_of_generations.append(i)
        ax = draw_simulation_genetic(num_of_generations)
        time.sleep(
        1)
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
        gbest_fitness = manager.list(gbest_fitness)
        for i in range(max_iterations):
            print("iteration: ", i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [
                    executor.submit(update_particle, i, population, velocity_vector, pbest_position, gbest_position,
                                    population_fitness, pbest_fitness, dead_cells_binary) for i in
                    range(population_size)]
                # wait for all results to be finished
                for f in concurrent.futures.as_completed(results):
                    print("finished: ", f.result())

            for k in range(population_size):
                neighbours = [(x + population_size) % population_size for x in
                              range(-neighbourhood_size, neighbourhood_size + 1)]
                best_fitness_index = min(neighbours, key=lambda x: population_fitness[x][0])
                gbest_position[k] = population[best_fitness_index].copy()
                gbest_fitness[k] = population_fitness[best_fitness_index][0]

            for j in range(population_size):
                if population_fitness[j][0] < best_fitness and population_fitness[j][2]:
                    best_fitness = population_fitness[j][0]
                    best_population = population[j].copy()
            print("best fitness: ", best_fitness)
            if(visualise):
                fitness_values = []
                for fitness in population_fitness:
                    fitness_values.append(fitness[0])
                myMax, myMin = update_plot_genetic(ax, i, fitness_values, None if i == 0 else myMax, None if i == 0 else myMin)


    end = time.perf_counter()
    return best_population, best_fitness, end - start


if __name__ == '__main__':
    bpopulation, bfitness, time = PSO(True)
    print("best population: ", bpopulation)
    print("best fitness: ", bfitness)
    print("time: ", time)
