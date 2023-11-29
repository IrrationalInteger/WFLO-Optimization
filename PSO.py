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
population_size = 50
w = 0.792
c1 = 1.494
c2 = 1.494
max_iterations = 100
neighbourhood_size = 10

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

def init_particle():
    #solution = generate_random_tuples(int(random.uniform(1, math.ceil(m / spacing_distance + 1) * math.ceil(n / spacing_distance + 1))), dead_cells, m, n, spacing_distance)
    solution = generate_random_tuples(random.randint(1, 5), dead_cells, m, n, spacing_distance)
    fitness = objective_function(solution, n, m)
    solution.sort(key=lambda x: (x[0], x[1]))
    return solution, fitness


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
        neighbours = [(x+population_size) % population_size for x in range(-neighbourhood_size, neighbourhood_size+1)]
        best_fitness_index = min(neighbours, key=lambda x: population_fitness[x][0])
        gbest_position[i] = population[best_fitness_index].copy()
        gbest_fitness[i] = population_fitness[best_fitness_index][0]
    return population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness


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


def add_velocity(list1, list2):
    result = []
    list1_new = [list1[i] for i in range(len(list1)) if list1[i][0] != '+' and list1[i][0] != '-']
    list2_new = [list2[i] for i in range(len(list2)) if list2[i][0] != '+' and list2[i][0] != '-']
    # Find the minimum length of the two lists
    min_length = min(len(list1_new), len(list2_new))

    # Perform element-wise subtraction up to the minimum length
    for i in range(min_length):
        tuple1 = list1_new[i]
        tuple2 = list2_new[i]
        result.append((tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))

    # append the remaining tuples in the first list
    for i in range(min_length, len(list1_new)):
        result.append(list1_new[i])
    for i in range(min_length, len(list2_new)):
        result.append(list2_new[i])

    for i in range(len(list1)):
        if list1[i][0] == '+' or list1[i][0] == '-':
            result.append(list1[i])

    for i in range(len(list2)):
        if list2[i][0] == '+' or list2[i][0] == '-':
            result.append(list2[i])

    return result


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

    # Check if scalar is less than 0
    if scalar < 1:
        # Delete half of the tuples randomly
        deleted_indices = random.sample(range(len(result)), k=len(result)-int(len(result) * scalar))
        # print(deleted_indices)
        # print(result)
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
        # print(result)
        # print("deleted")

    # Check if scalar is greater than 0
    elif scalar > 1:
        # Append a number of tuples of the type (0, '+') equal to half the length of the list
        for _ in range(int(len(result) * (scalar - 1))):
            result.append(('+', 0))

    return result


def update_particle(i, population, velocity_vector, pbest_position, gbest_position, population_fitness, pbest_fitness,
                    lookup_table_dead_space_offset):
    particle = population[i]
    error_cells = []
    def calculate_error_cells(x, y,j):
        particle.pop(j)
        error_cells_new = []
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx, y + dy) in particle:
                    # print("error cell: ", (x + dx, y + dy))
                    # print(dx, dy)
                    error_cells_new.append((x + dx, y + dy))
        #remove error cells from particle
        # print("error cells: ", error_cells_new)
        for error_cell in error_cells_new:
            # print("error cell: ", cell)
            # print("particle: ", particle)
            # print(error_cells)
            particle[particle.index(error_cell)] = (-100, -100)
            error_cells.append(error_cell)
        #add (x,y) to its position in j
        particle.insert(j, (x, y))
        return error_cells
    def is_valid(x, y):
        for dx in list(range(-spacing_distance, spacing_distance + 1)):
            for dy in list(range(-spacing_distance, spacing_distance + 1)):
                if (x + dx, y + dy) in particle:
                    return False
        return True
    print("particle: ", i)
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    # velocity_vector[i] = add_velocity(add_velocity(multiply_velocity(velocity_vector[i], w), multiply_velocity(
    #     subtract_solutions(pbest_position[i], particle), c1 * r1)), multiply_velocity(
    #     subtract_solutions(gbest_position[i], particle), c2 * r2))
    # Assuming the functions add_velocity, multiply_velocity, and subtract_solutions are defined

    # Step 1
    step1_result = multiply_velocity(velocity_vector[i], w)
    # print("Step 1 Result (multiply_velocity):", step1_result)
    # print(f"   Parameters: {velocity_vector[i]}, {w}")

    # Step 2
    step2_result = subtract_solutions(pbest_position[i], particle)
    # print("Step 2 Result (subtract_solutions):", step2_result)
    # print(f"   Parameters: {pbest_position[i]}, {particle}")

    # Step 3
    step3_result = multiply_velocity(step2_result, c1 * r1)
    # print("Step 3 Result (multiply_velocity):", step3_result)
    # print(f"   Parameters: {step2_result}, {c1 * r1}")

    # Step 4
    step4_result = add_velocity(step1_result, step3_result)
    # print("Step 4 Result (add_velocity):", step4_result)
    # print(f"   Parameters: {step1_result}, {step3_result}")

    # Step 5
    step5_result = subtract_solutions(gbest_position[i], particle)
    # print("Step 5 Result (subtract_solutions):", step5_result)
    # print(f"   Parameters: {gbest_position[i]}, {particle}")

    # Step 6
    step6_result = multiply_velocity(step5_result, c2 * r2)
    # print("Step 6 Result (multiply_velocity):", step6_result)
    # print(f"   Parameters: {step5_result}, {c2 * r2}")

    # Final step
    final_result = add_velocity(step4_result, step6_result)
    velocity_vector[i] = final_result
    # print("Final Result (add_velocity):", final_result)
    # print(f"   Parameters: {step4_result}, {step6_result}")
    # print("Updated velocity_vector[{}]: {}".format(i, velocity_vector[i]))

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
                calculate_error_cells(new_wt[0], new_wt[1], len(particle)-1)
            continue
        if velocity_vector[i][j][0] == '-' and velocity_vector[i][j][1] == 0:
            particle[random.randint(0, len(particle) - 1)] = (-100, -100)
            continue
        if velocity_vector[i][j][0] == '-':
            if velocity_vector[i][j][1] in particle:
                particle.remove(velocity_vector[i][j][1])
            continue
        # if j >= len(particle): # fuck
        #     continue
        if particle[j][0] == -100 and particle[j][1] == -100:
            continue
        # print(particle)
        # print(velocity_vector[i])
        # print(velocity_vector[i][j])
        # print(f'j;{j}')
        particle[j] = (((particle[j][0] + velocity_vector[i][j][0] + m - 0.5) % m)+0.5,
                       ((particle[j][1] + velocity_vector[i][j][1] + n - 0.5) % n)+0.5)
        if (int(particle[j][0]), int(particle[j][1])) in dead_cells:
            particle[j] = (-100, -100)
            error_cells.append(particle[j])
            continue
        calculate_error_cells(particle[j][0], particle[j][1], j)
    particle = [particle[j] for j in range(len(particle)) if particle[j][0] != -100]
    for cell in error_cells:
        for (dx, dy) in lookup_table_dead_space_offset:
            if m > math.floor(cell[0] + dx) >= 0 and n > math.floor(cell[1] + dy) >= 0:
                if is_valid(cell[0]+dx, cell[1]+dy):
                    particle.append((cell[0] + dx, cell[1] + dy))
                    break
    if len(particle) == 0:
        add_new_WT(particle, dead_cells, m, n)
    population_fitness[i] = objective_function(particle, n, m)
    length = 0
    for l in range(len(velocity_vector[i])):
        if velocity_vector[i][l][0] == '+' or velocity_vector[i][l][0] == '-':
            break
        length += 1
    velocity_vector_values = velocity_vector[i][:length]
    if len(velocity_vector_values) > len(particle):
        velocity_vector_values = velocity_vector_values[:len(particle)]
    velocity_vector_symbols = velocity_vector[i][length:]
    velocity_vector_symbols = [(velocity_vector_symbols[j][0], 0) for j in range(len(velocity_vector_symbols))]
    velocity_vector[i] = velocity_vector_values + velocity_vector_symbols
    print("fitness: ", population_fitness[i][0])
    if population_fitness[i][0] < pbest_fitness[i]:
        pbest_position[i] = particle.copy()
        pbest_fitness[i] = population_fitness[i][0]

    print("Num of Turbines for particle "+str(i), len(particle))
    particle.sort(key=lambda x: (x[0], x[1]))
    population[i] = particle
    return i


def PSO(visualise):
    start = time.perf_counter()
    population, population_fitness, velocity_vector, pbest_position, gbest_position, pbest_fitness, gbest_fitness = init_population()
    best_fitness = float('inf')
    best_population = []
    lookup_table_dead_space_offset_x = [x for x in range(-m, m + 1)]
    lookup_table_dead_space_offset_y = [x for x in range(-n, n + 1)]
    lookup_table_dead_space_offset = [(x, y) for x in lookup_table_dead_space_offset_x for y in
                                      lookup_table_dead_space_offset_y]
    lookup_table_dead_space_offset.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))
    if visualise:
        num_of_generations = []  # Num of generations used for plotting
        for i in range(1, max_iterations + 1):
            num_of_generations.append(i)
        ax = draw_simulation_genetic(num_of_generations)
        time.sleep(1)
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
                                    population_fitness, pbest_fitness, lookup_table_dead_space_offset) for i in
                    range(population_size)]
                # wait for all results to be finished
                for f in concurrent.futures.as_completed(results):
                    print("finished: ", f.result())
            # for j in range(population_size):
            #     #print("iteration: ", j)
            #     update_particle(j, population, velocity_vector, pbest_position, gbest_position,
            #                     population_fitness, pbest_fitness, lookup_table_dead_space_offset)

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
