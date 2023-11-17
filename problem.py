

import numpy as np
import math
import random
import matplotlib
from functions import generate_random_tuples, calculate_cone_grid
import matplotlib.pyplot as plt
import time


#constraints
n, m = 25,25 # grid size n*m
dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)] # no turbines can be placed in these cells
spacing_distance = 3
WT_max_number = math.ceil(m/spacing_distance+1)*math.ceil(n/spacing_distance+1) # user_defined
MAX_WT_number = math.ceil(m/spacing_distance+1)*math.ceil(n/spacing_distance+1) # defined by the grid size and spacing constraints

assert WT_max_number <= MAX_WT_number


# constants
POWER_COEFFICIENT = 0.3
POWER_THRESHOLD_COEFFICIENT = 0.927
ALPHA = 0.1
# variables
wind_speed = 12
wind_frequency=[1/36]*36

#decision variables
WT_list_length = 1 # or WT_max_number/2
WT_list = generate_random_tuples(WT_list_length,dead_cells, m, n, spacing_distance) # example : WT_list = [(7.5, 7.5), (9.5, 3.5), (9.5, 9.5), (6.5, 5.5), (4.5, 2.5), (3.5, 0.5), (7.5, 3.5), (5.5, 7.5), (0.5, 2.5), (4.5, 5.5), (8.5, 5.5)]
#print(f"WT_coordinates : {WT_list}")





# checks if the power generated in a specific wind direction is above the power threshold constraint (????????)
def satisfies_power_constraint(power_frequency, total_power):
  power_threshold = total_power * POWER_THRESHOLD_COEFFICIENT
  return all(power > power_threshold for power in power_frequency)

# calculates power of a wind turbine in the existence of wake. It first calculates the reduced wind speed due to the wake effect
def calculate_power(wind_speed, wind_speed_wake, distance, rotor_radius=25):
  speed = wind_speed
  speed *= (1-((1-((0.3*wind_speed_wake)/wind_speed))*((rotor_radius/(rotor_radius+(ALPHA*distance)))**2)))
  power = POWER_COEFFICIENT * (speed**3)
  return power,speed

# calculates power generated by a certain wind turbine, and reduced speed wind.
def calculate_power_cell(cone_grid,power_grid,WT_coordinate):
  WT_x, WT_y = WT_coordinate
  WT_x = int(WT_x-0.5)
  WT_y = int(WT_y-0.5)
  if(cone_grid[WT_x][WT_y] == None):
    power_grid[WT_x][WT_y] = (POWER_COEFFICIENT * (wind_speed**3),wind_speed)
  else:
    WT_wake_x, WT_wake_y = cone_grid[WT_x][WT_y][0]
    WT_wake_x = int(WT_wake_x-0.5)
    WT_wake_y = int(WT_wake_y-0.5)
    distance = math.dist([WT_x, WT_y],[WT_wake_x, WT_wake_y])
    if power_grid[WT_wake_x][WT_wake_y] == None:
      calculate_power_cell(cone_grid,power_grid, cone_grid[WT_x][WT_y][0])
    power_grid[WT_x][WT_y] = calculate_power(wind_speed=wind_speed,wind_speed_wake=power_grid[WT_wake_x][WT_wake_y][1],distance=distance*100)

# populates power_grid by power generated by each wind turbine and the reduced wind speed due to wake effect
def calculate_power_grid(cone_grid,WT_coordinates):
  power_grid = np.empty(cone_grid.shape, dtype = object)
  for WT in WT_coordinates:
    if(power_grid[int(WT[0]-0.5)][int(WT[1]-0.5)] != None):
      continue
    calculate_power_cell(cone_grid,power_grid,WT)
  return power_grid

# calculates the total power of all wind turbines for a certain wind direction
def calculate_total_power(cone_grid,WT_coordinates):
  power_grid = calculate_power_grid(cone_grid,WT_coordinates)
  total_power = 0
  for WT in WT_coordinates:
    WT_x,WT_y = WT
    WT_x = int(WT_x-0.5)
    WT_y = int(WT_y-0.5)
    total_power += power_grid[WT_x][WT_y][0]
  return total_power

# calculate the operational and maintainable cost of the wind farm
def calculate_WT_cost(WT_number):
  exponent = -0.00174 * (WT_number**2)
  result = np.exp([exponent])[0]
  cost = WT_number * (2/3 + (1/3 * result))
  return cost

# calculates the average total power generated by the wind farm. It calculates the cone grid for each wind direction and use it to calculate the power grid.
# It then calculates the weighted average of total power using wind_frequency (wind direction frequency)
def calculate_average_total_power(WT_coordinates,grid_x,grid_y):
  power_frequency=[0]*len(wind_frequency)
  for idx in range(len(wind_frequency)):
    #calculate cone_grid
    cone_grid = calculate_cone_grid(WT_coordinates, 7, idx * 10, grid_x,grid_y)
    power_frequency[idx] = calculate_total_power(cone_grid,WT_coordinates)
  average_total_power = sum(power_frequency * wind_frequency for power_frequency, wind_frequency in zip(power_frequency, wind_frequency))
  satisfies = satisfies_power_constraint(power_frequency, average_total_power)
  if(satisfies):
    print("The solution satisfies the power constraint")
  else:
    print("The solution does not satisfy the power constraint")
  return average_total_power,satisfies

# calculates fitness value of the solution
def objective_function(WT_coordinates,grid_x,grid_y):
  average_total_power,satisfies = calculate_average_total_power(WT_coordinates,grid_x,grid_y)
  print(f"Average total power : {average_total_power}")
  print(f"Average total power with no wake : {POWER_COEFFICIENT * len(WT_coordinates) * (wind_speed**3)}")
  total_cost = calculate_WT_cost(len(WT_coordinates))
  print(f"Total cost : {total_cost}")
  fitness_value = total_cost / average_total_power
  return fitness_value,average_total_power,satisfies


if __name__ == "__main__":
  start = time.perf_counter()
  solution = [(19.5, 13.5), (19.5, 9.5), (19.5, 19.5), (18.5, 0.5), (19.5, 5.5),
                 (0.5, 8.5), (0.5, 4.5), (0.5, 0.5), (0.5, 19.5), (0.5, 15.5), (8.5, 15.5), (15.5, 15.5),
                 (11.5, 11.5), (15.5, 9.5), (8.5, 7.5), (14.5, 4.5), (15.5, 19.5), (11.5, 19.5), (4.5, 11.5),
                 (4.5, 15.5), (4.5, 19.5), (4.5, 4.5), (10.5, 0.5), (14.5, 0.5), (6.5, 0.5)]
  print(objective_function(solution,m,n))
  end = time.perf_counter()
  print(f"Time taken : {end-start}")